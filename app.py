import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from autogluon.tabular import TabularPredictor
from sklearn.linear_model import LinearRegression
import altair as alt
import sys
import subprocess
import datetime
import os

# --- Config ---
st.set_page_config(
    page_title="花姑娘2.0 AI 投顾",
    page_icon="🌸",
    layout="wide"
)

# --- Authentication Gate ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("# 🌸 欢迎进入 AI 投顾系统")
        st.write("---")
        st.write("### 🔐 访问验证")
        answer = st.text_input("请输入通关口令以继续：", type="password", placeholder="请输入答案...")
        if st.button("立即解锁", use_container_width=True):
            if answer == "777":
                st.session_state.authenticated = True
                st.success("验证通过！正在为您加载系统...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("口令错误，无法进入系统。")
    st.stop()

# 获取当前脚本所在目录的绝对路径，确保模型加载不受运行环境影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Available Model Versions
MODEL_VERSIONS = {
    "最强王者": os.path.join(BASE_DIR, "AutogluonModels/ag-20260122_050556"),
    "进化失败": os.path.join(BASE_DIR, "AutogluonModels/ag-20260126_044254"),
    "绩效优化版未来10日": os.path.join(BASE_DIR, "AutogluonModels/performance_v1")
}

# Initialize session state for navigation and settings
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'selected_version' not in st.session_state or st.session_state.selected_version not in MODEL_VERSIONS:
    st.session_state.selected_version = list(MODEL_VERSIONS.keys())[0]

def navigate_to(page):
    st.session_state.page = page

MODEL_PATH = MODEL_VERSIONS[st.session_state.selected_version]
DATA_DIR = os.path.join(BASE_DIR, 'market_data')
WINDOWS = [3, 5, 10, 20, 23, 30, 60, 120]

NAME_MAP = {
    '513100.SH': '纳指100',
    '513520.SH': '日经ETF',
    '513500.SH': '标普500',
    '159915.SZ': '创业板',
    '588120.SH': '科创板',
    '588000.SH': '科创板', 
    '510180.SH': '上证180',
    '518880.SH': '黄金ETF',
    '511090.SH': '30年国债',
    '161129.SZ': '南方原油',
    '501018.SH': '南方原油'
}
VALID_ASSETS = list(set(NAME_MAP.values()))

# --- Helper Functions ---

@st.cache_data(ttl=3600)  # Add TTL to auto-refresh cache
def load_market_data():
    data = {}
    if not os.path.exists(DATA_DIR): return {}
    
    # Sort filenames to ensure deterministic loading order across platforms
    filenames = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    
    for filename in filenames:
        code = filename.split('_')[0]
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, filename))
            df['date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('date').reset_index(drop=True)
            if 'close_qfq' in df.columns: df['close'] = df['close_qfq']
            if 'open_qfq' in df.columns: df['open'] = df['open_qfq']
            if 'vol' in df.columns: df['volume'] = df['vol']
            
            name = None
            for k, v in NAME_MAP.items():
                if k in filename or k == code:
                    name = v
                    break
            if name:
                df['name'] = name
                df['code'] = code
                
                # 如果同一个资产名称对应多个文件（如 588120 和 588000 都是科创板），
                # 我们需要一个确定的逻辑来选择，避免在 Streamlit 上加载了不同的文件。
                # 逻辑：优先选择代码在 NAME_MAP 中靠前的，或者数据量更多的。
                if name in data:
                    # 如果当前文件的代码在映射表中更早出现，或者数据更长，则替换
                    existing_code = data[name]['code'].iloc[0]
                    if len(df) > len(data[name]):
                        data[name] = df
                else:
                    data[name] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return data

@st.cache_resource(show_spinner=False)
def load_model(path=None):
    target_path = path if path else MODEL_PATH
    predictor_file = os.path.join(target_path, 'predictor.pkl')
    models_dir = os.path.join(target_path, 'models')
    
    if not os.path.exists(predictor_file):
        return None
    
    # 额外检查 models 目录是否存在，防止上传 GitHub 时漏掉子目录
    if not os.path.exists(models_dir):
        st.error(f"⚠️ 核心模型目录缺失: {models_dir}。请检查 GitHub 仓库是否完整上传了 models 文件夹。")
        return None
        
    try:
        return TabularPredictor.load(target_path, require_version_match=False)
    except Exception as e:
        # 在 Streamlit UI 中显示错误，方便排查
        st.error(f"模型文件加载失败 ({target_path}): {e}")
        return None

@st.cache_data
def calc_max_drawdown(prices):
    """Calculate Maximum Drawdown for a price series."""
    if len(prices) < 1: return 0.0
    # Calculate cumulative max
    roll_max = np.maximum.accumulate(prices)
    # Avoid division by zero
    if roll_max[0] == 0: return 0.0
    # Calculate drawdown
    drawdown = (prices - roll_max) / roll_max
    # Return max drawdown (min value, since dd is negative)
    return drawdown.min()

def calculate_indicators(window_close, window_returns):
    """统一的特征计算逻辑，确保回测与单日决策完全一致"""
    if len(window_close) < 2:
        return {f: 0.0 for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']}
    
    # 收益率：当前价格 / 窗口起始价格 - 1
    ret = (window_close[-1] / window_close[0]) - 1
    # 波动率：收益率序列的标准差 * sqrt(252)
    vol = np.std(window_returns) * np.sqrt(252)
    # 趋势与拟合度
    slope = calc_slope(window_close)
    r2 = calc_r2(window_close)
    # 最大回撤
    mdd = calc_max_drawdown(window_close)
    # 复合指标
    sxr = slope * r2
    sharp = slope / (vol + 0.01)
    
    return {
        'ret': ret,
        'vol': vol,
        'slope': slope,
        'r2': r2,
        'mdd': mdd,
        'sxr': sxr,
        'sharp': sharp
    }

def prepare_all_features_cached(data_dict, windows, warmup=True, start_date=None):
    """
    预计算所有特征。
    """
    all_dates = set()
    for df in data_dict.values():
        all_dates.update(df['date'].tolist())
    sorted_dates = sorted(list(all_dates))
    
    processed_data = {} 
    
    for name, df in data_dict.items():
        sub = df.set_index('date').sort_index()
        df_feat = sub.copy()
        
        # 预计算每日收益率（用于波动率计算）
        df_feat['daily_ret'] = df_feat['close'].pct_change().fillna(0.0)
        
        close_vals = df_feat['close'].values
        ret_vals = df_feat['daily_ret'].values
        
        for w in windows:
            # 初始化列表
            feats = {f: [np.nan]*len(df_feat) for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']}
            
            for i in range(len(df_feat)):
                if i >= w - 1:
                    win_close = close_vals[i-w+1 : i+1]
                    win_ret = ret_vals[i-w+1 : i+1]
                    res = calculate_indicators(win_close, win_ret)
                    for k, v in res.items():
                        feats[k][i] = v
            
            # 写入 DataFrame
            for k, v in feats.items():
                df_feat[f'{k}_{w}'] = v
             
        processed_data[name] = df_feat
        
    return processed_data, sorted_dates

def run_backtest_range(predictor, data_dict, start_date, end_date, model_name, initial_holding=None, force_neutral=False, use_warmup=True):
    # 1. 预计算特征
    with st.spinner("正在预计算全量特征..."):
        s_str = str(start_date) if not use_warmup else None
        processed_data, all_dates = prepare_all_features_cached(data_dict, WINDOWS, warmup=use_warmup, start_date=s_str)
    
    # 过滤日期
    s_ts = pd.Timestamp(start_date)
    e_ts = pd.Timestamp(end_date)
    sim_dates = [d for d in all_dates if d >= s_ts and d <= e_ts]
    
    if not sim_dates:
        return None, "所选范围内没有交易日。"
        
    history = []
    current_holding = initial_holding
    
    # 找到第一个模拟日的前一个交易日索引
    # 我们需要 T-1 日的特征来决定 T 日的持仓
    progress_bar = st.progress(0)
    
    for i, d in enumerate(sim_dates):
        progress_bar.progress((i + 1) / len(sim_dates))
        
        # --- 核心逻辑：使用 T-1 日的数据决定 T 日持仓 ---
        # 1. 找到 d 日在 all_dates 中的索引
        d_idx = all_dates.index(d)
        if d_idx == 0:
            # 第一天没有 T-1，保持初始持仓
            top_pick = current_holding if current_holding else '现金'
            top_score = 1.0
        else:
            prev_d = all_dates[d_idx - 1] # T-1 日
            
            # 构建 T-1 日的特征矩阵
            daily_rows = []
            effective_holding = None if force_neutral else current_holding
            
            for name, df in processed_data.items():
                if prev_d in df.index:
                    row = df.loc[prev_d]
                    if pd.notnull(row['slope_23']):
                        feat = {'name': name, 'is_held': 1 if effective_holding == name else 0}
                        for w in WINDOWS:
                            for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']:
                                feat[f'{f}_{w}'] = row[f'{f}_{w}']
                        daily_rows.append(feat)
            
            # 现金资产特征
            cash_feat = {'name': '现金', 'is_held': 1 if effective_holding == '现金' else 0}
            for w in WINDOWS:
                for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']:
                    cash_feat[f'{f}_{w}'] = 0.0
            daily_rows.append(cash_feat)
            
            df_day = pd.DataFrame(daily_rows)
            # 计算排名
            for w in WINDOWS:
                for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']:
                    col = f'{f}_{w}'
                    df_day[f'rank_{col}'] = df_day[col].rank(pct=True)
            
            # 市场上下文
            non_cash = df_day[df_day['name'] != '现金']
            df_day['market_max_slope'] = non_cash['slope_23'].max() if not non_cash.empty else 0
            df_day['market_max_ret'] = non_cash['ret_23'].max() if not non_cash.empty else 0
            
            # 预测
            try:
                probs = predictor.predict_proba(df_day, model=model_name)
                score_col = 1 if 1 in probs.columns else probs.columns[-1]
                df_day['score'] = probs[score_col]
                df_day = df_day.sort_values('score', ascending=False)
                top_pick = df_day.iloc[0]['name']
                top_score = df_day.iloc[0]['score']
            except Exception as e:
                st.error(f"预测失败: {e}")
                st.stop()
        
        # --- 计算 T 日收益 ---
        # 逻辑：T-1日产生信号，T日开盘执行。
        # 如果发生调仓 (top_pick != current_holding)：使用 T日开盘价买入，计算 T日(收盘/开盘-1) 的收益。
        # 如果不调仓 (top_pick == current_holding)：计算 T日(收盘/昨收-1) 的全天收益。
        
        daily_ret = 0.0
        
        if top_pick != current_holding:
            # 调仓日：计算新持仓的日内收益 (Close/Open - 1)
            if top_pick != '现金' and top_pick in processed_data:
                asset_df = processed_data[top_pick]
                if d in asset_df.index:
                    row = asset_df.loc[d]
                    c = row['close']
                    o = row.get('open', c)
                    if o != 0:
                        daily_ret = (c / o) - 1
        else:
            # 持仓不变日：计算原持仓的全天收益 (Close/PrevClose - 1)
            if current_holding and current_holding != '现金':
                if current_holding in processed_data and d in processed_data[current_holding].index:
                    daily_ret = processed_data[current_holding].loc[d].get('daily_ret', 0.0)
        
        history.append({
            'date': d.date(),
            'holding': top_pick,
            'prev_holding': current_holding if current_holding else "空仓(初始)",
            'score': top_score,
            'action': 'Switch' if top_pick != current_holding else 'Hold',
            'daily_ret': daily_ret
        })
        
        # 更新状态：T 日收盘后的持仓变为 top_pick
        current_holding = top_pick
        
    return pd.DataFrame(history), None

def update_data_process():
    """Run update_data.py as a subprocess"""
    try:
        # Pass current environment + secrets to subprocess
        env = os.environ.copy()
        
        # Try to get Token from Streamlit Secrets
        try:
            if 'TS_TOKEN' in st.secrets:
                env['TS_TOKEN'] = st.secrets['TS_TOKEN']
        except:
            pass # Ignore if secrets not available (local dev)
            
        result = subprocess.run([sys.executable, 'update_data.py'], capture_output=True, text=True, env=env)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def calc_slope(y):
    # Safe check for NaN
    if len(y) < 2 or np.isnan(y).any(): return 0
    n = len(y)
    x = np.arange(n).reshape(-1, 1)
    
    # Avoid division by zero if y[0] is 0
    if y[0] == 0: return 0
    
    y_norm = y / y[0]
    model = LinearRegression().fit(x, y_norm)
    return model.coef_[0]

def calc_r2(y):
    if len(y) < 2 or np.isnan(y).any(): return 0
    n = len(y)
    x = np.arange(n).reshape(-1, 1)
    
    if y[0] == 0: return 0
    
    y_norm = y / y[0]
    model = LinearRegression().fit(x, y_norm)
    return model.score(x, y_norm)

def prepare_daily_features(data_dict, current_holding, target_date=None):
    all_dates = []
    for df in data_dict.values():
        all_dates.extend(df['date'].tolist())
    
    if not all_dates:
        return None, None, "No Data"
    
    # Filter dates
    unique_dates = sorted(list(set(all_dates)))
    
    if target_date is None:
        latest_date = unique_dates[-1]
    else:
        # Find closest date <= target_date
        target_ts = pd.Timestamp(target_date)
        valid_dates = [d for d in unique_dates if d <= target_ts]
        if not valid_dates:
            return None, None, "No data available before selected date"
        latest_date = valid_dates[-1]
    
    daily_snapshot = []
    
    # 1. Real Assets
    for name, df in data_dict.items():
        sub = df.set_index('date').sort_index()
        
        if latest_date in sub.index:
            idx = sub.index.get_loc(latest_date)
            if idx < 30: continue 
            
            # Logic for is_held
            is_held_val = 1 if current_holding == name else 0
            
            sample = {
                'name': name,
                'is_held': is_held_val
            }
            
            close_vals = sub['close'].values
            # 我们还需要 daily_ret 序列来计算波动率
            daily_rets = sub['close'].pct_change().fillna(0.0).values
            
            for w in WINDOWS:
                window_data = close_vals[idx-w+1 : idx+1]
                window_rets = daily_rets[idx-w+1 : idx+1]
                
                res = calculate_indicators(window_data, window_rets)
                for k, v in res.items():
                    sample[f'{k}_{w}'] = v
                
            daily_snapshot.append(sample)
            
    # 2. Cash Asset
    cash_is_held = 1 if current_holding == '现金' else 0
    cash_sample = {
        'name': '现金',
        'is_held': cash_is_held
    }
    for w in WINDOWS:
        cash_sample[f'ret_{w}'] = 0.0
        cash_sample[f'vol_{w}'] = 0.0
        cash_sample[f'slope_{w}'] = 0.0
        cash_sample[f'r2_{w}'] = 0.0
        cash_sample[f'mdd_{w}'] = 0.0
        cash_sample[f'sxr_{w}'] = 0.0
        cash_sample[f'sharp_{w}'] = 0.0
    daily_snapshot.append(cash_sample)
    
    # 3. Ranking Features
    df_day = pd.DataFrame(daily_snapshot)
    
    feature_cols = []
    for w in WINDOWS:
        feature_cols.extend([f'ret_{w}', f'vol_{w}', f'slope_{w}', f'r2_{w}', f'mdd_{w}', f'sxr_{w}', f'sharp_{w}'])
        
    for col in feature_cols:
        df_day[f'rank_{col}'] = df_day[col].rank(pct=True)
        
    # Market Context
    non_cash = df_day[df_day['name'] != '现金']
    if not non_cash.empty:
        df_day['market_max_slope'] = non_cash['slope_23'].max()
        df_day['market_max_ret'] = non_cash['ret_23'].max()
    else:
        df_day['market_max_slope'] = 0
        df_day['market_max_ret'] = 0
    
    return df_day, latest_date, None

def get_model_predictions(predictor, df_features, selected_models):
    """
    Returns a dict of {model_name: df_with_score}
    """
    results = {}
    
    for model_name in selected_models:
        # Clone df to avoid overwriting
        df_model = df_features.copy()
        
        # Predict
        try:
            probs = predictor.predict_proba(df_model, model=model_name)
            if 1 in probs.columns:
                score_col = 1
            else:
                score_col = probs.columns[-1]
            
            df_model['score'] = probs[score_col]
            df_model = df_model.sort_values('score', ascending=False)
            results[model_name] = df_model
        except Exception as e:
            st.warning(f"Model {model_name} prediction failed: {e}")
            
    return results

# --- UI ---

st.title("🌸 花姑娘 2.0 AI 投顾助手")
st.markdown("基于 **AutoGluon** 多模型集成与对比")

# Display current active model version
st.info(f"🧬 当前活跃模型版本: **{st.session_state.selected_version}**")
if not os.path.exists(os.path.join(MODEL_PATH, 'predictor.pkl')):
    st.error(f"Debug: 路径未找到 - {os.path.join(MODEL_PATH, 'predictor.pkl')}")

# Load Resources First to get model names
with st.spinner("正在加载模型与数据..."):
    data_dict = load_market_data()
    try:
        # 先检查核心文件是否存在，再进入缓存加载，避免缓存了错误的结果
        predictor_file = os.path.join(MODEL_PATH, 'predictor.pkl')
        if not os.path.exists(predictor_file):
            predictor = None
        else:
            predictor = load_model(MODEL_PATH)
            
        if predictor is None:
            st.warning(f"⚠️ 模型版本 **{st.session_state.selected_version}** 尚未训练完成，部分功能暂不可用。请在侧边栏切换版本或等待训练结束。")
            model_loaded = False
            available_models = []
        else:
            model_loaded = True
            available_models = predictor.model_names()
            # Default models: WeightedEnsemble_L2 (Best), CatBoost, XGBoost
            default_models = []
            best_model = predictor.model_best
            if best_model in available_models: default_models.append(best_model)
            if 'CatBoost' in available_models: default_models.append('CatBoost')
            
            # Fallback if specific names differ
            if not default_models: default_models = available_models[:1]
        
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        model_loaded = False
        available_models = []

# Sidebar
st.sidebar.header("⚙️ 参数设置")

# Model Version Selection
st.sidebar.subheader("🤖 模型版本")
selected_v = st.sidebar.selectbox(
    "选择 AI 模型版本",
    options=list(MODEL_VERSIONS.keys()),
    index=list(MODEL_VERSIONS.keys()).index(st.session_state.selected_version)
)
if selected_v != st.session_state.selected_version:
    st.session_state.selected_version = selected_v
    st.rerun()

# Navigation
st.sidebar.subheader("📍 导航")
if st.sidebar.button("📊 投顾控制台"):
    navigate_to("dashboard")
if st.sidebar.button("📚 关于模型原理"):
    navigate_to("about")
if st.sidebar.button("🎯 镜像策略中心", use_container_width=True):
    navigate_to("mirror")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.link_button("🌡️ 温度计指标 (外部跳转)", "https://robinindicator.streamlit.app/", use_container_width=True, help="跳转至外部温度计指标实时看板")
st.sidebar.markdown("---")

# Data Update
if st.sidebar.button("🔄 更新市场数据 (Tushare)"):
    with st.spinner("正在从 Tushare 拉取最新日线数据..."):
        success, logs = update_data_process()
        if success:
            st.sidebar.success("数据更新成功！")
            load_market_data.clear() # Clear cache to reload
            st.rerun()
        else:
            st.sidebar.error("数据更新失败，请查看日志")
            st.sidebar.text_area("错误日志", logs)

if st.sidebar.button("🧹 清除系统缓存"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("缓存已清除！")
    st.rerun()

# --- Main Routing ---

# Default to dashboard if page is not set
if 'page' not in st.session_state:
    st.session_state.page = "dashboard"

if st.session_state.page == "about":
    st.title("📚 花姑娘 2.0 项目白皮书")
    
    st.markdown("""
    欢迎使用 **花姑娘 2.0 AI 投顾系统**。本项目旨在通过机器学习技术，复刻并超越优秀的量化交易策略。
    
    ---
    
    ### 🗺️ 项目全景图
    
    本系统由三个核心模块组成，形成了一个完整的闭环：
    
    1.  **数据中心 (Data Hub)**: 负责从 Tushare 等数据源拉取全球核心资产的日线行情。
    2.  **AI 大脑 (Brain)**: 基于 AutoGluon 的集成学习模型，每日计算 96 个量化特征，输出买卖信号。
    3.  **决策终端 (Dashboard)**: 即您当前看到的界面，提供单日决策建议和历史回测验证。
    
    ---
    
    ### 🌊 数据流向 (Data Flow)
    
    1.  **原始数据**: `Open, High, Low, Close, Volume` (每日更新)
        ⬇️
    2.  **特征工程**: 计算 `Ret`, `Slope`, `R2`, `MaxDD`, `Vol` (8个时间窗口)
        ⬇️
    3.  **模型预测**: 输入特征矩阵 -> 多个模型并行打分 -> 加权集成
        ⬇️
    4.  **最终决策**: 输出 Score (0~1) -> 结合当前持仓生成操作指令 (买入/卖出/调仓)
    
    ---
    
    ### 🧠 核心模型原理
    
    #### 1. 行为克隆 (Behavioral Cloning)
    我们不直接预测股价涨跌，而是**模仿专家策略**。
    *   **专家**: 原始的“花姑娘规则E”策略（基于动量的趋势跟踪）。
    *   **学生**: AI 模型。它观察专家在历史上的每一次操作，学习其决策逻辑。
    
    #### 2. 特征体系 (96维)
    模型观察世界的“眼睛”由以下指标构成：
    
    | 维度 | 核心指标 | 作用 |
    | :--- | :--- | :--- |
    | **动量** | `ret_{w}` | 捕捉涨跌幅度 |
    | **趋势** | `slope_{w}` | 捕捉上涨速度 |
    | **稳健性** | `r2_{w}`, `sxr_{w}` | 剔除虚假突破 |
    | **风险** | **`mdd_{w}`** | **核心避险指标** (最大回撤) |
    | **波动** | `vol_{w}` | 衡量不确定性 |
    | **排名** | `rank_{feature}` | 寻找相对最强标的 |
    
    *注：`w` 代表时间窗口，覆盖 `[3, 5, 10, 20, 23, 30, 60, 120]` 日。*
    
    #### 3. 模型矩阵
    *   **WeightedEnsemble_L2**: 👑 综合能力最强，它会自动权衡各个子模型的意见。
    *   **CatBoost**: 反应敏捷，擅长处理突发特征。
    *   **LinearRegression**: 传统的线性基准，逻辑透明 (`0.5*收益 + 0.5*趋势`)。
    
    ---
    
    ### 📖 使用指南
    
    #### 场景 A: 每天早上怎么做？
    1.  点击左侧 **“🔄 更新市场数据”**，确保数据最新。
    2.  进入 **“📊 投顾控制台”** -> **“单日决策”**。
    3.  选择您的 **“当前持仓状态”** (例如：空仓，或持有纳指)。
    4.  点击 **“🚀 生成多模型决策”**。
    5.  **执行指令**:
        *   ✅ **建议买入**: 满仓买入推荐标的。
        *   🔄 **建议调仓**: 卖出当前持仓，买入新推荐标的。
        *   ⛔️ **建议观望/清仓**: 卖出所有持仓，持有现金。
    
    #### 场景 B: 验证策略靠谱吗？
    1.  进入 **“📊 投顾控制台”** -> **“区间回测”**。
    2.  选择一段历史时期 (如 2020-2023)。
    3.  勾选 **“🦅 狩猎模式”** (更严格的测试标准)。
    4.  点击回测，观察 **“最大回撤”** 和 **“年化收益”**。
    
    ---
    
    ### ⚠️ 风险提示
    *   **历史不代表未来**: AI 是基于历史规律训练的，遇到前所未见的黑天鹅事件可能会失效。
    *   **数据延迟**: 决策建议基于收盘价，实盘操作可能存在滑点。
    *   **非投资建议**: 本系统仅供辅助决策，盈亏自负。
    """)
    st.info("💡 提示：您可以在左侧导航栏返回【投顾控制台】进行实际操作。")

elif st.session_state.page == "mirror":
    st.title("🎯 镜像策略中心")
    st.markdown("---")
    st.caption("以下内容同步自外部优秀策略镜像，仅供对比参考。")
    
    # 使用 iframe 嵌入镜像网站，用户在地址栏只能看到当前网站的 URL
    # 这实现了“隐藏真实地址”的需求
    st.components.v1.iframe("https://168.unicornhunter.cn/", height=1000, scrolling=True)

elif st.session_state.page == "dashboard":
    # Mode Selection
    mode = st.sidebar.radio("选择模式", ["单日决策", "区间回测 (Backtest)"])
    
    if mode == "单日决策":
        current_holding_option = st.sidebar.selectbox(
            "当前持仓状态",
            ['空仓 (现金)'] + VALID_ASSETS
        )

        # Model Selection
        if model_loaded:
            st.sidebar.subheader("🧠 模型选择")
            selected_models = st.sidebar.multiselect(
                "选择对比模型",
                available_models,
                default=default_models
            )
            primary_model = st.sidebar.selectbox(
                "主决策模型",
                selected_models,
                index=0 if selected_models else 0
            )
            
            # Date Selection
            st.sidebar.subheader("📅 日期回溯")
            
            # Get max date from data
            all_d = []
            for df in data_dict.values(): all_d.extend(df['date'].tolist())
            max_d = max(all_d).date() if all_d else datetime.date.today()
            min_d = min(all_d).date() if all_d else max_d
            
            selected_date = st.sidebar.date_input(
                "选择决策日期",
                value=max_d,
                min_value=min_d,
                max_value=max_d
            )
        else:
            selected_models = []
            primary_model = None
            selected_date = None

        # Map UI selection to code
        if '空仓' in current_holding_option:
            current_holding = None # Fresh Entry
            holding_display = "现金/空仓"
        else:
            current_holding = current_holding_option
            holding_display = current_holding_option

        if model_loaded and selected_models:
            if st.button("🚀 生成多模型决策", type="primary"):
                df_features, date, err = prepare_daily_features(data_dict, current_holding, target_date=selected_date)
                
                if df_features is not None:
                    st.markdown(f"### 📅 决策基准日: {date.date()}")
                    
                    # Run Predictions
                    results = get_model_predictions(predictor, df_features, selected_models)
                    
                    if not results:
                        st.error("没有模型返回有效结果。")
                        st.stop()

                    # --- Primary Decision ---
                    df_primary = results[primary_model]
                    top_cand = df_primary.iloc[0]
                    top_name = top_cand['name']
                    top_score = top_cand['score']
                    
                    # Logic
                    action_color = "green"
                    action_text = ""
                    reason_text = ""
                    
                    if current_holding is None or current_holding == '现金':
                        if top_name == '现金':
                            action_text = "⛔️ 建议观望 (保持空仓)"
                            action_color = "gray"
                            reason_text = "市场风险较高，主模型认为持有现金是最优解。"
                        else:
                            action_text = f"✅ 建议买入: {top_name}"
                            action_color = "green"
                            reason_text = f"主模型 ({primary_model}) 综合评分最高 ({top_score:.4f})。"
                    else:
                        if top_name == current_holding:
                            action_text = f"🔒 建议持仓: {current_holding}"
                            action_color = "blue"
                            reason_text = f"当前持仓表现稳健 (得分 {top_score:.4f})。"
                        else:
                            if top_name == '现金':
                                action_text = f"⚠️ 建议清仓 -> 现金"
                                action_color = "red"
                                reason_text = f"持有标的转弱，建议避险。"
                            else:
                                action_text = f"🔄 建议调仓: {current_holding} -> {top_name}"
                                action_color = "orange"
                                reason_text = f"发现更优标的，得分优势显著 ({top_score:.4f})。"

                    st.info(f"**当前状态**: {holding_display}")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader(f"📢 核心指令 (基于 {primary_model})")
                        if action_color == 'green': st.success(action_text)
                        elif action_color == 'red': st.error(action_text)
                        elif action_color == 'blue': st.info(action_text)
                        elif action_color == 'orange': st.warning(action_text)
                        else: st.write(action_text)
                        st.markdown(f"**💡 决策理由**: {reason_text}")
                        
                    with col2:
                        st.metric("主模型确信度", f"{top_score:.2%}")

                    # --- Model Comparison Table ---
                    st.subheader("🤝 多模型共识分析")
                    st.caption(f"注：以下预测均基于当前活跃版本：{st.session_state.selected_version}")
                    
                    comp_data = []
                    for m_name, res_df in results.items():
                        top_row = res_df.iloc[0]
                        # Check consensus
                        action_type = "持仓" if top_row['name'] == current_holding else ("买入" if current_holding is None else "调仓")
                        if top_row['name'] == '现金' and current_holding is not None and current_holding != '现金':
                            action_type = "清仓"
                        elif top_row['name'] == '现金' and (current_holding is None or current_holding == '现金'):
                            action_type = "观望"
                        
                        comp_data.append({
                            "模型名称": m_name,
                            "首选标的": top_row['name'],
                            "确信度 (Score)": f"{top_row['score']:.2%}",
                            "建议动作": action_type,
                            "23日趋势": f"{top_row['rank_slope_23']:.2f}"
                        })
                    
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
                    
                    # --- Detailed Breakdown ---
                    st.subheader(f"📊 资产评分详情 ({primary_model})")
                    
                    # Chart
                    chart_df = df_primary.head(10).copy()
                    c = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('score', title='评分'),
                        y=alt.Y('name', sort='-x', title='资产'),
                        color=alt.condition(
                            alt.datum.name == top_name,
                            alt.value('orange'),
                            alt.value('steelblue')
                        ),
                        tooltip=['name', 'score', 'ret_23']
                    ).properties(height=350)
                    st.altair_chart(c, use_container_width=True)
                    
                else:
                    st.error("无法生成预测，请检查数据。")
            else:
                st.info("请点击按钮生成决策")

    elif mode == "区间回测 (Backtest)":
        st.header("📈 历史区间回测模拟")
        
        if not model_loaded:
            st.warning("⚠️ 当前模型版本未加载成功，无法进行回测。请在侧边栏切换版本。")
            st.stop()

        # Backtest Settings
        col1, col2 = st.columns(2)
        
        all_d = []
        for df in data_dict.values(): all_d.extend(df['date'].tolist())
        if not all_d:
            st.error("无数据")
            st.stop()
            
        max_d = max(all_d).date()
        min_d = min(all_d).date()
        
        # Enforce min date restriction
        limit_min_d = datetime.date(2017, 8, 1)
        if min_d < limit_min_d:
            min_d = limit_min_d
        
        with col1:
            start_date = st.date_input("开始日期", value=max_d - datetime.timedelta(days=365*2), min_value=min_d, max_value=max_d)
        with col2:
            end_date = st.date_input("结束日期", value=max_d, min_value=min_d, max_value=max_d)
            
        col3, col4 = st.columns(2)
        with col3:
            # Allow selecting across versions
            compare_versions = st.multiselect(
                "选择模型版本进行对比",
                options=list(MODEL_VERSIONS.keys()),
                default=[st.session_state.selected_version]
            )
            bt_models = st.multiselect("选择子模型 (各版本通用)", available_models, default=default_models)
        with col4:
            init_hold = st.selectbox("初始持仓", ["空仓 (Neutral)"] + VALID_ASSETS)
            force_neutral = st.checkbox("🦅 狩猎模式 (每日假设空仓，无视持仓Buffer)", value=True, help="勾选后，模型每天都会假设当前是空仓状态进行评分。")
            use_warmup = st.checkbox("🔥 使用历史数据预热", value=False, help="默认关闭。回测第一天将不使用开始日期之前的任何数据。")
            
        real_init = None if "空仓" in init_hold else init_hold
        
        if st.button("▶️ 开始回测", type="primary"):
            if start_date >= end_date:
                st.error("开始日期必须早于结束日期")
            elif not bt_models or not compare_versions:
                st.error("请至少选择一个版本和一个模型")
            else:
                results_df = []
                progress_text = st.empty()
                
                total_runs = len(compare_versions) * len(bt_models)
                run_idx = 0
                
                for v_name in compare_versions:
                    v_path = MODEL_VERSIONS[v_name]
                    with st.spinner(f"正在加载 {v_name}..."):
                        v_predictor = load_model(v_path)
                    
                    if v_predictor is None:
                        st.error(f"❌ 模型 {v_name} 尚未训练完成或路径不存在，请等待训练结束。")
                        continue
                    
                    for m_name in bt_models:
                        run_idx += 1
                        display_name = f"{v_name} - {m_name}" if len(compare_versions) > 1 else m_name
                        progress_text.text(f"正在回测: {display_name} ({run_idx}/{total_runs})...")
                        
                        df_hist, err = run_backtest_range(
                            v_predictor, data_dict, start_date, end_date, m_name,
                            initial_holding=real_init,
                            force_neutral=force_neutral,
                            use_warmup=use_warmup
                        )
                        
                        if df_hist is not None:
                            df_hist['Model'] = display_name
                            df_hist['cumulative_ret'] = (1 + df_hist['daily_ret']).cumprod()
                            results_df.append(df_hist)
                
                progress_text.empty()
                
                if results_df:
                    st.success("回测完成！")
                    
                    all_res = pd.concat(results_df)
                    
                    # --- Comparison Chart ---
                    st.subheader("📈 多模型净值对比")
                    
                    chart_comp = alt.Chart(all_res).mark_line().encode(
                        x=alt.X('date:T', title='日期'),
                        y=alt.Y('cumulative_ret', title='累计净值', scale=alt.Scale(zero=False)),
                        color='Model',
                        tooltip=['date', 'Model', 'cumulative_ret', 'holding']
                    ).interactive()
                    
                    st.altair_chart(chart_comp, use_container_width=True)
                    
                    # --- Metrics Table ---
                    # --- Metrics Table ---
                    st.subheader("📊 绩效指标对比")
                    
                    metrics_data = []
                    # Get unique model names in order of results_df
                    actual_models_list = [df['Model'].iloc[0] for df in results_df]
                    
                    for m_name in actual_models_list:
                        sub = all_res[all_res['Model'] == m_name]
                        total_days = (sub['date'].max() - sub['date'].min()).days
                        if total_days < 1: total_days = 1
                        
                        total_ret = sub['cumulative_ret'].iloc[-1] - 1
                        cagr = (1 + total_ret) ** (365 / total_days) - 1
                        
                        # Daily returns for risk metrics
                        rets = sub['daily_ret']
                        vol = rets.std() * np.sqrt(252)
                        
                        # Downside deviation for Sortino
                        downside_rets = rets[rets < 0]
                        downside_std = downside_rets.std() * np.sqrt(252)
                        
                        rf = 0.02
                        sharpe = (cagr - rf) / vol if vol != 0 else 0
                        sortino = (cagr - rf) / downside_std if downside_std != 0 else 0
                        
                        # Drawdown
                        roll_max = sub['cumulative_ret'].cummax()
                        dd = (sub['cumulative_ret'] - roll_max) / roll_max
                        max_dd = dd.min()
                        
                        # Calmar
                        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
                        
                        # Win Rate & Profit Factor
                        wins = rets[rets > 0]
                        losses = rets[rets < 0]
                        win_rate = len(wins) / len(rets[rets != 0]) if len(rets[rets != 0]) > 0 else 0
                        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')
                        
                        trade_count = len(sub[sub['action'] == 'Switch'])
                        
                        metrics_data.append({
                            "模型": m_name,
                            "总收益": f"{total_ret:.2%}",
                            "年化收益": f"{cagr:.2%}",
                            "夏普比率": f"{sharpe:.2f}",
                            "索提诺比率": f"{sortino:.2f}",
                            "卡玛比率": f"{calmar:.2f}",
                            "最大回撤": f"{max_dd:.2%}",
                            "胜率(日)": f"{win_rate:.2%}",
                            "盈亏比": f"{profit_factor:.2f}",
                            "交易次数": trade_count
                        })
                        
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

                    # --- Metrics Explanation ---
                    with st.expander("📚 点击查看金融绩效指标解释"):
                        st.markdown("""
                        | 指标 | 解释 | 通俗理解 |
                        | :--- | :--- | :--- |
                        | **总收益** | 回测期内的累计回报率。 | 最终赚了多少钱。 |
                        | **年化收益** | 将总收益转化成每年的平均收益。 | 相当于存银行的“年利率”。 |
                        | **夏普比率** | 每承担一单位总风险，所获得的超额收益。 | **越高越好**。反映了赚钱的“性价比”，1.0以上算不错。 |
                        | **索提诺比率** | 专门衡量承担“下跌风险”获得的收益。 | 相比夏普，它不惩罚向上的波动，更看重抗跌能力。 |
                        | **卡玛比率** | 年化收益与最大回撤的比值。 | 衡量“为了赚钱，你能忍受多大的亏损”，反映了收益风险比。 |
                        | **最大回撤** | 净值从最高点回落到最低点的最大幅度。 | 历史上“最惨”的时候亏了多少，考验投资者的心脏承受力。 |
                        | **胜率(日)** | 赚钱的天数占总交易天数的比例。 | 每天睁开眼，赚到钱的概率。 |
                        | **盈亏比** | 盈利总额与亏损总额的比值。 | 赚的时候赚多少，亏的时候亏多少。 |
                        | **交易次数** | 发生调仓（卖出旧标的买入新标的）的次数。 | 反映了策略的换手频率，次数太多可能产生较高的手续费。 |
                        """)

                    # --- Individual Details (Tabs) ---
                    st.subheader("📊 模型详细记录")
                    tabs = st.tabs(actual_models_list)
                    
                    for i, m_name in enumerate(actual_models_list):
                        with tabs[i]:
                            sub = all_res[all_res['Model'] == m_name].copy()
                            
                            # Max Drawdown for chart
                            roll_max = sub['cumulative_ret'].cummax()
                            sub['drawdown'] = (sub['cumulative_ret'] - roll_max) / roll_max
                            max_dd = sub['drawdown'].min()
                            
                            # Drawdown Chart
                            c_dd = alt.Chart(sub).mark_area(color='red', opacity=0.3).encode(
                                x='date:T',
                                y=alt.Y('drawdown', title='回撤', scale=alt.Scale(domain=[max_dd*1.1, 0])),
                                tooltip=['date', 'drawdown']
                            ).properties(height=150)
                            st.altair_chart(c_dd, use_container_width=True)
                            
                            # Table
                            st.dataframe(
                                sub[['date', 'holding', 'action', 'score', 'daily_ret', 'cumulative_ret']].style.format({
                                    'score': '{:.4f}',
                                    'daily_ret': '{:.2%}',
                                    'cumulative_ret': '{:.4f}'
                                }), 
                                use_container_width=True
                            )
                            
                            # Holding Pie
                            h_counts = sub['holding'].value_counts().reset_index()
                            h_counts.columns = ['Asset', 'Days']
                            c_pie = alt.Chart(h_counts).mark_arc().encode(
                                theta='Days', color='Asset', tooltip=['Asset', 'Days']
                            )
                            st.altair_chart(c_pie)
                else:
                    st.error("回测失败")
    else:
        st.error("模型未加载")


st.markdown("---")
st.caption("注：不同模型对风险的敏感度不同，WeightedEnsemble 通常最稳健，CatBoost 对类别特征更敏感。")
