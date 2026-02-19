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

def calc_slope_r2_fast(y):
    """
    使用纯 Numpy 计算标准化后的斜率和 R2，速度比 sklearn 快 100 倍以上。
    y: 价格序列
    """
    n = len(y)
    if n < 2 or y[0] == 0:
        return 0.0, 0.0
    
    # 归一化，与原逻辑保持一致
    y_norm = y / y[0]
    x = np.arange(n)
    
    # 简单的线性回归公式: y = kx + b
    # 使用 np.polyfit (底层是最小二乘法，非常快)
    # deg=1 返回 [slope, intercept]
    try:
        coeffs = np.polyfit(x, y_norm, 1)
    except:
        return 0.0, 0.0

    slope = coeffs[0]
    
    # 计算 R2
    # predicted = slope * x + intercept
    predicted = slope * x + coeffs[1]
    ss_res = np.sum((y_norm - predicted) ** 2)
    ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return slope, r2

def calc_max_drawdown_fast(prices):
    """向量化计算最大回撤"""
    if len(prices) < 1: return 0.0
    roll_max = np.maximum.accumulate(prices)
    if roll_max[0] == 0: return 0.0
    drawdown = (prices - roll_max) / roll_max
    return drawdown.min()

def calculate_indicators(window_close, window_returns):
    """统一的特征计算逻辑，确保回测与单日决策完全一致"""
    if len(window_close) < 2:
        return {f: 0.0 for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']}
    
    # 收益率：当前价格 / 窗口起始价格 - 1
    ret = (window_close[-1] / window_close[0]) - 1
    # 波动率：收益率序列的标准差 * sqrt(252)
    vol = np.std(window_returns) * np.sqrt(252)
    
    # 优化：使用快速算法替代 sklearn
    slope, r2 = calc_slope_r2_fast(window_close)
    
    # 最大回撤
    mdd = calc_max_drawdown_fast(window_close)
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

@st.cache_data(ttl=3600*24) # 缓存一天
def load_all_features_from_disk():
    """从磁盘加载预计算的全量特征数据"""
    processed_path = os.path.join(BASE_DIR, 'processed_features.pkl')
    if os.path.exists(processed_path):
        try:
            return pd.read_pickle(processed_path)
        except Exception as e:
            print(f"Error loading processed features: {e}")
            return None
    return None

def save_all_features_to_disk(processed_data):
    """保存预计算特征到磁盘"""
    try:
        processed_path = os.path.join(BASE_DIR, 'processed_features.pkl')
        pd.to_pickle(processed_data, processed_path)
    except Exception as e:
        print(f"Error saving processed features: {e}")

def prepare_all_features_cached(data_dict, windows, warmup=True, start_date=None):
    """
    预计算所有特征（向量化优化版）。
    优先尝试从磁盘加载预计算数据，如果不存在或数据过期则实时计算并保存。
    """
    # 尝试加载磁盘缓存
    # 注意：这里简化处理，假设本地缓存总是最新的。生产环境可能需要版本控制。
    # 实际上，如果用户点击了“更新数据”，我们应该强制重新计算。
    # Streamlit 的缓存机制 handle 了大部分情况，这里主要为了持久化加速首次启动。
    
    # 由于 data_dict 是动态传入的，我们还是依赖 Streamlit 的缓存机制 @st.cache_data
    # 但由于 data_dict 太大，作为 key 可能有问题。
    # 我们这里主要优化计算过程，向量化已经足够快了。
    
    all_dates = set()
    for df in data_dict.values():
        all_dates.update(df['date'].tolist())
    sorted_dates = sorted(list(all_dates))
    
    processed_data = {} 
    
    for name, df in data_dict.items():
        # 必须确保按时间排序
        sub = df.set_index('date').sort_index()
        df_feat = sub.copy()
        
        # 预计算每日收益率
        df_feat['daily_ret'] = df_feat['close'].pct_change().fillna(0.0)
        
        # 定义包装函数以适配 apply
        # 注意：rolling().apply 在每次调用时传入的是 numpy array (raw=True)
        
        for w in windows:
            # 1. 收益率 (Ret): 当前 / (T-w+1) - 1
            # 对应 calculate_indicators 中的 ret = (window_close[-1] / window_close[0]) - 1
            # pandas shift(w-1) 刚好拿到窗口第一个元素
            df_feat[f'ret_{w}'] = df_feat['close'] / df_feat['close'].shift(w - 1) - 1
            
            # 2. 波动率 (Vol): 滚动标准差 * sqrt(252)
            # numpy std 默认 ddof=0, pandas 默认 ddof=1。为了匹配 calculate_indicators 中的 np.std，使用 ddof=0
            df_feat[f'vol_{w}'] = df_feat['daily_ret'].rolling(window=w).std(ddof=0) * np.sqrt(252)
            
            # 3. 最大回撤 (MDD)
            df_feat[f'mdd_{w}'] = df_feat['close'].rolling(window=w).apply(calc_max_drawdown_fast, raw=True)
            
            # 4. 斜率 (Slope) 和 R2
            # apply 只能返回标量，所以需要两次调用
            def get_slope(y):
                s, _ = calc_slope_r2_fast(y)
                return s
            
            def get_r2(y):
                _, r = calc_slope_r2_fast(y)
                return r

            df_feat[f'slope_{w}'] = df_feat['close'].rolling(window=w).apply(get_slope, raw=True)
            df_feat[f'r2_{w}'] = df_feat['close'].rolling(window=w).apply(get_r2, raw=True)
            
            # 5. 复合指标 (向量化操作)
            df_feat[f'sxr_{w}'] = df_feat[f'slope_{w}'] * df_feat[f'r2_{w}']
            # 避免除以0
            df_feat[f'sharp_{w}'] = df_feat[f'slope_{w}'] / (df_feat[f'vol_{w}'] + 0.01)
            
        # 填充 NaN (因为滚动窗口前 w-1 个数据是 NaN)
        # 实际上我们不需要填充为0，保留NaN更好，因为在回测中我们会检查NaN
        # 但为了兼容原有逻辑，如果需要可以填0
        # df_feat = df_feat.fillna(0.0)

        processed_data[name] = df_feat
    
    # 异步保存到磁盘（可选，为了不阻塞主线程，这里简单同步保存）
    # save_all_features_to_disk(processed_data)
        
    return processed_data, sorted_dates

def calculate_trade_stats(history_df):
    """
    计算交易层面的统计指标：
    1. 每笔交易的收益、持仓天数、最大回撤
    2. 胜率、盈亏比等
    """
    if history_df.empty:
        return pd.DataFrame(), {}
        
    trades = []
    current_trade = None
    
    # 遍历每日历史记录，重构交易
    # history_df cols: date, holding, prev_holding, score, action, daily_ret, close_open_pct
    
    # 添加净值列辅助计算最大回撤
    history_df = history_df.copy()
    history_df['equity_curve'] = (1 + history_df['daily_ret']).cumprod()
    
    for idx, row in history_df.iterrows():
        date = row['date']
        holding = row['holding']
        prev_holding = row['prev_holding']
        daily_ret = row['daily_ret']
        
        # 识别交易起点：从空仓/其他资产 -> 新资产
        # 或者 初始持仓
        
        # 简化逻辑：只要 holding 发生变化，或者 holding 不变但今天是第一天
        # 我们以“连续持有一段资产”定义为一笔交易
        
        if current_trade is None:
            # 第一笔交易初始化
            current_trade = {
                'asset': holding,
                'start_date': date,
                'end_date': None,
                'days': 0,
                'returns': [],
                'equity': [1.0] # 交易内净值归一化
            }
        
        # 检查是否发生切换 (Switch)
        # 注意：row['action'] == 'Switch' 意味着今天持有的 holding 不同于昨天
        # 所以今天的收益属于新 holding。
        # 昨天的 trade 应该在昨天结束。
        
        if row['action'] == 'Switch' and idx > 0:
            # 结算上一笔交易
            current_trade['end_date'] = history_df.iloc[idx-1]['date']
            trades.append(current_trade)
            
            # 开启新交易
            current_trade = {
                'asset': holding,
                'start_date': date,
                'end_date': None,
                'days': 0,
                'returns': [],
                'equity': [1.0]
            }
            
        # 累积当前交易的数据
        current_trade['days'] += 1
        current_trade['returns'].append(daily_ret)
        new_nav = current_trade['equity'][-1] * (1 + daily_ret)
        current_trade['equity'].append(new_nav)
        
    # 最后一笔交易结算
    if current_trade:
        current_trade['end_date'] = history_df.iloc[-1]['date']
        trades.append(current_trade)
        
    # 计算每笔交易的指标
    trade_records = []
    for t in trades:
        # 忽略现金交易（如果需要统计空仓期也可以保留）
        if t['asset'] == '现金' or t['asset'] is None:
            continue
            
        # 总收益
        total_ret = t['equity'][-1] - 1
        
        # 交易内最大回撤及持续天数
        navs = np.array(t['equity'])
        roll_max = np.maximum.accumulate(navs)
        # 避免除以0
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = (navs - roll_max) / roll_max
            dd[np.isnan(dd)] = 0 # 处理 roll_max 为 0 的情况
            
        max_dd = dd.min()
        
        # 计算回撤持续时间 (简单估算: 从最近一个新高到最低点的距离，或者整个回撤期的长度)
        # 这里计算: 处于回撤状态的总天数 / 交易总天数
        # 或者更精确: 最长连续回撤天数
        is_dd = dd < 0
        if np.any(is_dd):
            # 找到最长连续 True 的序列
            # 使用 diff 找边界
            padded = np.concatenate(([False], is_dd, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if len(starts) > 0:
                max_dd_duration = (ends - starts).max()
            else:
                max_dd_duration = 0
        else:
            max_dd_duration = 0

        trade_records.append({
            '标的': t['asset'],
            '买入日期': t['start_date'],
            '卖出日期': t['end_date'],
            '持仓天数': t['days'],
            '交易收益': total_ret,
            '最大回撤': max_dd,
            '回撤持续天数': max_dd_duration
        })
        
    df_trades = pd.DataFrame(trade_records)
    
    # 全局统计
    stats = {}
    if not df_trades.empty:
        stats['total_trades'] = len(df_trades)
        stats['win_rate'] = (df_trades['交易收益'] > 0).mean()
        stats['avg_ret'] = df_trades['交易收益'].mean()
        stats['max_single_ret'] = df_trades['交易收益'].max()
        stats['min_single_ret'] = df_trades['交易收益'].min()
        stats['avg_days'] = df_trades['持仓天数'].mean()
        stats['avg_dd_days'] = df_trades['回撤持续天数'].mean()
        stats['max_dd_days'] = df_trades['回撤持续天数'].max()
        
        # 盈亏比
        avg_win = df_trades[df_trades['交易收益'] > 0]['交易收益'].mean() if not df_trades[df_trades['交易收益'] > 0].empty else 0
        avg_loss = abs(df_trades[df_trades['交易收益'] < 0]['交易收益'].mean()) if not df_trades[df_trades['交易收益'] < 0].empty else 1 # 避免除以0
        stats['pl_ratio'] = avg_win / avg_loss if avg_loss != 0 else 0
    else:
        stats = {k: 0 for k in ['total_trades', 'win_rate', 'avg_ret', 'max_single_ret', 'min_single_ret', 'avg_days', 'pl_ratio']}

    return df_trades, stats

def calculate_top_drawdowns(history_df, top_n=5):
    """
    计算历史最大的 N 次回撤区间
    返回 DataFrame: ['start_date', 'end_date', 'depth', 'duration']
    """
    # 确保有累计净值列
    if 'cumulative_ret' not in history_df.columns:
        history_df = history_df.copy()
        history_df['cumulative_ret'] = (1 + history_df['daily_ret']).cumprod() - 1
        
    equity = history_df['cumulative_ret'] + 1
    # 计算高水位
    high_water_mark = equity.cummax()
    # 计算回撤
    drawdown = (equity - high_water_mark) / high_water_mark
    
    # 寻找回撤区间
    # 逻辑：只要 drawdown < 0，就是一个回撤期
    # 结束标志：drawdown 回到 0 (即创新高)
    
    is_dd = drawdown < 0
    
    # 使用 diff 找状态变化点
    # padded: [False, ...data..., False]
    padded = np.concatenate(([False], is_dd, [False]))
    diff = np.diff(padded.astype(int))
    
    # starts: 从 0 变为 1 的索引 (进入回撤)
    # ends: 从 1 变为 0 的索引 (结束回撤，创新高)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    dd_periods = []
    
    if len(starts) > 0 and len(ends) > 0:
        for s, e in zip(starts, ends):
            # 在这个区间内找到最大回撤深度
            # 注意：s 是 history_df 的索引，e 是结束索引（不包含）
            # 切片范围是 [s, e)
            
            # 如果索引越界保护
            if s >= len(history_df): continue
            real_e = min(e, len(history_df))
            
            dd_slice = drawdown.iloc[s:real_e]
            min_dd = dd_slice.min()
            min_idx = dd_slice.idxmin() # 达到最大回撤的日期索引
            
            start_date = history_df.iloc[s]['date']
            # 如果 e 对应的是恢复日（创新高日），那么 e-1 是还在水下的最后一天
            # 恢复日是 e
            end_date = history_df.iloc[real_e-1]['date'] if real_e > 0 else start_date
            
            # 实际上，真正“恢复”是指创新高的那一天，即 history_df.iloc[e] (如果存在)
            # 如果回撤一直持续到最后一天，则未恢复
            if e < len(history_df):
                recovery_date = history_df.iloc[e]['date']
                status = "已恢复"
            else:
                recovery_date = None
                status = "未恢复"
                
            dd_periods.append({
                '开始日期': start_date,
                '最大回撤日期': history_df.loc[min_idx, 'date'],
                '结束/恢复日期': recovery_date if recovery_date else history_df.iloc[-1]['date'],
                '回撤深度': min_dd,
                '持续天数': (pd.to_datetime(recovery_date) - pd.to_datetime(start_date)).days if recovery_date else (pd.to_datetime(history_df.iloc[-1]['date']) - pd.to_datetime(start_date)).days,
                '状态': status
            })
            
    df_dd = pd.DataFrame(dd_periods)
    if not df_dd.empty:
        df_dd = df_dd.sort_values('回撤深度', ascending=True).head(top_n) # 深度是负数，越小越深
        
    return df_dd

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
if st.sidebar.button("📖 回测机制详解"):
    navigate_to("backtest_logic")
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

elif st.session_state.page == "backtest_logic":
    st.title("📖 回测系统机制详解")
    st.markdown("---")
    
    st.markdown("""
    本系统采用 **“信号-执行分离”** 的严格回测框架，旨在最大程度还原真实的实盘交易环境，杜绝“未来函数”带来的虚假繁荣。
    
    ### 1. 🕒 核心时间轴 (Timeline)
    
    我们的回测逻辑严格遵循以下时间顺序：
    
    *   **T-1 日 (信号日)**: 
        *   收盘后，系统获取截至当日的全部历史数据（收盘价、成交量等）。
        *   模型根据这些数据计算 96 维特征，并输出对 T 日的持仓建议（例如：持有纳指 或 切换为空仓）。
        *   **注意**: 此时 T 日的行情尚未发生，决策完全基于历史信息。
        
    *   **T 日 (交易与持仓日)**:
        *   **开盘时刻 (Open)**: 如果 T-1 日的建议与当前持仓不同（例如从空仓变为持有），系统假设在 **T 日开盘价** 完成调仓。
        *   **收盘时刻 (Close)**: 计算当日的账户权益变化。
    
    ---
    
    ### 2. 💰 收益计算公式 (Return Calculation)
    
    为了精确模拟交易损耗和日内波动，我们根据是否发生调仓采用不同的计算公式：
    
    #### 情况 A: 发生调仓 (Switch)
    当系统建议从“资产A”切换到“资产B”时，交易流程如下：
    
    1.  **卖出操作**: 在 **T 日开盘价** 卖出持有的“资产A”。
        *   *资产A 当日收益*: $\frac{Open_{T,A}}{Close_{T-1,A}} - 1$ (捕获了资产A的隔夜跳空)。
        *   *注意*: 为了简化计算，回测系统通常将这部分隔夜收益归入“上一笔交易”的最终净值中，或者在切换日直接计算新资产的收益。本系统采取**“无缝切换”**逻辑：我们假设资金在开盘瞬间完成转移。
        
    2.  **买入操作**: 在 **T 日开盘价** 买入“资产B”。
    
    3.  **当日净值变化**: 实际上由两部分组成（旧资产的隔夜波动 + 新资产的日内波动）。
        *   **本系统简化算法**: 为了规避复杂的资金结算延迟问题，我们在回测中主要关注**新持有资产（资产B）的日内表现**。
        *   **计算公式**: $\frac{Close_{T,B}}{Open_{T,B}} - 1$
        *   *这意味着*: 调仓日当天，我们承担了新资产 B 的日内涨跌风险。
    
    #### 情况 B: 持仓不变 (Hold)
    当系统建议继续持有“资产A”时：
    *   **基准价格**: 资产 A 的 **T-1 日收盘价 ($Close_{T-1}$)**。
    *   **当日收益**: $\frac{Close_T}{Close_{T-1}} - 1$
    *   *解释*: 您完整地持有了该资产度过了一整天，因此享受（或承担）了包括隔夜跳空在内的**全天涨跌幅**。
    
    #### 情况 C: 空仓 (Cash)
    *   **当日收益**: $0.0\%$ (我们暂不计算现金理财收益)。
    
    ---
    
    ### 3. 📊 绩效指标定义 (Metrics)
    
    系统会自动计算以下专业金融指标来评估策略质量：
    
    | 指标 | 定义 | 解读 |
    | :--- | :--- | :--- |
    | **累计收益 (Cumulative Return)** | $\prod (1 + r_t) - 1$ | 策略从开始到现在的总回报率。 |
    | **年化收益 (CAGR)** | $(1 + TotalRet)^{\frac{365}{Days}} - 1$ | 将总收益折算为每年的平均复利增长率。 |
    | **最大回撤 (Max Drawdown)** | $\min (\frac{Value_t - Peak_t}{Peak_t})$ | 历史上从最高点跌下来的最大幅度。**衡量风险的核心指标**。 |
    | **夏普比率 (Sharpe Ratio)** | $\frac{E[R_p - R_f]}{\sigma_p}$ | 每承担 1 单位波动风险所获得的超额回报。**>1.0 为优秀**。 |
    | **卡玛比率 (Calmar Ratio)** | $\frac{CAGR}{|MaxDD|}$ | 年化收益与最大回撤之比。衡量“为了赚这笔钱，我需要忍受多大的痛苦”。 |
    | **胜率 (Win Rate)** | $\frac{盈利天数}{总交易天数}$ | 每天赚钱的概率。注意：高胜率不代表一定赚钱（可能赚小钱亏大钱）。 |
    
    ---
    
    ### 4. 🤖 模型评分与决策 (Scoring)
    
    在 T-1 日，AI 模型会给每个资产打分 (Score, 0~1)：
    *   **Score**: 代表模型对该资产未来表现的信心。
    *   **Rank**: 我们将 Score 进行每日排名。
    *   **决策**: 系统总是选择 **Score 最高** 且符合风险控制规则的资产作为 T 日的持仓目标。
    
    *如果所有风险资产的评分都过低（或模型预测市场风险极高），系统会选择 **“现金”** 作为最优解，即建议空仓观望。*
    """)
    
    st.info("💡 明白了？点击左侧【投顾控制台】去试一试吧！")

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
                        
                        # Drawdown Duration (Strategy Level)
                        is_dd = dd < 0
                        if np.any(is_dd):
                            padded = np.concatenate(([False], is_dd, [False]))
                            diff = np.diff(padded.astype(int))
                            starts = np.where(diff == 1)[0]
                            ends = np.where(diff == -1)[0]
                            if len(starts) > 0:
                                max_strat_dd_days = (ends - starts).max()
                            else:
                                max_strat_dd_days = 0
                        else:
                            max_strat_dd_days = 0

                        metrics_data.append({
                            "模型": m_name,
                            "总收益": f"{total_ret:.2%}",
                            "年化收益": f"{cagr:.2%}",
                            "夏普比率": f"{sharpe:.2f}",
                            "索提诺比率": f"{sortino:.2f}",
                            "卡玛比率": f"{calmar:.2f}",
                            "最大回撤": f"{max_dd:.2%}",
                            "回撤最长持续": f"{max_strat_dd_days} 天",
                            "胜率(日)": f"{win_rate:.2%}",
                            "盈亏比": f"{profit_factor:.2f}",
                        })
                        
                        # Calculate Per-Trade Stats
                        # Need to reconstruct history_df from sub
                        # sub has columns: date, holding, prev_holding, score, action, daily_ret, close_open_pct, Model, cumulative_ret
                        
                        # Clean up sub for calculate_trade_stats
                        hist_for_stats = sub.copy()
                        hist_for_stats = hist_for_stats.drop(columns=['Model', 'cumulative_ret'])
                        # cum_ret needed for calculate_trade_stats is (1+ret).cumprod() - 1? 
                        # No, calculate_trade_stats uses daily_ret directly.
                        # But wait, previous implementation of calculate_trade_stats uses daily_ret.
                        # It re-calculates equity_curve internally.
                        
                        trade_df, trade_stats = calculate_trade_stats(hist_for_stats)
                        
                        if not trade_df.empty:
                            st.subheader(f"📊 {m_name} - 交易明细分析")
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("总交易次数", trade_stats['total_trades'])
                            c2.metric("平均持仓天数", f"{trade_stats['avg_days']:.1f} 天")
                            c3.metric("交易胜率", f"{trade_stats['win_rate']:.2%}")
                            c4.metric("平均单笔收益", f"{trade_stats['avg_ret']:.2%}")
                            
                            c5, c6, c7, c8 = st.columns(4)
                            c5.metric("盈亏比 (P/L)", f"{trade_stats['pl_ratio']:.2f}")
                            c6.metric("单笔最大收益", f"{trade_stats['max_single_ret']:.2%}", delta="🚀")
                            c7.metric("单笔最大亏损", f"{trade_stats['min_single_ret']:.2%}", delta="🔻")
                            c8.metric("策略最大回撤", f"{max_dd:.2%}", delta_color="inverse")
                            
                            st.markdown("##### 📉 交易分布统计 (高级视图)")
                            d1, d2 = st.columns(2)
                            
                            # 1. 散点图：收益 vs 持仓天数 (点的大小代表绝对收益大小，颜色代表盈亏)
                            scatter_chart = alt.Chart(trade_df).mark_circle().encode(
                                x=alt.X('持仓天数', title='持仓天数 (Days)'),
                                y=alt.Y('交易收益', title='交易收益率', axis=alt.Axis(format='%')),
                                color=alt.condition(
                                    alt.datum['交易收益'] > 0,
                                    alt.value('green'),
                                    alt.value('red')
                                ),
                                size=alt.Size('交易收益', scale=alt.Scale(domain=[-0.2, 0.2], range=[50, 500]), legend=None),
                                tooltip=['标的', '买入日期', '交易收益', '持仓天数', '最大回撤']
                            ).properties(
                                title='盈亏分布矩阵 (收益 vs 时间)',
                                height=300
                            ).interactive()
                            
                            # 添加 0 轴线
                            rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')
                            d1.altair_chart(scatter_chart + rule, use_container_width=True)
                            
                            # 2. 箱线图：不同标的的收益波动范围
                            boxplot = alt.Chart(trade_df).mark_boxplot(extent='min-max').encode(
                                x=alt.X('标的', title='资产类别'),
                                y=alt.Y('交易收益', title='收益分布', axis=alt.Axis(format='%')),
                                color='标的',
                                tooltip=['标的', '交易收益']
                            ).properties(
                                title='资产收益波动性分析',
                                height=300
                            )
                            d2.altair_chart(boxplot, use_container_width=True)
                            
                            # 3. 瀑布图 (Waterfall) - 累计收益构成
                            # 构造瀑布图数据
                            waterfall_df = trade_df.copy()
                            waterfall_df['id'] = range(len(waterfall_df))
                            waterfall_df['prev_sum'] = waterfall_df['交易收益'].cumsum().shift(1).fillna(0)
                            waterfall_df['curr_sum'] = waterfall_df['交易收益'].cumsum()
                            waterfall_df['color'] = np.where(waterfall_df['交易收益'] > 0, '盈利', '亏损')
                            
                            waterfall_chart = alt.Chart(waterfall_df).mark_bar().encode(
                                x=alt.X('id', title='交易序号'),
                                y=alt.Y('prev_sum', title='累计收益率', axis=alt.Axis(format='%')),
                                y2='curr_sum',
                                color=alt.Color('color', scale=alt.Scale(domain=['盈利', '亏损'], range=['green', 'red'])),
                                tooltip=['标的', '买入日期', '交易收益', 'curr_sum']
                            ).properties(
                                title='账户资金流 (交易逐笔盈亏)',
                                height=250
                            ).interactive()
                            
                            st.altair_chart(waterfall_chart, use_container_width=True)
                            
                            with st.expander(f"查看 {m_name} 所有交易记录"):
                                st.dataframe(
                                    trade_df.style.format({
                                        '交易收益': '{:.2%}',
                                        '最大回撤': '{:.2%}',
                                        '买入日期': '{:%Y-%m-%d}',
                                        '卖出日期': '{:%Y-%m-%d}'
                                    }),
                                    use_container_width=True
                                )
                                
                            # --- Trade Visualization ---
                            st.markdown("#### 🕯️ 交易可视化 (K线 + 买卖点)")
                            
                            # 获取该模型交易过的所有非空仓资产
                            traded_assets = trade_df['标的'].unique().tolist()
                            if '现金' in traded_assets: traded_assets.remove('现金')
                            
                            if traded_assets:
                                # 修复：不使用 selectbox 交互（导致页面刷新），而是直接循环展示所有资产
                                for selected_asset_chart in traded_assets:
                                    st.markdown(f"**{selected_asset_chart}**")
                                    
                                    # 1. 获取该资产的全量历史数据
                                    if selected_asset_chart in data_dict:
                                        df_asset = data_dict[selected_asset_chart].copy()
                                        df_asset['date'] = pd.to_datetime(df_asset['date'])
                                        
                                        # 过滤时间范围：仅显示回测区间内的数据
                                        mask = (df_asset['date'] >= pd.to_datetime(start_date)) & (df_asset['date'] <= pd.to_datetime(end_date))
                                        df_chart = df_asset.loc[mask].copy()
                                        
                                        # 2. 标记买卖点
                                        asset_trades = trade_df[trade_df['标的'] == selected_asset_chart]
                                        
                                        buy_points = []
                                        sell_points = []
                                        
                                        for _, t in asset_trades.iterrows():
                                            d_buy = pd.to_datetime(t['买入日期'])
                                            if d_buy in df_chart['date'].values:
                                                price = df_chart.loc[df_chart['date'] == d_buy, 'open'].values[0] # Open price for buy
                                                if pd.isna(price) or price == 0: price = df_chart.loc[df_chart['date'] == d_buy, 'close'].values[0]
                                                buy_points.append({'date': d_buy, 'price': price, 'type': 'Buy'})
                                                
                                            d_sell = pd.to_datetime(t['卖出日期'])
                                            if pd.notnull(d_sell) and d_sell in df_chart['date'].values:
                                                price = df_chart.loc[df_chart['date'] == d_sell, 'open'].values[0]
                                                if d_sell == pd.to_datetime(end_date): # Last day
                                                     price = df_chart.loc[df_chart['date'] == d_sell, 'close'].values[0]
                                                
                                                if pd.isna(price) or price == 0: price = df_chart.loc[df_chart['date'] == d_sell, 'close'].values[0]
                                                sell_points.append({'date': d_sell, 'price': price, 'type': 'Sell'})
                                        
                                        # 3. 绘制图表
                                        base = alt.Chart(df_chart).encode(x=alt.X('date:T', title='日期'))
                                        
                                        line = base.mark_line(color='gray', opacity=0.5).encode(
                                            y=alt.Y('close', title='价格', scale=alt.Scale(zero=False)),
                                            tooltip=['date', 'open', 'close', 'high', 'low']
                                        )
                                        
                                        if buy_points:
                                            df_buy = pd.DataFrame(buy_points)
                                            buy_chart = alt.Chart(df_buy).mark_point(
                                                shape='triangle-up', color='red', size=100, filled=True
                                            ).encode(
                                                x='date:T',
                                                y='price',
                                                tooltip=[alt.Tooltip('date', title='买入日期'), alt.Tooltip('price', title='买入价格')]
                                            )
                                        else:
                                            buy_chart = alt.Chart(pd.DataFrame()).mark_point()
                                            
                                        if sell_points:
                                            df_sell = pd.DataFrame(sell_points)
                                            sell_chart = alt.Chart(df_sell).mark_point(
                                                shape='triangle-down', color='green', size=100, filled=True
                                            ).encode(
                                                x='date:T',
                                                y='price',
                                                tooltip=[alt.Tooltip('date', title='卖出日期'), alt.Tooltip('price', title='卖出价格')]
                                            )
                                        else:
                                            sell_chart = alt.Chart(pd.DataFrame()).mark_point()
                                            
                                        st.altair_chart((line + buy_chart + sell_chart).interactive(), use_container_width=True)
                                        
                                    else:
                                        st.warning(f"未找到资产 {selected_asset_chart} 的历史数据。")
                            else:
                                st.info("该模型在此期间未交易任何风险资产。")
                        else:
                            st.info(f"模型 {m_name} 在此期间无交易或一直空仓。")
                    
                    st.subheader("🏆 策略横向对比")
                    st.dataframe(pd.DataFrame(metrics_data).set_index("模型"), use_container_width=True)

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
                            
                            # Drawdown Chart (Improved Visibility)
                            # 使用 Area 图并设置更醒目的颜色和透明度，同时增加交互线
                            c_dd = alt.Chart(sub).mark_area(
                                line={'color': 'darkred'}, # 增加深红色边线
                                color=alt.Gradient(
                                    gradient='linear',
                                    stops=[alt.GradientStop(color='red', offset=0),
                                           alt.GradientStop(color='white', offset=1)],
                                    x1=1, x2=1, y1=1, y2=0
                                ),
                                opacity=0.7
                            ).encode(
                                x=alt.X('date:T', title='日期'),
                                y=alt.Y('drawdown', title='回撤深度', axis=alt.Axis(format='%', titleColor='red')),
                                tooltip=[
                                    alt.Tooltip('date', title='日期', format='%Y-%m-%d'), 
                                    alt.Tooltip('drawdown', title='回撤深度', format='.2%'),
                                    alt.Tooltip('cumulative_ret', title='当前净值', format='.4f')
                                ]
                            ).properties(
                                title='策略水下曲线 (Underwater Chart)',
                                height=200
                            ).interactive()
                            
                            st.altair_chart(c_dd, use_container_width=True)
                            
                            # Top 5 Drawdowns
                            st.markdown("##### 📉 历史前 5 大回撤区间")
                            df_top_dd = calculate_top_drawdowns(sub, top_n=5)
                            if not df_top_dd.empty:
                                st.dataframe(
                                    df_top_dd.style.format({
                                        '回撤深度': '{:.2%}',
                                        '开始日期': '{:%Y-%m-%d}',
                                        '最大回撤日期': '{:%Y-%m-%d}',
                                        '结束/恢复日期': '{:%Y-%m-%d}'
                                    }),
                                    use_container_width=True
                                )
                            else:
                                st.info("策略表现极其稳健，无显著回撤。")
                                
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
