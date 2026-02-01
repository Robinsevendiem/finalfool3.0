import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

warnings.filterwarnings('ignore')

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
# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'market_data')
# 预测窗口：预测未来 10 个交易日的收益
PREDICT_WINDOW = 10 

def load_data():
    data = {}
    if not os.path.exists(DATA_DIR): return {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            code = filename.split('_')[0]
            df = pd.read_csv(os.path.join(DATA_DIR, filename))
            df['date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('date').reset_index(drop=True)
            if 'close_qfq' in df.columns: df['close'] = df['close_qfq']
            
            name = next((v for k, v in NAME_MAP.items() if k in filename or k == code), None)
            if name:
                df['name'] = name
                df['code'] = code
                data[name] = df
    return data

def calc_slope(y):
    if len(y) < 2: return 0
    x = np.arange(len(y)).reshape(-1, 1)
    y_norm = y / y[0]
    return LinearRegression().fit(x, y_norm).coef_[0]

def calc_r2(y):
    if len(y) < 2: return 0
    x = np.arange(len(y)).reshape(-1, 1)
    y_norm = y / y[0]
    return LinearRegression().fit(x, y_norm).score(x, y_norm)

def calc_max_drawdown(prices):
    if len(prices) < 1: return 0.0
    roll_max = np.maximum.accumulate(prices)
    # Avoid division by zero
    if roll_max[0] == 0: return 0.0
    return ((prices - roll_max) / roll_max).min()

def build_performance_dataset():
    print("🚀 正在构建【绩效优化型】数据集...")
    data_dict = load_data()
    WINDOWS = [3, 5, 10, 20, 23, 30, 60, 120]
    
    processed_data = {}
    print("1. 计算特征与未来收益...")
    for name, df in tqdm(data_dict.items(), desc="处理标的数据"):
        sub = df.set_index('date').sort_index()
        
        # 计算特征 (同之前逻辑)
        for w in WINDOWS:
            sub[f'ret_{w}'] = sub['close'].pct_change(w)
            sub[f'vol_{w}'] = sub['close'].pct_change().rolling(w).std() * np.sqrt(252)
            
            # 窗口统计量
            close_vals = sub['close'].values
            slopes, r2s, mdds = [], [], []
            for i in range(len(sub)):
                if i < w:
                    slopes.append(np.nan); r2s.append(np.nan); mdds.append(np.nan)
                else:
                    win = close_vals[i-w+1 : i+1]
                    slopes.append(calc_slope(win)); r2s.append(calc_r2(win)); mdds.append(calc_max_drawdown(win))
            
            sub[f'slope_{w}'], sub[f'r2_{w}'], sub[f'mdd_{w}'] = slopes, r2s, mdds
            sub[f'sxr_{w}'] = sub[f'slope_{w}'] * sub[f'r2_{w}']
            # 新增特征：风险调整后动量 (夏普比率思路)
            sub[f'sharp_{w}'] = sub[f'slope_{w}'] / (sub[f'vol_{w}'] + 0.01)
            
        # --- 核心修改：计算未来收益与未来风险 (Label) ---
        # future_ret: T 到 T+5 的收益率
        sub['future_ret'] = sub['close'].shift(-PREDICT_WINDOW) / sub['close'] - 1
        
        # future_mdd: T 到 T+5 期间的最大回撤
        f_mdds = []
        close_vals = sub['close'].values
        for i in range(len(sub)):
            if i + PREDICT_WINDOW >= len(sub):
                f_mdds.append(np.nan)
            else:
                # 考察未来窗口内的价格序列
                future_window = close_vals[i : i + PREDICT_WINDOW + 1]
                f_mdds.append(calc_max_drawdown(future_window))
        sub['future_mdd'] = f_mdds
        processed_data[name] = sub

    print("2. 跨标的对齐与标签生成...")
    all_dates = sorted(list(set(d for df in processed_data.values() for d in df.index)))
    dataset = []
    
    for i, date in enumerate(tqdm(all_dates, desc="对齐日期数据")):
        daily_snapshot = []
        
        # 1. 收集所有标的的未来表现
        for name, df in processed_data.items():
            if date in df.index:
                row = df.loc[date]
                if pd.notnull(row['slope_23']) and pd.notnull(row['future_ret']) and pd.notnull(row['future_mdd']):
                    sample = {
                        'date': date, 'name': name,
                        'future_ret': row['future_ret'],
                        'future_mdd': row['future_mdd']
                    }
                    for w in WINDOWS:
                        for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']:
                            sample[f'{f}_{w}'] = row[f'{f}_{w}']
                    daily_snapshot.append(sample)
        
        if not daily_snapshot: continue
        
        # 2. 加入现金选项 (未来收益固定为 0，回撤也为 0)
        cash_sample = {
            'date': date, 'name': '现金', 'future_ret': 0.0, 'future_mdd': 0.0
        }
        for w in WINDOWS:
            for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']: cash_sample[f'{f}_{w}'] = 0.0
        daily_snapshot.append(cash_sample)
        
        # 3. 生成【绩效+风控】综合标签
        # 目标：寻找未来预测窗口内“收益/风险”比最优的标的
        df_day = pd.DataFrame(daily_snapshot)
        
        # 核心逻辑更新：使用风险调整后收益 (未来收益 / 未来最大回撤的绝对值)
        # 增加风险惩罚因子：回撤权重翻倍，加 0.02 防止除以 0
        df_day['performance_score'] = df_day['future_ret'] / (df_day['future_mdd'].abs() * 2 + 0.02)
        
        max_score = df_day['performance_score'].max()
        
        # 只有当最优评分的未来收益大于 0 时，才标记为 1，否则全部选现金
        # 这确保了模型学习：如果全场都在跌，最优选择是现金
        best_row = df_day[df_day['performance_score'] == max_score].iloc[0]
        if best_row['future_ret'] > 0:
            df_day['target'] = (df_day['performance_score'] == max_score).astype(int)
        else:
            df_day['target'] = (df_day['name'] == '现金').astype(int)
        
        # 4. 特征排名 (Rank)
        for w in WINDOWS:
            for f in ['ret', 'vol', 'slope', 'r2', 'mdd', 'sxr', 'sharp']:
                df_day[f'rank_{f}_{w}'] = df_day[f'{f}_{w}'].rank(pct=True)
        
        # 市场大环境
        non_cash = df_day[df_day['name'] != '现金']
        df_day['market_max_slope'] = non_cash['slope_23'].max() if not non_cash.empty else 0
        df_day['market_max_ret'] = non_cash['ret_23'].max() if not non_cash.empty else 0
        
        dataset.extend(df_day.to_dict('records'))
        
    df_final = pd.DataFrame(dataset)
    # 移除未来收益和未来风险字段，避免模型泄露
    df_final = df_final.drop(columns=['future_ret', 'future_mdd', 'performance_score'])
    
    print(f"✅ 数据集构建完成：共 {len(df_final)} 条样本。")
    df_final.to_csv('performance_optimized_dataset.csv', index=False)
    print("💾 已保存至 performance_optimized_dataset.csv")

if __name__ == '__main__':
    build_performance_dataset()
