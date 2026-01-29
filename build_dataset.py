import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LinearRegression

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
DATA_DIR = 'market_data'

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
            if 'vol' in df.columns: df['volume'] = df['vol']
            
            name = None
            for k, v in NAME_MAP.items():
                if k in filename or k == code:
                    name = v
                    break
            if name:
                df['name'] = name
                df['code'] = code
                data[name] = df
    return data

def load_holdings():
    path = 'ori_strategy_info/2026-01-20T04-05_export持仓记录.csv'
    if not os.path.exists(path): return {}
    df = pd.read_csv(path)
    if '日期' in df.columns: df['date'] = pd.to_datetime(df['日期'])
    elif '交易日期' in df.columns: df['date'] = pd.to_datetime(df['交易日期'])
    if 'ETF名称' in df.columns: df['holding'] = df['ETF名称']
    elif '标的名称' in df.columns: df['holding'] = df['标的名称']
    df = df[['date', 'holding']].sort_values('date')
    return df.set_index('date')['holding'].to_dict()

def calc_slope(y):
    if len(y) < 2: return 0
    n = len(y)
    x = np.arange(n).reshape(-1, 1)
    y_norm = y / y[0]
    model = LinearRegression().fit(x, y_norm)
    return model.coef_[0]

def calc_r2(y):
    if len(y) < 2: return 0
    n = len(y)
    x = np.arange(n).reshape(-1, 1)
    y_norm = y / y[0]
    model = LinearRegression().fit(x, y_norm)
    return model.score(x, y_norm)

def calc_max_drawdown(prices):
    """Calculate Maximum Drawdown for a price series."""
    if len(prices) < 1: return 0.0
    # Calculate cumulative max
    roll_max = np.maximum.accumulate(prices)
    # Calculate drawdown
    drawdown = (prices - roll_max) / roll_max
    # Return max drawdown (min value, since dd is negative)
    return drawdown.min()

def build_dataset():
    print("1. Loading Data...")
    data_dict = load_data()
    holdings_map = load_holdings()
    
    # Windows to generate features for
    # Expanded window set: 3, 5, 10, 20, 23, 30, 60, 120
    WINDOWS = [3, 5, 10, 20, 23, 30, 60, 120]
    
    print("2. Generating Features...")
    # Structure: One row per (Date, Asset). 
    # Target: 1 if selected, 0 otherwise.
    
    dataset = []
    
    # Get all unique dates sorted
    all_dates = set()
    for df in data_dict.values():
        all_dates.update(df['date'])
    sorted_dates = sorted(list(all_dates))
    
    # Pre-calculate factors for efficiency
    processed_data = {}
    for name, df in data_dict.items():
        sub = df.set_index('date').sort_index()
        
        # Calculate features for multiple windows
        for w in WINDOWS:
            # Ret
            sub[f'ret_{w}'] = sub['close'].pct_change(w)
            
            # Vol
            sub[f'vol_{w}'] = sub['close'].pct_change().rolling(w).std() * np.sqrt(252)
            
            # Slope & R2 & MaxDD (Loop needed)
            slopes = []
            r2s = []
            mdds = []
            
            close_vals = sub['close'].values
            
            for i in range(len(sub)):
                if i < w:
                    slopes.append(np.nan)
                    r2s.append(np.nan)
                    mdds.append(np.nan)
                else:
                    window_data = close_vals[i-w+1 : i+1]
                    slopes.append(calc_slope(window_data))
                    r2s.append(calc_r2(window_data))
                    mdds.append(calc_max_drawdown(window_data))
            
            sub[f'slope_{w}'] = slopes
            sub[f'r2_{w}'] = r2s
            sub[f'mdd_{w}'] = mdds
            sub[f'slope_x_r2_{w}'] = sub[f'slope_{w}'] * sub[f'r2_{w}']
            
        processed_data[name] = sub

    # Build daily samples
    print("3. Assembling Dataset...")
    
    for i, date in enumerate(sorted_dates):
        if date not in holdings_map: continue
        real_holding = holdings_map[date]
        if real_holding == '未知': continue
        
        # Identify Previous Holding (for Buffer feature)
        prev_holding = "现金"
        if i > 0:
            prev_date = sorted_dates[i-1]
            if prev_date in holdings_map:
                prev_holding = holdings_map[prev_date]
        
        # Collect daily snapshot
        daily_snapshot = []
        
        # 1. Real Assets
        for name, df in processed_data.items():
            if date in df.index:
                row = df.loc[date]
                if pd.notnull(row['slope_23']): # Ensure main window exists
                    sample = {
                        'date': date,
                        'name': name,
                        'is_held': 1 if prev_holding == name else 0,
                        'target': 1 if real_holding == name else 0
                    }
                    
                    # Add features
                    for w in WINDOWS:
                        sample[f'ret_{w}'] = row[f'ret_{w}']
                        sample[f'vol_{w}'] = row[f'vol_{w}']
                        sample[f'slope_{w}'] = row[f'slope_{w}']
                        sample[f'r2_{w}'] = row[f'r2_{w}']
                        sample[f'mdd_{w}'] = row[f'mdd_{w}']
                        sample[f'sxr_{w}'] = row[f'slope_x_r2_{w}']
                    
                    daily_snapshot.append(sample)
        
        if not daily_snapshot: continue
        
        # 2. Add "Cash" Asset
        # Cash features: 0 slope, 0 ret, 0 vol... 
        # But wait, ML models need consistent scale. 
        # Cash is usually selected when others are bad.
        # So Cash features should be constant zeros? 
        # Yes, standardizing Cash as a zero-feature asset is reasonable.
        cash_sample = {
            'date': date,
            'name': '现金',
            'is_held': 1 if prev_holding == '现金' else 0,
            'target': 1 if real_holding == '现金' else 0
        }
        for w in WINDOWS:
            cash_sample[f'ret_{w}'] = 0.0
            cash_sample[f'vol_{w}'] = 0.0
            cash_sample[f'slope_{w}'] = 0.0
            cash_sample[f'r2_{w}'] = 0.0
            cash_sample[f'mdd_{w}'] = 0.0
            cash_sample[f'sxr_{w}'] = 0.0
            
        daily_snapshot.append(cash_sample)
        
        # 3. Add Cross-Sectional Ranks (Critical for Strategy)
        # We need to rank features WITHIN the day
        df_day = pd.DataFrame(daily_snapshot)
        
        # Rank features for each window
        feature_cols = []
        for w in WINDOWS:
            feature_cols.extend([f'ret_{w}', f'vol_{w}', f'slope_{w}', f'r2_{w}', f'mdd_{w}', f'sxr_{w}'])
            
        for col in feature_cols:
            # Rank excluding Cash? Or including? 
            # Strategy ranks excluding cash usually.
            # But here we want the model to learn comparison.
            # Let's rank everything. Cash with 0 will naturally fall in place.
            # Percentile Rank
            df_day[f'rank_{col}'] = df_day[col].rank(pct=True)
            
        # Add market context features (Max Slope, Max Ret of the day)
        # Identify non-cash assets
        non_cash = df_day[df_day['name'] != '现金']
        if not non_cash.empty:
            max_slope = non_cash['slope_23'].max()
            max_ret = non_cash['ret_23'].max()
        else:
            max_slope = 0
            max_ret = 0
            
        df_day['market_max_slope'] = max_slope
        df_day['market_max_ret'] = max_ret
        
        dataset.extend(df_day.to_dict('records'))
        
    df_final = pd.DataFrame(dataset)
    print(f"Dataset Built: {len(df_final)} samples (Asset-Days).")
    df_final.to_csv('imitation_dataset.csv', index=False)
    print("Saved to imitation_dataset.csv")

if __name__ == '__main__':
    build_dataset()
