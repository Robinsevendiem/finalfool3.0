import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime, timedelta

import os
import sys

# --- Config ---
# Try to get from Streamlit Secrets first, then Env Var, then fallback (for local dev)
TS_TOKEN = None

try:
    import streamlit as st
    # Check if secrets file exists or if we are running in a context where secrets might be available
    # But st.secrets access might raise FileNotFoundError if no .streamlit/secrets.toml exists locally
    try:
        if hasattr(st, 'secrets') and 'TS_TOKEN' in st.secrets:
            TS_TOKEN = st.secrets['TS_TOKEN']
    except Exception:
        pass # Secrets not available
except ImportError:
    pass

# Fallback to Env Var (Passed by app.py subprocess or set in CI/CD)
if not TS_TOKEN:
    TS_TOKEN = os.environ.get('TS_TOKEN', '') # Please set TS_TOKEN in environment or streamlit secrets

DATA_DIR = 'market_data'

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

# Mapping for Tushare: Some ETFs might need different codes or handling
# Tushare uses '513100.SH' format directly.
TARGET_CODES = list(NAME_MAP.keys())

def update_market_data():
    print("🚀 开始更新市场数据...")
    
    # 1. Init Tushare
    # Tushare tries to write to ~/.tushare/token.csv or current dir.
    # In restricted env, we set token in pro_api directly to avoid file write.
    pro = ts.pro_api(TS_TOKEN)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    today = datetime.now().strftime('%Y%m%d')
    start_date = '20230101' # Default start if no file
    
    updated_count = 0
    
    for code in TARGET_CODES:
        name = NAME_MAP[code]
        filename = f"{code}_{name}.csv"
        # Sanitize filename
        filename = filename.replace('/', '_')
        filepath = os.path.join(DATA_DIR, filename)
        
        # Check existing file to find last date
        if os.path.exists(filepath):
            try:
                df_old = pd.read_csv(filepath)
                # Ensure date format
                if 'trade_date' in df_old.columns:
                    df_old['trade_date'] = df_old['trade_date'].astype(str)
                    last_date = df_old['trade_date'].max()
                    
                    # Convert to datetime to add 1 day
                    last_dt = datetime.strptime(last_date, '%Y%m%d')
                    fetch_start = (last_dt + timedelta(days=1)).strftime('%Y%m%d')
                else:
                    fetch_start = start_date
            except:
                fetch_start = start_date
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()
            fetch_start = start_date
            
        if fetch_start > today:
            print(f"✅ {name} ({code}) 已是最新。")
            continue
            
        print(f"📥 更新 {name} ({code}) | 范围: {fetch_start} -> {today}")
        
        try:
            # Fetch Daily Data (Price)
            df_new = pro.fund_daily(ts_code=code, start_date=fetch_start, end_date=today)
            
            if df_new.empty:
                print(f"   ⚠️ 无新数据")
                continue
                
            # Fetch Adj Factor (for QFQ)
            df_adj = pro.fund_adj(ts_code=code, start_date=fetch_start, end_date=today)
            
            if not df_adj.empty:
                # Merge to calc QFQ
                # Tushare fund_daily has: close, open, high, low, pre_close, change, pct_chg, vol, amount
                # Need to calculate close_qfq. 
                # Formula: price * adj_factor
                df_new = pd.merge(df_new, df_adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
                df_new['adj_factor'] = df_new['adj_factor'].fillna(1.0)
                
                cols = ['open', 'high', 'low', 'close', 'pre_close']
                for c in cols:
                    df_new[f'{c}_qfq'] = df_new[c] * df_new['adj_factor']
            else:
                # Fallback if no adj factor (rare for ETFs)
                cols = ['open', 'high', 'low', 'close', 'pre_close']
                for c in cols:
                    df_new[f'{c}_qfq'] = df_new[c]
            
            # Combine
            if not df_old.empty:
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['trade_date']).sort_values('trade_date')
            else:
                df_final = df_new.sort_values('trade_date')
                
            # Save
            df_final.to_csv(filepath, index=False)
            print(f"   💾 已保存 ({len(df_new)} 条新记录)")
            updated_count += 1
            
            # Rate limit protection
            time.sleep(0.3)
            
        except Exception as e:
            print(f"   ❌ 更新失败: {e}")
            
    print(f"\n🎉 更新完成! 共更新 {updated_count} 个文件。")
    return updated_count

if __name__ == '__main__':
    update_market_data()
