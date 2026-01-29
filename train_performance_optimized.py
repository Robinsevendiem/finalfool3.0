import pandas as pd
from autogluon.tabular import TabularPredictor
import os
from tqdm import tqdm

def train_performance_model():
    print("1. Loading Performance Optimized Dataset...")
    if not os.path.exists('performance_optimized_dataset.csv'):
        print("❌ 错误：找不到 performance_optimized_dataset.csv，请先运行 build_performance_dataset.py")
        return
        
    df = pd.read_csv('performance_optimized_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 按照时间切分 训练集 和 测试集
    dates = df['date'].unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    print(f"Splitting at {split_date}...")
    
    train_df = df[df['date'] < split_date].drop(columns=['date', 'name'])
    test_df_full = df[df['date'] >= split_date].copy()
    test_df = test_df_full.drop(columns=['date', 'name'])
    
    label = 'target'
    
    print("2. Training AutoGluon (Optimizing for Returns)...")
    # 这里的目标不再是模仿，而是通过特征预测未来哪个标的表现最好
    save_path = 'AutogluonModels/performance_v1'
    
    # 优化配置：极致稳定性，确保模型能够顺利生成
    predictor = TabularPredictor(
        label=label, 
        eval_metric='accuracy',
        path=save_path,
        verbosity=2
    ).fit(
        train_df, 
        time_limit=600, 
        presets='medium_quality', # 使用中等质量，禁用复杂的堆叠和袋装以确保在 macOS 上稳定运行
        num_bag_folds=0,          # 禁用袋装
        num_stack_levels=0,       # 禁用堆叠
    )
    
    print("\n3. Leaderboard...")
    lb = predictor.leaderboard(test_df)
    print(lb[['model', 'score_test', 'score_val']])
    
    # 评价 Top-1 命中率（即预测的最优标的在现实中是否真的是最优的）
    print("\n4. Evaluating Ranking Accuracy (Performance Focus)...")
    probs = predictor.predict_proba(test_df)
    score_col = 1 if 1 in probs.columns else probs.columns[-1]
    test_df_full['score'] = probs[score_col]
    
    dates_test = test_df_full['date'].unique()
    matches = 0
    total = 0
    
    for d in tqdm(dates_test, desc="回测验证进度"):
        day_rows = test_df_full[test_df_full['date'] == d]
        if day_rows.empty: continue
        
        # 寻找这一天真正的“冠军”标的
        real_winners = day_rows[day_rows['target'] == 1]
        if real_winners.empty: continue
        real_winner_name = real_winners.iloc[0]['name']
        
        # 寻找模型预测的“冠军”标的
        pred_winner = day_rows.sort_values('score', ascending=False).iloc[0]
        pred_winner_name = pred_winner['name']
        
        if real_winner_name == pred_winner_name:
            matches += 1
        total += 1
        
    print(f"Daily Top-1 Success Rate (Performance Optimized): {matches/total:.2%}")
    print(f"Model saved to: {save_path}")

if __name__ == '__main__':
    train_performance_model()
