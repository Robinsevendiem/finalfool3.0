import pandas as pd
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import os

def run_autogluon():
    print("1. Loading Dataset...")
    df = pd.read_csv('imitation_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Split Train/Test (Time Series Split)
    dates = df['date'].unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    print(f"Splitting at {split_date}...")
    
    train_df = df[df['date'] < split_date].drop(columns=['date', 'name'])
    test_df_full = df[df['date'] >= split_date].copy() # Keep date/name for ranking eval
    test_df = test_df_full.drop(columns=['date', 'name'])
    
    # Target
    label = 'target'
    
    print("2. Training AutoGluon...")
    # Time limit: 10 minutes (600s) to allow more time for new features
    # Use 'best_quality' if possible, or stick to 'medium_quality'
    predictor = TabularPredictor(label=label, eval_metric='accuracy').fit(
        train_df, 
        time_limit=600, 
        presets='medium_quality'
    )
    
    print("\n3. Leaderboard...")
    lb = predictor.leaderboard(test_df)
    print(lb[['model', 'score_test', 'score_val']])
    
    # Evaluate Ranking Accuracy on Test Set
    # AutoGluon standard accuracy is row-wise. We need Ranking Accuracy.
    
    print("\n4. Evaluating Ranking Accuracy...")
    
    # Get probabilities for class 1
    # Check if target is 0/1 or categorical
    # predictor.predict_proba returns DataFrame
    probs = predictor.predict_proba(test_df)
    if 1 in probs.columns:
        score_col = 1
    else:
        # Maybe target is mapped? 
        score_col = probs.columns[-1] # Assume last column is positive class
        
    test_df_full['score'] = probs[score_col]
    
    dates_test = test_df_full['date'].unique()
    matches = 0
    total = 0
    
    for d in dates_test:
        day_rows = test_df_full[test_df_full['date'] == d]
        if day_rows.empty: continue
        
        real_targets = day_rows[day_rows['target'] == 1]
        if real_targets.empty: continue
        real_name = real_targets.iloc[0]['name']
        
        pred_row = day_rows.sort_values('score', ascending=False).iloc[0]
        pred_name = pred_row['name']
        
        if real_name == pred_name:
            matches += 1
        total += 1
        
    print(f"Daily Top-1 Ranking Accuracy (AutoGluon Best): {matches/total:.2%}")
    
    # Save Leaderboard
    lb.to_csv('autogluon_leaderboard.csv', index=False)
    print("Saved 'autogluon_leaderboard.csv'")

if __name__ == '__main__':
    run_autogluon()
