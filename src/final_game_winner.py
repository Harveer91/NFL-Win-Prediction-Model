import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sqlalchemy import text
from load_data import Database_engine
import joblib
from tqdm import tqdm

# Load your trained model
model = joblib.load("xgb_play_by_play_model.pkl")

# Load your labeled training data (this is what your model was actually trained on)
with Database_engine.connect() as conn:
    labeled_df = pd.read_sql(text("SELECT * FROM play_by_play_data_labeled"), con=conn)

# Get 2025 schedule
schedule_2025 = nfl.import_schedules([2025])

def get_training_data_stats(labeled_df):

    # Get available columns and their types
    available_cols = labeled_df.columns.tolist()
    
    stats = {}
    
    # Build stats for columns that actually exist in your labeled data
    if 'down' in available_cols:
        stats['down_dist'] = labeled_df['down'].value_counts(normalize=True).sort_index()
        
        if 'ydstogo' in available_cols:
            stats['ydstogo_stats'] = labeled_df.groupby('down')['ydstogo'].describe()
    
    if 'yardline_100' in available_cols:
        stats['yardline_stats'] = labeled_df['yardline_100'].describe()
    
    if 'qtr' in available_cols:
        stats['qtr_dist'] = labeled_df['qtr'].value_counts(normalize=True).sort_index()
    
    if 'game_seconds_remaining' in available_cols:
        stats['game_seconds_stats'] = labeled_df['game_seconds_remaining'].describe()
    
    if 'quarter_seconds_remaining' in available_cols:
        stats['quarter_seconds_stats'] = labeled_df['quarter_seconds_remaining'].describe()
    
    if 'half_seconds_remaining' in available_cols:
        stats['half_seconds_stats'] = labeled_df['half_seconds_remaining'].describe()
    
    # Check for score differential (might be in labeled data)
    score_cols = ['score_differential', 'home_score_diff', 'away_score_diff']
    for col in score_cols:
        if col in available_cols:
            stats['score_diff_stats'] = labeled_df[col].describe()
            stats['score_diff_col'] = col
            break
    
    return stats

def create_realistic_game_scenarios(home_team, away_team, labeled_df, n_scenarios=500):
    """
    Create realistic game scenarios based on your actual labeled data structure
    """
    # Get actual distributions from your labeled training data
    stats = get_training_data_stats(labeled_df)
    
    scenarios = []
    
    # Sample from multiple game phases to get balanced predictions
    game_phases = [
        {'phase': 'early', 'qtr_range': [1, 2], 'score_range': (-7, 7), 'weight': 0.3},
        {'phase': 'mid', 'qtr_range': [2, 3], 'score_range': (-14, 14), 'weight': 0.4}, 
        {'phase': 'late', 'qtr_range': [3, 4], 'score_range': (-21, 21), 'weight': 0.3}
    ]
    
    for phase in game_phases:
        phase_scenarios = int(n_scenarios * phase['weight'])
        
        for _ in range(phase_scenarios):
            
            # Down - use actual distribution if available
            if 'down_dist' in stats:
                down = np.random.choice(stats['down_dist'].index, p=stats['down_dist'].values)
            else:
                down = np.random.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.20, 0.05])
            
            # Yards to go - use actual stats if available
            if 'ydstogo_stats' in stats and down in stats['ydstogo_stats'].index:
                ydstogo_mean = stats['ydstogo_stats'].loc[down, 'mean']
                ydstogo_std = stats['ydstogo_stats'].loc[down, 'std']
                ydstogo = max(1, int(np.random.normal(ydstogo_mean, ydstogo_std)))
                ydstogo = min(ydstogo, 20)  # Cap at reasonable value
            else:
                # Default distribution based on down
                if down == 1:
                    ydstogo = 10
                elif down == 2:
                    ydstogo = np.random.randint(2, 15)
                elif down == 3:
                    ydstogo = np.random.randint(1, 20)
                else:  # down == 4
                    ydstogo = np.random.randint(1, 10)
            
            # Field position
            if 'yardline_stats' in stats:
                yardline_100 = max(1, min(99, int(np.random.normal(
                    stats['yardline_stats']['mean'], 
                    stats['yardline_stats']['std']
                ))))
            else:
                yardline_100 = np.random.randint(1, 100)
            
            # Quarter
            if 'qtr_dist' in stats:
                qtr = np.random.choice(stats['qtr_dist'].index, p=stats['qtr_dist'].values)
            else:
                qtr = np.random.choice(phase['qtr_range'])
            
            # Time remaining - use actual stats if available
            if 'game_seconds_stats' in stats:
                game_seconds_remaining = max(1, int(np.random.normal(
                    stats['game_seconds_stats']['mean'],
                    stats['game_seconds_stats']['std']
                )))
                game_seconds_remaining = min(game_seconds_remaining, 3600)  # Cap at 60 minutes
            else:
                # Default time distribution
                if qtr == 1:
                    game_seconds_remaining = np.random.randint(1800, 3600)
                elif qtr == 2:
                    game_seconds_remaining = np.random.randint(900, 1800)
                elif qtr == 3:
                    game_seconds_remaining = np.random.randint(900, 1800)
                else:  # qtr == 4
                    game_seconds_remaining = np.random.randint(0, 900)
            
            # Time remaining - use actual stats if available
            if 'game_seconds_stats' in stats:
                game_seconds_remaining = max(1, int(np.random.normal(
                    stats['game_seconds_stats']['mean'],
                    stats['game_seconds_stats']['std']
                )))
                game_seconds_remaining = min(game_seconds_remaining, 3600)  # Cap at 60 minutes
            else:
                # Default time distribution
                if qtr == 1:
                    game_seconds_remaining = np.random.randint(1800, 3600)
                elif qtr == 2:
                    game_seconds_remaining = np.random.randint(900, 1800)
                elif qtr == 3:
                    game_seconds_remaining = np.random.randint(900, 1800)
                else:  # qtr == 4
                    game_seconds_remaining = np.random.randint(0, 900)
            
            # Calculate other time fields
            quarter_seconds_remaining = game_seconds_remaining % 900
            half_seconds_remaining = game_seconds_remaining % 1800 if qtr <= 2 else (game_seconds_remaining % 1800) + 1800
            game_half = 1 if qtr <= 2 else 2
            
            # Create scenarios for both teams having possession
            for posteam in [home_team, away_team]:
                
                # Base scenario matching your cleaned data structure
                scenario = {
                    'posteam': posteam,
                    'defteam': away_team if posteam == home_team else home_team,
                    'down': down,
                    'ydstogo': ydstogo,
                    'yardline_100': yardline_100,
                    'qtr': qtr,
                    'game_seconds_remaining': game_seconds_remaining,
                    'quarter_seconds_remaining': quarter_seconds_remaining,
                    'half_seconds_remaining': half_seconds_remaining,
                    'game_half': game_half,
                }
                
                # Add default values for other columns that exist in your cleaned data
                # Based on your data_to_keep list
                default_values = {
                    'posteam_type': 'home' if posteam == home_team else 'away',
                    'season_type': 'REG',  # Regular season
                    'week': 1,  # Default week
                    'quarter_end': 0,
                    'drive': 1,
                    'goal_to_go': 1 if yardline_100 <= 10 else 0,
                    'shotgun': np.random.choice([0, 1], p=[0.4, 0.6]),
                    'no_huddle': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'qb_dropback': np.random.choice([0, 1], p=[0.5, 0.5]),
                    'qb_scramble': 0,
                    'qb_kneel': 0,
                    'qb_spike': 0,
                    'side_of_field': posteam,
                    'home_timeouts_remaining': 3,
                    'away_timeouts_remaining': 3,
                    'posteam_timeouts_remaining': 3,
                    'defteam_timeouts_remaining': 3,
                    'timeout': 0,
                    'timeout_team': None
                }
                
                # Add defaults to scenario
                for key, value in default_values.items():
                    scenario[key] = value
                
                # Add score differential with proper column name
                if 'score_diff_col' in stats:
                    adj_score_diff = score_differential if posteam == home_team else -score_differential
                    scenario[stats['score_diff_col']] = adj_score_diff
                
                # Add any remaining columns from your actual labeled data with their median/mode values
                for col in labeled_df.columns:
                    if col not in scenario and col not in ['game_id', 'play_id', 'home_team', 'away_team', 'winning_team', 'posteam_win']:
                        # Set reasonable defaults for missing features
                        if labeled_df[col].dtype in ['int64', 'float64']:
                            scenario[col] = labeled_df[col].median()
                        else:
                            mode_values = labeled_df[col].mode()
                            scenario[col] = mode_values.iloc[0] if len(mode_values) > 0 else None
                
                scenarios.append(scenario)
    
    return pd.DataFrame(scenarios)

def prepare_features_for_model(scenarios_df, model_features):
    """
    Prepare the scenario features to match your model's expected input
    This must match EXACTLY how you prepared your training data
    """
    # Start with the scenarios dataframe
    X = scenarios_df.copy()
    
    # Remove the same columns you removed during training
    columns_to_remove = ["game_id", "play_id", "home_team", "away_team", "posteam", "defteam", "winning_team", "posteam_win"]
    
    # Only remove columns that actually exist in the dataframe
    cols_to_drop = [col for col in columns_to_remove if col in X.columns]
    X = X.drop(columns=cols_to_drop)
    
    # Apply pd.get_dummies to ALL remaining features (just like in training)
    X = pd.get_dummies(X)
    
    # Add any missing columns that your model expects (set to 0)
    missing_cols = [col for col in model_features if col not in X.columns]
    for col in missing_cols:
        X[col] = 0
    
    # Remove any extra columns that weren't in training and reorder to match model
    X = X[[col for col in model_features if col in X.columns]]
    
    # If still missing some columns, add them as zeros
    final_missing = [col for col in model_features if col not in X.columns]
    for col in final_missing:
        X[col] = 0
    
    # Final reorder to match model exactly
    X = X[model_features]
    
    return X

def simulate_game(home_team, away_team, model, model_features, clean_df, n_scenarios=500):
    """
    Simulate a single game by creating multiple scenarios and averaging predictions
    """
    # Create realistic scenarios
    scenarios = create_realistic_game_scenarios(home_team, away_team, labeled_df, n_scenarios)
    
    # Prepare features for the model
    X = prepare_features_for_model(scenarios, model_features)
    
    # Get predictions for all scenarios
    try:
        predictions = model.predict_proba(X)[:, 1]  # Probability of class 1 (win)
    except:
        # Fallback to regular predict if predict_proba doesn't work
        predictions = model.predict(X)
    
    # Calculate win probability for each team
    home_scenarios = scenarios[scenarios['posteam'] == home_team]
    away_scenarios = scenarios[scenarios['posteam'] == away_team]
    
    home_indices = scenarios['posteam'] == home_team
    away_indices = scenarios['posteam'] == away_team
    
    home_win_prob = predictions[home_indices].mean()
    away_win_prob = predictions[away_indices].mean()
    
    # The team with higher average win probability wins
    if home_win_prob > away_win_prob:
        return home_team, home_win_prob
    else:
        return away_team, away_win_prob

# Debug: Check what columns are available in your labeled training data
print("Available columns in your labeled training data:")
print(labeled_df.columns.tolist())
print(f"\nTotal columns: {len(labeled_df.columns)}")

# Debug: Check model features
print("\nModel expects these features:")
model_features = model.get_booster().feature_names
print(f"Total features: {len(model_features)}")
print("Sample model features:", model_features[:10])

# Simulate all games in the 2025 schedule
print("Simulating 2025 NFL season games...")
results = []

for idx, game in tqdm(schedule_2025.iterrows(), total=len(schedule_2025), desc="Simulating games"):
    home_team = game['home_team']
    away_team = game['away_team']
    game_id = game['game_id']
    
    # Simulate the game
    winner, win_prob = simulate_game(home_team, away_team, model, model_features, labeled_df)
    
    results.append({
        'game_id': game_id,
        'home_team': home_team,
        'away_team': away_team,
        'projected_winner': winner,
        'win_probability': win_prob,
        'week': game['week'] if 'week' in game else None
    })

# Create final predictions dataframe
final_predictions = pd.DataFrame(results)

# Display results
print("\nSample predictions:")
print(final_predictions[['game_id', 'home_team', 'away_team', 'projected_winner', 'win_probability']].head(20))

# Check prediction distribution
home_wins = (final_predictions['projected_winner'] == final_predictions['home_team']).sum()
away_wins = (final_predictions['projected_winner'] == final_predictions['away_team']).sum()

print(f"\nPrediction Distribution:")
print(f"Home team wins: {home_wins}")
print(f"Away team wins: {away_wins}")
print(f"Home win percentage: {home_wins / len(final_predictions) * 100:.1f}%")

# Show win probability distribution
print(f"\nWin Probability Stats:")
print(f"Mean win probability: {final_predictions['win_probability'].mean():.3f}")
print(f"Min win probability: {final_predictions['win_probability'].min():.3f}")
print(f"Max win probability: {final_predictions['win_probability'].max():.3f}")

# Save results
final_predictions.to_csv('2025_season_predictions.csv', index=False)
print("\nPredictions saved to '2025_season_predictions.csv'")
