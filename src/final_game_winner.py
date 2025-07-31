import pandas as pd
import nfl_data_py as nfl
from sqlalchemy import text
from load_data import Database_engine
import joblib

#loading database (labeled-play-by-play-data)
with Database_engine.connect() as conn:
    df = pd.read_sql(text("SELECT * FROM play_by_play_data_labeled"), con=conn)

#load trained model
model = joblib.load("xgb_play_by_play_model.pkl")

#2025 schedule 
schedule_2025 = nfl.import_schedules([2025])
#print(schedule_2025.head(256)) only to confirm if the 2025 exists in wrapper

#simulates 
home_teams_df = schedule_2025.copy()
home_teams_df['posteam'] = home_teams_df['home_team']
home_teams_df['possession_side'] = 'home'

away_teams_df = schedule_2025.copy()
away_teams_df['posteam'] = away_teams_df['away_team']
away_teams_df['possession_side'] = 'away'

prediction_df = pd.concat([home_teams_df, away_teams_df], ignore_index=True)

# One-hot encode the 'posteam' column
X = pd.get_dummies(prediction_df['posteam'])


model_features = model.get_booster().feature_names

for col in model_features:
    if col not in X.columns:
        X[col] = 0

# Reorder columns to match model's expected input
X = X[model_features]

prediction_df['posteam_win_pred'] = model.predict(X)

# aggregate to find final winner
def get_winner(group):
    home_row = group[group['possession_side'] == 'home']
    away_row = group[group['possession_side'] == 'away']
    
    if home_row['posteam_win_pred'].values[0] > away_row['posteam_win_pred'].values[0]:
        return home_row['posteam'].values[0]
    else:
        return away_row['posteam'].values[0]

final_predictions = prediction_df.groupby('game_id').apply(get_winner).reset_index(name='projected_winner')

# Merge with schedule
final_predictions = final_predictions.merge(
    schedule_2025[['game_id', 'home_team', 'away_team']],
    on='game_id',
    how='left'
)
#test to see if the model works 
print(final_predictions.head(17))
