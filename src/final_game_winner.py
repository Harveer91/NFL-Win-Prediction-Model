import pandas as pd
from sqlalchemy import text
from load_data import Database_engine
import joblib

#loading database (labeled-play-by-play-data)
with Database_engine.connect() as conn:
    df = pd.read_sql(text("SELECT * FROM play_by_play_data_labeled"), con=conn)

#load trained model
model = joblib.load("xgb_play_by_play_model.pkl")

#removing cols from input 
columns_to_remove = ["game_id", "play_id", "home_team", "away_team", "posteam", "defteam", "winning_team", "posteam_win"]
feature_cols = []

for col in df.columns:
    if col not in columns_to_remove:
        feature_cols.append(col)

#creating the input features
X = df[feature_cols]
#translating df info for ml by converting categorical collous into binary coloumns(one-hot encoding)
X = pd.get_dummies(X)

# Ensure all expected model input columns exist in X (add missing ones as 0s)
model_features = model.get_booster().feature_names

for col in model_features:
    if col not in X.columns:
        X[col] = 0


X = X[model_features]

#Prediction
df['predicted_posteam_win'] = model.predict(X)

def decide_final_winner(game_df):
    home_team = game_df['home_team'].iloc[0]
    away_team = game_df['away_team'].iloc[0]

    home_win_plays = 0
    away_win_plays = 0

    for _, row in game_df.iterrows():
        if row['predicted_posteam_win'] == 1:
            if row['posteam'] == home_team:
                home_win_plays += 1
            elif row['posteam'] == away_team:
                away_win_plays += 1

    if home_win_plays > away_win_plays:
        return home_team
    elif away_win_plays > home_win_plays:
        return away_team
    else:
        return "Tie"
    
final_game_winners = df.groupby('game_id').apply(decide_final_winner).reset_index(name='projected_winner')

print(final_game_winners.head(17))