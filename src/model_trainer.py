import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import Database_engine

df = pd.read_sql("SELECT * FROM play_by_play_data_labeled", con=Database_engine)

columns_to_remove = ["game_id", "play_id", "home_team", "away_team", "posteam", "defteam", "winning_team", "posteam_win"]

cols_feature = []

for col in df.columns:
    if col not in columns_to_remove:
        cols_feature.append(col)

