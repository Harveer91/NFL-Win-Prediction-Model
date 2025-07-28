import nfl_data_py as nfl
import pandas as pd
from load_data import Database_engine


clean_df = pd.read_sql("SELECT * FROM play_by_play_data_cleaned_version", con=Database_engine)
raw_df = raw_df = pd.read_sql("SELECT * FROM play_by_play_data", con=Database_engine)

#create new dataframe that groups all play_by_play data from each game into a group with the important info of the home team, away team & the final score
game_results = raw_df.groupby('game_id').agg({
    "home_team": "first",
    "away_team": "first",
    "total_home_score": "last",
    "total_away_score": "last"
}).reset_index()

