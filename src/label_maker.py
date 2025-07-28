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

#function that determines the winner of each game
def get_winning_team(row):
    if row["total_home_score"] > row["total_away_score"]:
        return row["home_team"]
    elif row["total_home_score"] < row["total_away_score"]:
        return row["away_team"]
    else:
        return "Tie"

#applys the get_winning_team function to every row in the game_results DataFrame
game_results["winning_team"] = game_results.apply(get_winning_team, axis=1)

#Merge game results into the cleaned play-by-play data
labeled_df = clean_df.merge(game_results[["game_id", "winning_team"]], on="game_id", how="left")

#Add a new column that is in binary based on posteam result
labeled_df["posteam_win"] = (labeled_df["posteam"] == labeled_df["winning_team"]).astype(int)
 
labeled_df = labeled_df[labeled_df["winning_team"] != "Tie"]

#Saved data to database
labeled_df.to_sql("play_by_play_data_labeled", con=Database_engine, if_exists="replace", index=False)
