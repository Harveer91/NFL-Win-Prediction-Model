import pandas as pd

def clean_data(df):

    df = df[
        #ensures playtype is focused on standard plays, not marginal cases 
        (df["play_type"].isin(['run', 'pass'])) &
        #ensures their is an active field postion for the data
        (df["yardline_100"].notna()) &
        #ensures a team actually has possision of the ball (1 is on offense, other on defense)
        (df["posteam"].notna()) &
        #ensures data has a down corresponding to it
        (df["down"].notna()) &
        #ensures data is from during gametime 
        (df["game_seconds_remaining"] > 0)
    ]

    #keeping all preplay coloums, filters out already existing probability models and result oriented data
    data_to_keep = [
        "game_id", "play_id", "home_team", "away_team", "posteam", "defteam", "posteam_type",
        "season_type", "week", "qtr", "quarter_seconds_remaining", "half_seconds_remaining",
        "game_seconds_remaining", "game_half", "quarter_end", "drive",
        "down", "goal_to_go", "yardline_100", "ydstogo",
        "shotgun", "no_huddle", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike",
        "side_of_field", "yrdln", "time",
        "home_timeouts_remaining", "away_timeouts_remaining",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "timeout", "timeout_team"
    ]


    cols_to_keep = []

    #filters through dataframe to keep the coloumns that are 
    for col in data_to_keep:
        if col in df.columns:
            cols_to_keep.append(col)
    
    #drops the unwanted cols while keeping the wanted ones 
    df = df[cols_to_keep]

    return df 



