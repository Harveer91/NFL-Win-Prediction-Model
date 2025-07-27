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

    #removing certain cols from database due to these coloumns showing predictive outcomes 
    data_to_be_removed = [
        "desc", "yards_gained", "epa", "ep", "score_differential_post",
        "posteam_score_post", "defteam_score_post",
        "td_team", "td_player_name", "td_player_id",
        "win_prob", "home_win_prob", "away_win_prob",
        "no_score_prob", "opp_fg_prob", "opp_safety_prob", "opp_td_prob",
        "fg_prob", "safety_prob", "td_prob",
        "extra_point_prob", "two_point_conversion_prob",
        "total_home_score", "total_away_score", "posteam_score", "defteam_score",
        "total_home_epa", "total_away_epa", "total_home_rush_epa", "total_away_rush_epa",
        "field_goal_result", "kick_distance", "extra_point_result", "two_point_conv_result"
    ]

    cols_to_drop = []

    #filters through dataframe to remove cols that were requested, if doesn't appear will skip it 
    for col in data_to_be_removed:
        if col in df.columns:
            cols_to_drop.append(col)
    
    #drops the unwanted columns
    df = df.drop(columns = cols_to_drop)

    return df 



