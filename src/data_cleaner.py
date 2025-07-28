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
    #removing data that would be unknown prior to a play starting in order to avoid breaking model 
    data_to_be_removed = data_to_be_removed = [
    # Descriptive text & outcome summary
    "desc", "yards_gained",

    # EPA-related columns
    "epa", "ep", "total_home_epa", "total_away_epa",
    "total_home_rush_epa", "total_away_rush_epa",
    "total_home_pass_epa", "total_away_pass_epa",
    "air_epa", "yac_epa", "comp_air_epa", "comp_yac_epa",
    "total_home_comp_air_epa", "total_away_comp_air_epa",
    "total_home_comp_yac_epa", "total_away_comp_yac_epa",
    "total_home_raw_air_epa", "total_away_raw_air_epa",
    "total_home_raw_yac_epa", "total_away_raw_yac_epa",

    # Win Probability columns 
    "wp", "def_wp", "home_wp", "away_wp",
    "home_wp_post", "away_wp_post",
    "vegas_wp", "vegas_home_wp",

    # Win Probability Added columns 
    "wpa", "vegas_wpa", "vegas_home_wpa",
    "air_wpa", "yac_wpa", "comp_air_wpa", "comp_yac_wpa",
    "total_home_comp_air_wpa", "total_away_comp_air_wpa",
    "total_home_comp_yac_wpa", "total_away_comp_yac_wpa",
    "total_home_raw_air_wpa", "total_away_raw_air_wpa",
    "total_home_raw_yac_wpa", "total_away_raw_yac_wpa",
    "total_home_pass_wpa", "total_away_pass_wpa",
    "total_home_rush_wpa", "total_away_rush_wpa",

    # Score/post-play outcome values
    "score_differential_post",
    "posteam_score_post", "defteam_score_post",
    "total_home_score", "total_away_score", "posteam_score", "defteam_score",

    # Touchdown info (outcome of play)
    "td_team", "td_player_name", "td_player_id",

    # FG/XP/2pt outcomes (post-play)
    "field_goal_result", "kick_distance",
    "extra_point_result", "two_point_conv_result",

    # Win prob subcomponents (leakage)
    "win_prob", "home_win_prob", "away_win_prob",
    "no_score_prob", "opp_fg_prob", "opp_safety_prob", "opp_td_prob",
    "fg_prob", "safety_prob", "td_prob",
    "extra_point_prob", "two_point_conversion_prob",

    # Play results 
    "incomplete_pass", "interception",
    "touchback", "punt_inside_twenty", "punt_downed", "punt_fair_catch",
    "punt_in_endzone", "punt_out_of_bounds",
    "kickoff_inside_twenty", "kickoff_in_endzone",
    "kickoff_out_of_bounds", "kickoff_downed",

    # First/third/fourth down conversion results (outcomes)
    "first_down_rush", "first_down_pass", "first_down_penalty",
    "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed"
]


    cols_to_drop = []

    #filters through dataframe to remove cols that were requested, if doesn't appear will skip it 
    for col in data_to_be_removed:
        if col in df.columns:
            cols_to_drop.append(col)
    
    #drops the unwanted columns
    df = df.drop(columns = cols_to_drop)

    return df 



