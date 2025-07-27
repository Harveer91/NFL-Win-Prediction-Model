import pandas as pd

def clean_data(df):
    df = df[
        (df["play_type"].isin(['run', 'pass'])) &
        (df["yardline_100"].notna()) &
        (df["posteam"].notna()) &
        (df["down"].notna()) &
        (df["game_seconds_remaining"] > 0)
    ]