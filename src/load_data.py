import os 
import nfl_data_py as nfl 
import pandas as pd
from sqlalchemy import create_engine 
from dotenv import load_dotenv

load_dotenv()

#using load_dotenv to get database connection without leaking private info
PGHOST = os.getenv("PGHOST") 
PGPORT = os.getenv("PGPORT")
PGDATABASE =  os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
#check if user requires password, note to add it later if it does

DATABASE_url = f"postgresql+psycopg2://{PGUSER}@{PGHOST}:{PGPORT}/{PGDATABASE}"
 
Database_engine = create_engine(DATABASE_url)

#gets the play-by-play data for the 2024 NFL season 
nfl_pbp_DataFrame = nfl.import_pbp_data([2024]) 

team_df = nfl.import_seasonal_data([2024])

nfl_pbp_DataFrame.to_sql("play_by_play_data",Database_engine,if_exists='replace', index=False)
team_df.to_sql("seasonal_team_level_data", Database_engine, if_exists='replace', index=False)



