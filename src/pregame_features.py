"""Pre-game team-strength features built only from games played *before* kickoff.

Every feature is expressed as a home-minus-away difference so the model sees
both teams symmetrically; home field is a separate, explicit `venue` feature.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"

FIRST_SEASON = 1999
PBP_COLUMNS = ["game_id", "posteam", "defteam", "epa", "play_type", "season_type"]

# Relocated franchises share one rating history.
TEAM_ALIASES = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

ELO_K = 20.0
ELO_MEAN = 1500.0
ELO_SEASON_CARRYOVER = 2 / 3
EWM_ALPHA = 0.12
EWM_SEASON_CARRYOVER = 0.6
REST_CAP = 14

FEATURES = ["elo_diff", "off_epa_diff", "def_epa_diff", "pt_diff_diff", "rest_diff", "venue"]


def _download(url, path, refresh=False):
    if refresh or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_games(refresh=True):
    path = _download(GAMES_URL, os.path.join(CACHE_DIR, "games.csv"), refresh=refresh)
    games = pd.read_csv(path)
    games = games[games["season"] >= FIRST_SEASON].copy()
    for col in ("home_team", "away_team"):
        games[col] = games[col].replace(TEAM_ALIASES)
    games["gameday"] = pd.to_datetime(games["gameday"])
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_team_game_epa(seasons, refresh_seasons=()):
    """Offensive EPA/play for every team in every game (run + pass plays only)."""
    frames = []
    for season in seasons:
        path = os.path.join(CACHE_DIR, "pbp", f"play_by_play_{season}.parquet")
        try:
            _download(PBP_URL.format(year=season), path, refresh=season in refresh_seasons)
        except Exception as exc:  # season not published yet
            print(f"  skipping pbp {season}: {exc}")
            continue
        pbp = pd.read_parquet(path, columns=PBP_COLUMNS)
        pbp = pbp[pbp["play_type"].isin(["run", "pass"]) & pbp["epa"].notna()]
        frames.append(pbp.groupby(["game_id", "posteam"], as_index=False)["epa"].mean())
    epa = pd.concat(frames, ignore_index=True)
    epa["posteam"] = epa["posteam"].replace(TEAM_ALIASES)
    return epa.rename(columns={"posteam": "team", "epa": "off_epa"})


def _elo_mov_multiplier(margin, elo_gap):
    return np.log(abs(margin) + 1) * 2.2 / (elo_gap * 0.001 + 2.2)


def build_features(games, team_epa):
    """Walk through games chronologically, recording each team's ratings *before* the game."""
    epa_lookup = {(g, t): v for g, t, v in team_epa[["game_id", "team", "off_epa"]].itertuples(index=False)}

    elo, off_epa, def_epa, pt_diff = {}, {}, {}, {}
    last_season = {}
    rows = []

    def ewm(store, team, value):
        prev = store.get(team)
        store[team] = value if prev is None else (1 - EWM_ALPHA) * prev + EWM_ALPHA * value

    for g in games.itertuples(index=False):
        home, away, season = g.home_team, g.away_team, g.season
        for team in (home, away):
            if last_season.get(team) not in (None, season):
                elo[team] = ELO_MEAN + ELO_SEASON_CARRYOVER * (elo[team] - ELO_MEAN)
                for store in (off_epa, def_epa, pt_diff):
                    if team in store:
                        store[team] *= EWM_SEASON_CARRYOVER
            last_season[team] = season
            elo.setdefault(team, ELO_MEAN)

        venue = 0 if g.location == "Neutral" else 1
        rest_h = min(g.home_rest, REST_CAP) if pd.notna(g.home_rest) else 7
        rest_a = min(g.away_rest, REST_CAP) if pd.notna(g.away_rest) else 7
        rows.append({
            "game_id": g.game_id,
            "elo_diff": elo[home] - elo[away],
            "off_epa_diff": off_epa.get(home, 0.0) - off_epa.get(away, 0.0),
            "def_epa_diff": def_epa.get(home, 0.0) - def_epa.get(away, 0.0),
            "pt_diff_diff": pt_diff.get(home, 0.0) - pt_diff.get(away, 0.0),
            "rest_diff": rest_h - rest_a,
            "venue": venue,
            "home_elo": elo[home],
            "away_elo": elo[away],
        })

        if pd.isna(g.result):
            continue

        margin = g.result  # home score - away score
        expected_home = 1 / (1 + 10 ** (-(elo[home] - elo[away]) / 400))
        actual_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_gap = (elo[home] - elo[away]) if margin > 0 else (elo[away] - elo[home])
        shift = ELO_K * _elo_mov_multiplier(margin, winner_gap) * (actual_home - expected_home) if margin else 0.0
        elo[home] += shift
        elo[away] -= shift

        home_off, away_off = epa_lookup.get((g.game_id, home)), epa_lookup.get((g.game_id, away))
        if home_off is not None and away_off is not None:
            ewm(off_epa, home, home_off)
            ewm(off_epa, away, away_off)
            # Defensive rating: higher is better, i.e. negative EPA allowed.
            ewm(def_epa, home, -away_off)
            ewm(def_epa, away, -home_off)
        ewm(pt_diff, home, margin)
        ewm(pt_diff, away, -margin)

    feats = pd.DataFrame(rows)
    ratings = pd.DataFrame({
        "team": list(elo),
        "elo": [elo[t] for t in elo],
        "off_epa": [off_epa.get(t, 0.0) for t in elo],
        "def_epa": [def_epa.get(t, 0.0) for t in elo],
        "pt_diff": [pt_diff.get(t, 0.0) for t in elo],
    })
    return games.merge(feats, on="game_id"), ratings
