"""Predict every NFL game and export model-performance data for the frontend.

    python src/final_game_winner.py

Steps: download nflverse schedules + play-by-play, build pre-game team ratings,
back-test the symmetric model season by season (train on past seasons only),
fit on all completed games, then predict the rest of the current season.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from pregame_features import DATA_DIR, FEATURES, build_features, load_games, load_team_game_epa
from pregame_model import TRAIN_START_SEASON, SymmetricGameModel, walk_forward

ROOT = os.path.dirname(DATA_DIR)
FRONTEND_DATA_DIR = os.path.join(ROOT, "Frontend", "public", "data")
MODEL_PATH = os.path.join(ROOT, "models", "pregame_model.pkl")

FIRST_REPORT_SEASON = 2009
ATS_EDGE_THRESHOLD = 1.5  # points between model margin and the spread before we "bet"
WIN_PAYOUT = 100 / 110  # standard -110 juice
CALIBRATION_BINS = np.linspace(0, 1, 11)


def _ats_outcome(row):
    edge = row["pred_margin"] - row["spread_line"]
    if pd.isna(row["result"]) or pd.isna(edge) or abs(edge) < ATS_EDGE_THRESHOLD:
        return None, None
    side = "home" if edge > 0 else "away"
    cover = row["result"] - row["spread_line"]
    if cover == 0:
        return side, "push"
    won = (cover > 0) == (side == "home")
    return side, "win" if won else "loss"


def annotate(preds):
    preds = preds.copy()
    preds["pick"] = np.where(preds["home_win_prob"] >= 0.5, preds["home_team"], preds["away_team"])
    preds["pick_prob"] = np.maximum(preds["home_win_prob"], 1 - preds["home_win_prob"])
    played = preds["result"].notna()
    preds["winner"] = np.where(
        preds["result"] > 0, preds["home_team"], np.where(preds["result"] < 0, preds["away_team"], "TIE")
    )
    preds.loc[~played, "winner"] = None
    graded = played & (preds["result"] != 0)
    preds["correct"] = (preds["pick"] == preds["winner"]).astype(object).where(graded, None)
    ats = preds.apply(_ats_outcome, axis=1, result_type="expand")
    preds["ats_side"], preds["ats_result"] = ats[0], ats[1]
    preds["units"] = preds["ats_result"].map({"win": WIN_PAYOUT, "loss": -1.0, "push": 0.0})
    return preds


def summarize(preds):
    decided = preds[preds["result"].notna() & (preds["result"] != 0)]
    y = (decided["result"] > 0).astype(int)
    p = decided["home_win_prob"].clip(1e-6, 1 - 1e-6)
    market = decided[decided["spread_line"].notna() & (decided["spread_line"] != 0)]
    with_line = preds[preds["result"].notna() & preds["spread_line"].notna()]
    bets = preds[preds["ats_result"].notna()]
    wins, losses = (bets["ats_result"] == "win").sum(), (bets["ats_result"] == "loss").sum()
    n = len(decided)
    if n == 0:
        return None
    su = float((decided["correct"] == True).mean())  # noqa: E712
    market_su = float(((market["spread_line"] > 0) == (market["result"] > 0)).mean()) if len(market) else None
    mae = float((decided["result"] - decided["pred_margin"]).abs().mean())
    market_mae = float((with_line["result"] - with_line["spread_line"]).abs().mean()) if len(with_line) else None
    return {
        "games": int(n),
        "su": su,
        "market_su": market_su,
        "su_vs_market": su - market_su if market_su is not None else None,
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "mae": mae,
        "market_mae": market_mae,
        "mae_vs_market": mae - market_mae if market_mae is not None else None,
        "ats_bets": int(len(bets)),
        "ats_wins": int(wins),
        "ats_losses": int(losses),
        "ats_pushes": int((bets["ats_result"] == "push").sum()),
        "ats_pct": float(wins / (wins + losses)) if wins + losses else None,
        "units": float(bets["units"].sum()),
        "home_pick_rate": float((decided["home_win_prob"] >= 0.5).mean()),
        "market_home_fav_rate": float((market["spread_line"] > 0).mean()) if len(market) else None,
        "home_win_rate": float(y.mean()),
        "mean_home_prob": float(decided["home_win_prob"].mean()),
    }


def calibration(preds):
    decided = preds[preds["result"].notna() & (preds["result"] != 0)]
    bins = pd.cut(decided["home_win_prob"], CALIBRATION_BINS, include_lowest=True)
    grouped = decided.groupby(bins, observed=True)
    return [
        {
            "bin": f"{int(iv.left * 100)}-{int(iv.right * 100)}%",
            "predicted": float(g["home_win_prob"].mean()),
            "actual": float((g["result"] > 0).mean()),
            "games": int(len(g)),
        }
        for iv, g in grouped
    ]


def cumulative_units(preds):
    bets = preds[preds["ats_result"].notna()].sort_values(["season", "week", "gameday"])
    weekly = bets.groupby(["season", "week"], as_index=False)["units"].sum()
    weekly["cumulative"] = weekly["units"].cumsum()
    return [
        {"season": int(r.season), "week": int(r.week), "units": round(float(r.units), 3),
         "cumulative": round(float(r.cumulative), 3)}
        for r in weekly.itertuples()
    ]


def power_ratings(model, ratings, active_teams):
    """Points better than a league-average team on a neutral field."""
    coefs = model.margin_coefficients()
    ratings = ratings[ratings["team"].isin(active_teams)].copy()
    centred = {
        "elo_diff": ratings["elo"] - ratings["elo"].mean(),
        "off_epa_diff": ratings["off_epa"] - ratings["off_epa"].mean(),
        "def_epa_diff": ratings["def_epa"] - ratings["def_epa"].mean(),
        "pt_diff_diff": ratings["pt_diff"] - ratings["pt_diff"].mean(),
    }
    ratings["rating"] = sum(coefs[k] * v for k, v in centred.items())
    ratings = ratings.sort_values("rating", ascending=False).reset_index(drop=True)
    ratings["rank"] = ratings.index + 1
    return [
        {"rank": int(r.rank), "team": r.team, "rating": round(float(r.rating), 2), "elo": round(float(r.elo), 1),
         "off_epa": round(float(r.off_epa), 4), "def_epa": round(float(r.def_epa), 4),
         "pt_diff": round(float(r.pt_diff), 2)}
        for r in ratings.itertuples()
    ]


def game_records(preds):
    cols = ["game_id", "season", "game_type", "week", "gameday", "gametime", "home_team", "away_team",
            "home_score", "away_score", "result", "spread_line", "home_win_prob", "pred_margin",
            "pick", "pick_prob", "correct", "ats_side", "ats_result", "units", "home_elo", "away_elo"]
    out = preds[cols].copy()
    out["gameday"] = out["gameday"].dt.strftime("%Y-%m-%d")
    out["home_win_prob"] = out["home_win_prob"].round(4)
    out["pick_prob"] = out["pick_prob"].round(4)
    out["pred_margin"] = out["pred_margin"].round(1)
    out["home_elo"] = out["home_elo"].round(0)
    out["away_elo"] = out["away_elo"].round(0)
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict(orient="records")


def write_json(name, payload):
    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    with open(os.path.join(FRONTEND_DATA_DIR, name), "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def main():
    print("Loading schedules...")
    games = load_games()
    current_season = int(games.loc[games["result"].notna(), "season"].max())
    print("Loading play-by-play EPA...")
    team_epa = load_team_game_epa(range(games["season"].min(), current_season + 1),
                                  refresh_seasons=(current_season,))
    features, ratings = build_features(games, team_epa)

    print(f"Back-testing {FIRST_REPORT_SEASON}-{current_season} (train on prior seasons only)...")
    backtest = annotate(walk_forward(features, FIRST_REPORT_SEASON, current_season))
    completed_backtest = backtest[backtest["result"].notna()]

    print("Fitting final model on all completed games...")
    model = SymmetricGameModel().fit(features[(features["season"] >= TRAIN_START_SEASON)])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    current = features[features["season"] == current_season].copy()
    upcoming_mask = current["result"].isna()
    # Completed games keep their out-of-sample back-test prediction; future games use the final model.
    current = current.merge(completed_backtest[["game_id", "home_win_prob", "pred_margin"]], on="game_id", how="left")
    current.loc[upcoming_mask.values, "home_win_prob"] = model.predict_home_win_prob(current[upcoming_mask.values])
    current.loc[upcoming_mask.values, "pred_margin"] = model.predict_home_margin(current[upcoming_mask.values])
    current = annotate(current)
    all_preds = pd.concat([backtest[backtest["season"] < current_season], current], ignore_index=True)

    seasons = []
    for season, group in all_preds.groupby("season"):
        stats = summarize(group)
        if stats:
            seasons.append({"season": int(season), **stats})
    upcoming_weeks = current.loc[current["result"].isna(), "week"]
    current_week = int(upcoming_weeks.min()) if len(upcoming_weeks) else int(current["week"].max())

    win_coefs = model.coefficients()
    performance = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_season": current_season,
        "current_week": current_week,
        "first_season": FIRST_REPORT_SEASON,
        "ats_edge_threshold": ATS_EDGE_THRESHOLD,
        "model": {
            "name": "Symmetric pre-game logistic model",
            "features": FEATURES,
            "coefficients": win_coefs,
            "home_field_points": model.margin_coefficients()["venue"],
            "home_field_win_prob": float(1 / (1 + np.exp(-win_coefs["venue"]))),
        },
        "summary": {
            "all_time": summarize(all_preds),
            "current_season": summarize(current),
        },
        "seasons": seasons,
        "cumulative_units": cumulative_units(all_preds),
        "calibration": calibration(all_preds),
    }
    write_json("performance.json", performance)
    recent = all_preds[all_preds["season"] >= current_season - 1]
    write_json("games.json", {"current_season": current_season, "current_week": current_week,
                              "games": game_records(recent)})
    write_json("ratings.json", {"season": current_season, "as_of_week": current_week,
                                "teams": power_ratings(model, ratings, set(current["home_team"]))})

    for season in (current_season - 1, current_season):
        rows = all_preds[all_preds["season"] == season]
        rows[["game_id", "week", "home_team", "away_team", "home_win_prob", "pred_margin", "pick", "pick_prob",
              "spread_line", "result"]].round(4).to_csv(os.path.join(ROOT, f"{season}_season_predictions.csv"), index=False)

    s = performance["summary"]["all_time"]
    print(f"\n{FIRST_REPORT_SEASON}-{current_season} out-of-sample: {s['games']} games")
    print(f"  Straight-up accuracy {s['su']:.1%} (market favourite {s['market_su']:.1%})")
    print(f"  Home picks {s['home_pick_rate']:.1%} | market home favourites {s['market_home_fav_rate']:.1%}"
          f" | home teams actually won {s['home_win_rate']:.1%}")
    print(f"  Brier {s['brier']:.4f} | MAE {s['mae']:.2f} (market {s['market_mae']:.2f})")
    print(f"  ATS {s['ats_wins']}-{s['ats_losses']}-{s['ats_pushes']} ({s['ats_pct']:.1%}), units {s['units']:+.1f}")

if __name__ == "__main__":
    main()
