"""Symmetric pre-game win-probability + margin model.

Each game is used twice in training: once from the home team's view and once
mirrored from the away team's view (features negated, venue flipped, label
flipped). With no intercept, swapping the two teams gives exactly 1 - p, so
the model can't favour the home side unless the explicit `venue` term earns it.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pregame_features import FEATURES

TRAIN_START_SEASON = 2001  # 1999-2000 used only to warm up ratings
# Older seasons count less, so the home-field term tracks today's (smaller) edge.
SEASON_WEIGHT_HALF_LIFE = 8


def _mirror(X, y=None, margin=None):
    Xm = pd.concat([X, -X], ignore_index=True)
    out = [Xm]
    if y is not None:
        out.append(np.concatenate([y, 1 - y]))
    if margin is not None:
        out.append(np.concatenate([margin, -margin]))
    return out


class SymmetricGameModel:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, games):
        played = games[games["result"].notna() & (games["result"] != 0)]
        X = played[FEATURES].astype(float)
        y = (played["result"] > 0).astype(int).to_numpy()
        margin = played["result"].to_numpy(dtype=float)
        Xm, ym, mm = _mirror(X, y, margin)
        age = played["season"].max() - played["season"].to_numpy()
        weight = np.tile(0.5 ** (age / SEASON_WEIGHT_HALF_LIFE), 2)

        # Scale without centring so the mirrored data stays symmetric around zero.
        self.win_model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(C=self.C, fit_intercept=False, max_iter=1000),
        ).fit(Xm, ym, logisticregression__sample_weight=weight)
        self.margin_model = make_pipeline(
            StandardScaler(with_mean=False),
            LinearRegression(fit_intercept=False),
        ).fit(Xm, mm, linearregression__sample_weight=weight)
        return self

    def predict_home_win_prob(self, games):
        return self.win_model.predict_proba(games[FEATURES].astype(float))[:, 1]

    def predict_home_margin(self, games):
        return self.margin_model.predict(games[FEATURES].astype(float))

    def coefficients(self):
        lr = self.win_model[-1]
        scale = self.win_model[0].scale_
        return dict(zip(FEATURES, (lr.coef_[0] / scale).tolist()))

    def margin_coefficients(self):
        lr = self.margin_model[-1]
        scale = self.margin_model[0].scale_
        return dict(zip(FEATURES, (lr.coef_ / scale).tolist()))


def walk_forward(games, first_test_season, last_test_season):
    """Train on every season before S, predict season S. Fully out-of-sample."""
    preds = []
    for season in range(first_test_season, last_test_season + 1):
        train = games[(games["season"] >= TRAIN_START_SEASON) & (games["season"] < season)]
        test = games[games["season"] == season].copy()
        model = SymmetricGameModel().fit(train)
        test["home_win_prob"] = model.predict_home_win_prob(test)
        test["pred_margin"] = model.predict_home_margin(test)
        preds.append(test)
    return pd.concat(preds, ignore_index=True)
