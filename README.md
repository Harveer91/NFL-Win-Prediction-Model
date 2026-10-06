# NFL-Win-Prediction-Model

Pre-game NFL win probabilities and projected spreads, plus a React dashboard that tracks how the model has done against real results and the Vegas line.

## Model

`src/final_game_winner.py` runs the whole pipeline:

1. Downloads nflverse schedules (1999 onward) and play-by-play data into `data/cache/`. This folder is gitignored.
2. Builds each team's ratings using only games played *before* kickoff (`src/pregame_features.py`):
   - margin-of-victory Elo
   - exponentially weighted offensive and defensive EPA/play
   - point differential
   - rest days
3. Back-tests one season at a time from 2009: each season is predicted by a model trained only on earlier seasons (`src/pregame_model.py`).
4. Fits the final model on every completed game and predicts the rest of the current season.
5. Writes the outputs:
   - `models/pregame_model.pkl`
   - `<season>_season_predictions.csv`
   - `Frontend/public/data/*.json`

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/final_game_winner.py
```

### Design

- Every feature is a home-minus-away difference.
- Each training game is used twice, once mirrored with the teams swapped.
- There is no intercept, so swapping the two teams always gives `1 - p`.
- Home field is a single explicit `venue` term. Older seasons get less weight in training, so this term follows today's smaller home edge.

## Frontend

`Frontend/` is a Vite + React dashboard with five pages:

- **Model Performance:** summary cards, unit returns, accuracy vs Vegas, a season table, a home-pick monitor and calibration
- **Games:** weekly projections with win-probability bars and model vs Vegas spreads
- **Power Ratings**
- **Fraud-o-Meter:** season-to-date record vs the model's expected wins and Pythagorean wins, as a gauge, scatter and ranked board
- **Methodology**

It reads the JSON files that the pipeline writes.

```bash
cd Frontend && npm install && npm run dev
```
