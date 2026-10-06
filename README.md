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

### Home-team bias fix

The earlier version simulated random in-game situations with a play-by-play XGBoost model and picked the home team in **96%** of 2025 games. Its accuracy was 53%. That model had no team-strength inputs and dropped score differential. As a result, `posteam_type = home` was the only thing that separated the two teams. Each side was also scored on its own, so the two win probabilities didn't add up to 1. Those predictions are kept in `data/legacy_2025_season_predictions.csv` for comparison.

How the current model avoids the bias:

- Every feature is a home-minus-away difference.
- Each training game is used twice, once mirrored with the teams swapped.
- There is no intercept, so swapping the two teams always gives `1 - p`.
- Home field is a single explicit `venue` term. Older seasons get less weight in training, so this term follows today's smaller home edge.

On the same 2025 games the current model picks the home side 59% of the time and is 62% accurate.

The original play-by-play pipeline (`load_data.py`, `data_cleaner.py`, `label_maker.py`, `model_trainer.py`) still exists but needs Postgres. The pre-game model doesn't use it.

## Frontend

`Frontend/` is a Vite + React dashboard with four pages:

- **Model Performance:** summary cards, unit returns, accuracy vs Vegas, a season table, a home-bias monitor and calibration
- **Games:** weekly projections with win-probability bars and model vs Vegas spreads
- **Power Ratings**
- **Methodology**

It reads the JSON files that the pipeline writes.

```bash
cd Frontend && npm install && npm run dev
```
