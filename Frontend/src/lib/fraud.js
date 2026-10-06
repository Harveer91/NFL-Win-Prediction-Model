const PYTH_EXP = 2.37
const MODEL_WEIGHT = 1 / 3
const SHRINK_GAMES = 6
const SCALE = 0.15

export const FRAUD_TIERS = [
  { min: 75, label: 'Certified Fraud', tone: 'fraud' },
  { min: 60, label: 'Fraud Watch', tone: 'watch' },
  { min: 40, label: 'Legit', tone: 'legit' },
  { min: 25, label: 'Underrated', tone: 'under' },
  { min: -Infinity, label: 'Snakebitten', tone: 'snake' },
]

export const fraudTier = (score) => FRAUD_TIERS.find((t) => score >= t.min)

export const completedWeeks = (games, season) =>
  [...new Set(games.filter((g) => g.season === season && g.result != null).map((g) => g.week))].sort((a, b) => a - b)

// Record vs what the model expected (sum of pre-game win probabilities) and vs point differential.
export function fraudTable(games, season, throughWeek) {
  const rows = {}
  const row = (team) =>
    (rows[team] ??= { team, games: 0, wins: 0, losses: 0, ties: 0, expWins: 0, pf: 0, pa: 0, predMargin: 0 })

  games
    .filter((g) => g.season === season && g.result != null && g.week <= throughWeek)
    .forEach((g) => {
      ;[
        [g.home_team, g.home_win_prob, g.home_score, g.away_score, g.pred_margin],
        [g.away_team, 1 - g.home_win_prob, g.away_score, g.home_score, -g.pred_margin],
      ].forEach(([team, prob, pf, pa, margin]) => {
        const r = row(team)
        r.games += 1
        r.expWins += prob
        r.pf += pf
        r.pa += pa
        r.predMargin += margin
        if (pf > pa) r.wins += 1
        else if (pf < pa) r.losses += 1
        else r.ties += 1
      })
    })

  return Object.values(rows)
    .map((r) => {
      const actual = r.wins + r.ties / 2
      const pythWins = r.pf + r.pa > 0 ? (r.games * r.pf ** PYTH_EXP) / (r.pf ** PYTH_EXP + r.pa ** PYTH_EXP) : r.games / 2
      const overModel = actual - r.expWins
      const overPyth = actual - pythWins
      const index = (MODEL_WEIGHT * overModel + (1 - MODEL_WEIGHT) * overPyth) / (r.games + SHRINK_GAMES)
      const score = 50 + 50 * Math.tanh(index / SCALE)
      return {
        ...r,
        actual,
        winPct: actual / r.games,
        expWinPct: r.expWins / r.games,
        pythWins,
        pointDiff: r.pf - r.pa,
        overModel,
        overPyth,
        score,
        tier: fraudTier(score),
      }
    })
    .sort((a, b) => b.score - a.score)
}
