import Loading from '../components/Loading'
import Panel from '../components/Panel'
import { num, pct } from '../lib/format'
import { useData } from '../lib/useData'

const FEATURE_INFO = {
  elo_diff: ['Elo difference', 'Margin-of-victory Elo, regressed one third to the mean each offseason.'],
  off_epa_diff: ['Offensive EPA/play', 'Exponentially weighted EPA per run/pass play, from nflverse play-by-play.'],
  def_epa_diff: ['Defensive EPA/play', 'Same, for EPA allowed (positive = better defence).'],
  pt_diff_diff: ['Point differential', 'Exponentially weighted scoring margin.'],
  rest_diff: ['Rest days', 'Home rest minus away rest (capped at 14).'],
  venue: ['Venue', '+1 for the home team, 0 at neutral sites. The only place home field enters the model.'],
}

export default function Methodology() {
  const { data, error } = useData('performance')
  if (!data) return <Loading error={error} />
  return (
    <div className="stack narrow">
      <div className="hero">
        <div>
          <p className="eyebrow">About</p>
          <h1>Methodology</h1>
          <p className="lede">What the model uses, how it's tested, and how the Fraud-o-Meter is scored.</p>
        </div>
      </div>

      <Panel title="How it works">
        <div className="prose">
          <ul>
            <li><strong>Pre-game team strength.</strong> Ratings come only from games played before kickoff.</li>
            <li>
              <strong>Symmetric by construction.</strong> Every feature is a home-minus-away difference. Each training game
              appears twice, once mirrored, and there's no intercept. Swapping the teams gives exactly 1 − p.
            </li>
            <li>
              <strong>Home field is explicit.</strong> One <code>venue</code> term worth{' '}
              {num(data.model.home_field_points, 1)} points ({pct(data.model.home_field_win_prob, 1)} between equal teams).
              Older seasons get less weight, so it follows the smaller home edge of recent years.
            </li>
            <li>
              <strong>Honest back-test.</strong> Walk-forward by season, never training on the season being graded.
            </li>
          </ul>
        </div>
      </Panel>

      <Panel title="Features">
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Feature</th><th className="left">Description</th><th>Coefficient</th></tr></thead>
            <tbody>
              {data.model.features.map((f) => (
                <tr key={f}>
                  <td><strong>{FEATURE_INFO[f]?.[0] ?? f}</strong></td>
                  <td className="muted wrap">{FEATURE_INFO[f]?.[1]}</td>
                  <td>{num(data.model.coefficients[f], 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Grading">
        <div className="prose">
          <ul>
            <li><strong>Accuracy:</strong> the side with &gt;50% win probability won. Ties are excluded.</li>
            <li><strong>Vegas:</strong> the closing spread favourite from nflverse schedules.</li>
            <li><strong>MAE:</strong> mean absolute error between projected and actual home margin.</li>
            <li>
              <strong>ATS:</strong> a play is made when the projected margin differs from the spread by at least{' '}
              {data.ats_edge_threshold} points. Units assume −110 odds; break-even is 52.4%.
            </li>
          </ul>
        </div>
      </Panel>

      <Panel title="Fraud-o-Meter">
        <div className="prose">
          <p>
            For each team, season to date, the meter blends two gaps: actual wins minus the wins the model expected (the
            sum of its pre-game win probabilities), and actual wins minus Pythagorean wins from points scored and allowed.
            The point-differential gap counts for two thirds, the model gap for one third. The result is shrunk toward zero early in the season, then mapped to 0–100. 50 is a record that matches
            the underlying play; 75+ is a <strong>Certified Fraud</strong>, 25 or below is <strong>Snakebitten</strong>.
          </p>
        </div>
      </Panel>
    </div>
  )
}
