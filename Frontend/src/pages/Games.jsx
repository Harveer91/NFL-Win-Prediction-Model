import { useMemo, useState } from 'react'
import Loading from '../components/Loading'
import Panel from '../components/Panel'
import ProbBar from '../components/ProbBar'
import Segmented from '../components/Segmented'
import TeamBadge from '../components/TeamBadge'
import { TEAMS } from '../data/teams'
import { pct, spread, weekLabel } from '../lib/format'
import { useData } from '../lib/useData'

const fmtDate = (d, t) => {
  const date = new Date(`${d}T12:00:00`)
  const day = date.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
  return t ? `${day} · ${t} ET` : day
}

function GameCard({ g, threshold }) {
  const played = g.result != null
  const edge = g.spread_line != null ? g.pred_margin - g.spread_line : null
  const atsSide = edge != null && Math.abs(edge) >= threshold ? (edge > 0 ? g.home_team : g.away_team) : null
  return (
    <article className={`game ${played ? 'game--final' : ''}`}>
      <header className="game__head">
        <span>{fmtDate(g.gameday, played ? null : g.gametime)}</span>
        {played ? (
          <span className={`pill ${g.correct ? 'pill--good' : g.correct === false ? 'pill--bad' : ''}`}>
            {g.correct ? 'Correct' : g.correct === false ? 'Missed' : 'Tie'}
          </span>
        ) : (
          <span className="pill pill--live">Upcoming</span>
        )}
      </header>

      <div className="game__teams">
        {[['away', g.away_team, 1 - g.home_win_prob, g.away_score], ['home', g.home_team, g.home_win_prob, g.home_score]].map(
          ([side, abbr, prob, score]) => (
            <div key={side} className={`game__row ${g.pick === abbr ? 'is-pick' : ''}`}>
              <TeamBadge abbr={abbr} showName />
              <span className="game__side">{side === 'home' ? 'home' : ''}</span>
              <span className="game__prob">{pct(prob, 0)}</span>
              <span className="game__score">{played ? score : ''}</span>
            </div>
          ),
        )}
      </div>

      <ProbBar home={g.home_team} away={g.away_team} homeProb={g.home_win_prob} />

      <dl className="game__lines">
        <div>
          <dt>Model</dt>
          <dd>{spread(g.pred_margin, g.home_team, g.away_team)}</dd>
        </div>
        <div>
          <dt>Vegas</dt>
          <dd>{spread(g.spread_line, g.home_team, g.away_team)}</dd>
        </div>
        <div>
          <dt>ATS play</dt>
          <dd className={g.ats_result ? `tone-${g.ats_result === 'win' ? 'good' : g.ats_result === 'loss' ? 'bad' : 'neutral'}` : ''}>
            {atsSide ? `${atsSide}${g.ats_result ? ` · ${g.ats_result}` : ''}` : 'No play'}
          </dd>
        </div>
      </dl>
    </article>
  )
}

export default function Games() {
  const { data, error } = useData('games')
  const { data: perf } = useData('performance')
  const [season, setSeason] = useState(null)
  const [week, setWeek] = useState(null)
  const [teamFilter, setTeamFilter] = useState('')
  const [view, setView] = useState('all')

  const activeSeason = season ?? data?.current_season
  const seasonGames = useMemo(() => data?.games.filter((g) => g.season === activeSeason) ?? [], [data, activeSeason])
  const weeks = useMemo(() => {
    const map = new Map()
    seasonGames.forEach((g) => map.set(g.week, g.game_type))
    return [...map.entries()].sort((a, b) => a[0] - b[0])
  }, [seasonGames])

  if (!data) return <Loading error={error} />

  const defaultWeek = activeSeason === data.current_season ? data.current_week : weeks.at(-1)?.[0]
  const activeWeek = week ?? defaultWeek
  const weekIdx = weeks.findIndex(([w]) => w === activeWeek)
  const threshold = perf?.ats_edge_threshold ?? 1.5

  let games = teamFilter
    ? seasonGames.filter((g) => g.home_team === teamFilter || g.away_team === teamFilter)
    : seasonGames.filter((g) => g.week === activeWeek)
  if (view === 'upcoming') games = games.filter((g) => g.result == null)
  if (view === 'final') games = games.filter((g) => g.result != null)

  const graded = games.filter((g) => g.correct != null)
  const right = graded.filter((g) => g.correct).length

  return (
    <div className="stack">
      <div className="hero">
        <div>
          <p className="eyebrow">Model · Games</p>
          <h1>Game projections</h1>
          <p className="lede">
            Win probabilities and projected spreads for every game. Completed games show the prediction the model made
            before kickoff.
          </p>
        </div>
      </div>

      <div className="toolbar">
        <Segmented
          label="Season"
          value={activeSeason}
          onChange={(s) => {
            setSeason(s)
            setWeek(null)
          }}
          options={[data.current_season - 1, data.current_season].map((s) => ({ value: s, label: String(s) }))}
        />
        {!teamFilter && (
          <div className="weekpicker">
            <button type="button" disabled={weekIdx <= 0} onClick={() => setWeek(weeks[weekIdx - 1][0])} aria-label="Previous week">‹</button>
            <select value={activeWeek} onChange={(e) => setWeek(Number(e.target.value))} aria-label="Week">
              {weeks.map(([w, type]) => <option key={w} value={w}>{weekLabel(w, type)}</option>)}
            </select>
            <button type="button" disabled={weekIdx >= weeks.length - 1} onClick={() => setWeek(weeks[weekIdx + 1][0])} aria-label="Next week">›</button>
          </div>
        )}
        <select className="select" value={teamFilter} onChange={(e) => setTeamFilter(e.target.value)} aria-label="Filter by team">
          <option value="">All teams</option>
          {Object.entries(TEAMS).sort((a, b) => a[1].name.localeCompare(b[1].name)).map(([abbr, t]) => (
            <option key={abbr} value={abbr}>{t.name}</option>
          ))}
        </select>
        <Segmented label="Status" value={view} onChange={setView}
          options={[{ value: 'all', label: 'All' }, { value: 'upcoming', label: 'Upcoming' }, { value: 'final', label: 'Final' }]} />
      </div>

      <Panel
        title={teamFilter ? `${TEAMS[teamFilter]?.name} · ${activeSeason}` : `${weekLabel(activeWeek, weeks[weekIdx]?.[1])} · ${activeSeason}`}
        subtitle={graded.length ? `Model went ${right}-${graded.length - right} straight up` : `${games.length} games`}
      >
        {games.length ? (
          <div className="game-grid">
            {games.map((g) => <GameCard key={g.game_id} g={g} threshold={threshold} />)}
          </div>
        ) : (
          <p className="empty">No games match these filters.</p>
        )}
      </Panel>
    </div>
  )
}
