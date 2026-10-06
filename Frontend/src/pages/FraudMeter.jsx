import { useMemo, useState } from 'react'
import { CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import Loading from '../components/Loading'
import Panel from '../components/Panel'
import Segmented from '../components/Segmented'
import TeamBadge from '../components/TeamBadge'
import { team, TEAMS } from '../data/teams'
import { completedWeeks, FRAUD_TIERS, fraudTable } from '../lib/fraud'
import { num, pct, signed, weekLabel } from '../lib/format'
import { useData } from '../lib/useData'

const C = { grid: '#1f2a44', axis: '#64748b' }
const TONE_COLOR = { fraud: '#fb7185', watch: '#fbbf24', legit: '#94a3b8', under: '#7dd3fc', snake: '#2dd4bf' }
const axis = { stroke: C.axis, fontSize: 11, tickLine: false, axisLine: false }
const record = (r) => `${r.wins}-${r.losses}${r.ties ? `-${r.ties}` : ''}`

function Gauge({ score }) {
  const theta = Math.PI * (1 - score / 100)
  const nx = 100 + 68 * Math.cos(theta)
  const ny = 100 - 68 * Math.sin(theta)
  return (
    <svg viewBox="0 0 200 124" className="gauge" role="img" aria-label={`Fraud score ${Math.round(score)} of 100`}>
      <defs>
        <linearGradient id="gaugeFill" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor={TONE_COLOR.snake} />
          <stop offset="30%" stopColor={TONE_COLOR.under} />
          <stop offset="50%" stopColor={TONE_COLOR.legit} />
          <stop offset="70%" stopColor={TONE_COLOR.watch} />
          <stop offset="100%" stopColor={TONE_COLOR.fraud} />
        </linearGradient>
      </defs>
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#1a2742" strokeWidth="18" strokeLinecap="round" />
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeFill)" strokeWidth="14" strokeLinecap="round" />
      {[25, 40, 60, 75].map((t) => {
        const a = Math.PI * (1 - t / 100)
        return (
          <line key={t} x1={100 + 88 * Math.cos(a)} y1={100 - 88 * Math.sin(a)} x2={100 + 94 * Math.cos(a)}
            y2={100 - 94 * Math.sin(a)} stroke="#64748b" strokeWidth="1.5" />
        )
      })}
      <line x1="100" y1="100" x2={nx} y2={ny} stroke="#e6edf7" strokeWidth="3" strokeLinecap="round"
        style={{ transition: 'all .5s ease' }} />
      <text x="4" y="122" textAnchor="start" className="gauge__end">UNDERRATED</text>
      <text x="196" y="122" textAnchor="end" className="gauge__end">FRAUD</text>
      <circle cx="100" cy="100" r="7" fill="#e6edf7" />
      <circle cx="100" cy="100" r="3" fill="#070d1a" />
    </svg>
  )
}

function MeterCard({ r, season, weekText }) {
  const t = team(r.team)
  return (
    <Panel title="Fraud-o-Meter" subtitle={`${t.name} · ${season} through ${weekText}`} className="meter">
      <div className="meter__body">
        <div className="meter__dial">
          <Gauge score={r.score} />
          <div className="meter__score">
            <strong style={{ color: TONE_COLOR[r.tier.tone] }}>{Math.round(r.score)}</strong>
            <span className={`tier tier--${r.tier.tone}`}>{r.tier.label}</span>
          </div>
        </div>
        <dl className="meter__stats">
          <div><dt>Record</dt><dd>{record(r)}</dd></div>
          <div><dt>Model expected wins</dt><dd>{num(r.expWins, 1)}</dd></div>
          <div><dt>Wins vs model</dt><dd className={r.overModel > 0 ? 'tone-bad' : r.overModel < 0 ? 'tone-good' : ''}>{signed(r.overModel, 1)}</dd></div>
          <div><dt>Point differential</dt><dd>{signed(r.pointDiff, 0)}</dd></div>
          <div><dt>Pythagorean wins</dt><dd>{num(r.pythWins, 1)}</dd></div>
          <div><dt>Wins vs points</dt><dd className={r.overPyth > 0 ? 'tone-bad' : r.overPyth < 0 ? 'tone-good' : ''}>{signed(r.overPyth, 1)}</dd></div>
        </dl>
      </div>
    </Panel>
  )
}

function ScatterTip({ active, payload }) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  return (
    <div className="charttip">
      <div className="charttip__label">{team(r.team).name} · {record(r)}</div>
      <div className="charttip__row"><span>Actual win %</span><strong>{pct(r.winPct, 0)}</strong></div>
      <div className="charttip__row"><span>Model expected</span><strong>{pct(r.expWinPct, 0)}</strong></div>
      <div className="charttip__row"><span>Fraud score</span><strong>{Math.round(r.score)}</strong></div>
    </div>
  )
}

function RecordScatter({ rows, selected, onSelect }) {
  return (
    <Panel title="Record vs Model Expectation" subtitle="Above the line = winning more than the model's pre-game odds said they should">
      <div className="chart">
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 10, right: 16, left: -12, bottom: 4 }}>
            <CartesianGrid stroke={C.grid} />
            <XAxis dataKey="expWinPct" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} {...axis}
              name="Model expected win %" />
            <YAxis dataKey="winPct" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} {...axis} name="Win %" />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.axis} strokeDasharray="4 4" />
            <Tooltip content={<ScatterTip />} cursor={{ strokeDasharray: '3 3', stroke: C.axis }} />
            <Scatter data={rows} onClick={(p) => onSelect(p.team ?? p.payload?.team)} isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.team} fill={TONE_COLOR[r.tier.tone]} stroke={r.team === selected ? '#fff' : 'none'}
                  strokeWidth={2} r={r.team === selected ? 8 : 5} style={{ cursor: 'pointer' }} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function FraudBoard({ rows, selected, onSelect }) {
  return (
    <Panel title="Fraud Board" subtitle="Every team, most fraudulent first. Bars show distance from a legit 50.">
      <div className="fraudboard">
        <div className="fraudboard__row fraudboard__row--head">
          <span>#</span><span>Team</span><span>Record</span><span>xW</span><span>PD</span>
          <span className="fraudboard__bar">Underrated ← → Fraud</span><span>Score</span><span>Verdict</span>
        </div>
        {rows.map((r, i) => {
          const d = r.score - 50
          return (
            <button key={r.team} type="button" onClick={() => onSelect(r.team)}
              className={`fraudboard__row ${r.team === selected ? 'is-selected' : ''}`}>
              <span className="muted">{i + 1}</span>
              <span><TeamBadge abbr={r.team} showName /></span>
              <span className="mono">{record(r)}</span>
              <span className="mono">{num(r.expWins, 1)}</span>
              <span className={`mono tone-${r.pointDiff > 0 ? 'good' : r.pointDiff < 0 ? 'bad' : 'neutral'}`}>{signed(r.pointDiff, 0)}</span>
              <span className="fraudboard__bar">
                <span className="divbar">
                  <span className="divbar__fill" style={{
                    left: d >= 0 ? '50%' : `${50 + d}%`, width: `${Math.abs(d)}%`, background: TONE_COLOR[r.tier.tone],
                  }} />
                </span>
              </span>
              <span className="mono"><strong>{Math.round(r.score)}</strong></span>
              <span><span className={`tier tier--${r.tier.tone}`}>{r.tier.label}</span></span>
            </button>
          )
        })}
      </div>
    </Panel>
  )
}

export default function FraudMeter() {
  const { data, error } = useData('games')
  const [season, setSeason] = useState(null)
  const [week, setWeek] = useState(null)
  const [selected, setSelected] = useState(null)

  const activeSeason = season ?? data?.current_season
  const weeks = useMemo(() => (data ? completedWeeks(data.games, activeSeason) : []), [data, activeSeason])
  const activeWeek = week ?? weeks.at(-1)
  const rows = useMemo(
    () => (data && activeWeek != null ? fraudTable(data.games, activeSeason, activeWeek) : []),
    [data, activeSeason, activeWeek],
  )
  if (!data) return <Loading error={error} />

  const weekType = (w) => data.games.find((g) => g.season === activeSeason && g.week === w)?.game_type
  const pick = rows.find((r) => r.team === selected) ?? rows[0]
  const weekText = activeWeek != null ? weekLabel(activeWeek, weekType(activeWeek)) : ''

  return (
    <div className="stack">
      <div className="hero">
        <div>
          <p className="eyebrow">Model · Fraud-o-Meter</p>
          <h1>Is that record real?</h1>
          <p className="lede">
            Compares each team's record to the wins the model expected from its pre-game win probabilities and to what
            its point differential says it should have. Teams winning more than they "should" are frauds; teams losing
            more are underrated.
          </p>
        </div>
      </div>

      <div className="toolbar">
        <Segmented label="Season" value={activeSeason}
          onChange={(s) => { setSeason(s); setWeek(null) }}
          options={[data.current_season - 1, data.current_season].map((s) => ({ value: s, label: String(s) }))} />
        <label className="field">
          <span>Through</span>
          <select className="select" value={activeWeek ?? ''} onChange={(e) => setWeek(Number(e.target.value))} aria-label="Through week">
            {weeks.map((w) => <option key={w} value={w}>{weekLabel(w, weekType(w))}</option>)}
          </select>
        </label>
        <select className="select" value={pick?.team ?? ''} onChange={(e) => setSelected(e.target.value)} aria-label="Team">
          {rows.map((r) => r.team).sort((a, b) => (TEAMS[a]?.name ?? a).localeCompare(TEAMS[b]?.name ?? b)).map((t) => (
            <option key={t} value={t}>{TEAMS[t]?.name ?? t}</option>
          ))}
        </select>
        <div className="legend">
          {FRAUD_TIERS.map((t) => <span key={t.tone} className={`tier tier--${t.tone}`}>{t.label}</span>)}
        </div>
      </div>

      {pick ? (
        <>
          <div className="grid-2">
            <MeterCard r={pick} season={activeSeason} weekText={weekText} />
            <RecordScatter rows={rows} selected={pick.team} onSelect={setSelected} />
          </div>
          <FraudBoard rows={rows} selected={pick.team} onSelect={setSelected} />
        </>
      ) : (
        <Panel><p className="empty">No completed games yet this season.</p></Panel>
      )}
    </div>
  )
}
