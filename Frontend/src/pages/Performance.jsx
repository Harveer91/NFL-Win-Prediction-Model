import { useMemo, useState } from 'react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from 'recharts'
import ChartTooltip from '../components/ChartTooltip'
import Loading from '../components/Loading'
import Panel from '../components/Panel'
import Segmented from '../components/Segmented'
import StatCard from '../components/StatCard'
import { num, pct, signed, signedPct, tone } from '../lib/format'
import { useData } from '../lib/useData'

const C = { teal: '#2dd4bf', amber: '#fbbf24', rose: '#fb7185', sky: '#7dd3fc', grid: '#1f2a44', axis: '#64748b' }
const axis = { stroke: C.axis, fontSize: 11, tickLine: false, axisLine: false }

const Icon = ({ d }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
)

function SummaryCards({ s }) {
  return (
    <div className="stat-grid">
      <StatCard
        label="Accuracy"
        value={pct(s.su, 1)}
        caption="% of winners picked straight up"
        icon={<Icon d="M20 6 9 17l-5-5" />}
        rows={[
          { label: 'Vegas favourite', value: pct(s.market_su) },
          { label: 'vs Vegas', value: signedPct(s.su_vs_market), tone: tone(s.su_vs_market) },
        ]}
      />
      <StatCard
        label="Against the Spread"
        accent="amber"
        value={pct(s.ats_pct, 1)}
        caption={`${s.ats_wins}-${s.ats_losses}-${s.ats_pushes} on ${s.ats_bets} plays`}
        icon={<Icon d="M3 17l6-6 4 4 8-8M14 7h7v7" />}
        rows={[
          { label: 'Units (−110)', value: signed(s.units, 1), tone: tone(s.units) },
          { label: 'Break-even', value: '52.4%' },
        ]}
      />
      <StatCard
        label="Prediction Error"
        accent="sky"
        value={num(s.mae, 2)}
        caption="Mean absolute error of projected margin"
        icon={<Icon d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 6a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" />}
        rows={[
          { label: 'Vegas MAE', value: num(s.market_mae, 2) },
          { label: 'vs Vegas', value: signed(s.mae_vs_market, 2), tone: tone(s.mae_vs_market, true) },
        ]}
      />
      <StatCard
        label="Home-Pick Rate"
        accent="rose"
        value={pct(s.home_pick_rate, 1)}
        caption="Share of games where the model took the home side"
        icon={<Icon d="M3 11 12 3l9 8v10H3z" />}
        rows={[
          { label: 'Vegas home favourites', value: pct(s.market_home_fav_rate) },
          { label: 'Home teams actually won', value: pct(s.home_win_rate) },
        ]}
      />
    </div>
  )
}

function UnitsChart({ data, seasons }) {
  const [mode, setMode] = useState('cumulative')
  const points = useMemo(() => data.map((d, i) => ({ ...d, i, label: `${d.season} W${d.week}` })), [data])
  const ticks = useMemo(() => {
    const firstIdx = {}
    points.forEach((p) => (firstIdx[p.season] ??= p.i))
    return Object.values(firstIdx).filter((_, k) => k % 3 === 0)
  }, [points])
  const seasonUnits = seasons.map((s) => ({ season: s.season, units: s.units }))
  const last = points.at(-1)?.cumulative ?? 0
  const color = last >= 0 ? C.teal : C.rose

  return (
    <Panel
      title="Unit Returns"
      subtitle={`Betting the model side whenever its margin differs from the spread by 1.5+ pts`}
      actions={
        <Segmented
          label="Units view"
          value={mode}
          onChange={setMode}
          options={[{ value: 'cumulative', label: 'Cumulative' }, { value: 'season', label: 'By season' }]}
        />
      }
    >
      <div className="chart">
        <ResponsiveContainer width="100%" height={300}>
          {mode === 'cumulative' ? (
            <AreaChart data={points} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
              <defs>
                <linearGradient id="unitsFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="i" type="number" domain={['dataMin', 'dataMax']} ticks={ticks}
                tickFormatter={(i) => points[i]?.season} {...axis} />
              <YAxis {...axis} />
              <ReferenceLine y={0} stroke={C.axis} strokeDasharray="4 4" />
              <Tooltip content={<ChartTooltip labelFormatter={(i) => points[i]?.label} formatter={(v) => signed(v, 1)} />} />
              <Area type="monotone" dataKey="cumulative" name="Cumulative units" stroke={color} strokeWidth={2}
                fill="url(#unitsFill)" dot={false} isAnimationActive={false} />
            </AreaChart>
          ) : (
            <BarChart data={seasonUnits} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="season" {...axis} />
              <YAxis {...axis} />
              <ReferenceLine y={0} stroke={C.axis} />
              <Tooltip cursor={{ fill: '#ffffff08' }} content={<ChartTooltip formatter={(v) => signed(v, 1)} />} />
              <Bar dataKey="units" name="Units" radius={[4, 4, 0, 0]}>
                {seasonUnits.map((s) => <Cell key={s.season} fill={s.units >= 0 ? C.teal : C.rose} />)}
              </Bar>
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function AccuracyChart({ seasons }) {
  return (
    <Panel title="Accuracy vs Vegas" subtitle="Straight-up winners picked, by season">
      <div className="chart">
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={seasons} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="season" {...axis} />
            <YAxis domain={[0.5, 0.8]} ticks={[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]} tickFormatter={(v) => pct(v, 0)} {...axis} />
            <Tooltip cursor={{ fill: '#ffffff08' }} content={<ChartTooltip formatter={(v) => pct(v)} />} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: C.axis }} />
            <Bar dataKey="su" name="Model" fill={C.teal} radius={[4, 4, 0, 0]} barSize={14} />
            <Line dataKey="market_su" name="Vegas favourite" stroke={C.amber} strokeWidth={2} dot={{ r: 2.5 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function HomePickChart({ seasons }) {
  return (
    <Panel title="Home-Pick Monitor" subtitle="How often the model picked the home team vs Vegas home favourites and actual home wins">
      <div className="chart">
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={seasons} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="season" {...axis} />
            <YAxis domain={[0.4, 0.8]} ticks={[0.4, 0.5, 0.6, 0.7, 0.8]} tickFormatter={(v) => pct(v, 0)} {...axis} />
            <Tooltip content={<ChartTooltip formatter={(v) => pct(v)} />} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: C.axis }} />
            <Line dataKey="home_pick_rate" name="Model home picks" stroke={C.teal} strokeWidth={2} dot={false} />
            <Line dataKey="market_home_fav_rate" name="Vegas home favs" stroke={C.amber} strokeWidth={2} dot={false} strokeDasharray="5 4" />
            <Line dataKey="home_win_rate" name="Home win rate" stroke={C.rose} strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function CalibrationChart({ bins }) {
  const data = bins.filter((b) => b.games >= 10)
  return (
    <Panel title="Calibration" subtitle="When the model says X%, the home team should win X% of the time">
      <div className="chart">
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={data} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} />
            <XAxis dataKey="predicted" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} {...axis} />
            <YAxis dataKey="actual" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} {...axis} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.axis} strokeDasharray="4 4" />
            <Tooltip
              content={<ChartTooltip
                labelFormatter={(_, p) => `${p?.[0]?.payload.bin} bucket · ${p?.[0]?.payload.games} games`}
                formatter={(v) => pct(v)} />}
            />
            <Line dataKey="actual" name="Actual home win %" stroke={C.sky} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Scatter dataKey="actual" name="Actual home win %" fill={C.sky} legendType="none" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

const COLS = [
  { key: 'season', label: 'Season', fmt: (v) => v },
  { key: 'games', label: 'Games', fmt: (v) => v },
  { key: 'su', label: 'Model SU', fmt: (v) => pct(v) },
  { key: 'su_vs_market', label: 'vs Vegas', fmt: (v) => signedPct(v), tone: (v) => tone(v) },
  { key: 'mae', label: 'MAE', fmt: (v) => num(v) },
  { key: 'mae_vs_market', label: 'vs Vegas', fmt: (v) => signed(v, 2), tone: (v) => tone(v, true) },
  { key: 'brier', label: 'Brier', fmt: (v) => num(v, 3) },
  { key: 'ats_bets', label: 'Plays', fmt: (v) => v },
  { key: 'ats_pct', label: 'ATS', fmt: (v) => pct(v), tone: (v) => (v == null ? 'neutral' : v >= 0.524 ? 'good' : 'bad') },
  { key: 'units', label: 'Units', fmt: (v) => signed(v, 1), tone: (v) => tone(v) },
  { key: 'home_pick_rate', label: 'Home picks', fmt: (v) => pct(v, 0) },
]

function SeasonTable({ seasons, selected, onSelect }) {
  const [sort, setSort] = useState({ key: 'season', dir: -1 })
  const rows = useMemo(
    () => [...seasons].sort((a, b) => ((a[sort.key] ?? -Infinity) - (b[sort.key] ?? -Infinity)) * sort.dir),
    [seasons, sort],
  )
  const toggle = (key) => setSort((s) => ({ key, dir: s.key === key ? -s.dir : -1 }))
  return (
    <Panel title="Season Detail" subtitle="Click a season to focus the summary cards on it">
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              {COLS.map((c) => (
                <th key={c.key + c.label} onClick={() => toggle(c.key)} className={sort.key === c.key ? 'sorted' : ''}>
                  {c.label}
                  {sort.key === c.key && <span className="sort">{sort.dir > 0 ? '▲' : '▼'}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.season} onClick={() => onSelect(selected === r.season ? 'all' : r.season)}
                className={selected === r.season ? 'selected' : ''}>
                {COLS.map((c) => (
                  <td key={c.key + c.label} className={c.tone ? `tone-${c.tone(r[c.key])}` : ''}>{c.fmt(r[c.key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

export default function Performance() {
  const { data, error } = useData('performance')
  const [scope, setScope] = useState('all')
  if (!data) return <Loading error={error} />

  const scoped =
    scope === 'all' ? data.summary.all_time
      : scope === 'current' ? data.summary.current_season
        : data.seasons.find((s) => s.season === scope)
  const scopeLabel = scope === 'all' ? `${data.first_season}–${data.current_season}`
    : scope === 'current' ? `${data.current_season} season` : `${scope} season`
  const segValue = scope === 'all' || scope === 'current' ? scope : 'pinned'
  const options = [
    { value: 'all', label: 'All Time' },
    { value: 'current', label: 'This Season' },
    ...(segValue === 'pinned' ? [{ value: 'pinned', label: String(scope) }] : []),
  ]

  return (
    <div className="stack">
      <div className="hero">
        <div>
          <p className="eyebrow">Model · Performance</p>
          <h1>How the model has done</h1>
          <p className="lede">
            Every number below is out-of-sample: each season is predicted by a model trained only on the seasons
            before it, then graded against final scores and the Vegas closing spread.
          </p>
        </div>
      </div>

      <Panel
        title="Performance Summary"
        subtitle={scopeLabel}
        actions={<Segmented label="Summary scope" value={segValue} onChange={(v) => v !== 'pinned' && setScope(v)} options={options} />}
      >
        <SummaryCards s={scoped} />
      </Panel>

      <div className="grid-2">
        <UnitsChart data={data.cumulative_units} seasons={data.seasons} />
        <AccuracyChart seasons={data.seasons} />
      </div>

      <SeasonTable seasons={data.seasons} selected={scope} onSelect={setScope} />

      <div className="grid-2">
        <HomePickChart seasons={data.seasons} />
        <CalibrationChart bins={data.calibration} />
      </div>
    </div>
  )
}
