import { useMemo, useState } from 'react'
import Loading from '../components/Loading'
import Panel from '../components/Panel'
import TeamBadge from '../components/TeamBadge'
import { team } from '../data/teams'
import { num, signed } from '../lib/format'
import { useData } from '../lib/useData'

const COLS = [
  { key: 'rank', label: 'Rank' },
  { key: 'team', label: 'Team' },
  { key: 'rating', label: 'Rating' },
  { key: 'elo', label: 'Elo' },
  { key: 'off_epa', label: 'Off EPA/play' },
  { key: 'def_epa', label: 'Def EPA/play' },
  { key: 'pt_diff', label: 'Pt diff (EWM)' },
]

export default function Ratings() {
  const { data, error } = useData('ratings')
  const [sort, setSort] = useState({ key: 'rank', dir: 1 })
  const rows = useMemo(() => {
    if (!data) return []
    return [...data.teams].sort((a, b) => {
      const av = a[sort.key]
      const bv = b[sort.key]
      return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * sort.dir
    })
  }, [data, sort])
  if (!data) return <Loading error={error} />

  const maxAbs = Math.max(...data.teams.map((t) => Math.abs(t.rating)))
  const toggle = (key) => setSort((s) => ({ key, dir: s.key === key ? -s.dir : key === 'rank' || key === 'team' ? 1 : -1 }))

  return (
    <div className="stack">
      <div className="hero">
        <div>
          <p className="eyebrow">Teams · Power Ratings</p>
          <h1>Power ratings</h1>
          <p className="lede">
            Points better or worse than a league-average team on a neutral field, from the model's current Elo, EPA and
            point-differential inputs. As of {data.season} week {data.as_of_week}.
          </p>
        </div>
      </div>
      <Panel title="All 32 teams">
        <div className="table-wrap">
          <table className="table table--ratings">
            <thead>
              <tr>
                {COLS.map((c) => (
                  <th key={c.key} onClick={() => toggle(c.key)} className={sort.key === c.key ? 'sorted' : ''}>
                    {c.label}
                    {sort.key === c.key && <span className="sort">{sort.dir > 0 ? '▲' : '▼'}</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <tr key={t.team}>
                  <td className="muted">{t.rank}</td>
                  <td><TeamBadge abbr={t.team} showName /></td>
                  <td>
                    <div className="ratingbar">
                      <div className="ratingbar__track">
                        <div
                          className="ratingbar__fill"
                          style={{
                            width: `${(Math.abs(t.rating) / maxAbs) * 50}%`,
                            [t.rating >= 0 ? 'left' : 'right']: '50%',
                            background: t.rating >= 0 ? team(t.team).color : 'var(--rose)',
                          }}
                        />
                      </div>
                      <span className={t.rating >= 0 ? 'tone-good' : 'tone-bad'}>{signed(t.rating, 1)}</span>
                    </div>
                  </td>
                  <td>{num(t.elo, 0)}</td>
                  <td className={t.off_epa >= 0 ? 'tone-good' : 'tone-bad'}>{signed(t.off_epa, 3)}</td>
                  <td className={t.def_epa >= 0 ? 'tone-good' : 'tone-bad'}>{signed(t.def_epa, 3)}</td>
                  <td>{signed(t.pt_diff, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )
}
