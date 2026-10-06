import { matchupColors } from '../data/teams'

export default function ProbBar({ home, away, homeProb }) {
  const colors = matchupColors(home, away)
  const awayPct = (1 - homeProb) * 100
  return (
    <div className="probbar" role="img" aria-label={`${away} ${awayPct.toFixed(0)}%, ${home} ${(homeProb * 100).toFixed(0)}%`}>
      <div className="probbar__seg" style={{ width: `${awayPct}%`, background: colors.away }} />
      <div className="probbar__seg" style={{ width: `${100 - awayPct}%`, background: colors.home }} />
    </div>
  )
}
