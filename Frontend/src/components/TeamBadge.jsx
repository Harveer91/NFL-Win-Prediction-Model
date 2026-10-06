import { team } from '../data/teams'

export default function TeamBadge({ abbr, size = 'md', showName = false }) {
  const t = team(abbr)
  return (
    <span className={`teambadge teambadge--${size}`}>
      <span className="teambadge__chip" style={{ background: t.color }}>
        {abbr}
      </span>
      {showName && <span className="teambadge__name">{t.short}</span>}
    </span>
  )
}
