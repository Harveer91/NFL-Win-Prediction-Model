const LINKS = [
  { id: 'performance', label: 'Model Performance' },
  { id: 'games', label: 'Games' },
  { id: 'ratings', label: 'Power Ratings' },
  { id: 'fraud', label: 'Fraud-o-Meter' },
  { id: 'method', label: 'Methodology' },
]

export default function TopNav({ route, meta }) {
  return (
    <header className="topnav">
      <div className="topnav__inner">
        <a className="brand" href="#/performance">
          <span className="brand__mark" aria-hidden>
            <svg viewBox="0 0 32 32" width="28" height="28">
              <ellipse cx="16" cy="16" rx="11" ry="7" transform="rotate(-35 16 16)" fill="currentColor" />
              <path d="M11 21l10-10M13 15l2 2M15 13l2 2M17 11l2 2" stroke="var(--bg)" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </span>
          <span className="brand__name">redzone<span>model</span></span>
        </a>
        <nav className="topnav__links" aria-label="Primary">
          {LINKS.map((l) => (
            <a key={l.id} href={`#/${l.id}`} className={route === l.id ? 'active' : ''}>
              {l.label}
            </a>
          ))}
        </nav>
        {meta && (
          <div className="topnav__status">
            <span className="pulse" aria-hidden />
            {meta.current_season} · Week {meta.current_week}
          </div>
        )}
      </div>
    </header>
  )
}
