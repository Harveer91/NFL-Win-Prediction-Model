export default function StatCard({ label, value, caption, rows = [], accent = 'teal', icon }) {
  return (
    <article className={`stat stat--${accent}`}>
      <div className="stat__top">
        <span className="stat__label">{label}</span>
        {icon && <span className="stat__icon">{icon}</span>}
      </div>
      <div className="stat__value">{value}</div>
      {caption && <div className="stat__caption">{caption}</div>}
      {rows.length > 0 && (
        <dl className="stat__rows">
          {rows.map((r) => (
            <div key={r.label}>
              <dt>{r.label}</dt>
              <dd className={`tone-${r.tone ?? 'neutral'}`}>{r.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </article>
  )
}
