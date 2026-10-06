export default function Panel({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <div className="panel__head">
          <div>
            {title && <h2 className="panel__title">{title}</h2>}
            {subtitle && <p className="panel__subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="panel__actions">{actions}</div>}
        </div>
      )}
      <div className="panel__body">{children}</div>
    </section>
  )
}
