export default function ChartTooltip({ active, payload, label, formatter, labelFormatter }) {
  if (!active || !payload?.length) return null
  return (
    <div className="charttip">
      <div className="charttip__label">{labelFormatter ? labelFormatter(label, payload) : label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="charttip__row">
          <span className="charttip__dot" style={{ background: p.color ?? p.fill ?? p.stroke }} />
          <span>{p.name}</span>
          <strong>{formatter ? formatter(p.value, p.dataKey) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}
