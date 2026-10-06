export const pct = (v, digits = 1) => (v == null ? '—' : `${(v * 100).toFixed(digits)}%`)

export const signed = (v, digits = 1, suffix = '') =>
  v == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}${suffix}`

export const signedPct = (v, digits = 1) => (v == null ? '—' : signed(v * 100, digits, '%'))

export const num = (v, digits = 1) => (v == null ? '—' : v.toFixed(digits))

// Positive is good unless `lowerIsBetter`.
export const tone = (v, lowerIsBetter = false) => {
  if (v == null || Math.abs(v) < 1e-9) return 'neutral'
  return (v > 0) !== lowerIsBetter ? 'good' : 'bad'
}

export const spread = (homeMargin, home, away) => {
  if (homeMargin == null) return '—'
  if (Math.abs(homeMargin) < 0.05) return 'PK'
  return homeMargin > 0 ? `${home} −${homeMargin.toFixed(1)}` : `${away} −${(-homeMargin).toFixed(1)}`
}

export const weekLabel = (week, gameType) => {
  const playoff = { WC: 'Wild Card', DIV: 'Divisional', CON: 'Conference', SB: 'Super Bowl' }
  return playoff[gameType] ?? `Week ${week}`
}
