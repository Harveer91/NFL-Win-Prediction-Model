// Display colours are tuned to stay readable on the dark theme.
export const TEAMS = {
  ARI: { name: 'Arizona Cardinals', short: 'Cardinals', color: '#B0213F', alt: '#FFB612' },
  ATL: { name: 'Atlanta Falcons', short: 'Falcons', color: '#D0213B', alt: '#A5ACAF' },
  BAL: { name: 'Baltimore Ravens', short: 'Ravens', color: '#6A4FD1', alt: '#C9A227' },
  BUF: { name: 'Buffalo Bills', short: 'Bills', color: '#2F6BFF', alt: '#C60C30' },
  CAR: { name: 'Carolina Panthers', short: 'Panthers', color: '#0085CA', alt: '#BFC0BF' },
  CHI: { name: 'Chicago Bears', short: 'Bears', color: '#E2581E', alt: '#4B6CB7' },
  CIN: { name: 'Cincinnati Bengals', short: 'Bengals', color: '#FB4F14', alt: '#E5E5E5' },
  CLE: { name: 'Cleveland Browns', short: 'Browns', color: '#FF6A1A', alt: '#9B6B43' },
  DAL: { name: 'Dallas Cowboys', short: 'Cowboys', color: '#4C7BD9', alt: '#869397' },
  DEN: { name: 'Denver Broncos', short: 'Broncos', color: '#FB4F14', alt: '#3E64B5' },
  DET: { name: 'Detroit Lions', short: 'Lions', color: '#0076B6', alt: '#B0B7BC' },
  GB: { name: 'Green Bay Packers', short: 'Packers', color: '#2E8B57', alt: '#FFB612' },
  HOU: { name: 'Houston Texans', short: 'Texans', color: '#C8102E', alt: '#3B6EA5' },
  IND: { name: 'Indianapolis Colts', short: 'Colts', color: '#3D7DCA', alt: '#A2AAAD' },
  JAX: { name: 'Jacksonville Jaguars', short: 'Jaguars', color: '#00A0B0', alt: '#D7A22A' },
  KC: { name: 'Kansas City Chiefs', short: 'Chiefs', color: '#E31837', alt: '#FFB81C' },
  LV: { name: 'Las Vegas Raiders', short: 'Raiders', color: '#A5ACAF', alt: '#E0E0E0' },
  LAC: { name: 'Los Angeles Chargers', short: 'Chargers', color: '#0080C6', alt: '#FFC20E' },
  LA: { name: 'Los Angeles Rams', short: 'Rams', color: '#2B5FD9', alt: '#FFD100' },
  MIA: { name: 'Miami Dolphins', short: 'Dolphins', color: '#008E97', alt: '#FC4C02' },
  MIN: { name: 'Minnesota Vikings', short: 'Vikings', color: '#7A4FC9', alt: '#FFC62F' },
  NE: { name: 'New England Patriots', short: 'Patriots', color: '#3A5BA0', alt: '#C60C30' },
  NO: { name: 'New Orleans Saints', short: 'Saints', color: '#D3BC8D', alt: '#A0A0A0' },
  NYG: { name: 'New York Giants', short: 'Giants', color: '#2E5BBA', alt: '#A71930' },
  NYJ: { name: 'New York Jets', short: 'Jets', color: '#1E8F5A', alt: '#E0E0E0' },
  PHI: { name: 'Philadelphia Eagles', short: 'Eagles', color: '#12808A', alt: '#A5ACAF' },
  PIT: { name: 'Pittsburgh Steelers', short: 'Steelers', color: '#FFB612', alt: '#A5ACAF' },
  SF: { name: 'San Francisco 49ers', short: '49ers', color: '#C8102E', alt: '#B3995D' },
  SEA: { name: 'Seattle Seahawks', short: 'Seahawks', color: '#69BE28', alt: '#4B7BD1' },
  TB: { name: 'Tampa Bay Buccaneers', short: 'Buccaneers', color: '#D50A0A', alt: '#FF7900' },
  TEN: { name: 'Tennessee Titans', short: 'Titans', color: '#4B92DB', alt: '#C8102E' },
  WAS: { name: 'Washington Commanders', short: 'Commanders', color: '#A3354B', alt: '#FFB612' },
}

export const team = (abbr) => TEAMS[abbr] ?? { name: abbr, short: abbr, color: '#64748b', alt: '#94a3b8' }

const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16))

// Pick colours for a matchup, switching the away side to its alt colour if the two clash.
export function matchupColors(home, away) {
  const h = team(home)
  const a = team(away)
  const [r1, g1, b1] = rgb(h.color)
  const [r2, g2, b2] = rgb(a.color)
  const distance = Math.hypot(r1 - r2, g1 - g2, b1 - b2)
  return { home: h.color, away: distance < 110 ? a.alt : a.color }
}
