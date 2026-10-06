import TopNav from './components/TopNav'
import { useData } from './lib/useData'
import { useHashRoute } from './lib/useHashRoute'
import Performance from './pages/Performance'
import Games from './pages/Games'
import Ratings from './pages/Ratings'
import Methodology from './pages/Methodology'
import FraudMeter from './pages/FraudMeter'

const PAGES = { performance: Performance, games: Games, ratings: Ratings, fraud: FraudMeter, method: Methodology }

export default function App() {
  const route = useHashRoute()
  const { data: performance } = useData('performance')
  const Page = PAGES[route] ?? Performance
  return (
    <div className="app">
      <TopNav route={PAGES[route] ? route : 'performance'} meta={performance} />
      <main className="page">
        <Page />
      </main>
      <footer className="footer">
        Data: <a href="https://github.com/nflverse" target="_blank" rel="noreferrer">nflverse</a> · Model
        predictions for entertainment purposes only.
        {performance && <> · Updated {new Date(performance.generated_at).toLocaleString()}</>}
      </footer>
    </div>
  )
}
