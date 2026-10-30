import { useState } from 'react'
import { Menu, RefreshCw } from 'lucide-react'
import { Route, Routes, useLocation } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Analytics from './pages/Analytics'
import FleetManagement from './pages/FleetManagement'
import Forecast from './pages/Forecast'
import Model from './pages/Model'
import Overview from './pages/Overview'
import './App.css'

const pageMeta: Record<string, { title: string; label: string }> = {
  '/': { title: 'Overview', label: 'OPERATIONS' },
  '/forecast': { title: 'Forecast', label: 'LIVE INFERENCE' },
  '/analytics': { title: 'Analytics', label: 'PERFORMANCE' },
  '/fleet': { title: 'Fleet', label: 'FLEET' },
  '/model': { title: 'Model', label: 'WORKSPACE' },
}

function App() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const location = useLocation()
  const currentPage = pageMeta[location.pathname] ?? pageMeta['/']

  return (
    <div className="app-shell">
      <Sidebar mobileNavOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-title">
            <button
              type="button"
              className="icon-button mobile-menu"
              onClick={() => setMobileNavOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={18} />
            </button>

            <div>
              <span className="eyebrow">{currentPage.label}</span>
              <h1>{currentPage.title}</h1>
            </div>

            <div className="topbar-actions">
              <span className="live-indicator">
                <span className="status-dot" />
                Local model ready
              </span>
              <button type="button" className="refresh-button" aria-label="Refresh dashboard">
                <RefreshCw size={16} />
              </button>
            </div>
          </div>
        </header>

        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/fleet" element={<FleetManagement />} />
          <Route path="/model" element={<Model />} />
          <Route path="*" element={<Overview />} />
        </Routes>
      </main>
    </div>
  )
}

export default App