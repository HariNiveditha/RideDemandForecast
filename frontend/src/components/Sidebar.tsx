import { ArrowUpRight, BarChart3, LayoutDashboard, Network, Target, X } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { CheckerStrip, TaxiIcon } from './TaxiIcons'

const navItems = [
  { label: 'Overview', to: '/', icon: LayoutDashboard, end: true },
  { label: 'Forecast', to: '/forecast', icon: Target },
  { label: 'Analytics', to: '/analytics', icon: BarChart3 },
  { label: 'Model', to: '/model', icon: Network },
]

type SidebarProps = {
  mobileNavOpen: boolean
  onClose: () => void
}

function Sidebar({ mobileNavOpen, onClose }: SidebarProps) {
  return (
    <aside className={`sidebar ${mobileNavOpen ? 'is-open' : ''}`}>
      <div className="brand">
        <div className="brand-mark">
          <TaxiIcon size={24} />
        </div>
        <div>
          <strong>RideCast</strong>
          <span>NYC demand intelligence</span>
        </div>
        <button type="button" className="icon-button mobile-close" onClick={onClose} aria-label="Close navigation">
          <X size={18} />
        </button>
      </div>
      <CheckerStrip className="sidebar-checker" />

      <div className="workspace-label">WORKSPACE</div>
      <nav className="nav-list" aria-label="Main navigation">
        {navItems.map(({ label, to, icon: Icon, end }) => (
          <NavLink
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            end={end}
            key={label}
            to={to}
            onClick={onClose}
          >
            <Icon size={17} />
            <span>{label}</span>
            {label === 'Forecast' && <span className="nav-badge">Live</span>}
            <TaxiIcon size={16} className="nav-taxi" />
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="model-status">
          <span className="status-dot" />
          <div>
            <strong>Model workspace</strong>
            <small>Waiting for data upload</small>
          </div>
        </div>
        <div className="profile">
          <div className="avatar">RD</div>
          <div>
            <strong>Research project</strong>
            <small>Local environment</small>
          </div>
          <ArrowUpRight size={15} />
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
