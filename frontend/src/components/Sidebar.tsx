import { Activity, ArrowUpRight, BarChart3, LayoutDashboard, Network, Target, X } from 'lucide-react'
import { NavLink } from 'react-router-dom'

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
      <div className="brand"><div className="brand-mark"><Activity size={18} /></div><div><strong>RideCast</strong><span>Demand intelligence</span></div><button className="icon-button mobile-close" onClick={onClose} aria-label="Close navigation"><X size={18} /></button></div>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="nav-list" aria-label="Main navigation">
        {navItems.map(({ label, to, icon: Icon, end }) => <NavLink className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`} end={end} key={label} to={to} onClick={onClose}><Icon size={17} /><span>{label}</span>{label === 'Forecast' && <span className="nav-badge">Live</span>}</NavLink>)}
      </nav>
      <div className="sidebar-bottom"><div className="model-status"><span className="status-dot" /><div><strong>Model workspace</strong><small>Waiting for data upload</small></div></div><div className="profile"><div className="avatar">RD</div><div><strong>Research project</strong><small>Local environment</small></div><ArrowUpRight size={15} /></div></div>
    </aside>
  )
}

export default Sidebar
