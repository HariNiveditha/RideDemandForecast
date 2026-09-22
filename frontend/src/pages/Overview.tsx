import { useEffect, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, Gauge, MapPin, Network } from 'lucide-react'
import { Link } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

type HistoricalResponse = {
  date_range: { start: string; end: string }
  summary: {
    avg_demand: number
    peak_demand: number
    peak_at: string
    peak_location: number
  }
  hourly_series: Array<{ pickup_hour: string; total_rides: number }>
}

function Overview() {
  const [data, setData] = useState<HistoricalResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadHistorical = async () => {
      try {
        setLoading(true)
        setError('')

        const response = await fetch('/api/historical')
        if (!response.ok) {
          throw new Error('Historical data endpoint unavailable')
        }

        const payload = (await response.json()) as HistoricalResponse
        setData(payload)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load historical demand data.')
      } finally {
        setLoading(false)
      }
    }

    void loadHistorical()
  }, [])

  if (loading) {
    return <div className="view-notice"><strong>Loading demand data…</strong></div>
  }

  if (error || !data) {
    return <div className="view-notice"><div><strong>Historical data unavailable</strong><span>{error || 'The historical endpoint could not be loaded.'}</span></div></div>
  }

  const startDate = new Date(data.date_range.start).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
  const endDate = new Date(data.date_range.end).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })

  const kpis = [
    { label: 'Average demand', icon: Activity, value: Number(data.summary.avg_demand).toFixed(1), note: 'Average hourly rides' },
    { label: 'Peak demand', icon: Gauge, value: Number(data.summary.peak_demand).toFixed(1), note: `Peak at ${new Date(data.summary.peak_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric' })}` },
    { label: 'Peak location', icon: MapPin, value: `Zone ${data.summary.peak_location}`, note: 'Highest total demand' },
    { label: 'Date range', icon: BarChart3, value: `${startDate}`, note: `to ${endDate}` },
  ]

  return (
    <>
      <section className="hero-banner"><div><span className="eyebrow light">WEATHER + LAG MODEL</span><h2>See demand before<br />the city gets busy.</h2><p>Forecast hourly ride demand with a Random Forest model trained on temporal patterns, pickup zones, weather, and historical lag features.</p><Link className="primary-button" to="/forecast">Open forecast <ArrowUpRight size={17} /></Link></div><div className="hero-orbit"><div className="orbit-ring ring-one" /><div className="orbit-ring ring-two" /><div className="orbit-core"><Gauge size={31} /><span>Model<br />ready</span></div><span className="orbit-label label-a">LAG SIGNALS</span><span className="orbit-label label-b">WEATHER</span><span className="orbit-label label-c">LOCATION</span></div></section>
      <section className="section-heading"><div><span className="eyebrow">CURRENT SNAPSHOT</span><h2>Demand at a glance</h2></div><span className="muted-pill">Live dataset</span></section>
      <section className="kpi-grid">{kpis.map(({ label, icon: Icon, value, note }) => <article className="kpi-card" key={label}><div className="kpi-top"><span>{label}</span><span className="kpi-icon"><Icon size={16} /></span></div><strong>{value}</strong><small>{note}</small></article>)}</section>
      <section className="dashboard-grid"><article className="panel trend-panel" style={{ minHeight: 360 }}><div className="panel-heading"><div><span className="eyebrow">DEMAND SERIES</span><h3>Hourly demand over time</h3></div><span className="muted-pill">{data.hourly_series.length} points</span></div><div style={{ width: '100%', height: 260, marginTop: 18 }}><ResponsiveContainer><LineChart data={data.hourly_series} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="pickup_hour" tick={{ fontSize: 10 }} minTickGap={20} /><YAxis tick={{ fontSize: 10 }} /><Tooltip formatter={(value) => { const numericValue = Array.isArray(value) ? value[0] : value; return [typeof numericValue === 'number' ? Number(numericValue).toFixed(1) : String(numericValue ?? ''), 'Total rides']; }} labelFormatter={(label) => new Date(String(label ?? '')).toLocaleString()} /><Line type="monotone" dataKey="total_rides" stroke="#2aa6a4" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div></article><article className="panel insight-panel"><div className="panel-heading"><div><span className="eyebrow">MODEL SIGNALS</span><h3>Feature families</h3></div><Network size={19} className="panel-icon" /></div><div className="feature-list">{[['Lag history', 'lag_1 · lag_24 · lag_168', 'primary'], ['Weather', 'temperature · precipitation · wind', 'accent'], ['Time patterns', 'hour · day of week · seasonality', 'neutral'], ['Location', 'pickup zone identifier', 'soft']].map(([title, detail, tone]) => <div className="feature-row" key={title}><span className={`feature-bar ${tone}`} /><div><strong>{title}</strong><small>{detail}</small></div><span className="feature-state">Ready</span></div>)}</div></article></section>
    </>
  )
}

export default Overview
