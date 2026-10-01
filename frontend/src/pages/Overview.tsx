import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { ArrowUpRight, CalendarRange, Network } from 'lucide-react'
import { Link } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import SectionHeading from '../components/SectionHeading'
import StateCard from '../components/StateCard'
import { CheckerStrip, TaxiIcon, TaxiIllustration } from '../components/TaxiIcons'

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

type Kpi = { label: string; icon: ReactNode; value: string; note: string; cab?: boolean }

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

  const hero = (
    <section className="hero-banner">
      <div className="hero-copy">
        <span className="eyebrow light">WEATHER + LAG MODEL</span>
        <h2>
          See demand before
          <br />
          the city gets busy.
        </h2>
        <p>
          Forecast hourly NYC taxi demand with a Random Forest model trained on temporal patterns, pickup zones, weather,
          and historical lag features.
        </p>
        <Link className="primary-button" to="/forecast">
          Open forecast <ArrowUpRight size={17} />
        </Link>
      </div>
      <div className="hero-orbit" aria-hidden="true">
        <div className="orbit-ring ring-one" />
        <div className="orbit-ring ring-two" />
        <span className="hero-ready">
          <span className="status-dot" />
          Model ready
        </span>
        <div className="hero-taxi">
          <TaxiIllustration width={210} motion />
        </div>
        <span className="hero-road" />
        <span className="orbit-label label-a">LAG SIGNALS</span>
        <span className="orbit-label label-b">WEATHER</span>
        <span className="orbit-label label-c">LOCATION</span>
      </div>
      <CheckerStrip className="hero-checker" />
    </section>
  )

  if (loading) {
    return (
      <>
        {hero}
        <StateCard tone="loading" title="Hailing demand data…" description="Fetching historical ride volumes from the inference service." />
      </>
    )
  }

  if (error || !data) {
    return (
      <>
        {hero}
        <StateCard
          tone="error"
          title="Historical data unavailable"
          description={error || 'The historical endpoint could not be loaded. Start the inference service and refresh.'}
        />
      </>
    )
  }

  const startDate = new Date(data.date_range.start).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
  const endDate = new Date(data.date_range.end).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })

  const kpis: Kpi[] = [
    { label: 'Average demand', icon: <TaxiIcon size={20} />, value: Number(data.summary.avg_demand).toFixed(1), note: 'Average hourly rides', cab: true },
    {
      label: 'Peak demand',
      icon: <TaxiIcon size={20} badge="peak" />,
      value: Number(data.summary.peak_demand).toFixed(1),
      note: `Peak at ${new Date(data.summary.peak_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric' })}`,
      cab: true,
    },
    { label: 'Peak location', icon: <TaxiIcon size={20} badge="pin" />, value: `Zone ${data.summary.peak_location}`, note: 'Highest total demand', cab: true },
    { label: 'Date range', icon: <CalendarRange size={16} />, value: startDate, note: `to ${endDate}` },
  ]

  const featureFamilies = [
    ['Lag history', 'lag_1 · lag_24 · lag_168', 'primary'],
    ['Weather', 'temperature · precipitation · wind', 'accent'],
    ['Time patterns', 'hour · day of week · seasonality', 'neutral'],
    ['Location', 'pickup zone identifier', 'soft'],
  ]

  return (
    <>
      {hero}

      <SectionHeading eyebrow="CURRENT SNAPSHOT" title="Demand at a glance">
        <span className="muted-pill">Live dataset</span>
      </SectionHeading>

      <section className="kpi-grid">
        {kpis.map(({ label, icon, value, note, cab }) => (
          <article className="kpi-card" key={label}>
            <div className="kpi-top">
              <span>{label}</span>
              <span className={`kpi-icon ${cab ? 'is-cab' : ''}`}>{icon}</span>
            </div>
            <strong>{value}</strong>
            <small>{note}</small>
          </article>
        ))}
      </section>

      <section className="dashboard-grid">
        <article className="panel trend-panel" style={{ minHeight: 360 }}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">DEMAND SERIES</span>
              <h3>Hourly demand over time</h3>
            </div>
            <span className="muted-pill">{data.hourly_series.length} points</span>
          </div>
          <div style={{ width: '100%', height: 260, marginTop: 18 }}>
            <ResponsiveContainer>
              <LineChart data={data.hourly_series} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="pickup_hour" tick={{ fontSize: 10 }} minTickGap={20} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(value) => {
                    const numericValue = Array.isArray(value) ? value[0] : value
                    return [typeof numericValue === 'number' ? Number(numericValue).toFixed(1) : String(numericValue ?? ''), 'Total rides']
                  }}
                  labelFormatter={(label) => new Date(String(label ?? '')).toLocaleString()}
                />
                <Line type="monotone" dataKey="total_rides" stroke="#2aa6a4" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel insight-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">MODEL SIGNALS</span>
              <h3>Feature families</h3>
            </div>
            <Network size={19} className="panel-icon" />
          </div>
          <div className="feature-list">
            {featureFamilies.map(([title, detail, tone]) => (
              <div className="feature-row" key={title}>
                <span className={`feature-bar ${tone}`} />
                <div>
                  <strong>{title}</strong>
                  <small>{detail}</small>
                </div>
                <span className="feature-state">Ready</span>
              </div>
            ))}
          </div>
        </article>
      </section>
    </>
  )
}

export default Overview
