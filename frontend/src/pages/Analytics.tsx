import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

type HistoricalResponse = {
  demand_by_hour_of_day: Array<{ hour: number; avg_rides: number }>
  demand_by_location: Array<{ PULocationID: number; total_rides: number }>
}

function Analytics() {
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
          throw new Error('Historical statistics unavailable')
        }

        const payload = (await response.json()) as HistoricalResponse
        setData(payload)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load analytics data.')
      } finally {
        setLoading(false)
      }
    }

    void loadHistorical()
  }, [])

  if (loading) {
    return <div className="view-notice"><strong>Loading analytics…</strong></div>
  }

  if (error || !data) {
    return <div className="view-notice"><div><strong>Analytics unavailable</strong><span>{error || 'The historical endpoint could not be loaded.'}</span></div></div>
  }

  const peakHour = data.demand_by_hour_of_day.reduce((best, current) => (current.avg_rides > best.avg_rides ? current : best), data.demand_by_hour_of_day[0])
  const topLocations = [...data.demand_by_location].slice(0, 10)

  return (
    <section className="forecast-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">PERFORMANCE</span>
          <h2>Demand analytics</h2>
        </div>
      </div>

      <div className="dashboard-grid">
        <article className="panel" style={{ minHeight: 320 }}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">HOURLY PROFILE</span>
              <h3>Average rides by hour of day</h3>
            </div>
          </div>

          <div style={{ width: '100%', height: 260, marginTop: 18 }}>
            <ResponsiveContainer>
              <BarChart data={data.demand_by_hour_of_day} margin={{ top: 12, right: 12, left: 0, bottom: 18 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(value) => {
                    const numericValue = Array.isArray(value) ? value[0] : value
                    return typeof numericValue === 'number' ? Number(numericValue).toFixed(1) : String(numericValue ?? '')
                  }}
                />
                <Bar dataKey="avg_rides" fill={peakHour ? '#2aa6a4' : '#9ecdc9'} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel" style={{ minHeight: 320 }}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">TOP LOCATIONS</span>
              <h3>Demand by pickup zone</h3>
            </div>
          </div>

          <div style={{ width: '100%', height: 260, marginTop: 18 }}>
            <ResponsiveContainer>
              <BarChart data={topLocations} layout="vertical" margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="PULocationID" width={68} tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(value) => {
                    const numericValue = Array.isArray(value) ? value[0] : value
                    return typeof numericValue === 'number' ? Number(numericValue).toFixed(1) : String(numericValue ?? '')
                  }}
                />
                <Bar dataKey="total_rides" fill="#2aa6a4" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>
    </section>
  )
}

export default Analytics
