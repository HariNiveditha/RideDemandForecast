import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import SectionHeading from '../components/SectionHeading'
import StateCard from '../components/StateCard'
import { ASPHALT, CAB_YELLOW, TaxiIcon } from '../components/TaxiIcons'

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
    return <StateCard tone="loading" title="Loading analytics…" description="Tallying rides by hour and pickup zone." />
  }

  if (error || !data) {
    return <StateCard tone="error" title="Analytics unavailable" description={error || 'The historical endpoint could not be loaded.'} />
  }

  const peakHour = data.demand_by_hour_of_day.reduce((best, current) => (current.avg_rides > best.avg_rides ? current : best), data.demand_by_hour_of_day[0])
  const topLocations = [...data.demand_by_location].slice(0, 10)

  return (
    <section className="forecast-section">
      <SectionHeading eyebrow="PERFORMANCE" title="Demand analytics" />

      <div className="dashboard-grid">
        <article className="panel" style={{ minHeight: 320 }}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">HOURLY PROFILE</span>
              <h3>Average rides by hour of day</h3>
            </div>
            {peakHour && (
              <span className="cab-chip">
                <TaxiIcon size={16} badge="peak" />
                Peak {String(peakHour.hour).padStart(2, '0')}:00
              </span>
            )}
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
                <Bar dataKey="avg_rides" fill="#2aa6a4" radius={[4, 4, 0, 0]}>
                  {data.demand_by_hour_of_day.map((entry) => (
                    <Cell
                      key={entry.hour}
                      fill={entry.hour === peakHour?.hour ? CAB_YELLOW : '#2aa6a4'}
                      stroke={entry.hour === peakHour?.hour ? ASPHALT : undefined}
                    />
                  ))}
                </Bar>
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
            {topLocations[0] && (
              <span className="cab-chip">
                <TaxiIcon size={16} badge="pin" />
                Top zone {topLocations[0].PULocationID}
              </span>
            )}
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
                <Bar dataKey="total_rides" fill="#2aa6a4" radius={[0, 4, 4, 0]}>
                  {topLocations.map((entry, index) => (
                    <Cell key={entry.PULocationID} fill={index === 0 ? CAB_YELLOW : '#2aa6a4'} stroke={index === 0 ? ASPHALT : undefined} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>
    </section>
  )
}

export default Analytics
