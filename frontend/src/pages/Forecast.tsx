import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { ArrowUpRight, Calendar, Clock, Info, Layers, RefreshCw, ShieldCheck } from 'lucide-react'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import SectionHeading from '../components/SectionHeading'
import StateCard from '../components/StateCard'
import { TaxiIcon } from '../components/TaxiIcons'

type PickupLocation = {
  PULocationID: number
  Borough: string
  Zone: string
}

type ForecastRecord = {
  zone_id: number
  forecast_origin: string
  target_timestamp: string
  horizon_minutes: number
  predicted_rides: number
  lower_bound?: number | null
  upper_bound?: number | null
  model_version?: string
}

type RecentHistoryPoint = {
  timestamp: string
  actual_rides: number
}

type MultiHorizonResponse = {
  zone_id: number
  forecast_origin: string
  forecasts: ForecastRecord[]
  recent_history: RecentHistoryPoint[]
  model_version: string
  target_description: string
}

type HistoricalRange = {
  start: string
  end: string
}

type ChartPoint = {
  timeLabel: string
  isoTime: string
  actual: number | null
  predicted: number | null
  lowerBound: number | null
  upperBound: number | null
  isOrigin: boolean
}

function dateIsWithinRange(value: string, start: string, end: string) {
  if (!value || !start || !end) return false
  const selected = new Date(`${value}T00:00:00`)
  const minDate = new Date(`${start.slice(0, 10)}T00:00:00`)
  const maxDate = new Date(`${end.slice(0, 10)}T00:00:00`)
  return selected >= minDate && selected <= maxDate
}

function getDefaultDate(start: string, end: string) {
  if (!start || !end) return '2021-06-15'
  const today = new Date()
  const minDate = new Date(`${start.slice(0, 10)}T00:00:00`)
  const maxDate = new Date(`${end.slice(0, 10)}T00:00:00`)
  if (today >= minDate && today <= maxDate) {
    return today.toISOString().slice(0, 10)
  }
  const midpoint = new Date((minDate.getTime() + maxDate.getTime()) / 2)
  return midpoint.toISOString().slice(0, 10)
}

function Forecast() {
  const [locationId, setLocationId] = useState('161') // Default to Midtown Center (Zone 161)
  const [locations, setLocations] = useState<PickupLocation[]>([])
  const [date, setDate] = useState('2021-06-15')
  const [time, setTime] = useState('12:00')
  const [horizonFilter, setHorizonFilter] = useState<'all' | '60' | '180' | '360'>('all')
  const [weather, setWeather] = useState({ temp: '72', humidity: '55', precip: '0.0' })
  const [dateRange, setDateRange] = useState<HistoricalRange>({ start: '2021-04-01', end: '2021-06-30' })
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [forecastResult, setForecastResult] = useState<MultiHorizonResponse | null>(null)

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [histRes, locsRes] = await Promise.all([
          fetch('/api/historical'),
          fetch('/api/locations'),
        ])

        if (!histRes.ok || !locsRes.ok) {
          throw new Error('Inference backend metadata unavailable.')
        }

        const histData = await histRes.json()
        const locsData = (await locsRes.json()) as { locations: PickupLocation[] }

        const range = {
          start: histData.date_range.start.slice(0, 10),
          end: histData.date_range.end.slice(0, 10),
        }
        setDateRange(range)
        setLocations(locsData.locations)
        setDate(getDefaultDate(range.start, range.end))
      } catch (err) {
        setStatus(err instanceof Error ? err.message : 'Failed to connect to backend.')
      }
    }

    void loadInitialData()
  }, [])

  const selectedZone = locations.find((l) => l.PULocationID === Number(locationId))

  const handlePredict = async (e?: FormEvent) => {
    if (e) e.preventDefault()
    if (!locationId || !date) {
      setStatus('Please select a valid zone and date.')
      return
    }

    if (!dateIsWithinRange(date, dateRange.start, dateRange.end)) {
      setStatus(`Date must be within the available historical training/testing range (${dateRange.start} to ${dateRange.end}).`)
      return
    }

    setLoading(true)
    setStatus('Generating multi-horizon demand forecasts...')

    const selectedTime = time || '12:00'
    const originIso = `${date}T${selectedTime}:00`

    const payload = {
      zone_id: Number(locationId),
      forecast_origin: originIso,
      horizon_minutes: horizonFilter === 'all' ? null : Number(horizonFilter),
      temp: weather.temp ? Number(weather.temp) : null,
      humidity: weather.humidity ? Number(weather.humidity) : null,
      precip: weather.precip ? Number(weather.precip) : null,
    }

    try {
      const res = await fetch('/api/forecast/multi-horizon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errorDetail = await res.text()
        throw new Error(`Inference error (${res.status}): ${errorDetail}`)
      }

      const data = (await res.json()) as MultiHorizonResponse
      setForecastResult(data)
      setStatus('')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Forecast request failed'
      setStatus(msg)
    } finally {
      setLoading(false)
    }
  }

  // Auto-run forecast once initial locations load
  useEffect(() => {
    if (locations.length > 0 && !forecastResult) {
      void handlePredict()
    }
  }, [locations])

  // Build chart timeline combining recent history and future forecast intervals
  const chartData: ChartPoint[] = []
  if (forecastResult) {
    if (forecastResult.recent_history && forecastResult.recent_history.length > 0) {
      for (const h of forecastResult.recent_history) {
        chartData.push({
          timeLabel: new Date(h.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
          isoTime: h.timestamp,
          actual: h.actual_rides,
          predicted: null,
          lowerBound: null,
          upperBound: null,
          isOrigin: h.timestamp === forecastResult.forecast_origin,
        })
      }
    }

    // Connect from origin point to forecasts
    const originPoint = chartData[chartData.length - 1]
    if (originPoint) {
      originPoint.predicted = originPoint.actual
      originPoint.lowerBound = originPoint.actual
      originPoint.upperBound = originPoint.actual
    }

    for (const f of forecastResult.forecasts) {
      chartData.push({
        timeLabel: `+${f.horizon_minutes / 60}h (${new Date(f.target_timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })})`,
        isoTime: f.target_timestamp,
        actual: null,
        predicted: f.predicted_rides,
        lowerBound: f.lower_bound ?? null,
        upperBound: f.upper_bound ?? null,
        isOrigin: false,
      })
    }
  }

  return (
    <section className="forecast-section">
      <SectionHeading eyebrow="DEMAND FORECASTING" title="Multi-Horizon Ride Demand Forecast">
        <div className="heading-chips">
          <span className="model-chip">
            <span className="status-dot" />
            Direct Forecast Models · 1h, 3h & 6h
          </span>
          <span className="range-chip">
            Data window: {dateRange.start} to {dateRange.end}
          </span>
        </div>
      </SectionHeading>

      {/* Control Form */}
      <form className="forecast-card" onSubmit={handlePredict}>
        <div className="form-fields">
          <label>
            <span className="label-with-icon">
              <TaxiIcon size={15} badge="pin" />
              Pickup location
            </span>
            <select value={locationId} onChange={(e) => setLocationId(e.target.value)}>
              {locations.map((zone) => (
                <option key={zone.PULocationID} value={zone.PULocationID}>
                  {zone.Zone} (ID {zone.PULocationID}) — {zone.Borough}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span className="label-with-icon">
              <Layers size={14} />
              Forecast horizon
            </span>
            <select
              value={horizonFilter}
              onChange={(e) => setHorizonFilter(e.target.value as 'all' | '60' | '180' | '360')}
            >
              <option value="all">All Horizons (1h, 3h, 6h)</option>
              <option value="60">1 Hour Ahead (60 min)</option>
              <option value="180">3 Hours Ahead (180 min)</option>
              <option value="360">6 Hours Ahead (360 min)</option>
            </select>
          </label>

          <label>
            <span className="label-with-icon">
              <Calendar size={14} />
              Origin date
            </span>
            <input
              type="date"
              value={date}
              min={dateRange.start}
              max={dateRange.end}
              onChange={(e) => setDate(e.target.value)}
            />
          </label>

          <label>
            <span className="label-with-icon">
              <Clock size={14} />
              Origin hour
            </span>
            <input type="time" value={time} step="3600" onChange={(e) => setTime(e.target.value)} />
          </label>

          <label>
            Temp (°C / °F)
            <input
              type="number"
              placeholder="e.g. 72"
              value={weather.temp}
              onChange={(e) => setWeather({ ...weather, temp: e.target.value })}
            />
          </label>

          <label>
            Precip (mm)
            <input
              type="number"
              step="0.1"
              placeholder="e.g. 0.0"
              value={weather.precip}
              onChange={(e) => setWeather({ ...weather, precip: e.target.value })}
            />
          </label>
        </div>

        <div className="form-footer">
          <span>
            {status ||
              'Forecast origin T establishes causal cutoff. Lags and weather are resolved without leakage.'}
          </span>
          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? <RefreshCw size={15} className="spin" /> : <ArrowUpRight size={17} />}
            {loading ? 'Forecasting...' : 'Generate Multi-Horizon Forecast'}
          </button>
        </div>
      </form>

      {/* Main Results Display */}
      {loading && !forecastResult ? (
        <StateCard
          tone="loading"
          title="Running Multi-Horizon Demand Forecast"
          description="Running leak-free direct estimators with calibrated conformal uncertainty bounds..."
        />
      ) : forecastResult && selectedZone ? (
        <>
          {/* Horizon Cards Grid */}
          <div className="horizon-cards-grid">
            {forecastResult.forecasts.map((f) => {
              const targetDate = new Date(f.target_timestamp)
              const formattedTime = targetDate.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
              const formattedDate = targetDate.toLocaleDateString([], { month: 'short', day: 'numeric' })
              return (
                <article key={f.horizon_minutes} className="horizon-card active-horizon" style={{ height: 'auto', minHeight: 'auto' }}>
                  <div className="horizon-card-header">
                    <span className="horizon-badge">
                      HORIZON: +{f.horizon_minutes / 60}H ({f.horizon_minutes} MIN)
                    </span>
                    <span className="eyebrow">Multi-Horizon Forecast</span>
                  </div>

                  <strong className="horizon-predicted" style={{ margin: '4px 0 2px' }}>
                    {Math.round(f.predicted_rides).toLocaleString()}{' '}
                    <span>completed rides</span>
                  </strong>

                  <div className="meta-item" style={{ padding: '6px 10px', marginTop: 2, flex: '0 0 auto', minHeight: 'auto' }}>
                    <Clock size={14} />
                    <div>
                      <small style={{ fontSize: 10 }}>Target interval start</small>
                      <strong style={{ fontSize: 12 }}>
                        {formattedDate} · {formattedTime}
                      </strong>
                    </div>
                  </div>

                  {f.lower_bound != null && f.upper_bound != null && (
                    <div className="horizon-bounds" style={{ marginTop: 2 }}>
                      <ShieldCheck size={13} />
                      <span>
                        80% Conformal CI: [{Math.round(f.lower_bound)} – {Math.round(f.upper_bound)} rides]
                      </span>
                    </div>
                  )}
                </article>
              )
            })}
          </div>

          <div className="disclaimer-banner" style={{ marginTop: 10 }}>
            <Info size={15} />
            <span>
              <strong>Target Definition:</strong> Predicted values strictly represent expected completed passenger
              pickups originating in <strong>{selectedZone.Zone} (ID {selectedZone.PULocationID})</strong> during the
              respective 1-hour interval. They do not represent available fleet vehicles or dispatch quotas.
            </span>
          </div>

          {/* Interactive Multi-Horizon Time-Series Chart */}
          <div className="forecast-chart-container">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">DEMAND TRAJECTORY</span>
                <h3>Recent History & Future Multi-Horizon Forecasts</h3>
              </div>
              <span className="muted-pill">
                Zone {selectedZone.PULocationID} ({selectedZone.Zone})
              </span>
            </div>

            <div style={{ width: '100%', height: 320, marginTop: 18 }}>
              <ResponsiveContainer>
                <ComposedChart data={chartData} margin={{ top: 12, right: 24, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="timeLabel" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} label={{ value: 'Rides / hr', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    formatter={(val, name) => {
                      if (val === null || val === undefined) return ['', '']
                      const num = Number(val).toFixed(1)
                      if (name === 'actual') return [`${num} rides`, 'Actual Demand (Observed)']
                      if (name === 'predicted') return [`${num} rides`, 'Forecasted Demand']
                      if (name === 'upperBound') return [`${num} rides`, '80% Upper Bound']
                      if (name === 'lowerBound') return [`${num} rides`, '80% Lower Bound']
                      return [num, String(name)]
                    }}
                  />
                  {/* Historical demand solid line */}
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#1e3a5f"
                    strokeWidth={2.5}
                    dot={{ r: 3 }}
                    name="actual"
                    connectNulls={false}
                  />
                  {/* Future predicted demand dashed line */}
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#f59e0b"
                    strokeWidth={3}
                    strokeDasharray="5 5"
                    dot={{ r: 5, fill: '#f59e0b' }}
                    name="predicted"
                    connectNulls={false}
                  />
                  {/* Upper Bound Envelope */}
                  <Line
                    type="monotone"
                    dataKey="upperBound"
                    stroke="#10b981"
                    strokeWidth={1.5}
                    strokeDasharray="2 2"
                    dot={false}
                    name="upperBound"
                  />
                  {/* Lower Bound Envelope */}
                  <Line
                    type="monotone"
                    dataKey="lowerBound"
                    stroke="#10b981"
                    strokeWidth={1.5}
                    strokeDasharray="2 2"
                    dot={false}
                    name="lowerBound"
                  />
                  <ReferenceLine
                    x={chartData.find((p) => p.isOrigin)?.timeLabel}
                    stroke="#ef4444"
                    strokeDasharray="3 3"
                    label={{ value: 'Forecast Origin (T)', fill: '#ef4444', fontSize: 10, position: 'top' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Forecast Records Table */}
          <div className="forecast-table-container">
            <div className="panel-heading" style={{ marginBottom: 12 }}>
              <div>
                <span className="eyebrow">FORECAST RECORDS</span>
                <h3>Forecast Records for Fleet Allocation</h3>
              </div>
              <span className="muted-pill">{forecastResult.forecasts.length} horizons</span>
            </div>

            <table className="forecast-table">
              <thead>
                <tr>
                  <th>Horizon</th>
                  <th>Pickup Zone</th>
                  <th>Forecast Origin (T)</th>
                  <th>Target Timestamp</th>
                  <th>Predicted Rides</th>
                  <th>Lower Bound (80%)</th>
                  <th>Upper Bound (80%)</th>
                  <th>Model Version</th>
                </tr>
              </thead>
              <tbody>
                {forecastResult.forecasts.map((f) => (
                  <tr key={f.horizon_minutes}>
                    <td>
                      <span className="horizon-badge">
                        +{f.horizon_minutes / 60}h ({f.horizon_minutes}m)
                      </span>
                    </td>
                    <td>
                      <strong>{selectedZone.Zone}</strong>
                      <small style={{ display: 'block', color: 'var(--muted)' }}>
                        {selectedZone.Borough} · ID {f.zone_id}
                      </small>
                    </td>
                    <td>{new Date(f.forecast_origin).toLocaleString()}</td>
                    <td>
                      <strong>{new Date(f.target_timestamp).toLocaleString()}</strong>
                    </td>
                    <td>
                      <strong style={{ color: 'var(--navy)', fontSize: 14 }}>
                        {f.predicted_rides.toFixed(1)} rides
                      </strong>
                    </td>
                    <td>{f.lower_bound != null ? `${f.lower_bound.toFixed(1)} rides` : '—'}</td>
                    <td>{f.upper_bound != null ? `${f.upper_bound.toFixed(1)} rides` : '—'}</td>
                    <td>
                      <code style={{ fontSize: 11, background: '#f0f4f6', padding: '2px 5px', borderRadius: 4 }}>
                        Multi-Horizon Forecast
                      </code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="forecast-empty">
          <StateCard
            compact
            title="Ready to generate forecasts"
            description="Select a NYC taxi zone, forecast origin time, and horizons to generate multi-horizon demand forecasts."
          />
        </div>
      )}
    </section>
  )
}

export default Forecast
