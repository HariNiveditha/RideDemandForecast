import { useEffect, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  Clock,
  Filter,
  Layers,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { CAB_YELLOW } from '../components/TaxiIcons'

// Types
type TabType = 'anomalies' | 'errors' | 'historical'

type AnomalyRecord = {
  zone_id: number
  zone_name?: string | null
  borough?: string | null
  timestamp: string
  actual_demand?: number | null
  baseline_demand?: number | null
  predicted_demand?: number | null
  lower_bound?: number | null
  upper_bound?: number | null
  deviation?: number | null
  deviation_pct?: number | null
  z_score?: number | null
  anomaly_type: 'demand_spike' | 'demand_drop' | 'forecast_deviation' | 'persistent_anomaly' | 'data_quality'
  severity: 'low' | 'medium' | 'high' | 'critical'
  explanation: string
  is_persistent: boolean
  consecutive_count: number
}

type AnomalySummary = {
  total_anomalies: number
  by_type: Record<string, number>
  by_severity: Record<string, number>
  top_zones: Array<{
    zone_id: number
    zone_name: string
    borough: string
    anomaly_count: number
  }>
  date_range: { start: string; end: string }
}

type ForecastMetricSummary = {
  horizon_minutes: number
  horizon_hours: number
  mae: number
  rmse: number
  mean_bias: number
  wape_pct: number
  sample_count: number
  empirical_coverage_pct: number
  mean_interval_width: number
  interpretation: string
}

type ForecastPerformanceResponse = {
  horizons: ForecastMetricSummary[]
  model_version: string
  evaluation_period: { start: string; end: string }
  error_convention: string
}

type ZoneErrorRecord = {
  zone_id: number
  zone_name: string
  borough: string
  mae: number
  rmse: number
  mean_bias: number
  wape_pct: number
  sample_count: number
}

type TimeSeriesErrorPoint = {
  date: string
  horizon_minutes: number
  mae: number
  rmse: number
  mean_bias: number
  sample_count: number
}

type LocationItem = {
  PULocationID: number
  Borough: string
  Zone: string
}

type HistoricalResponse = {
  demand_by_hour_of_day: Array<{ hour: number; avg_rides: number }>
  demand_by_location: Array<{ PULocationID: number; total_rides: number }>
}

function Analytics() {
  const [activeTab, setActiveTab] = useState<TabType>('anomalies')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // Anomaly State
  const [summary, setSummary] = useState<AnomalySummary | null>(null)
  const [anomalies, setAnomalies] = useState<AnomalyRecord[]>([])
  const [selectedType, setSelectedType] = useState<string>('all')
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all')
  const [selectedZone, setSelectedZone] = useState<string>('all')
  const [locations, setLocations] = useState<LocationItem[]>([])

  // Forecast Error State
  const [performance, setPerformance] = useState<ForecastPerformanceResponse | null>(null)
  const [selectedHorizon, setSelectedHorizon] = useState<number>(60)
  const [zoneSort, setZoneSort] = useState<'mae' | 'bias_asc' | 'bias_desc'>('mae')
  const [zoneErrors, setZoneErrors] = useState<ZoneErrorRecord[]>([])
  const [timeseriesErrors, setTimeseriesErrors] = useState<TimeSeriesErrorPoint[]>([])

  // Historical Baseline State
  const [historicalData, setHistoricalData] = useState<HistoricalResponse | null>(null)

  // Load Initial Metadata & Summaries
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true)
        setError('')

        const [summaryRes, perfRes, locRes] = await Promise.all([
          fetch('/api/anomalies/summary'),
          fetch('/api/forecast-performance'),
          fetch('/api/locations'),
        ])

        if (summaryRes.ok) {
          const s = (await summaryRes.json()) as AnomalySummary
          setSummary(s)
        }

        if (perfRes.ok) {
          const p = (await perfRes.json()) as ForecastPerformanceResponse
          setPerformance(p)
        }

        if (locRes.ok) {
          const l = (await locRes.json()) as { locations: LocationItem[] }
          setLocations(l.locations || [])
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to initialize analytics diagnostics.')
      } finally {
        setLoading(false)
      }
    }

    void fetchInitialData()
  }, [])

  // Load Anomalies Table when filters change
  useEffect(() => {
    const loadAnomalies = async () => {
      try {
        const params = new URLSearchParams()
        params.append('limit', '50')
        if (selectedType !== 'all') params.append('anomaly_type', selectedType)
        if (selectedSeverity !== 'all') params.append('severity', selectedSeverity)
        if (selectedZone !== 'all') params.append('zone_id', selectedZone)

        const res = await fetch(`/api/anomalies?${params.toString()}`)
        if (res.ok) {
          const data = (await res.json()) as { anomalies: AnomalyRecord[] }
          setAnomalies(data.anomalies || [])
        }
      } catch (err) {
        console.error('Error fetching filtered anomalies:', err)
      }
    }

    void loadAnomalies()
  }, [selectedType, selectedSeverity, selectedZone])

  // Load Forecast Error Details when horizon or sort changes
  useEffect(() => {
    const loadErrorBreakdowns = async () => {
      try {
        const [zRes, tRes] = await Promise.all([
          fetch(`/api/forecast-errors/zones?horizon=${selectedHorizon}&sort_by=${zoneSort}&limit=10`),
          fetch(`/api/forecast-errors/timeseries?horizon=${selectedHorizon}`),
        ])

        if (zRes.ok) {
          const zData = (await zRes.json()) as { zones: ZoneErrorRecord[] }
          setZoneErrors(zData.zones || [])
        }

        if (tRes.ok) {
          const tData = (await tRes.json()) as { timeseries: TimeSeriesErrorPoint[] }
          setTimeseriesErrors(tData.timeseries || [])
        }
      } catch (err) {
        console.error('Error fetching error breakdowns:', err)
      }
    }

    if (activeTab === 'errors') {
      void loadErrorBreakdowns()
    }
  }, [activeTab, selectedHorizon, zoneSort])

  // Load Historical Profile if tab chosen
  useEffect(() => {
    if (activeTab === 'historical' && !historicalData) {
      const loadHist = async () => {
        try {
          const res = await fetch('/api/historical')
          if (res.ok) {
            const h = (await res.json()) as HistoricalResponse
            setHistoricalData(h)
          }
        } catch (err) {
          console.error('Error loading historical stats:', err)
        }
      }
      void loadHist()
    }
  }, [activeTab, historicalData])

  if (loading) {
    return <StateCard tone="loading" title="Loading Demand Analytics…" description="Loading anomaly detection and multi-horizon forecast performance." />
  }

  if (error) {
    return <StateCard tone="error" title="Analytics Unavailable" description={error} />
  }

  const selectedPerf = performance?.horizons.find((h) => h.horizon_minutes === selectedHorizon) || performance?.horizons[0]

  return (
    <section className="forecast-section" style={{ maxWidth: 1200, margin: '0 auto', paddingBottom: 60 }}>
      <SectionHeading eyebrow="ANOMALY DETECTION & ERROR ANALYSIS" title="Demand Diagnostics & Performance" />

      {/* Primary Tab Navigation */}
      <div style={{ display: 'flex', gap: 10, margin: '20px 0 24px', borderBottom: '1px solid var(--line)', paddingBottom: 14 }}>
        <button
          type="button"
          className={`filter-pill ${activeTab === 'anomalies' ? 'active' : ''}`}
          onClick={() => setActiveTab('anomalies')}
          style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600, padding: '8px 16px', borderRadius: 20, cursor: 'pointer' }}
        >
          <Activity size={16} />
          Demand Anomaly Detection
          {summary && <span style={{ background: '#dbe8ee', color: 'var(--navy)', borderRadius: 10, padding: '2px 7px', fontSize: 11 }}>{summary.total_anomalies.toLocaleString()}</span>}
        </button>

        <button
          type="button"
          className={`filter-pill ${activeTab === 'errors' ? 'active' : ''}`}
          onClick={() => setActiveTab('errors')}
          style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600, padding: '8px 16px', borderRadius: 20, cursor: 'pointer' }}
        >
          <BarChart3 size={16} />
          Forecast Error Analysis
        </button>

        <button
          type="button"
          className={`filter-pill ${activeTab === 'historical' ? 'active' : ''}`}
          onClick={() => setActiveTab('historical')}
          style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600, padding: '8px 16px', borderRadius: 20, cursor: 'pointer' }}
        >
          <Layers size={16} />
          Historical Baseline Profile
        </button>
      </div>

      {/* TAB 1: ANOMALY DETECTION */}
      {activeTab === 'anomalies' && (
        <>
          {/* KPI Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, marginBottom: 16 }}>
            <article className="compact-metric-card" style={{ borderLeft: '4px solid #f97316' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="eyebrow" style={{ color: '#c2410c' }}>DEMAND SPIKES</span>
                <ArrowUpRight size={16} color="#ea580c" />
              </div>
              <div className="metric-val">
                {summary?.by_type.demand_spike?.toLocaleString() ?? 0}
              </div>
              <p className="metric-desc">Unusually high volume (&gt;2.5 MAD z-score)</p>
            </article>

            <article className="compact-metric-card" style={{ borderLeft: '4px solid #0284c7' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="eyebrow" style={{ color: '#0369a1' }}>DEMAND DROPS</span>
                <ArrowDownRight size={16} color="#0284c7" />
              </div>
              <div className="metric-val">
                {summary?.by_type.demand_drop?.toLocaleString() ?? 0}
              </div>
              <p className="metric-desc">Unusually low volume (&lt;-2.5 MAD z-score)</p>
            </article>

            <article className="compact-metric-card" style={{ borderLeft: '4px solid #8b5cf6' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="eyebrow" style={{ color: '#6d28d9' }}>FORECAST DEVIATIONS</span>
                <AlertTriangle size={16} color="#8b5cf6" />
              </div>
              <div className="metric-val">
                {summary?.by_type.forecast_deviation?.toLocaleString() ?? 0}
              </div>
              <p className="metric-desc">Breached 80% conformal bounds</p>
            </article>

            <article className="compact-metric-card" style={{ borderLeft: '4px solid #eab308' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="eyebrow" style={{ color: '#a16207' }}>PERSISTENT (&ge;3h)</span>
                <Clock size={16} color="#ca8a04" />
              </div>
              <div className="metric-val">
                {summary?.by_type.persistent_anomaly?.toLocaleString() ?? 0}
              </div>
              <p className="metric-desc">3+ consecutive anomalous hours</p>
            </article>

            <article className="compact-metric-card" style={{ borderLeft: '4px solid #64748b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="eyebrow" style={{ color: '#475569' }}>DATA QUALITY</span>
                <ShieldAlert size={16} color="#64748b" />
              </div>
              <div className="metric-val">
                {summary?.by_type.data_quality?.toLocaleString() ?? 0}
              </div>
              <p className="metric-desc">Missing, invalid, duplicate, or extreme</p>
            </article>
          </div>

          {/* Anomaly Records Explorer with Integrated Filters */}
          <article className="integrated-table-card">
            {/* Header: Title and Observation Count */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div>
                <span className="eyebrow">AUDIT TRAIL</span>
                <h3 style={{ margin: '2px 0 0', fontSize: 17, color: 'var(--navy)' }}>Detected Demand Anomalies</h3>
              </div>
              <span style={{ fontSize: 11, color: 'var(--navy)', background: '#eef3f6', padding: '3px 9px', borderRadius: 10, fontWeight: 700 }}>
                {anomalies.length} observations
              </span>
            </div>

            {/* Filter Toolbar (Directly Above Table with Zero Gap) */}
            <div className="filter-toolbar">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--navy)', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <Filter size={13} /> Filters:
                </span>

                {/* Anomaly Type */}
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  style={{ padding: '5px 10px', borderRadius: 6, border: '1px solid var(--line)', fontSize: 12, background: '#fff' }}
                >
                  <option value="all">All Anomaly Types</option>
                  <option value="demand_spike">Demand Spikes</option>
                  <option value="demand_drop">Demand Drops</option>
                  <option value="forecast_deviation">Forecast Deviations</option>
                  <option value="persistent_anomaly">Persistent (3+ hrs)</option>
                  <option value="data_quality">Data Quality Defects</option>
                </select>

                {/* Severity */}
                <select
                  value={selectedSeverity}
                  onChange={(e) => setSelectedSeverity(e.target.value)}
                  style={{ padding: '5px 10px', borderRadius: 6, border: '1px solid var(--line)', fontSize: 12, background: '#fff' }}
                >
                  <option value="all">All Severities</option>
                  <option value="low">Low (|z| &ge; 2.5)</option>
                  <option value="medium">Medium (|z| &ge; 3.5)</option>
                  <option value="high">High (|z| &ge; 5.0)</option>
                  <option value="critical">Critical (|z| &ge; 7.0)</option>
                </select>

                {/* Zone Selector */}
                <select
                  value={selectedZone}
                  onChange={(e) => setSelectedZone(e.target.value)}
                  style={{ padding: '5px 10px', borderRadius: 6, border: '1px solid var(--line)', fontSize: 12, maxWidth: 220, background: '#fff' }}
                >
                  <option value="all">All Zones (NYC)</option>
                  {locations.map((loc) => (
                    <option key={loc.PULocationID} value={loc.PULocationID}>
                      {loc.PULocationID} - {loc.Zone} ({loc.Borough})
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="button"
                onClick={() => {
                  setSelectedType('all')
                  setSelectedSeverity('all')
                  setSelectedZone('all')
                }}
                style={{ padding: '5px 10px', border: '1px solid var(--line)', background: '#fff', borderRadius: 6, cursor: 'pointer', fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}
              >
                <RefreshCw size={11} /> Reset Filters
              </button>
            </div>

            {/* Results Table OR Compact Empty State */}
            {anomalies.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '24px 16px', color: 'var(--muted)' }}>
                <CheckCircle2 size={26} color="#10b981" style={{ margin: '0 auto 6px' }} />
                <p style={{ fontWeight: 600, color: 'var(--navy)', margin: '0 0 2px', fontSize: 13 }}>No anomalies match the selected filters.</p>
                <p style={{ fontSize: 11, margin: 0 }}>Try clearing filters or selecting another zone or severity level.</p>
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--line)', color: 'var(--muted)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      <th style={{ padding: '8px 6px' }}>Timestamp</th>
                      <th style={{ padding: '8px 6px' }}>Zone</th>
                      <th style={{ padding: '8px 6px' }}>Type</th>
                      <th style={{ padding: '8px 6px' }}>Severity</th>
                      <th style={{ padding: '8px 6px', textAlign: 'right' }}>Observed</th>
                      <th style={{ padding: '8px 6px', textAlign: 'right' }}>Baseline</th>
                      <th style={{ padding: '8px 6px', textAlign: 'right' }}>Deviation</th>
                      <th style={{ padding: '8px 6px', textAlign: 'right' }}>Z-Score</th>
                      <th style={{ padding: '8px 6px' }}>Explanation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {anomalies.map((rec, idx) => {
                      const typeBadge = {
                        demand_spike: { label: 'Spike', bg: '#ffedd5', text: '#c2410c' },
                        demand_drop: { label: 'Drop', bg: '#e0f2fe', text: '#0369a1' },
                        forecast_deviation: { label: 'Forecast Dev', bg: '#ede9fe', text: '#6d28d9' },
                        persistent_anomaly: { label: `Persistent (${rec.consecutive_count}h)`, bg: '#fef3c7', text: '#b45309' },
                        data_quality: { label: 'Data Quality', bg: '#f1f5f9', text: '#334155' },
                      }[rec.anomaly_type]

                      const sevBadge = {
                        low: { bg: '#f3f4f6', text: '#4b5563' },
                        medium: { bg: '#fef9c3', text: '#854d0e' },
                        high: { bg: '#ffedd5', text: '#c2410c' },
                        critical: { bg: '#fee2e2', text: '#991b1b' },
                      }[rec.severity]

                      return (
                        <tr key={`${rec.zone_id}-${rec.timestamp}-${idx}`} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '8px 6px', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                            {rec.timestamp.replace('T', ' ')}
                          </td>
                          <td style={{ padding: '8px 6px' }}>
                            <strong>{rec.zone_name || `Zone ${rec.zone_id}`}</strong>
                            <small style={{ display: 'block', color: 'var(--muted)', fontSize: 10 }}>{rec.borough}</small>
                          </td>
                          <td style={{ padding: '8px 6px' }}>
                            <span style={{ background: typeBadge.bg, color: typeBadge.text, padding: '2px 7px', borderRadius: 10, fontSize: 10, fontWeight: 700 }}>
                              {typeBadge.label}
                            </span>
                          </td>
                          <td style={{ padding: '8px 6px' }}>
                            <span style={{ background: sevBadge.bg, color: sevBadge.text, padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 700, textTransform: 'uppercase' }}>
                              {rec.severity}
                            </span>
                          </td>
                          <td style={{ padding: '8px 6px', textAlign: 'right', fontWeight: 700 }}>
                            {rec.actual_demand !== null && rec.actual_demand !== undefined ? Math.round(rec.actual_demand) : '—'}
                          </td>
                          <td style={{ padding: '8px 6px', textAlign: 'right', color: 'var(--muted)' }}>
                            {rec.baseline_demand !== null && rec.baseline_demand !== undefined ? Math.round(rec.baseline_demand) : '—'}
                          </td>
                          <td style={{ padding: '8px 6px', textAlign: 'right', fontWeight: 700, color: (rec.deviation || 0) > 0 ? '#c2410c' : '#0369a1' }}>
                            {rec.deviation !== null && rec.deviation !== undefined ? `${rec.deviation > 0 ? '+' : ''}${Math.round(rec.deviation)}` : '—'}
                          </td>
                          <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'monospace' }}>
                            {rec.z_score !== null && rec.z_score !== undefined ? `${rec.z_score > 0 ? '+' : ''}${rec.z_score.toFixed(2)}` : '—'}
                          </td>
                          <td style={{ padding: '8px 6px', fontSize: 11, maxWidth: 300, color: 'var(--ink)' }}>
                            {rec.explanation}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        </>
      )}
      {/* TAB 2: FORECAST ERROR ANALYSIS */}
      {activeTab === 'errors' && (
        <>
          {/* Horizon Pills Toolbar */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 14 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--navy)' }}>Forecast Horizon:</span>
            {[
              { min: 60, label: '1 Hour Ahead (T+1h)' },
              { min: 180, label: '3 Hours Ahead (T+3h)' },
              { min: 360, label: '6 Hours Ahead (T+6h)' },
            ].map((h) => (
              <button
                key={h.min}
                type="button"
                className={`filter-pill ${selectedHorizon === h.min ? 'active' : ''}`}
                onClick={() => setSelectedHorizon(h.min)}
                style={{ padding: '5px 12px', borderRadius: 14, cursor: 'pointer', fontWeight: 600, fontSize: 11 }}
              >
                {h.label}
              </button>
            ))}
          </div>

          {/* Model Performance Cards (Content-Driven & Compact) */}
          {selectedPerf && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, marginBottom: 16 }}>
              <article className="compact-metric-card">
                <span className="eyebrow">MAE (ACCURACY)</span>
                <div className="metric-val">
                  {selectedPerf.mae.toFixed(2)} <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--muted)' }}>rides/hr</span>
                </div>
                <p className="metric-desc">Mean Absolute Error on June test split</p>
              </article>

              <article className="compact-metric-card">
                <span className="eyebrow">RMSE (SPREAD)</span>
                <div className="metric-val">
                  {selectedPerf.rmse.toFixed(2)} <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--muted)' }}>rides/hr</span>
                </div>
                <p className="metric-desc">Penalizes large outlier forecast errors</p>
              </article>

              <article className="compact-metric-card" style={{ borderLeft: `3px solid ${Math.abs(selectedPerf.mean_bias) < 0.25 ? '#10b981' : '#f59e0b'}` }}>
                <span className="eyebrow">SIGNED MEAN BIAS</span>
                <div className="metric-val" style={{ color: selectedPerf.mean_bias > 0 ? '#b45309' : '#0369a1' }}>
                  {selectedPerf.mean_bias > 0 ? `+${selectedPerf.mean_bias.toFixed(2)}` : selectedPerf.mean_bias.toFixed(2)}{' '}
                  <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--muted)' }}>rides</span>
                </div>
                <p className="metric-desc" style={{ color: 'var(--ink)' }}>
                  {selectedPerf.mean_bias > 0 ? 'Underpredicting (actual > pred)' : 'Overpredicting (pred > actual)'}
                </p>
              </article>

              <article className="compact-metric-card">
                <span className="eyebrow">WAPE</span>
                <div className="metric-val">
                  {selectedPerf.wape_pct.toFixed(1)}%
                </div>
                <p className="metric-desc">Volume-weighted percentage error</p>
              </article>

              <article className="compact-metric-card">
                <span className="eyebrow">80% INTERVAL COVERAGE</span>
                <div className="metric-val" style={{ color: '#059669' }}>
                  {selectedPerf.empirical_coverage_pct.toFixed(1)}%
                </div>
                <p className="metric-desc">Target: 80% conformal coverage</p>
              </article>
            </div>
          )}

          {/* Time Series Error Trend Chart */}
          <article className="integrated-table-card" style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div>
                <span className="eyebrow">TEMPORAL ERROR TREND</span>
                <h3 style={{ margin: '2px 0 0', fontSize: 16, color: 'var(--navy)' }}>Daily Forecast Error & Signed Bias across June Test Month</h3>
              </div>
              <div style={{ display: 'flex', gap: 12, fontSize: 11 }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 10, height: 10, background: '#2aa6a4', borderRadius: 2 }} /> Daily MAE
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 12, height: 3, background: '#e11d48' }} /> Signed Bias (actual - pred)
                </span>
              </div>
            </div>

            <div style={{ width: '100%', height: 240 }}>
              <ResponsiveContainer>
                <ComposedChart data={timeseriesErrors} margin={{ top: 8, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(d) => d.slice(5)} />
                  <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
                  <YAxis yAxisId="right" orientation="right" domain={[-1.5, 1.5]} tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(val, name) => {
                      const num = Number(val)
                      return [num.toFixed(2), name === 'mae' ? 'Daily MAE (rides)' : 'Signed Mean Bias (rides)']
                    }}
                  />
                  <ReferenceLine yAxisId="right" y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                  <Bar yAxisId="left" dataKey="mae" fill="#2aa6a4" opacity={0.8} radius={[3, 3, 0, 0]} />
                  <Line yAxisId="right" type="monotone" dataKey="mean_bias" stroke="#e11d48" strokeWidth={2} dot={{ r: 2 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }}>
              Note: Error convention strictly follows <code>error = actual_demand - predicted_demand</code>. A positive bias means underprediction; negative bias means overprediction.
            </div>
          </article>

          {/* Top Error Zones Table & Comparison */}
          <article className="integrated-table-card">
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div>
                <span className="eyebrow">ZONE-LEVEL BREAKDOWN</span>
                <h3 style={{ margin: '2px 0 0', fontSize: 16, color: 'var(--navy)' }}>Top 10 Problem Zones by Accuracy & Bias</h3>
              </div>

              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  type="button"
                  className={`filter-pill ${zoneSort === 'mae' ? 'active' : ''}`}
                  onClick={() => setZoneSort('mae')}
                  style={{ fontSize: 11, padding: '4px 10px', borderRadius: 12, cursor: 'pointer' }}
                >
                  Highest MAE
                </button>
                <button
                  type="button"
                  className={`filter-pill ${zoneSort === 'bias_asc' ? 'active' : ''}`}
                  onClick={() => setZoneSort('bias_asc')}
                  style={{ fontSize: 11, padding: '4px 10px', borderRadius: 12, cursor: 'pointer' }}
                >
                  Most Overpredicted
                </button>
                <button
                  type="button"
                  className={`filter-pill ${zoneSort === 'bias_desc' ? 'active' : ''}`}
                  onClick={() => setZoneSort('bias_desc')}
                  style={{ fontSize: 11, padding: '4px 10px', borderRadius: 12, cursor: 'pointer' }}
                >
                  Most Underpredicted
                </button>
              </div>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--line)', color: 'var(--muted)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    <th style={{ padding: '8px 6px' }}>Zone Name</th>
                    <th style={{ padding: '8px 6px' }}>Borough</th>
                    <th style={{ padding: '8px 6px', textAlign: 'right' }}>MAE (Rides)</th>
                    <th style={{ padding: '8px 6px', textAlign: 'right' }}>RMSE</th>
                    <th style={{ padding: '8px 6px', textAlign: 'right' }}>Signed Mean Bias</th>
                    <th style={{ padding: '8px 6px', textAlign: 'right' }}>WAPE</th>
                    <th style={{ padding: '8px 6px', textAlign: 'right' }}>Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {zoneErrors.map((z) => (
                    <tr key={z.zone_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>
                        {z.zone_name} <span style={{ color: 'var(--muted)', fontWeight: 400, fontSize: 10 }}>#{z.zone_id}</span>
                      </td>
                      <td style={{ padding: '8px 6px', color: 'var(--muted)' }}>{z.borough}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'right', fontWeight: 700 }}>{z.mae.toFixed(2)}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'right', color: 'var(--muted)' }}>{z.rmse.toFixed(2)}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'right', fontWeight: 700, color: z.mean_bias > 0 ? '#b45309' : '#0369a1' }}>
                        {z.mean_bias > 0 ? `+${z.mean_bias.toFixed(2)}` : z.mean_bias.toFixed(2)}
                      </td>
                      <td style={{ padding: '8px 6px', textAlign: 'right' }}>{z.wape_pct.toFixed(1)}%</td>
                      <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'monospace' }}>{z.sample_count.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </>
      )}

      {/* TAB 3: HISTORICAL BASELINE */}
      {activeTab === 'historical' && (
        <div className="dashboard-grid">
          <article className="panel" style={{ minHeight: 320 }}>
            <div className="panel-heading">
              <div>
                <span className="eyebrow">HOURLY PROFILE</span>
                <h3>Average rides by hour of day</h3>
              </div>
            </div>

            <div style={{ width: '100%', height: 260, marginTop: 18 }}>
              {historicalData ? (
                <ResponsiveContainer>
                  <BarChart data={historicalData.demand_by_hour_of_day} margin={{ top: 12, right: 12, left: 0, bottom: 18 }}>
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
                      {historicalData.demand_by_hour_of_day.map((entry) => (
                        <Cell key={entry.hour} fill="#2aa6a4" />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ textAlign: 'center', paddingTop: 80, color: 'var(--muted)' }}>Loading hourly profile…</div>
              )}
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
              {historicalData ? (
                <ResponsiveContainer>
                  <BarChart data={historicalData.demand_by_location.slice(0, 10)} margin={{ top: 12, right: 12, left: 0, bottom: 18 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="PULocationID" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip
                      formatter={(value) => {
                        const numericValue = Array.isArray(value) ? value[0] : value
                        return typeof numericValue === 'number' ? Number(numericValue).toLocaleString() : String(numericValue ?? '')
                      }}
                    />
                    <Bar dataKey="total_rides" fill={CAB_YELLOW} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ textAlign: 'center', paddingTop: 80, color: 'var(--muted)' }}>Loading top zones…</div>
              )}
            </div>
          </article>
        </div>
      )}
    </section>
  )
}

export default Analytics
