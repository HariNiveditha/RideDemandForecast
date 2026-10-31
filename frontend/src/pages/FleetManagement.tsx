import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Compass,
  Download,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  Truck,
  Users,
  Zap,
} from 'lucide-react'
import SectionHeading from '../components/SectionHeading'
import StateCard from '../components/StateCard'

type ZoneMetrics = {
  zone_id: number
  predicted_rides: number
  idle_cabs: number
  busy_cabs: number
  repositioning_cabs: number
  estimated_cabs_required: number
  surplus: number
  shortage_estimate: number
  surplus_after_own_demand: number
}

type ZoneAvailability = {
  idle_cabs: number
  busy_cabs: number
  total_cabs: number
  repositioning_cabs?: number
  estimated_cabs_required?: number
  shortage_estimate?: number
  surplus_after_own_demand?: number
}

type FleetState = {
  simulation_id: string
  timestamp: string
  fleet_size: number
  availability: Record<string, ZoneAvailability>
  predicted_rides?: Record<string, number> | Record<number, number>
}

type FleetMetrics = {
  zones: ZoneMetrics[]
  recommended_movements: {
    from_zone: number
    to_zone: number
    cab_count: number
    reason?: string
  }[]
  unmet_demand_estimate: {
    total_unmet_rides: number
    affected_zones: number[]
  }
  demand_source?: DemandSource
}

/** Demand source selector values. 'historical' is the default behaviour. */
type ForecastHorizon = 'historical' | 60 | 180 | 360

/** Mirrors the `demand_source` block returned by POST /fleet/recommendations. */
type DemandSource = {
  type: 'historical' | 'forecast' | 'historical_fallback'
  requested_horizon_minutes: number | null
  horizon_minutes: number | null
  forecast_origin: string | null
  target_timestamp: string | null
  model_version: string | null
  zones_forecasted: number
  zones_from_history: number
  total_predicted_rides: number | null
  description: string
  forecast_error: string | null
  data_basis?: string
  dataset_latest_timestamp?: string | null
  system_time?: string | null
  is_live?: boolean
  freshness_note?: string
  origin_is_dataset_latest?: boolean
  target_beyond_dataset_end?: boolean
}

/** Mirrors zone-level demand anomaly records returned by GET /anomalies. */
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

function describeDataBasis(source: DemandSource): string {
  const basis = source.is_live ? 'Live demand feed' : 'Historical dataset (not live)'
  const latest = source.dataset_latest_timestamp ? `; latest demand data ${source.dataset_latest_timestamp}` : ''
  const clock = source.system_time ? `; response generated ${source.system_time}` : ''
  return `${basis}${latest}${clock}.`
}

function describeDemandSource(source: DemandSource): string {
  if (source.type === 'forecast') {
    const anchor = source.origin_is_dataset_latest ? ' (measured from newest data hour)' : ''
    return `Demand Forecast, +${source.horizon_minutes! / 60}H (${source.horizon_minutes}m ahead)${anchor}`
  }
  if (source.type === 'historical_fallback') {
    const reason = source.forecast_error ? ` — ${source.forecast_error}` : ''
    return `Historical hourly average (requested ${source.requested_horizon_minutes}m forecast unavailable${reason})`
  }
  return 'Historical hourly average (default baseline)'
}

interface FleetManagementProps {
  backendUrl?: string
}

type ApiFleetState = Partial<FleetState> & {
  available?: Record<string, ZoneAvailability>
  availability?: Record<string, ZoneAvailability>
}

function normalizeFleetState(data: ApiFleetState): FleetState {
  return {
    simulation_id: data.simulation_id ?? '',
    timestamp: data.timestamp ?? new Date().toISOString(),
    fleet_size: Number(data.fleet_size ?? 0),
    availability: data.availability ?? data.available ?? {},
    predicted_rides: data.predicted_rides ?? {},
  }
}

function FleetManagement({ backendUrl = '/api' }: FleetManagementProps) {
  const [initialized, setInitialized] = useState(false)
  const [fleetState, setFleetState] = useState<FleetState | null>(null)
  const [metrics, setMetrics] = useState<FleetMetrics | null>(null)
  const [anomalies, setAnomalies] = useState<AnomalyRecord[]>([])
  const [loading, setLoading] = useState<
    'idle' | 'initializing' | 'refreshing' | 'advancing' | 'resetting' | 'exporting' | 'recommending'
  >('idle')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  // Demand source used when generating recommendations (historical is default)
  const [demandHorizon, setDemandHorizon] = useState<ForecastHorizon>('historical')

  // Collapsible section state
  const [zoneSectionExpanded, setZoneSectionExpanded] = useState(false)
  const [demandSectionExpanded, setDemandSectionExpanded] = useState(false)
  const [selectedFleetSize, setSelectedFleetSize] = useState<number>(20000)

  // Filter and search state for zone tables
  const [zoneSearch, setZoneSearch] = useState('')
  const [zoneFilter, setZoneFilter] = useState<'all' | 'shortage' | 'surplus' | 'anomaly'>('all')

  useEffect(() => {
    if (fleetState?.fleet_size && fleetState.fleet_size > 0) {
      setSelectedFleetSize(fleetState.fleet_size)
    }
  }, [fleetState?.fleet_size])

  // Map of zone_id -> latest anomaly record from Demand Anomaly Detection
  const zoneAnomalyMap = useMemo(() => {
    const map = new Map<number, AnomalyRecord>()
    for (const anom of anomalies) {
      const existing = map.get(anom.zone_id)
      if (!existing || new Date(anom.timestamp) > new Date(existing.timestamp)) {
        map.set(anom.zone_id, anom)
      }
    }
    return map
  }, [anomalies])

  const fetchAnomalies = useCallback(async () => {
    try {
      const resp = await fetch(`${backendUrl}/anomalies?limit=250`)
      if (resp.ok) {
        const data = (await resp.json()) as { anomalies: AnomalyRecord[] }
        if (Array.isArray(data.anomalies)) {
          setAnomalies(data.anomalies)
        }
      }
    } catch {
      // Non-blocking: Fleet simulation operates even if anomaly endpoint is unreachable
    }
  }, [backendUrl])

  const fetchRecommendations = useCallback(
    async (simulationId: string, horizon?: ForecastHorizon) => {
      if (!simulationId) return
      const activeHorizon = horizon !== undefined ? horizon : demandHorizon
      const body: Record<string, unknown> = { simulation_id: simulationId }
      if (activeHorizon !== 'historical') body.horizon_minutes = activeHorizon
      const resp = await fetch(`${backendUrl}/fleet/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const payload = (await resp.json().catch(() => null)) as { detail?: unknown } | null
        const detail = payload?.detail
        throw new Error(
          typeof detail === 'string' ? detail : `Recommendations request failed (HTTP ${resp.status})`,
        )
      }
      const data = await resp.json()
      setMetrics({
        zones: Array.isArray(data.zones) ? data.zones : [],
        recommended_movements: Array.isArray(data.recommended_movements)
          ? data.recommended_movements
          : [],
        unmet_demand_estimate: data.unmet_demand_estimate ?? {
          total_unmet_rides: 0,
          affected_zones: [],
        },
        demand_source: data.demand_source ?? undefined,
      })
      return (data.demand_source ?? undefined) as DemandSource | undefined
    },
    [backendUrl, demandHorizon],
  )

  const fetchFleetCurrent = useCallback(
    async (showLoading = true) => {
      if (showLoading) setLoading('refreshing')
      setError(null)
      setNotice(null)
      try {
        const resp = await fetch(`${backendUrl}/fleet/current`)
        if (!resp.ok) {
          if (resp.status === 503 || resp.status === 404) {
            setInitialized(false)
            setFleetState(null)
            setMetrics(null)
            return
          }
          throw new Error(`Unable to fetch fleet state (HTTP ${resp.status})`)
        }
        const data = normalizeFleetState((await resp.json()) as ApiFleetState)
        setFleetState(data)
        setInitialized(Boolean(data.simulation_id))
        void fetchAnomalies()
        if (data.simulation_id) {
          try {
            await fetchRecommendations(data.simulation_id, demandHorizon)
          } catch (recError) {
            setMetrics(null)
            setError(
              recError instanceof Error ? recError.message : 'Recommendations could not be loaded',
            )
          }
        }
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Unable to load fleet state')
      } finally {
        if (showLoading) setLoading('idle')
      }
    },
    [backendUrl, demandHorizon, fetchAnomalies, fetchRecommendations],
  )

  useEffect(() => {
    void fetchFleetCurrent(false)
  }, [fetchFleetCurrent])

  async function handleInit(fleetSizeOverride?: number) {
    setLoading('initializing')
    setError(null)
    setNotice(null)
    const targetSize = fleetSizeOverride ?? selectedFleetSize
    try {
      const resp = await fetch(`${backendUrl}/fleet/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fleet_size: targetSize }),
      })
      if (!resp.ok) throw new Error(`Fleet initialization failed (HTTP ${resp.status})`)
      const data = await resp.json()
      const simulationId = String(data.simulation_id ?? '')
      if (!simulationId) throw new Error('Initialization response did not include a simulation ID.')
      await fetchFleetCurrent(false)
      setInitialized(true)
      setNotice(`Fleet simulation initialized with ${targetSize.toLocaleString()} cabs.`)
      setFleetState((current) => current ?? normalizeFleetState(data))
      void fetchAnomalies()
      if (simulationId) {
        try {
          await fetchRecommendations(simulationId, demandHorizon)
        } catch (recError) {
          setMetrics(null)
          setError(
            recError instanceof Error ? recError.message : 'Recommendations could not be loaded',
          )
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to initialize fleet')
    } finally {
      setLoading('idle')
    }
  }

  async function handleRecommendations() {
    if (!fleetState?.simulation_id) {
      setError('Initialize the fleet simulation first.')
      return
    }
    setLoading('recommending')
    setError(null)
    setNotice(null)
    try {
      const source = await fetchRecommendations(fleetState.simulation_id, demandHorizon)
      setNotice(
        source
          ? `Recommendations refreshed using ${describeDemandSource(source)}. No vehicle movements were executed.`
          : 'Recommendations refreshed. No vehicle movements were executed.',
      )
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to generate recommendations')
    } finally {
      setLoading('idle')
    }
  }

  async function handleAdvance() {
    if (!fleetState?.simulation_id) {
      setError('Initialize the fleet simulation first.')
      return
    }
    setLoading('advancing')
    setError(null)
    setNotice(null)
    try {
      const resp = await fetch(`${backendUrl}/fleet/advance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simulation_id: fleetState.simulation_id }),
      })
      if (!resp.ok) {
        const detail = await resp.text()
        throw new Error(detail || `Simulation advance failed (HTTP ${resp.status})`)
      }
      await fetchFleetCurrent(false)
      setNotice('Simulation step advanced (+1 hour). Fleet availability and recommendations refreshed.')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to advance simulation')
    } finally {
      setLoading('idle')
    }
  }

  async function handleReset() {
    if (!fleetState?.simulation_id) {
      setError('Initialize the fleet simulation first.')
      return
    }
    if (!window.confirm('Reset the fleet simulation to its initial state?')) return

    setLoading('resetting')
    setError(null)
    setNotice(null)
    try {
      const resp = await fetch(`${backendUrl}/fleet/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simulation_id: fleetState.simulation_id }),
      })
      if (!resp.ok) {
        const detail = await resp.text()
        throw new Error(detail || `Simulation reset failed (HTTP ${resp.status})`)
      }
      setMetrics(null)
      await fetchFleetCurrent(false)
      setNotice('Simulation reset to initial fleet state.')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to reset simulation')
    } finally {
      setLoading('idle')
    }
  }

  async function handleExport() {
    setLoading('exporting')
    setError(null)
    setNotice(null)
    try {
      const resp = await fetch(`${backendUrl}/fleet/export`)
      if (!resp.ok) {
        const detail = await resp.text()
        throw new Error(detail || `Fleet export failed (HTTP ${resp.status})`)
      }
      const data = await resp.json()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `fleet-snapshot-${data.simulation_id ?? 'export'}.json`
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
      setNotice('Fleet snapshot export downloaded successfully.')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to export fleet snapshots')
    } finally {
      setLoading('idle')
    }
  }

  const formatNumber = (value: number | undefined) =>
    Number(value ?? 0).toLocaleString()

  const zoneEntries = (Object.entries(fleetState?.availability ?? {}) as [string, ZoneAvailability][]).map(
    ([zoneId, value]) => ({
      zoneId: Number(zoneId),
      ...value,
    }),
  )

  const filteredZones = zoneEntries.filter((zone) => {
    const zoneIdStr = zone.zoneId.toString()
    const searchMatch = zoneSearch === '' || zoneIdStr.includes(zoneSearch)
    if (!searchMatch) return false

    if (zoneFilter === 'shortage') {
      if (!metrics?.zones) return false
      const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
      if (!metric || metric.shortage_estimate <= 0) return false
    } else if (zoneFilter === 'surplus') {
      if (!metrics?.zones) return false
      const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
      if (!metric || metric.surplus_after_own_demand <= 0) return false
    } else if (zoneFilter === 'anomaly') {
      if (!zoneAnomalyMap.has(zone.zoneId)) return false
    }
    return true
  })

  const totalCabs = zoneEntries.reduce((sum, zone) => sum + Number(zone.total_cabs ?? 0), 0)
  const totalIdle = zoneEntries.reduce((sum, zone) => sum + Number(zone.idle_cabs ?? 0), 0)
  const totalBusy = zoneEntries.reduce((sum, zone) => sum + Number(zone.busy_cabs ?? 0), 0)
  const totalRepositioning = zoneEntries.reduce(
    (sum, zone) => sum + Number(zone.repositioning_cabs ?? 0),
    0,
  )

  // Demand-source consistent total predicted demand
  const totalPredicted =
    metrics?.demand_source?.total_predicted_rides ??
    (metrics?.zones && metrics.zones.length > 0
      ? metrics.zones.reduce((sum, zone) => sum + Number(zone.predicted_rides ?? 0), 0)
      : (Object.values(fleetState?.predicted_rides ?? {}) as number[]).reduce(
          (sum, rides) => sum + Number(rides ?? 0),
          0,
        ))

  const shortageZones = metrics?.zones.filter((zone) => zone.shortage_estimate > 0).length ?? 0
  const surplusZones =
    metrics?.zones.filter((zone) => zone.surplus_after_own_demand > 0).length ?? 0
  const isBusy = loading !== 'idle'

  // Uninitialized state view
  if (!initialized) {
    return (
      <section className="fleet-management-section">
        <SectionHeading eyebrow="FLEET OPERATIONS" title="Fleet Management" />
        <div style={{ maxWidth: 640, margin: '40px auto 0' }}>
          <StateCard
            title="Fleet Simulation Uninitialized"
            description="Initialize the simulated fleet to begin monitoring zone availability, demand analysis, and repositioning recommendations."
            tone="empty"
          >
            <div style={{ marginTop: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
              {error && <div className="fleet-notice-banner fleet-notice-error">⚠️ {error}</div>}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
                <div className="fleet-select-wrapper" style={{ flexDirection: 'row', alignItems: 'center' }}>
                  <span style={{ fontSize: 13, fontWeight: 600 }}>Fleet Size:</span>
                  <select
                    value={selectedFleetSize}
                    onChange={(e) => setSelectedFleetSize(Number(e.target.value))}
                    disabled={isBusy}
                    style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid var(--line)', background: '#fff' }}
                  >
                    <option value={13000}>13,000 cabs (Baseline / Deficit)</option>
                    <option value={16000}>16,000 cabs (Balanced Supply)</option>
                    <option value={20000}>20,000 cabs (Recommended — 16 recs)</option>
                    <option value={25000}>25,000 cabs (Abundant Supply)</option>
                  </select>
                </div>
                <button className="primary-button" onClick={() => void handleInit()} disabled={isBusy}>
                  {loading === 'initializing' ? <RefreshCw size={14} className="spin" /> : <Play size={14} />}
                  {loading === 'initializing' ? 'Initializing...' : 'Initialize Fleet Simulation'}
                </button>
              </div>
              <small style={{ color: 'var(--muted)', textAlign: 'center' }}>
                Fleet availability is simulated and does not represent live NYC cab availability.
              </small>
            </div>
          </StateCard>
        </div>
      </section>
    )
  }

  // Render zone rows with search/filter highlighting
  const renderZoneRows = (zones: typeof zoneEntries) => {
    if (zones.length === 0) {
      return (
        <tr>
          <td colSpan={5} className="empty-state">
            No zone availability records matched the current filter.
          </td>
        </tr>
      )
    }
    return zones.map((zone) => {
      const zoneIdStr = zone.zoneId.toString()
      const searchMatch = zoneSearch === '' || zoneIdStr.includes(zoneSearch)
      const showsShortage = zoneFilter === 'shortage'
      const showsSurplus = zoneFilter === 'surplus'
      const showsAnomaly = zoneFilter === 'anomaly'

      let matchesFilter = true
      if (showsShortage && metrics?.zones) {
        const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
        matchesFilter = Boolean(metric && metric.shortage_estimate > 0)
      }
      if (showsSurplus && metrics?.zones) {
        const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
        matchesFilter = Boolean(metric && metric.surplus_after_own_demand > 0)
      }
      if (showsAnomaly) {
        matchesFilter = zoneAnomalyMap.has(zone.zoneId)
      }

      if (!searchMatch || !matchesFilter) {
        return null
      }

      const anom = zoneAnomalyMap.get(zone.zoneId)

      return (
        <tr key={zone.zoneId} className="zone-row">
          <td>
            <strong>Zone {zone.zoneId}</strong>
            {anom && (
              <span
                style={{
                  marginLeft: 8,
                  fontSize: 10,
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: 4,
                  background:
                    anom.severity === 'critical'
                      ? '#fee2e2'
                      : anom.severity === 'high'
                        ? '#ffedd5'
                        : anom.severity === 'medium'
                          ? '#fef9c3'
                          : '#f3f4f6',
                  color:
                    anom.severity === 'critical'
                      ? '#991b1b'
                      : anom.severity === 'high'
                        ? '#c2410c'
                        : anom.severity === 'medium'
                          ? '#854d0e'
                          : '#4b5563',
                }}
                title={anom.explanation}
              >
                ⚠️ {anom.anomaly_type.replace('_', ' ')}
              </span>
            )}
          </td>
          <td>{formatNumber(zone.idle_cabs)}</td>
          <td>{formatNumber(zone.busy_cabs)}</td>
          <td>{formatNumber(zone.repositioning_cabs)}</td>
          <td><strong>{formatNumber(zone.total_cabs)}</strong></td>
        </tr>
      )
    })
  }

  return (
    <section className="fleet-management-section">
      <SectionHeading eyebrow="FLEET OPERATIONS" title="Fleet Management" />

      {/* Notifications */}
      {error && (
        <div className="fleet-notice-banner fleet-notice-error">
          <AlertTriangle size={16} /> {error}
        </div>
      )}
      {notice && (
        <div className="fleet-notice-banner fleet-notice-success">
          <CheckCircle2 size={16} /> {notice}
        </div>
      )}

      {/* ======================================================== */}
      {/* SECTION A: FLEET SUMMARY (OPERATIONAL KPI CARDS)         */}
      {/* ======================================================== */}
      <section className="fleet-kpi-grid">
        <article className="fleet-kpi-card">
          <div className="fleet-kpi-top">
            <span>Total Fleet</span>
            <span className="fleet-kpi-icon"><Truck size={16} /></span>
          </div>
          <strong>{formatNumber(fleetState?.fleet_size)}</strong>
          <small>Total registered fleet capacity</small>
        </article>

        <article className="fleet-kpi-card">
          <div className="fleet-kpi-top">
            <span>Idle / Available</span>
            <span className="fleet-kpi-icon" style={{ color: '#059669' }}><CheckCircle2 size={16} /></span>
          </div>
          <strong style={{ color: '#059669' }}>{formatNumber(totalIdle)}</strong>
          <small>Unassigned & available for dispatch</small>
        </article>

        <article className="fleet-kpi-card">
          <div className="fleet-kpi-top">
            <span>In Service / Busy</span>
            <span className="fleet-kpi-icon" style={{ color: '#2563eb' }}><Users size={16} /></span>
          </div>
          <strong style={{ color: '#2563eb' }}>{formatNumber(totalBusy)}</strong>
          <small>Passenger trips currently in progress</small>
        </article>

        <article className="fleet-kpi-card">
          <div className="fleet-kpi-top">
            <span>Repositioning</span>
            <span className="fleet-kpi-icon" style={{ color: '#d97706' }}><RotateCcw size={16} /></span>
          </div>
          <strong style={{ color: '#d97706' }}>{formatNumber(totalRepositioning)}</strong>
          <small>Cabs relocating between zones</small>
        </article>

        <article className="fleet-kpi-card">
          <div className="fleet-kpi-top">
            <span>Active Zones</span>
            <span className="fleet-kpi-icon"><Compass size={16} /></span>
          </div>
          <strong>{zoneEntries.length}</strong>
          <small>Monitored taxi dispatch zones</small>
        </article>

        <article className="fleet-kpi-card">
          <div className="fleet-kpi-top">
            <span>
              {metrics?.demand_source?.type === 'forecast'
                ? `Predicted Demand (+${metrics.demand_source.horizon_minutes! / 60}H)`
                : 'Predicted Demand'}
            </span>
            <span className="fleet-kpi-icon" style={{ color: 'var(--teal)' }}><Zap size={16} /></span>
          </div>
          <strong style={{ color: 'var(--navy)' }}>{formatNumber(totalPredicted)}</strong>
          <small>
            {metrics?.demand_source?.type === 'forecast'
              ? `${metrics.demand_source.horizon_minutes}m multi-horizon forecast`
              : 'Historical hourly baseline'}
          </small>
        </article>
      </section>

      {/* Operational Status Strip */}
      <div className="fleet-status-bar">
        <div className="fleet-status-group">
          <span>Simulation: <strong>{fleetState?.simulation_id || 'Active'}</strong></span>
          <span>•</span>
          <span>Time: <strong>{fleetState?.timestamp ? new Date(fleetState.timestamp).toLocaleString() : 'Active'}</strong></span>
          <span>•</span>
          <span className="fleet-pill fleet-pill-success">
            <ShieldCheck size={13} /> Fleet Conserved: {formatNumber(totalCabs)} / {formatNumber(fleetState?.fleet_size)}
          </span>
          {zoneAnomalyMap.size > 0 && (
            <span className="fleet-pill fleet-pill-warning">
              <AlertTriangle size={13} /> {zoneAnomalyMap.size} Demand Anomalies Flagged
            </span>
          )}
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={() => void fetchFleetCurrent()}
          disabled={isBusy}
          style={{ padding: '6px 12px', fontSize: 11 }}
        >
          <RefreshCw size={13} className={loading === 'refreshing' ? 'spin' : ''} /> Refresh Status
        </button>
      </div>

      {/* ======================================================== */}
      {/* SECTION B: SIMULATION CONTROLS (OPERATIONAL CONTROL BAR) */}
      {/* ======================================================== */}
      <div className="fleet-control-panel">
        <div className="fleet-control-inputs">
          <div className="fleet-select-wrapper">
            <label>Fleet Size</label>
            <select
              value={selectedFleetSize}
              onChange={(e) => {
                const newSize = Number(e.target.value)
                setSelectedFleetSize(newSize)
                void handleInit(newSize)
              }}
              disabled={isBusy}
            >
              <option value={13000}>13,000 (Baseline / Deficit)</option>
              <option value={16000}>16,000 (Balanced Supply)</option>
              <option value={20000}>20,000 (Recommended — 16 recs)</option>
              <option value={25000}>25,000 (Abundant Supply)</option>
            </select>
          </div>

          <div className="fleet-select-wrapper">
            <label>Demand Source</label>
            <select
              value={demandHorizon}
              onChange={(e) => {
                const value = e.target.value
                const newHorizon = value === 'historical' ? 'historical' : (Number(value) as ForecastHorizon)
                setDemandHorizon(newHorizon)
                if (fleetState?.simulation_id) {
                  setLoading('recommending')
                  setError(null)
                  setNotice(null)
                  void fetchRecommendations(fleetState.simulation_id, newHorizon)
                    .then((source) => {
                      setNotice(
                        source
                          ? `Recommendations and Demand & Supply Analysis updated using ${describeDemandSource(source)}.`
                          : 'Recommendations and Demand & Supply Analysis updated.',
                      )
                    })
                    .catch((err) => {
                      setError(err instanceof Error ? err.message : 'Failed to update recommendations')
                    })
                    .finally(() => {
                      setLoading('idle')
                    })
                }
              }}
              disabled={isBusy}
            >
              <option value="historical">Historical Demand (default baseline)</option>
              <option value={60}>Demand Forecast — 60m (+1H)</option>
              <option value={180}>Demand Forecast — 180m (+3H)</option>
              <option value={360}>Demand Forecast — 360m (+6H)</option>
            </select>
          </div>
        </div>

        <div className="fleet-control-actions">
          <button className="primary-button" onClick={handleAdvance} disabled={isBusy}>
            {loading === 'advancing' ? <RefreshCw size={14} className="spin" /> : <Play size={14} />}
            {loading === 'advancing' ? 'Advancing...' : 'Advance Simulation (+1H)'}
          </button>
          <button className="secondary-button" onClick={handleRecommendations} disabled={isBusy}>
            <Compass size={14} />
            {loading === 'recommending' ? 'Generating...' : 'Generate Recommendations'}
          </button>
          <button className="secondary-button" onClick={handleReset} disabled={isBusy}>
            <RotateCcw size={14} /> Reset
          </button>
          <button className="secondary-button" onClick={handleExport} disabled={isBusy}>
            <Download size={14} /> Export Snapshots
          </button>
        </div>
      </div>

      {/* ======================================================== */}
      {/* SECTION C: REPOSITIONING RECOMMENDATIONS                 */}
      {/* ======================================================== */}
      <div className="fleet-card">
        <div className="fleet-card-header">
          <div>
            <h3 className="fleet-card-title">Repositioning Recommendations</h3>
            <span className="fleet-card-subtitle">
              {metrics?.recommended_movements.length ?? 0} proposed vehicle movement(s) • Generating recommendations does not execute vehicle transfers
            </span>
          </div>
          {metrics?.demand_source && (
            <span
              className={`fleet-pill ${
                metrics.demand_source.type === 'forecast' ? 'fleet-pill-info' : 'fleet-pill-warning'
              }`}
            >
              {describeDemandSource(metrics.demand_source)}
            </span>
          )}
        </div>

        {/* Demand Source Disclosure Banner */}
        {metrics?.demand_source && (
          <div className="fleet-disclosure-box">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', fontWeight: 600, color: 'var(--navy)' }}>
              <span>Demand Source:</span>
              <span className="fleet-pill fleet-pill-info">
                {metrics.demand_source.type === 'forecast'
                  ? `Forecast Horizon +${metrics.demand_source.horizon_minutes! / 60}H (${metrics.demand_source.horizon_minutes} min)`
                  : 'Historical Average Demand'}
              </span>
              {metrics.demand_source.type === 'forecast' && (
                <span style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'monospace' }}>
                  (Origin: {metrics.demand_source.forecast_origin} → Target: {metrics.demand_source.target_timestamp})
                </span>
              )}
            </div>
            <div style={{ marginTop: 6, color: 'var(--muted)', fontSize: 11 }}>
              <span>Data basis: {describeDataBasis(metrics.demand_source)}</span>
              {metrics.demand_source.target_beyond_dataset_end &&
                ' Target timestamp is past the end of the historical dataset (model extrapolates).'}
            </div>
          </div>
        )}

        {/* Recommendations Table */}
        {metrics && metrics.recommended_movements.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="fleet-recs-table">
              <thead>
                <tr>
                  <th style={{ width: '16%' }}>Donor Zone</th>
                  <th style={{ width: '4%', textAlign: 'center' }}></th>
                  <th style={{ width: '20%' }}>Shortage Zone</th>
                  <th style={{ width: '15%' }}>Recommended Vehicles</th>
                  <th style={{ width: '15%' }}>Estimated Shortage</th>
                  <th style={{ width: '30%' }}>Operational Dispatch Reason</th>
                </tr>
              </thead>
              <tbody>
                {metrics.recommended_movements.map((movement, idx) => {
                  const toAnom = zoneAnomalyMap.get(movement.to_zone)
                  const fromAnom = zoneAnomalyMap.get(movement.from_zone)
                  return (
                    <tr key={`${movement.from_zone}-${movement.to_zone}-${idx}`}>
                      <td>
                        <strong>Zone {movement.from_zone}</strong>
                        {(() => {
                          const donorMetric = metrics.zones.find((z) => z.zone_id === movement.from_zone)
                          return donorMetric ? (
                            <small style={{ display: 'block', color: 'var(--muted)', fontSize: 11 }}>
                              Surplus: {formatNumber(donorMetric.surplus_after_own_demand)} cabs
                            </small>
                          ) : null
                        })()}
                        {fromAnom && (
                          <span
                            className="fleet-pill fleet-pill-warning"
                            style={{ marginTop: 3, fontSize: 10 }}
                            title={fromAnom.explanation}
                          >
                            Origin: {fromAnom.anomaly_type.replace('_', ' ')}
                          </span>
                        )}
                      </td>
                      <td style={{ textAlign: 'center', color: 'var(--muted)' }}>
                        <ArrowRight size={14} />
                      </td>
                      <td>
                        <strong>Zone {movement.to_zone}</strong>
                        {toAnom && (
                          <span
                            className="fleet-pill fleet-pill-danger"
                            style={{ marginLeft: 6, fontSize: 10 }}
                            title={toAnom.explanation}
                          >
                            ⚠️ {toAnom.anomaly_type.replace('_', ' ')} ({toAnom.severity})
                          </span>
                        )}
                      </td>
                      <td>
                        <span className="fleet-cab-badge">
                          <Truck size={13} /> {formatNumber(movement.cab_count)} cab(s)
                        </span>
                      </td>
                      <td>
                        {(() => {
                          const recipientMetric = metrics.zones.find((z) => z.zone_id === movement.to_zone)
                          return (
                            <span style={{ color: '#b91c1c', fontWeight: 700, fontSize: 12 }}>
                              {recipientMetric ? `${formatNumber(recipientMetric.shortage_estimate)} rides` : '—'}
                            </span>
                          )
                        })()}
                      </td>
                      <td style={{ color: 'var(--ink)' }}>
                        {movement.reason || 'Surplus rebalance toward predicted demand'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '30px 10px' }}>
            <Truck size={28} style={{ color: 'var(--muted)', marginBottom: 8 }} />
            <p style={{ margin: 0, fontWeight: 600 }}>No repositioning movements currently recommended.</p>
            <small style={{ color: 'var(--muted)' }}>
              Current available vehicles across all zones sufficiently cover predicted demand without critical shortages.
            </small>
          </div>
        )}
      </div>

      {/* ======================================================== */}
      {/* SECTION D: ZONE-LEVEL AVAILABILITY (COLLAPSIBLE)         */}
      {/* ======================================================== */}
      <div className="collapsible-section">
        <button
          type="button"
          className="collapsible-button"
          onClick={() => setZoneSectionExpanded((prev) => !prev)}
          aria-expanded={zoneSectionExpanded}
          aria-controls="zone-availability-section"
        >
          <span className="collapsible-label">Zone-Level Fleet Availability</span>
          <span style={{ fontSize: 12, color: 'var(--muted)', fontWeight: 400 }}>
            ({zoneEntries.length} Zones Monitored)
          </span>
          <svg className="collapsible-chevron" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" style={{ marginLeft: 'auto' }}>
            <path d={zoneSectionExpanded ? 'M10 6L16 12H8L10 6Z' : 'M6 10L12 16H4L6 10Z'} />
          </svg>
        </button>

        {zoneSectionExpanded && (
          <div id="zone-availability-section" className="collapsible-content">
            <div className="fleet-card" style={{ marginBottom: 0 }}>
              <div className="zone-search-filters">
                <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
                  <Search size={14} style={{ position: 'absolute', left: 12, top: 12, color: 'var(--muted)' }} />
                  <input
                    type="text"
                    className="zone-search-input"
                    style={{ paddingLeft: 34 }}
                    placeholder="Filter by zone ID (e.g. 61, 79)..."
                    value={zoneSearch}
                    onChange={(e) => setZoneSearch(e.target.value)}
                    aria-label="Filter zones by ID"
                  />
                </div>
                <div className="zone-filter-buttons">
                  <button
                    type="button"
                    className={zoneFilter === 'all' ? 'zone-filter-btn active' : 'zone-filter-btn'}
                    onClick={() => setZoneFilter('all')}
                  >
                    All Zones ({zoneEntries.length})
                  </button>
                  <button
                    type="button"
                    className={zoneFilter === 'shortage' ? 'zone-filter-btn active' : 'zone-filter-btn'}
                    onClick={() => setZoneFilter('shortage')}
                  >
                    Shortage Zones ({shortageZones})
                  </button>
                  <button
                    type="button"
                    className={zoneFilter === 'surplus' ? 'zone-filter-btn active' : 'zone-filter-btn'}
                    onClick={() => setZoneFilter('surplus')}
                  >
                    Surplus Zones ({surplusZones})
                  </button>
                  <button
                    type="button"
                    className={zoneFilter === 'anomaly' ? 'zone-filter-btn active' : 'zone-filter-btn'}
                    onClick={() => setZoneFilter('anomaly')}
                  >
                    Anomaly Zones ({zoneAnomalyMap.size})
                  </button>
                </div>
              </div>

              <div className="zone-table-wrapper">
                <table className="zone-availability-table">
                  <thead>
                    <tr>
                      <th className="sticky-header">Zone</th>
                      <th className="sticky-header">Available / Idle</th>
                      <th className="sticky-header">Busy</th>
                      <th className="sticky-header">Repositioning</th>
                      <th className="sticky-header">Total Cabs</th>
                    </tr>
                  </thead>
                  <tbody>{renderZoneRows(filteredZones)}</tbody>
                </table>
              </div>

              <p className="fleet-note" style={{ marginTop: 12, marginBottom: 0, fontSize: 11, color: 'var(--muted)' }}>
                Fleet availability is simulated for prototype operations and does not represent actual live NYC cab availability.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ======================================================== */}
      {/* SECTION E: DEMAND AND SUPPLY ANALYSIS (COLLAPSIBLE)      */}
      {/* ======================================================== */}
      <div className="collapsible-section">
        <button
          type="button"
          className="collapsible-button"
          onClick={() => setDemandSectionExpanded((prev) => !prev)}
          aria-expanded={demandSectionExpanded}
          aria-controls="demand-analysis-section"
        >
          <span className="collapsible-label">Demand and Supply Analysis</span>
          <span style={{ fontSize: 12, color: 'var(--muted)', fontWeight: 400 }}>
            (Zone-Level Supply Requirements & Shortages)
          </span>
          <svg className="collapsible-chevron" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" style={{ marginLeft: 'auto' }}>
            <path d={demandSectionExpanded ? 'M10 6L16 12H8L10 6Z' : 'M6 10L12 16H4L6 10Z'} />
          </svg>
        </button>

        {demandSectionExpanded && (
          <div id="demand-analysis-section" className="collapsible-content">
            <div className="fleet-card" style={{ marginBottom: 0 }}>
              {/* Summary Structured Metric Cards */}
              <div className="fleet-analysis-grid">
                <article className="fleet-analysis-card">
                  <span className="fleet-analysis-label">Demand Source</span>
                  <strong>
                    {metrics?.demand_source?.type === 'forecast'
                      ? `Forecast +${metrics.demand_source.horizon_minutes! / 60}H`
                      : 'Historical Baseline'}
                  </strong>
                  <small>
                    {metrics?.demand_source?.type === 'forecast'
                      ? `${metrics.demand_source.horizon_minutes}m multi-horizon forecast`
                      : 'Hourly average baseline'}
                  </small>
                </article>

                <article className="fleet-analysis-card">
                  <span className="fleet-analysis-label">Monitored Zones</span>
                  <strong>{metrics?.zones.length ?? 0}</strong>
                  <small>Total active dispatch zones</small>
                </article>

                <article className="fleet-analysis-card">
                  <span className="fleet-analysis-label">Shortage Zones</span>
                  <strong style={{ color: shortageZones > 0 ? '#b91c1c' : 'var(--navy)' }}>
                    {shortageZones}
                  </strong>
                  <small>Zones requiring additional cabs</small>
                </article>

                <article className="fleet-analysis-card">
                  <span className="fleet-analysis-label">Surplus Zones</span>
                  <strong style={{ color: '#059669' }}>
                    {surplusZones}
                  </strong>
                  <small>Potential donor zones</small>
                </article>

                <article className="fleet-analysis-card">
                  <span className="fleet-analysis-label">Demand Anomalies</span>
                  <strong style={{ color: zoneAnomalyMap.size > 0 ? '#d97706' : 'var(--navy)' }}>
                    {zoneAnomalyMap.size}
                  </strong>
                  <small>Flagged outlier zones</small>
                </article>

                <article className="fleet-analysis-card">
                  <span className="fleet-analysis-label">Total Unmet Demand</span>
                  <strong style={{ color: (metrics?.unmet_demand_estimate.total_unmet_rides ?? 0) > 0 ? '#b91c1c' : 'var(--navy)' }}>
                    {formatNumber(metrics?.unmet_demand_estimate.total_unmet_rides)}
                  </strong>
                  <small>Estimated unserved passenger rides</small>
                </article>
              </div>

              {/* Analysis Table */}
              {metrics && metrics.zones.length > 0 ? (
                <div className="zone-table-wrapper">
                  <table className="recommendations-table">
                    <thead>
                      <tr>
                        <th className="sticky-header" style={{ textAlign: 'left' }}>Zone</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Predicted Demand</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Available Cabs</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Required Cabs</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Estimated Shortage</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Estimated Surplus</th>
                        <th className="sticky-header" style={{ textAlign: 'center' }}>Anomaly Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.zones.map((zone) => {
                        const anom = zoneAnomalyMap.get(zone.zone_id)
                        return (
                          <tr key={zone.zone_id} className="recommendation-row">
                            <td style={{ textAlign: 'left' }}>
                              <strong>Zone {zone.zone_id}</strong>
                            </td>
                            <td style={{ textAlign: 'right' }}>{formatNumber(zone.predicted_rides)} rides</td>
                            <td style={{ textAlign: 'right' }}>{formatNumber(zone.idle_cabs)}</td>
                            <td style={{ textAlign: 'right' }}>{formatNumber(zone.estimated_cabs_required)}</td>
                            <td style={{ textAlign: 'right' }}>
                              <span style={{ color: zone.shortage_estimate > 0 ? '#b91c1c' : 'inherit', fontWeight: zone.shortage_estimate > 0 ? 700 : 400 }}>
                                {formatNumber(zone.shortage_estimate)}
                              </span>
                            </td>
                            <td style={{ textAlign: 'right' }}>
                              <span style={{ color: zone.surplus_after_own_demand > 0 ? '#059669' : 'inherit', fontWeight: zone.surplus_after_own_demand > 0 ? 700 : 400 }}>
                                {formatNumber(zone.surplus_after_own_demand)}
                              </span>
                            </td>
                            <td>
                              {anom ? (
                                <span
                                  style={{
                                    background:
                                      anom.severity === 'critical'
                                        ? '#fee2e2'
                                        : anom.severity === 'high'
                                          ? '#ffedd5'
                                          : anom.severity === 'medium'
                                            ? '#fef9c3'
                                            : '#f3f4f6',
                                    color:
                                      anom.severity === 'critical'
                                        ? '#991b1b'
                                        : anom.severity === 'high'
                                          ? '#c2410c'
                                          : anom.severity === 'medium'
                                            ? '#854d0e'
                                            : '#4b5563',
                                    padding: '2px 7px',
                                    borderRadius: 4,
                                    fontSize: 10,
                                    fontWeight: 700,
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: 3,
                                  }}
                                  title={anom.explanation}
                                >
                                  <AlertTriangle size={11} aria-hidden="true" />
                                  {anom.anomaly_type.replace('_', ' ')} ({anom.severity})
                                </span>
                              ) : (
                                <span style={{ color: 'var(--muted)', fontSize: 11 }}>Normal</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state">
                  <p>Generate recommendations to populate zone-level demand and supply metrics.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <p className="fleet-note" style={{ marginTop: 24, fontSize: 11, color: 'var(--muted)' }}>
        Fleet availability is simulated for prototype purposes and does not represent actual live NYC cab availability.
        Demand predictions are anchored to historical dataset 2021-06-30T23:00:00.
      </p>
    </section>
  )
}

export default FleetManagement
