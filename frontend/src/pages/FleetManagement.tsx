import { useCallback, useEffect, useState } from 'react'
import { RefreshCw, Truck } from 'lucide-react'
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
  // Backend versions may call this field "availability" or "available".
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

function FleetManagement({ backendUrl = 'http://localhost:8000' }: FleetManagementProps) {
  const [initialized, setInitialized] = useState(false)
  const [fleetState, setFleetState] = useState<FleetState | null>(null)
  const [metrics, setMetrics] = useState<FleetMetrics | null>(null)
  const [loading, setLoading] = useState<
    'idle' | 'initializing' | 'refreshing' | 'advancing' | 'resetting' | 'exporting' | 'recommending'
  >('idle')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  // Collapsible section state
  const [zoneSectionExpanded, setZoneSectionExpanded] = useState(false)
  const [demandSectionExpanded, setDemandSectionExpanded] = useState(false)
  const [selectedFleetSize, setSelectedFleetSize] = useState<number>(20000)

  useEffect(() => {
    if (fleetState?.fleet_size && fleetState.fleet_size > 0) {
      setSelectedFleetSize(fleetState.fleet_size)
    }
  }, [fleetState?.fleet_size])

  const fetchRecommendations = useCallback(async (simulationId: string) => {
    if (!simulationId) return
    const resp = await fetch(`${backendUrl}/fleet/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ simulation_id: simulationId }),
    })
    if (!resp.ok) {
      const detail = await resp.text()
      throw new Error(detail || `Recommendations request failed (HTTP ${resp.status})`)
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
    })
  }, [backendUrl])

  const fetchFleetCurrent = useCallback(async (showLoading = true) => {
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
      const data = normalizeFleetState(await resp.json() as ApiFleetState)
      setFleetState(data)
      setInitialized(Boolean(data.simulation_id))
      if (data.simulation_id) {
        try {
          await fetchRecommendations(data.simulation_id)
        } catch {
          // Keep the current fleet state visible even if recommendations are unavailable.
          setMetrics(null)
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unable to load fleet state')
    } finally {
      if (showLoading) setLoading('idle')
    }
  }, [backendUrl, fetchRecommendations])

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
      // If current-state retrieval did not provide a valid state, keep the init response visible.
      setFleetState((current) => current ?? normalizeFleetState(data))
      if (simulationId) {
        try {
          await fetchRecommendations(simulationId)
        } catch {
          setMetrics(null)
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
      await fetchRecommendations(fleetState.simulation_id)
      setNotice('Recommendations refreshed. No cab movements were executed.')
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
      // Re-fetch the authoritative state rather than guessing which fields changed.
      await fetchFleetCurrent(false)
      setNotice('Simulation advanced and the displayed state was refreshed.')
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
      setNotice('Simulation reset.')
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
      setNotice('Fleet snapshot export downloaded.')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to export fleet snapshots')
    } finally {
      setLoading('idle')
    }
  }

  const formatNumber = (value: number | undefined) =>
    Number(value ?? 0).toLocaleString()

  // Filter and search state for zone tables
  const [zoneSearch, setZoneSearch] = useState('')
  const [zoneFilter, setZoneFilter] = useState<'all' | 'shortage' | 'surplus'>('all')

  const zoneEntries = (Object.entries(fleetState?.availability ?? {}) as [string, ZoneAvailability][]).map(([zoneId, value]) => ({
    zoneId: Number(zoneId),
    ...value,
  }))

  // Filter zones based on search and filter criteria
  const filteredZones = zoneEntries.filter((zone) => {
    const zoneIdStr = zone.zoneId.toString()
    const searchMatch = zoneSearch === '' || zoneIdStr.includes(zoneSearch)
    if (!searchMatch) return false

    // Apply zone filter (shortage/surplus)
    if (zoneFilter === 'shortage') {
      // Check if this zone has shortage based on metrics data
      if (!metrics?.zones) return false
      const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
      if (!metric || metric.shortage_estimate <= 0) return false
    } else if (zoneFilter === 'surplus') {
      if (!metrics?.zones) return false
      const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
      if (!metric || metric.surplus_after_own_demand <= 0) return false
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
  const totalPredicted = (Object.values(fleetState?.predicted_rides ?? {}) as number[]).reduce(
    (sum, rides) => sum + Number(rides ?? 0),
    0,
  )
  const shortageZones = metrics?.zones.filter((zone) => zone.shortage_estimate > 0).length ?? 0
  const surplusZones =
    metrics?.zones.filter((zone) => zone.surplus_after_own_demand > 0).length ?? 0
  const busyTone = totalBusy > totalIdle * 1.5 ? 'error' : totalIdle > totalBusy * 1.5 ? 'loading' : 'empty'
  const isBusy = loading !== 'idle'

  if (!initialized) {
    return (
      <section className="fleet-management-section">
        <SectionHeading eyebrow="FLEET SIMULATION" title="Fleet Management">
          <p>Initialize the simulated fleet to begin.</p>
          {error && <p role="alert" className="fleet-error">{error}</p>}
          <div className="fleet-controls" style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', fontWeight: 500 }}>
              Fleet Size:
              <select
                value={selectedFleetSize}
                onChange={(e) => setSelectedFleetSize(Number(e.target.value))}
                disabled={isBusy}
                style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #ccc', background: 'white' }}
              >
                <option value={13000}>13,000 cabs (Baseline / Deficit)</option>
                <option value={16000}>16,000 cabs (Balanced Supply)</option>
                <option value={20000}>20,000 cabs (Recommended — 16 recs)</option>
                <option value={25000}>25,000 cabs (Abundant Supply)</option>
              </select>
            </label>
            <button className="primary-button" onClick={() => void handleInit()} disabled={isBusy}>
              {loading === 'initializing' ? 'Initializing' : 'Initialize Fleet Simulation'}
            </button>
          </div>
          <p className="fleet-note">
            Fleet availability is simulated and does not represent live NYC cab availability.
          </p>
        </SectionHeading>
      </section>
    )
  }

  // Render zone rows with search/filter highlighting
  const renderZoneRows = (zones: typeof zoneEntries) => {
    if (zones.length === 0) {
      return <tr><td colSpan={5} className="empty-state">No zone availability data was returned by the backend.</td></tr>
    }
    return zones.map((zone) => {
      const zoneIdStr = zone.zoneId.toString()
      const searchMatch = zoneSearch === '' || zoneIdStr.includes(zoneSearch)
      const showsShortage = zoneFilter !== 'all' && zoneFilter === 'shortage'
      const showsSurplus = zoneFilter !== 'all' && zoneFilter === 'surplus'

      // Determine if zone matches filter criteria
      let matchesFilter = true
      if (showsShortage && metrics?.zones) {
        const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
        matchesFilter = Boolean(metric && metric.shortage_estimate > 0)
      }
      if (showsSurplus && metrics?.zones) {
        const metric = metrics.zones.find((m) => m.zone_id === zone.zoneId)
        matchesFilter = Boolean(metric && metric.surplus_after_own_demand > 0)
      }

      if (!searchMatch || !matchesFilter) {
        return null
      }

      return (
        <tr key={zone.zoneId} className="zone-row">
          <td><strong>Zone {zone.zoneId}</strong></td>
          <td>{formatNumber(zone.idle_cabs)}</td>
          <td>{formatNumber(zone.busy_cabs)}</td>
          <td>{formatNumber(zone.repositioning_cabs)}</td>
          <td>{formatNumber(zone.total_cabs)}</td>
        </tr>
      )
    })
  }

  return (
    <section className="fleet-management-section">
      <SectionHeading eyebrow="FLEET SIMULATION" title="Fleet Management">
        <div className="fleet-toolbar">
          <span>Simulation: {fleetState?.simulation_id || 'Unknown'}</span>
          <span>Time: {fleetState?.timestamp ? new Date(fleetState.timestamp).toLocaleString() : 'Unavailable'}</span>
          <button
            type="button"
            className="secondary-button"
            onClick={() => void fetchFleetCurrent()}
            disabled={isBusy}
          >
            <RefreshCw size={16} aria-hidden="true" /> Refresh State
          </button>
        </div>

        {error && <p role="alert" className="fleet-error">{error}</p>}
        {notice && <p role="status" className="fleet-notice">{notice}</p>}

        <StateCard title="Fleet Summary" tone={busyTone}>
          <div className="fleet-summary-grid">
            <div className="summary-card"><div>Total Fleet</div><div>{formatNumber(fleetState?.fleet_size)}</div></div>
            <div className="summary-card"><div>Idle / Available</div><div>{formatNumber(totalIdle)}</div></div>
            <div className="summary-card"><div>Busy</div><div>{formatNumber(totalBusy)}</div></div>
            <div className="summary-card"><div>Repositioning</div><div>{formatNumber(totalRepositioning)}</div></div>
            <div className="summary-card"><div>Zones</div><div>{zoneEntries.length}</div></div>
            <div className="summary-card"><div>Predicted Rides</div><div>{formatNumber(totalPredicted)}</div></div>
          </div>
          <small className="fleet-conservation">
            Fleet accounting: {formatNumber(totalCabs)} of {formatNumber(fleetState?.fleet_size)} cabs represented by zone totals.
            {totalCabs === fleetState?.fleet_size ? ' ✓ Conserved' : ' — check backend state fields'}
          </small>
        </StateCard>

        <div className="controls-panel" style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 500 }}>
            Fleet Size:
            <select
              value={selectedFleetSize}
              onChange={(e) => {
                const newSize = Number(e.target.value)
                setSelectedFleetSize(newSize)
                void handleInit(newSize)
              }}
              disabled={isBusy}
              style={{ padding: '6px 10px', borderRadius: '4px', border: '1px solid #ccc', background: 'white' }}
            >
              <option value={13000}>13,000 (Baseline / Deficit)</option>
              <option value={16000}>16,000 (Balanced Supply)</option>
              <option value={20000}>20,000 (Recommended — 16 recs)</option>
              <option value={25000}>25,000 (Abundant Supply)</option>
            </select>
          </label>
          <button className="primary-button" onClick={handleAdvance} disabled={isBusy}>
            {loading === 'advancing' ? 'Advancing' : 'Advance Simulation'}
          </button>
          <button className="secondary-button" onClick={handleRecommendations} disabled={isBusy}>
            {loading === 'recommending' ? 'Generating' : 'Generate Recommendations'}
          </button>
          <button className="secondary-button" onClick={handleReset} disabled={isBusy}>
            {loading === 'resetting' ? 'Resetting' : 'Reset Simulation'}
          </button>
          <button className="secondary-button" onClick={handleExport} disabled={isBusy}>
            {loading === 'exporting' ? 'Exporting' : 'Export Snapshots'}
          </button>
        </div>

        {/* Repositioning Recommendations moved directly below controls */}
        <StateCard title="Repositioning Recommendations" tone="empty">
          <p>{metrics?.recommended_movements.length} proposed movement(s). Generating recommendations does not execute them.</p>
          <ul className="recommendations-list">
            {metrics?.recommended_movements.map((movement, index) => (
              <li key={`${movement.from_zone}-${movement.to_zone}-${index}`}>
                <Truck size={16} aria-hidden="true" /> Zone {movement.from_zone} → Zone {movement.to_zone}: {formatNumber(movement.cab_count)} cab(s)
                {movement.reason ? ` — ${movement.reason}` : ''}
              </li>
            ))}
          </ul>
          {(!metrics || metrics.recommended_movements.length === 0) && <p>No recommendations are currently available. Generate recommendations after loading the fleet state.</p>}
        </StateCard>

        {/* Collapsible Zone-level Availability section */}
        <div className="collapsible-section">
          <button
            type="button"
            className="collapsible-button"
            onClick={() => setZoneSectionExpanded((prev) => !prev)}
            aria-expanded={zoneSectionExpanded}
            aria-controls="zone-availability-section"
          >
            <span className="collapsible-label">Zone-level Availability</span>
            <svg
              className="collapsible-chevron"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d={zoneSectionExpanded ? 'M10 6L16 12H8L10 6Z' : 'M6 10L12 16H4L6 10Z'} />
            </svg>
          </button>

          {zoneSectionExpanded && (
            <div id="zone-availability-section" className="collapsible-content">
              <StateCard title="Zone-level Availability" tone="empty">
                <div className="zone-search-filters">
                  <input
                    type="text"
                    className="zone-search-input"
                    placeholder="Search zone ID..."
                    value={zoneSearch}
                    onChange={(e) => setZoneSearch(e.target.value)}
                    aria-label="Search zones by ID"
                  />
                  <div className="zone-filter-buttons">
                    <button
                      type="button"
                      className={zoneFilter === 'all' ? 'zone-filter-btn active' : 'zone-filter-btn'}
                      onClick={() => setZoneFilter('all')}
                    >
                      All Zones
                    </button>
                    <button
                      type="button"
                      className={zoneFilter === 'shortage' ? 'zone-filter-btn active' : 'zone-filter-btn'}
                      onClick={() => setZoneFilter('shortage')}
                    >
                      Shortage Zones
                    </button>
                    <button
                      type="button"
                      className={zoneFilter === 'surplus' ? 'zone-filter-btn active' : 'zone-filter-btn'}
                      onClick={() => setZoneFilter('surplus')}
                    >
                      Surplus Zones
                    </button>
                  </div>
                </div>

                <div className="zone-table-wrapper">
                  <table className="zone-availability-table">
                    <thead>
                      <tr>
                        <th className="sticky-header">Zone</th>
                        <th className="sticky-header">Available</th>
                        <th className="sticky-header">Busy</th>
                        <th className="sticky-header">Repositioning</th>
                        <th className="sticky-header">Total</th>
                      </tr>
                    </thead>
                    <tbody>
                      {renderZoneRows(filteredZones)}
                    </tbody>
                  </table>
                </div>

                <p className="fleet-note">
                  Fleet availability is simulated and does not represent actual live NYC cab availability.
                </p>
              </StateCard>
            </div>
          )}
        </div>

        {/* Collapsible Demand & Supply Analysis section */}
        <div className="collapsible-section">
          <button
            type="button"
            className="collapsible-button"
            onClick={() => setDemandSectionExpanded((prev) => !prev)}
            aria-expanded={demandSectionExpanded}
            aria-controls="demand-analysis-section"
          >
            <span className="collapsible-label">Demand & Supply Analysis</span>
            <svg
              className="collapsible-chevron"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d={demandSectionExpanded ? 'M10 6L16 12H8L10 6Z' : 'M6 10L12 16H4L6 10Z'} />
            </svg>
          </button>

          {demandSectionExpanded && (
            <div id="demand-analysis-section" className="collapsible-content">
              <StateCard title="Demand & Supply Analysis" tone="empty">
                <div className="recommendations-summary">
                  <p>Zones with demand metrics: {metrics?.zones.length ?? 0}</p>
                  <p>Zones with shortage: {shortageZones}</p>
                  <p>Zones with surplus: {surplusZones}</p>
                  <p>Estimated unmet rides: {formatNumber(metrics?.unmet_demand_estimate.total_unmet_rides)}</p>
                </div>
                {metrics && metrics.zones.length > 0 ? (
                  <div className="zone-table-wrapper">
                    <table className="recommendations-table">
                      <thead>
                        <tr>
                          <th className="sticky-header">Zone</th>
                          <th className="sticky-header">Predicted Rides</th>
                          <th className="sticky-header">Idle Cabs</th>
                          <th className="sticky-header">Required Cabs</th>
                          <th className="sticky-header">Shortage</th>
                          <th className="sticky-header">Surplus</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.zones.map((zone) => (
                          <tr key={zone.zone_id} className="recommendation-row">
                            <td><strong>Zone {zone.zone_id}</strong></td>
                            <td>{formatNumber(zone.predicted_rides)}</td>
                            <td>{formatNumber(zone.idle_cabs)}</td>
                            <td>{formatNumber(zone.estimated_cabs_required)}</td>
                            <td>{formatNumber(zone.shortage_estimate)}</td>
                            <td>{formatNumber(zone.surplus_after_own_demand)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p>Generate recommendations to load demand and supply metrics.</p>
                )}
              </StateCard>
            </div>
          )}
        </div>

        <p className="fleet-note">
          Fleet availability is simulated for prototype purposes and does not represent actual live NYC cab availability.
          Demand values may use a heuristic fallback if actual model forecasts are not connected.
        </p>
      </SectionHeading>
    </section>
  )
}

export default FleetManagement