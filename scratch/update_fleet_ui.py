import re

with open('frontend/src/pages/FleetManagement.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\r\n', '\n')

# 1. Update Section C table headers and cells
old_thead = '''              <thead>
                <tr>
                  <th style={{ width: '18%' }}>Donor Zone</th>
                  <th style={{ width: '5%', textAlign: 'center' }}></th>
                  <th style={{ width: '22%' }}>Shortage Zone</th>
                  <th style={{ width: '16%' }}>Transfer Quantity</th>
                  <th style={{ width: '39%' }}>Operational Dispatch Reason</th>
                </tr>
              </thead>'''

new_thead = '''              <thead>
                <tr>
                  <th style={{ width: '16%' }}>Donor Zone</th>
                  <th style={{ width: '4%', textAlign: 'center' }}></th>
                  <th style={{ width: '20%' }}>Shortage Zone</th>
                  <th style={{ width: '15%' }}>Recommended Vehicles</th>
                  <th style={{ width: '15%' }}>Estimated Shortage</th>
                  <th style={{ width: '30%' }}>Operational Dispatch Reason</th>
                </tr>
              </thead>'''

assert old_thead in content, 'old_thead not found'
content = content.replace(old_thead, new_thead)

old_row = '''                    <tr key={`${movement.from_zone}-${movement.to_zone}-${idx}`}>
                      <td>
                        <strong>Zone {movement.from_zone}</strong>
                        {fromAnom && (
                          <span
                            className="fleet-pill fleet-pill-warning"
                            style={{ marginLeft: 6, fontSize: 10 }}
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
                      <td style={{ color: 'var(--ink)' }}>
                        {movement.reason || 'Surplus rebalance toward predicted demand'}
                      </td>
                    </tr>'''

new_row = '''                    <tr key={`${movement.from_zone}-${movement.to_zone}-${idx}`}>
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
                    </tr>'''

assert old_row in content, 'old_row not found'
content = content.replace(old_row, new_row)

# 2. Update Section E Summary Strip to Analysis Grid
old_strip = '''              {/* Summary Metrics Strip */}
              <div className="fleet-stats-strip">
                <div>
                  Active Demand Source:{' '}
                  <strong>
                    {metrics?.demand_source?.type === 'forecast'
                      ? `Forecast (+${metrics.demand_source.horizon_minutes! / 60}H ahead)`
                      : 'Historical Hourly Baseline'}
                  </strong>
                </div>
                <div>•</div>
                <div>Monitored Zones: <strong>{metrics?.zones.length ?? 0}</strong></div>
                <div>•</div>
                <div>Shortage Zones: <strong style={{ color: shortageZones > 0 ? '#b91c1c' : 'inherit' }}>{shortageZones}</strong></div>
                <div>•</div>
                <div>Surplus Zones: <strong style={{ color: '#059669' }}>{surplusZones}</strong></div>
                <div>•</div>
                <div>Demand Anomalies: <strong style={{ color: zoneAnomalyMap.size > 0 ? '#b91c1c' : 'inherit' }}>{zoneAnomalyMap.size}</strong></div>
                <div>•</div>
                <div>Total Unmet Demand: <strong style={{ color: '#b91c1c' }}>{formatNumber(metrics?.unmet_demand_estimate.total_unmet_rides)} rides</strong></div>
              </div>'''

new_grid = '''              {/* Summary Structured Metric Cards */}
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
              </div>'''

assert old_strip in content, 'old_strip not found'
content = content.replace(old_strip, new_grid)

# 3. Clean table header alignments in Section E
old_analysis_thead = '''                    <thead>
                      <tr>
                        <th className="sticky-header">Zone</th>
                        <th className="sticky-header">Predicted Demand</th>
                        <th className="sticky-header">Available Cabs</th>
                        <th className="sticky-header">Required Cabs</th>
                        <th className="sticky-header">Estimated Shortage</th>
                        <th className="sticky-header">Estimated Surplus</th>
                        <th className="sticky-header">Anomaly Status</th>
                      </tr>
                    </thead>'''

new_analysis_thead = '''                    <thead>
                      <tr>
                        <th className="sticky-header" style={{ textAlign: 'left' }}>Zone</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Predicted Demand</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Available Cabs</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Required Cabs</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Estimated Shortage</th>
                        <th className="sticky-header" style={{ textAlign: 'right' }}>Estimated Surplus</th>
                        <th className="sticky-header" style={{ textAlign: 'center' }}>Anomaly Status</th>
                      </tr>
                    </thead>'''

assert old_analysis_thead in content, 'old_analysis_thead not found'
content = content.replace(old_analysis_thead, new_analysis_thead)

# 4. Clean row cell alignments in Section E
old_analysis_row = '''                          <tr key={zone.zone_id} className="recommendation-row">
                            <td>
                              <strong>Zone {zone.zone_id}</strong>
                            </td>
                            <td>{formatNumber(zone.predicted_rides)} rides</td>
                            <td>{formatNumber(zone.idle_cabs)}</td>
                            <td>{formatNumber(zone.estimated_cabs_required)}</td>
                            <td>
                              <span style={{ color: zone.shortage_estimate > 0 ? '#b91c1c' : 'inherit', fontWeight: zone.shortage_estimate > 0 ? 700 : 400 }}>
                                {formatNumber(zone.shortage_estimate)}
                              </span>
                            </td>
                            <td>
                              <span style={{ color: zone.surplus_after_own_demand > 0 ? '#059669' : 'inherit', fontWeight: zone.surplus_after_own_demand > 0 ? 700 : 400 }}>
                                {formatNumber(zone.surplus_after_own_demand)}
                              </span>
                            </td>'''

new_analysis_row = '''                          <tr key={zone.zone_id} className="recommendation-row">
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
                            </td>'''

assert old_analysis_row in content, 'old_analysis_row not found'
content = content.replace(old_analysis_row, new_analysis_row)

with open('frontend/src/pages/FleetManagement.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated FleetManagement.tsx successfully')
