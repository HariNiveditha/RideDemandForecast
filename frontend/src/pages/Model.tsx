import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

type MetricsResponse = {
  rmse: number
  mae: number
  r2: number
  feature_importances: Array<{ feature: string; importance: number }>
}

type PredictionSampleResponse = {
  points: Array<{ timestamp: string; actual: number; predicted: number }>
}

function Model() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [sample, setSample] = useState<PredictionSampleResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [predictionLoading, setPredictionLoading] = useState(true)
  const [error, setError] = useState('')
  const [predictionError, setPredictionError] = useState('')

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        setLoading(true)
        setError('')

        const response = await fetch('/api/metrics')
        if (!response.ok) {
          throw new Error('Metrics endpoint unavailable')
        }

        const data = (await response.json()) as MetricsResponse
        setMetrics(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load model metrics.')
        setMetrics(null)
      } finally {
        setLoading(false)
      }
    }

    void loadMetrics()
  }, [])

  useEffect(() => {
    const loadPredictionSample = async () => {
      try {
        setPredictionLoading(true)
        setPredictionError('')

        const response = await fetch('/api/predictions-sample')
        if (!response.ok) {
          throw new Error('Prediction sample endpoint unavailable')
        }

        const data = (await response.json()) as PredictionSampleResponse
        setSample(data)
      } catch (err) {
        setPredictionError(err instanceof Error ? err.message : 'Unable to load prediction sample.')
        setSample(null)
      } finally {
        setPredictionLoading(false)
      }
    }

    void loadPredictionSample()
  }, [])

  if (loading) {
    return <div className="view-notice"><strong>Loading model metrics…</strong></div>
  }

  if (error || !metrics) {
    return <div className="view-notice"><div><strong>Model metrics unavailable</strong><span>{error || 'metrics.json was not found or could not be loaded.'}</span></div></div>
  }

  const chartData = [...metrics.feature_importances]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10)
    .map((entry) => ({ ...entry, importance: Number(entry.importance) }))

  return (
    <section className="forecast-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">MODEL PERFORMANCE</span>
          <h2>Model metrics</h2>
        </div>
      </div>

      <div className="kpi-grid">
        <article className="kpi-card">
          <div className="kpi-top"><span>RMSE</span></div>
          <strong>{Number(metrics.rmse).toFixed(4)}</strong>
          <small>Root mean squared error</small>
        </article>

        <article className="kpi-card">
          <div className="kpi-top"><span>MAE</span></div>
          <strong>{Number(metrics.mae).toFixed(4)}</strong>
          <small>Mean absolute error</small>
        </article>

        <article className="kpi-card">
          <div className="kpi-top"><span>R²</span></div>
          <strong>{Number(metrics.r2).toFixed(4)}</strong>
          <small>Coefficient of determination</small>
        </article>
      </div>

      <div className="panel" style={{ marginTop: '20px', padding: '20px' }}>
        <div className="panel-heading">
          <div>
            <span className="eyebrow">PREDICTION SAMPLE</span>
            <h3>Actual vs predicted demand</h3>
          </div>
        </div>

        {predictionLoading ? (
          <div className="view-notice"><strong>Loading prediction sample…</strong></div>
        ) : predictionError || !sample ? (
          <div className="view-notice"><div><strong>Prediction sample unavailable</strong><span>{predictionError || 'The prediction sample could not be loaded.'}</span></div></div>
        ) : (
          <div style={{ width: '100%', height: 260, marginTop: 18 }}>
            <ResponsiveContainer>
              <LineChart data={sample.points} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="timestamp"
                  tick={{ fontSize: 10 }}
                  minTickGap={18}
                  tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(value, name) => {
                    const numericValue = Array.isArray(value) ? value[0] : value
                    return [typeof numericValue === 'number' ? Number(numericValue).toFixed(1) : String(numericValue ?? ''), String(name)]
                  }}
                  labelFormatter={(label) => new Date(String(label)).toLocaleString()}
                />
                <Legend />
                <Line type="monotone" dataKey="actual" name="Actual" stroke="#2aa6a4" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#9ecdc9" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="panel" style={{ marginTop: '20px', padding: '20px' }}>
        <div className="panel-heading">
          <div>
            <span className="eyebrow">FEATURE IMPORTANCE</span>
            <h3>Top-ranked signals</h3>
          </div>
        </div>

        <div style={{ width: '100%', height: 260, marginTop: 18 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 12, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="feature" angle={-25} textAnchor="end" height={60} interval={0} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip
                formatter={(value) => {
                  const numericValue = Array.isArray(value) ? value[0] : value
                  return typeof numericValue === 'number' ? Number(numericValue).toFixed(4) : String(numericValue ?? '')
                }}
              />
              <Bar dataKey="importance" fill="#2aa6a4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  )
}

export default Model
