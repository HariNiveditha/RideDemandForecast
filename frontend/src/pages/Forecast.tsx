import { useEffect, useState } from 'react'
import { ArrowUpRight } from 'lucide-react'

const locationMap: Record<string, number> = {
  Manhattan: 4,
  Queens: 7,
  Brooklyn: 11,
  Bronx: 3,
}

type HistoricalRange = {
  start: string
  end: string
}

type FeatureImportance = {
  feature: string
  importance: number
}

type MetricsResponse = {
  feature_importances: FeatureImportance[]
}

function dateIsWithinRange(value: string, start: string, end: string) {
  if (!value || !start || !end) {
    return false
  }

  const selected = new Date(`${value}T00:00:00`)
  const minDate = new Date(`${start.slice(0, 10)}T00:00:00`)
  const maxDate = new Date(`${end.slice(0, 10)}T00:00:00`)

  return selected >= minDate && selected <= maxDate
}

function getDefaultDate(start: string, end: string) {
  if (!start || !end) {
    return ''
  }

  const today = new Date()
  const minDate = new Date(`${start.slice(0, 10)}T00:00:00`)
  const maxDate = new Date(`${end.slice(0, 10)}T00:00:00`)

  if (today >= minDate && today <= maxDate) {
    return today.toISOString().slice(0, 10)
  }

  const midpoint = new Date((minDate.getTime() + maxDate.getTime()) / 2)
  return midpoint.toISOString().slice(0, 10)
}

function getFeatureInfluenceText(featureImportances: FeatureImportance[]) {
  const ranked = [...featureImportances].sort((a, b) => b.importance - a.importance)
  const topThree = ranked.slice(0, 3)

  if (!topThree.length) {
    return 'This estimate is based on the model inputs and recent historical demand signals.'
  }

  const topNames = topThree.map((entry) => entry.feature)
  const mostImportant = topThree[0].feature
  const lagBased = topNames.filter((feature) => feature.startsWith('lag_'))
  const weatherBased = topNames.filter((feature) => ['temp', 'humidity', 'precip', 'windgust', 'windspeed', 'sealevelpressure', 'cloudcover'].includes(feature))

  if (lagBased.length > 0 && lagBased.includes(mostImportant)) {
    return `This estimate is most influenced by recent demand patterns (${lagBased.join(', ')}) rather than current weather conditions.`
  }

  if (weatherBased.length > 0 && weatherBased.includes(mostImportant)) {
    return `This estimate is most influenced by weather conditions (${weatherBased.join(', ')}) rather than recent demand history.`
  }

  return `This estimate is most influenced by ${topNames.join(', ')} based on the model's real feature importance rankings.`
}

function Forecast() {
  const [location, setLocation] = useState('Select a pickup zone')
  const [date, setDate] = useState('')
  const [time, setTime] = useState('')
  const [weather, setWeather] = useState({ temp: '', humidity: '', precip: '' })
  const [dateRange, setDateRange] = useState<HistoricalRange>({ start: '', end: '' })
  const [status, setStatus] = useState('')
  const [prediction, setPrediction] = useState<{ rides: number; context: string; influence: string } | null>(null)
  const [featureImportances, setFeatureImportances] = useState<FeatureImportance[]>([])

  useEffect(() => {
    const loadForecastConstraints = async () => {
      try {
        const [historicalResponse, metricsResponse] = await Promise.all([
          fetch('/api/historical'),
          fetch('/api/metrics'),
        ])

        if (!historicalResponse.ok || !metricsResponse.ok) {
          throw new Error('Unable to load the forecast window and model metrics.')
        }

        const historicalData = await historicalResponse.json()
        const metricsData = (await metricsResponse.json()) as MetricsResponse

        const range = {
          start: historicalData.date_range.start.slice(0, 10),
          end: historicalData.date_range.end.slice(0, 10),
        }

        setDateRange(range)
        setFeatureImportances(metricsData.feature_importances ?? [])
        setDate((currentDate) => {
          if (currentDate && dateIsWithinRange(currentDate, range.start, range.end)) {
            return currentDate
          }
          return getDefaultDate(range.start, range.end)
        })
      } catch (error) {
        const message = error instanceof Error ? error.message : 'The forecast window could not be loaded.'
        setStatus(message)
      }
    }

    void loadForecastConstraints()
  }, [])

  const dateHelperText = dateRange.start && dateRange.end
    ? `Predictions available between ${dateRange.start} and ${dateRange.end}`
    : 'Loading the available prediction window...'

  const submitPrediction = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const selectedLocation = locationMap[location]
    const timestamp = date && time ? new Date(`${date}T${time}:00`).toISOString() : ''

    if (!dateRange.start || !dateRange.end) {
      setStatus('The forecast window is still loading. Please wait a moment and try again.')
      return
    }

    if (!dateIsWithinRange(date, dateRange.start, dateRange.end)) {
      setStatus(dateHelperText)
      return
    }

    if (!selectedLocation || !timestamp || !weather.temp || !weather.humidity || !weather.precip) {
      setStatus('Please choose a location, date, time, and required weather fields before predicting.')
      return
    }

    const payload = {
      PULocationID: selectedLocation,
      datetime: timestamp,
      temp: Number(weather.temp),
      feelslike: Number(weather.temp),
      humidity: Number(weather.humidity),
      precip: Number(weather.precip),
      precipprob: 0,
      snow: 0,
      windgust: 15,
      windspeed: 10,
      sealevelpressure: 1012,
      cloudcover: 40,
    }

    setStatus('Connecting to the model...')

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Inference service unavailable')
      }

      const result = await response.json()
      const predictedRides = Math.round(Number(result.predicted_demand))
      const contextDate = new Date(`${date}T${time}:00`)
      const formattedDate = contextDate.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
      const formattedTime = contextDate.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
      const tempValue = Number(weather.temp)
      const precipValue = Number(weather.precip)

      setPrediction({
        rides: predictedRides,
        context: `For ${location} on ${formattedDate} at ${formattedTime}, with ${tempValue.toFixed(1)}°C and ${precipValue.toFixed(1)}mm precipitation, the model expects ${predictedRides} rides.`,
        influence: getFeatureInfluenceText(featureImportances),
      })
      setStatus('')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Inference service unavailable'
      setStatus(`Model connection is pending. Upload the data and start the inference service to enable predictions. ${message}`)
    }
  }

  return (
    <section className="forecast-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">LIVE INFERENCE</span>
          <h2>Run a demand forecast</h2>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span className="model-chip" style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <span className="status-dot" />
            Random Forest · weather + lags
          </span>
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              padding: '6px 10px',
              borderRadius: 999,
              border: '1px solid #d6e7eb',
              background: '#edf6f7',
              color: '#4d6d7d',
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: '.02em',
            }}
          >
            {dateHelperText}
          </span>
        </div>
      </div>

      <form className="forecast-card" onSubmit={submitPrediction}>
        <div className="form-fields">
          <label>
            Pickup location
            <select value={location} onChange={(event) => setLocation(event.target.value)}>
              <option>Select a pickup zone</option>
              <option>Manhattan</option>
              <option>Queens</option>
              <option>Brooklyn</option>
              <option>Bronx</option>
            </select>
          </label>

          <label>
            Date
            <input
              type="date"
              value={date}
              min={dateRange.start}
              max={dateRange.end}
              onChange={(event) => setDate(event.target.value)}
            />
          </label>

          <label>
            Time
            <input type="time" value={time} onChange={(event) => setTime(event.target.value)} />
          </label>

          <label>
            Temperature (°C)
            <input
              type="number"
              placeholder="Required"
              value={weather.temp}
              onChange={(event) => setWeather({ ...weather, temp: event.target.value })}
            />
          </label>

          <label>
            Humidity (%)
            <input
              type="number"
              placeholder="Required"
              value={weather.humidity}
              onChange={(event) => setWeather({ ...weather, humidity: event.target.value })}
            />
          </label>

          <label>
            Precipitation
            <input
              type="number"
              step="0.1"
              placeholder="Required"
              value={weather.precip}
              onChange={(event) => setWeather({ ...weather, precip: event.target.value })}
            />
          </label>
        </div>

        <div className="form-footer">
          <span>{status || 'Weather and lag features are resolved by the inference service.'}</span>
          <button className="primary-button" type="submit">
            Predict ride demand <ArrowUpRight size={17} />
          </button>
        </div>
      </form>

      {prediction && (
        <div style={{ marginTop: 22, borderRadius: 16, background: '#edf9f7', border: '1px solid #c8eae5', padding: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: '1.35px', textTransform: 'uppercase', color: '#587286', fontWeight: 700 }}>
            Prediction result
          </div>
          <div style={{ marginTop: 10, fontSize: 36, lineHeight: 1.1, fontWeight: 800, color: '#10263f' }}>
            {prediction.rides} rides expected
          </div>
          <p style={{ marginTop: 12, marginBottom: 8, color: '#1c3e59', fontSize: 15 }}>{prediction.context}</p>
          <p style={{ margin: 0, color: '#577386', fontSize: 14 }}>
            {prediction.influence}
          </p>
        </div>
      )}
    </section>
  )
}

export default Forecast
