import carCsv from './car-details.csv?raw'
import cropCsv from './crop-recommendation.csv?raw'

function readCSV(source) {
  const rows = []
  let row = [], cell = '', quoted = false
  for (let i = 0; i < source.length; i++) {
    const c = source[i], next = source[i + 1]
    if (c === '"' && quoted && next === '"') { cell += '"'; i++ }
    else if (c === '"') quoted = !quoted
    else if (c === ',' && !quoted) { row.push(cell.trim()); cell = '' }
    else if ((c === '\n' || c === '\r') && !quoted) {
      if (c === '\r' && next === '\n') i++
      row.push(cell.trim())
      if (row.some(Boolean)) rows.push(row)
      row = []; cell = ''
    } else cell += c
  }
  row.push(cell.trim())
  if (row.some(Boolean)) rows.push(row)
  const headers = rows.shift()
  return rows.filter(r => r.length >= headers.length).map(r =>
    Object.fromEntries(headers.map((h, i) => [h, r[i]]))
  )
}

const carRows = readCSV(carCsv)
const cropRows = readCSV(cropCsv)

export const realDatasets = {
  cars: {
    label: 'CarDekho used cars',
    description: 'Classify transmission type using vehicle details.',
    targetLabel: 'Transmission',
    defaultX: 'year',
    defaultY: 'km_driven',
    featureOptions: [
      { value: 'year', label: 'Year' },
      { value: 'selling_price', label: 'Selling price (₹)' },
      { value: 'km_driven', label: 'Kilometres driven' }
    ],
    points: carRows.map(r => ({
      features: {
        year: Number(r.year),
        selling_price: Number(r.selling_price),
        km_driven: Number(r.km_driven)
      },
      label: r.transmission
    })).filter(p => Object.values(p.features).every(Number.isFinite) && p.label)
  },
  crops: {
    label: 'Crop recommendation',
    description: 'Classify the crop label from soil and environmental measurements.',
    targetLabel: 'Crop',
    defaultX: 'N',
    defaultY: 'P',
    featureOptions: [
      { value: 'N', label: 'Nitrogen (N)' },
      { value: 'P', label: 'Phosphorus (P)' },
      { value: 'K', label: 'Potassium (K)' },
      { value: 'temperature', label: 'Temperature (°C)' },
      { value: 'humidity', label: 'Humidity (%)' },
      { value: 'ph', label: 'Soil pH' },
      { value: 'rainfall', label: 'Rainfall (mm)' }
    ],
    points: cropRows.map(r => ({
      features: {
        N: Number(r.N), P: Number(r.P), K: Number(r.K),
        temperature: Number(r.temperature), humidity: Number(r.humidity),
        ph: Number(r.ph), rainfall: Number(r.rainfall)
      },
      label: r.label
    })).filter(p => Object.values(p.features).every(Number.isFinite) && p.label)
  }
}
