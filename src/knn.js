import { realDatasets } from './data/realDatasets'

export const datasets = {
  flowers: { label: 'Flower classification', xLabel: 'Petal length', yLabel: 'Petal width', points: [
    {x:1.2,y:.3,label:'Setosa'},{x:1.5,y:.4,label:'Setosa'},{x:1.7,y:.5,label:'Setosa'},{x:1.4,y:.2,label:'Setosa'},{x:1.8,y:.6,label:'Setosa'},{x:1.3,y:.4,label:'Setosa'},
    {x:4.1,y:1.3,label:'Versicolor'},{x:4.5,y:1.5,label:'Versicolor'},{x:4.7,y:1.4,label:'Versicolor'},{x:4.3,y:1.2,label:'Versicolor'},{x:4.9,y:1.7,label:'Versicolor'},{x:4,y:1.1,label:'Versicolor'},
    {x:5.4,y:2,label:'Virginica'},{x:5.8,y:2.2,label:'Virginica'},{x:6.1,y:2.1,label:'Virginica'},{x:5.6,y:1.9,label:'Virginica'},{x:6.3,y:2.4,label:'Virginica'},{x:5.9,y:2,label:'Virginica'}
  ]},
  students: { label: 'Student performance', xLabel: 'Study hours', yLabel: 'Practice tests', points: [
    {x:1,y:1,label:'Needs support'},{x:2,y:1,label:'Needs support'},{x:2,y:2,label:'Needs support'},{x:3,y:1,label:'Needs support'},{x:2,y:3,label:'Needs support'},{x:3,y:2,label:'Needs support'},
    {x:4,y:4,label:'Average'},{x:5,y:3,label:'Average'},{x:5,y:5,label:'Average'},{x:6,y:4,label:'Average'},{x:4,y:5,label:'Average'},{x:6,y:5,label:'Average'},
    {x:7,y:7,label:'High performer'},{x:8,y:7,label:'High performer'},{x:8,y:8,label:'High performer'},{x:9,y:8,label:'High performer'},{x:7,y:9,label:'High performer'},{x:9,y:9,label:'High performer'}
  ]},
  ...realDatasets
}

export function get2DView(dataset, xFeature, yFeature) {
  if (!dataset.featureOptions) return {
    points: dataset.points, xLabel: dataset.xLabel, yLabel: dataset.yLabel,
    maxX: dataset.maxX || (dataset.xLabel === 'Petal length' ? 7 : 10),
    maxY: dataset.maxY || (dataset.yLabel === 'Petal width' ? 3 : 10)
  }
  const xs = dataset.points.map(p => p.features[xFeature])
  const ys = dataset.points.map(p => p.features[yFeature])
  const minX = Math.min(...xs), maxRawX = Math.max(...xs)
  const minY = Math.min(...ys), maxRawY = Math.max(...ys)
  const scale = (v, min, max) => max === min ? 5 : ((v - min) / (max - min)) * 10
  const xName = dataset.featureOptions.find(f => f.value === xFeature)?.label || xFeature
  const yName = dataset.featureOptions.find(f => f.value === yFeature)?.label || yFeature
  return {
    points: dataset.points.map((p, index) => ({
      x: scale(p.features[xFeature], minX, maxRawX),
      y: scale(p.features[yFeature], minY, maxRawY),
      rawX: p.features[xFeature], rawY: p.features[yFeature],
      label: p.label, index
    })),
    xLabel: xName + ' (scaled 0–10)',
    yLabel: yName + ' (scaled 0–10)',
    maxX: 10, maxY: 10
  }
}

export function distance(a,b,metric='euclidean') {
  const dx=Math.abs(a.x-b.x), dy=Math.abs(a.y-b.y)
  return metric==='manhattan' ? dx+dy : Math.hypot(dx,dy)
}
export function classify(points,query,k,metric) {
  const nearest=points.map((p,index)=>({...p,index,distance:distance(p,query,metric)})).sort((a,b)=>a.distance-b.distance)
  const selected=nearest.slice(0,Math.max(1,Math.min(Number(k)||1,nearest.length)))
  const counts={}
  selected.forEach(p=>counts[p.label]=(counts[p.label]||0)+1)
  const prediction=Object.entries(counts).sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0]))[0]?.[0]??'Unknown'
  return {nearest,selected,prediction,counts}
}
