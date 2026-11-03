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
  ]}
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