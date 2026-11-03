import { useMemo, useState } from 'react'
import { Upload, FileSpreadsheet, BarChart3, Footprints, CheckCircle2, AlertCircle } from 'lucide-react'
import { datasets, classify, distance } from './knn'

function parseCSV(text) {
  const rows=[]; let row=[], cell='', quoted=false
  for(let i=0;i<text.length;i++){
    const c=text[i], n=text[i+1]
    if(c==='"'&&quoted&&n==='"'){cell+='"';i++}
    else if(c==='"') quoted=!quoted
    else if(c===','&&!quoted){row.push(cell.trim());cell=''}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&n==='\n')i++;row.push(cell.trim());if(row.some(Boolean))rows.push(row);row=[];cell=''}
    else cell+=c
  }
  row.push(cell.trim());if(row.some(Boolean))rows.push(row)
  if(rows.length<4)throw new Error('Include a header and at least three data rows.')
  const headers=rows[0], data=rows.slice(1).filter(r=>r.length>=headers.length)
  const numeric=headers.map((_,i)=>data.every(r=>r[i]!==''&&Number.isFinite(Number(r[i]))))
  const features=headers.map((_,i)=>i).filter(i=>numeric[i]), label=headers.findIndex((_,i)=>!numeric[i])
  if(features.length<2||label<0)throw new Error('CSV needs two numeric feature columns and one text class/label column.')
  const points=data.map(r=>({x:Number(r[features[0]]),y:Number(r[features[1]]),label:r[label]})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.label)
  if(points.length<3)throw new Error('At least three valid data rows are required.')
  return {label:'Uploaded CSV',xLabel:headers[features[0]],yLabel:headers[features[1]],labelColumn:headers[label],points}
}
function evaluate(points,k,metric){
  const labels=[...new Set(points.map(p=>p.label))].sort()
  const matrix=Object.fromEntries(labels.map(a=>[a,Object.fromEntries(labels.map(b=>[b,0]))]))
  let correct=0
  points.forEach((p,i)=>{const train=points.filter((_,j)=>i!==j);const pred=classify(train,p,Math.min(k,train.length),metric).prediction;matrix[p.label][pred]=(matrix[p.label][pred]||0)+1;if(pred===p.label)correct++})
  return {labels,matrix,correct,total:points.length,accuracy:correct/points.length}
}
export default function AdvancedLab(){
  const [kind,setKind]=useState('flowers'),[custom,setCustom]=useState(null),[message,setMessage]=useState('')
  const [k,setK]=useState(3),[metric,setMetric]=useState('euclidean'),[step,setStep]=useState(0)
  const data=kind==='custom'&&custom?custom:datasets[kind], labels=[...new Set(data.points.map(p=>p.label))].sort()
  const report=useMemo(()=>evaluate(data.points,Math.min(k,data.points.length-1),metric),[data,k,metric])
  const query={x:4.8,y:1.6}, result=classify(data.points,query,k,metric)
  const steps=[
    ['Calculate distances','Compute the selected distance from the query point to every training sample.'],
    ['Sort the distances','Arrange all samples from the smallest distance to the largest distance.'],
    ['Select K neighbors',`Take the first ${k} rows from the sorted distance list.`],
    ['Majority vote',`Count each selected class. The predicted class is ${result.prediction}.`]
  ]
  async function upload(e){const file=e.target.files?.[0];if(!file)return;try{const parsed=parseCSV(await file.text());setCustom(parsed);setKind('custom');setMessage(`Loaded ${parsed.points.length} rows · X: ${parsed.xLabel} · Y: ${parsed.yLabel} · Label: ${parsed.labelColumn}`)}catch(err){setMessage(err.message)}e.target.value=''}
  return <section className="panel text-panel advanced-lab">
    <div className="section-icon"><BarChart3/></div><h2>Dataset & model evaluation</h2>
    <p>Upload your own labelled CSV or use a sample dataset. The evaluation uses leave-one-out cross-validation: each row is predicted using the remaining rows.</p>
    <div className="advanced-controls"><label>DATASET<select value={kind} onChange={e=>setKind(e.target.value)}><option value="flowers">Flower classification</option><option value="students">Student performance</option>{custom&&<option value="custom">Uploaded CSV</option>}</select></label><label>K VALUE<select value={k} onChange={e=>setK(+e.target.value)}>{Array.from({length:Math.min(15,data.points.length-1)},(_,i)=>i+1).map(n=><option key={n} value={n}>{n}</option>)}</select></label><label>DISTANCE<select value={metric} onChange={e=>setMetric(e.target.value)}><option value="euclidean">Euclidean</option><option value="manhattan">Manhattan</option></select></label></div>
    <div className="upload-panel"><div><h3><FileSpreadsheet size={17}/> Upload a CSV dataset</h3><p>At least two numeric columns and one categorical label column are required.</p></div><label className="upload-button"><Upload size={16}/> Choose CSV<input type="file" accept=".csv,text/csv" onChange={upload}/></label>{message&&<p className={message.startsWith('Loaded')?'upload-success':'upload-error'}>{message.startsWith('Loaded')?<CheckCircle2 size={15}/>:<AlertCircle size={15}/>} {message}</p>}<small>Your CSV is processed locally in your browser; it is not sent to a server. The first two numeric columns and first text column are used.</small></div>
    <div className="metric-cards"><article><small>LOOCV ACCURACY</small><strong>{(report.accuracy*100).toFixed(1)}%</strong><span>{report.correct} correct / {report.total} samples</span></article><article><small>VALIDATION</small><strong>LOOCV</strong><span>One sample held out each round</span></article><article><small>MODEL SETUP</small><strong>K = {k}</strong><span>{metric==='euclidean'?'Euclidean':'Manhattan'} distance</span></article></div>
    <h3 className="subheading">Confusion matrix</h3><p className="hint">Rows are actual classes; columns are predicted classes. Diagonal cells are correct predictions.</p><div className="table-wrap"><table className="matrix"><thead><tr><th>ACTUAL ↓ / PREDICTED →</th>{report.labels.map(l=><th key={l}>{l}</th>)}</tr></thead><tbody>{report.labels.map(a=><tr key={a}><th>{a}</th>{report.labels.map(b=><td key={b} className={a===b?'matrix-correct':''}>{report.matrix[a][b]||0}</td>)}</tr>)}</tbody></table></div>
    <div className="note"><b>Interpretation:</b> Accuracy is the percentage of held-out samples classified correctly. For a formal project report, use a separate test set or stratified cross-validation. Feature scaling is important when measurement units differ.</div>
    <div className="walkthrough-heading"><div className="section-icon"><Footprints/></div><div><h2>KNN step-by-step walkthrough</h2><p>Example query point: ({query.x}, {query.y}) using the current dataset and settings.</p></div></div>
    <div className="walk-progress">{steps.map(([title],i)=><button key={title} className={step===i?'walk-active':step>i?'walk-done':''} onClick={()=>setStep(i)}><b>{i+1}</b><span>{title}</span></button>)}</div>
    <div className="walk-card"><small>STEP {step+1} OF 4</small><h3>{steps[step][0]}</h3><p>{steps[step][1]}</p>{step===0&&<div className="walk-detail">{data.points.length} distances calculated using {metric} distance.</div>}{step===1&&<div className="walk-detail">Three smallest distances: {result.nearest.slice(0,3).map(p=>p.distance.toFixed(3)).join(' · ')}</div>}{step===2&&<div className="walk-detail">Selected classes: {result.selected.map(p=>p.label).join(', ')}</div>}{step===3&&<div className="walk-detail">Vote counts: {Object.entries(result.counts).map(([l,n])=>l+': '+n).join(' · ')}</div>}</div>
    <div className="walk-actions"><button className="reset" disabled={step===0} onClick={()=>setStep(s=>s-1)}>Previous</button><button className="primary" disabled={step===3} onClick={()=>setStep(s=>s+1)}>Next step</button></div>
    <h3 className="subheading">Nearest-neighbor ranking for example query</h3><div className="table-wrap"><table><thead><tr><th>RANK</th><th>POINT</th><th>CLASS</th><th>DISTANCE</th><th>STATUS</th></tr></thead><tbody>{result.nearest.slice(0,Math.min(10,data.points.length)).map((p,i)=><tr key={i}><td>{i+1}</td><td>({p.x}, {p.y})</td><td>{p.label}</td><td>{p.distance.toFixed(3)}</td><td>{i<k?'Selected':'Not selected'}</td></tr>)}</tbody></table></div>
  </section>
}
