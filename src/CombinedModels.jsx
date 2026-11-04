import { useMemo, useState } from 'react'
import { BrainCircuit, GitCompareArrows } from 'lucide-react'
import { trainSVM, evaluateSelectedModels } from './svm'
import { classify } from './knn'

const colors={Setosa:'#60a5fa',Versicolor:'#a78bfa',Virginica:'#34d399','Needs support':'#fb7185',Average:'#fbbf24','High performer':'#34d399'}
const colorFor=label=>colors[label]||`hsl(${[...String(label)].reduce((a,c)=>a+c.charCodeAt(0),0)%360} 65% 55%)`
function trainingSample(points,limit=4){
  const groups=new Map()
  points.forEach(p=>{if(!groups.has(p.label))groups.set(p.label,[]);groups.get(p.label).push(p)})
  return [...groups.values()].flatMap(group=>group.length>limit?Array.from({length:limit},(_,i)=>group[Math.floor(i*group.length/limit)]):group)
}
export default function CombinedModels({points,query,k,metric,maxX,maxY}){
  const [kernel,setKernel]=useState('rbf'),[C,setC]=useState(2),[gamma,setGamma]=useState(.5)
  const sampled=useMemo(()=>trainingSample(points),[points])
  const model=useMemo(()=>{try{return trainSVM(sampled,kernel,C,gamma)}catch(error){return {error:error.message}}},[sampled,kernel,C,gamma])
  const prediction=model.error?null:model.predict(query)
  const knnPrediction=classify(points,query,k,metric).prediction
  const comparison=useMemo(()=>evaluateSelectedModels(points,k,metric,C,gamma),[points,k,metric,C,gamma])
  const cells=useMemo(()=>{
    if(model.error)return []
    const cols=48,rows=28,out=[]
    for(let iy=0;iy<rows;iy++)for(let ix=0;ix<cols;ix++){
      const x=(ix+.5)/cols*maxX,y=(1-(iy+.5)/rows)*maxY
      const scores=model.decisionScores({x,y}),winner=scores[0]
      out.push({x:ix*600/cols,y:iy*330/rows,label:winner.label,margin:Math.abs(winner.score)})
    }
    return out
  },[model,maxX,maxY])
  const support=model.supportVectors||[]
  const X=x=>x/maxX*600,Y=y=>330-y/maxY*330
  return <section className="panel combined-lab">
    <div className="combined-heading"><div className="section-icon"><BrainCircuit/></div><div><h2>Combined KNN + SVM Model Workbench</h2><p>Both algorithms use the same selected dataset and feature pair. Change the SVM kernel and compare its prediction with KNN.</p></div></div>
    <div className="combined-grid">
      <div className="combined-controls">
        <h3>Train the SVM model</h3>
        <label>DECISION BOUNDARY / KERNEL<select value={kernel} onChange={e=>setKernel(e.target.value)}><option value="linear">Linear SVM</option><option value="polynomial">Polynomial Kernel (degree 3)</option><option value="rbf">RBF / Gaussian Kernel</option></select></label>
        <div className="range-title"><label>REGULARIZATION (C)</label><b>{C.toFixed(1)}</b></div><input type="range" min=".2" max="8" step=".2" value={C} onChange={e=>setC(+e.target.value)}/>
        {kernel!=='linear'&&<><div className="range-title"><label>GAMMA</label><b>{gamma.toFixed(2)}</b></div><input type="range" min=".1" max="2" step=".05" value={gamma} onChange={e=>setGamma(+e.target.value)}/></>}
        <div className="svm-parameter-note"><b>What changes?</b><p>C controls the penalty for margin violations. Gamma controls the reach of the RBF kernel; a larger value can create more local, complex boundaries.</p></div>
      </div>
      <div className="combined-prediction">
        <h3>Prediction for the same query point</h3>
        <div className="prediction-pair"><article><small>KNN · K={k}</small><strong style={{color:colorFor(knnPrediction)}}>{knnPrediction}</strong><b>{metric} · majority vote</b></article><article><small>{kernel.toUpperCase()} SVM</small><strong style={{color:prediction?colorFor(prediction):undefined}}>{prediction||'Unavailable'}</strong><b>{support.length} support-vector entries</b></article></div>
        {model.error&&<p className="upload-error">{model.error}</p>}
        <p className="hint">The KNN prediction for the movable orange query point is shown in the main simulator above. The SVM prediction updates when you change the query point, kernel, C or gamma.</p>
      </div>
    </div>
    <h3 className="subheading">SVM decision-region visualisation</h3>
    <p className="hint">The finer background grid shows predicted class regions. Amber cells mark the approximate one-vs-rest margin band (decision score near ±1); ringed points are support vectors, and the orange point is the query. Margin bands are illustrative for this multiclass educational solver, not exact geometric distances.</p>
    <div className="combined-chart"><svg viewBox="0 0 600 330" role="img" aria-label={`${kernel} SVM decision regions and data points`}>{cells.map((cell,i)=><rect key={i} x={cell.x} y={cell.y} width={600/48+.5} height={330/28+.5} fill={colorFor(cell.label)} opacity=".16"/>)}{cells.filter(cell=>cell.margin<=1.15).map((cell,i)=><rect key={'m'+i} x={cell.x} y={cell.y} width={600/48+.5} height={330/28+.5} fill="#f59e0b" opacity=".32"/>)}{points.map((p,i)=><circle key={i} cx={X(p.x)} cy={Y(p.y)} r="3.5" fill={colorFor(p.label)} stroke="#fff" strokeWidth=".7"><title>{p.label}</title></circle>)}{support.slice(0,160).map((p,i)=><circle key={'s'+i} cx={X(p.x)} cy={Y(p.y)} r="6" fill="none" stroke="#111827" strokeWidth="1.1"/>)}<circle cx={X(query.x)} cy={Y(query.y)} r="8" fill="#f97316" stroke="#fff" strokeWidth="2.5"/><text x={Math.min(550,X(query.x)+10)} y={Math.max(14,Y(query.y)-10)} fontSize="10" fill="#c2410c" fontWeight="700">QUERY</text></svg></div>
    <div className="legend">{[...new Set(points.map(p=>p.label))].slice(0,8).map(label=><span key={label}><i style={{background:colorFor(label)}}/>{label}</span>)}<span><i className="support-dot"/>Support vectors</span><span><i className="query-dot"/>Query</span></div>
    <h3 className="subheading">Same-split performance comparison</h3>
    {comparison.error?<p className="note">{comparison.error}</p>:<><p className="hint">All four models use the same deterministic stratified hold-out split. To keep large datasets responsive, up to six representative rows per class are selected before splitting. SVM uses one-vs-rest classification for multiple classes.</p><div className="table-wrap"><table className="model-comparison"><thead><tr><th>MODEL</th><th>ACCURACY</th><th>MACRO PRECISION</th><th>MACRO RECALL</th><th>MACRO F1</th></tr></thead><tbody><tr><td><b>KNN (K={k})</b><small>{metric} distance</small></td><td>{comparison.knn.accuracy.toFixed(1)}%</td><td>{comparison.knn.precision.toFixed(1)}%</td><td>{comparison.knn.recall.toFixed(1)}%</td><td>{comparison.knn.f1.toFixed(1)}%</td></tr>{[['Linear SVM','linear'],['Polynomial SVM','polynomial'],['RBF Kernel SVM','rbf']].map(([name,key])=><tr key={key}><td><b>{name}</b><small>{key==='polynomial'?'Degree 3':key==='rbf'?'Gaussian kernel':'Straight hyperplane'}</small></td><td>{comparison.svm[key].accuracy.toFixed(1)}%</td><td>{comparison.svm[key].precision.toFixed(1)}%</td><td>{comparison.svm[key].recall.toFixed(1)}%</td><td>{comparison.svm[key].f1.toFixed(1)}%</td></tr>)}</tbody></table></div><div className="svm-result-meta"><span>Training rows: <b>{comparison.trainCount}</b></span><span>Test rows: <b>{comparison.testCount}</b></span><span>Classes: <b>{comparison.labels.length}</b></span></div></>}
    <div className="conclusion-panel"><h3>Experiment interpretation</h3><p>KNN predicts using nearby training examples, whereas SVM learns a separating boundary. Linear SVM creates a straight boundary; polynomial and RBF kernels can represent non-linear patterns. Compare accuracy together with macro precision, recall and F1 rather than selecting a model from accuracy alone. The displayed hold-out scores are educational estimates on a small deterministic sample, not a substitute for repeated stratified cross-validation.</p></div>
  </section>
}
