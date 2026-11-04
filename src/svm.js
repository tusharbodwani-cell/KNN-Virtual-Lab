import { classify } from './knn'

const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0)
const kernel=(a,b,type,gamma)=>type==='linear'?dot(a,b):type==='polynomial'?(1+dot(a,b))**3:Math.exp(-gamma*a.reduce((s,v,i)=>s+(v-b[i])**2,0))
const statsFor=points=>{
  const mean=[0,1].map(j=>points.reduce((s,p)=>s+p[j===0?'x':'y'],0)/points.length)
  const scale=[0,1].map(j=>Math.sqrt(points.reduce((s,p)=>s+(p[j===0?'x':'y']-mean[j])**2,0)/points.length)||1)
  return {mean,scale}
}
const transform=(p,stats)=>[(p.x-stats.mean[0])/stats.scale[0],(p.y-stats.mean[1])/stats.scale[1]]
function binarySMO(X,y,type,C,gamma){
  const n=X.length, alpha=Array(n).fill(0), K=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>kernel(X[i],X[j],type,gamma)))
  let bias=0,passes=0,steps=0
  const decision=i=>{let s=bias;for(let q=0;q<n;q++)if(alpha[q]>0)s+=alpha[q]*y[q]*K[q][i];return s}
  const errors=()=>X.map((_,i)=>decision(i)-y[i])
  const maxSteps=n>100?80:1800\n  while(passes<6&&steps<maxSteps){
    let changed=0
    const E=errors()
    for(let i=0;i<n;i++){
      const Ei=E[i]
      if(!((y[i]*Ei < -0.001&&alpha[i]<C-1e-5)||(y[i]*Ei>0.001&&alpha[i]>1e-5)))continue
      let j=-1,best=-1
      for(let q=0;q<n;q++)if(q!==i){const gap=Math.abs(Ei-E[q]);if(gap>best){best=gap;j=q}}
      if(j<0)continue
      const Ej=E[j],ai=alpha[i],aj=alpha[j]
      let L,H
      if(y[i]!==y[j]){L=Math.max(0,aj-ai);H=Math.min(C,C+aj-ai)}
      else{L=Math.max(0,ai+aj-C);H=Math.min(C,ai+aj)}
      if(H-L<1e-8)continue
      const eta=2*K[i][j]-K[i][i]-K[j][j]
      if(eta>=0)continue
      let nextJ=aj-y[j]*(Ei-Ej)/eta
      nextJ=Math.max(L,Math.min(H,nextJ))
      if(Math.abs(nextJ-aj)<1e-5)continue
      const nextI=ai+y[i]*y[j]*(aj-nextJ)
      const b1=bias-Ei-y[i]*(nextI-ai)*K[i][i]-y[j]*(nextJ-aj)*K[i][j]
      const b2=bias-Ej-y[i]*(nextI-ai)*K[i][j]-y[j]*(nextJ-aj)*K[j][j]
      bias=nextI>1e-5&&nextI<C-1e-5?b1:nextJ>1e-5&&nextJ<C-1e-5?b2:(b1+b2)/2
      alpha[i]=nextI;alpha[j]=nextJ;changed++
    }
    passes=changed===0?passes+1:0
    steps++
  }
  const support=X.map((x,i)=>({x,y:y[i],a:alpha[i]})).filter(v=>v.a>1e-6)
  return {support,bias,score(point){return support.reduce((s,v)=>s+v.a*v.y*kernel(v.x,point,type,gamma),bias)}}
}
export function trainSVM(points,type='rbf',C=2,gamma=0.5){
  const labels=[...new Set(points.map(p=>p.label))]
  if(points.length<4||labels.length<2)throw new Error('SVM needs at least four labelled samples from two or more classes.')
  const stats=statsFor(points),X=points.map(p=>transform(p,stats))
  const models=labels.map(label=>({label,...binarySMO(X,points.map(p=>p.label===label?1:-1),type,Number(C),Number(gamma))}))
  const predict=point=>{
    const z=transform(point,stats)
    return models.map(m=>({label:m.label,score:m.score(z)})).sort((a,b)=>b.score-a.score)[0].label
  }
  return {predict,labels,supportVectors:models.flatMap(m=>m.support.map((v,i)=>({x:v.x[0]*stats.scale[0]+stats.mean[0],y:v.x[1]*stats.scale[1]+stats.mean[1],label:m.label}))),kernel:type,C,gamma}
}
function boundedSplit(points,capPerClass=4){
  const groups=new Map()
  points.forEach((p,i)=>{if(!groups.has(p.label))groups.set(p.label,[]);groups.get(p.label).push({...p,_index:i})})
  const train=[],test=[]
  for(const group of groups.values()){
    const chosen=group.length>capPerClass?Array.from({length:capPerClass},(_,i)=>group[Math.floor(i*group.length/capPerClass)]):group
    chosen.forEach((p,i)=>(i%4===0?test:train).push(p))
  }
  return {train,test,sampled:train.length+test.length<points.length}
}
const metricSummary=(actual,predicted,labels)=>{
  let correct=0
  const rows=labels.map(label=>{
    let tp=0,fp=0,fn=0
    actual.forEach((a,i)=>{if(a===label&&predicted[i]===label)tp++;else if(a!==label&&predicted[i]===label)fp++;else if(a===label&&predicted[i]!==label)fn++})
    correct+=tp
    const precision=tp/(tp+fp||1),recall=tp/(tp+fn||1)
    return {label,precision,recall,f1:2*precision*recall/(precision+recall||1)}
  })
  return {accuracy:correct/(actual.length||1)*100,correct,total:actual.length,precision:rows.reduce((s,r)=>s+r.precision,0)/rows.length*100,recall:rows.reduce((s,r)=>s+r.recall,0)/rows.length*100,f1:rows.reduce((s,r)=>s+r.f1,0)/rows.length*100}
}
export function compareKNNandSVM(points,k=5){
  if(points.length<6)return {error:'At least six labelled samples are required for a comparison.'}
  const {train,test,sampled}=boundedSplit(points)
  if(!test.length||!train.length)return {error:'Not enough samples to create a training and test split.'}
  const labels=[...new Set(points.map(p=>p.label))],actual=test.map(p=>p.label)
  const knn=metricSummary(actual,test.map(p=>classify(train,p,Math.min(k,train.length),'euclidean').prediction),labels)
  const svm={}
  for(const type of ['linear','polynomial','rbf']){
    const model=trainSVM(train,type)
    svm[type]=metricSummary(actual,test.map(model.predict),labels)
  }
  return {knn,svm,trainCount:train.length,testCount:test.length,labels,sampled}
}
export function evaluateSelectedModels(points,k=5,metric='euclidean',C=2,gamma=0.5){
  const split=boundedSplit(points)
  if(split.train.length<4||!split.test.length)return {error:'Not enough labelled samples for a train/test comparison.'}
  const labels=[...new Set(points.map(p=>p.label))],actual=split.test.map(p=>p.label)
  const knn=metricSummary(actual,split.test.map(p=>classify(split.train,p,Math.min(k,split.train.length),metric).prediction),labels)
  const svm={}
  for(const type of ['linear','polynomial','rbf']){
    const model=trainSVM(split.train,type,C,gamma)
    svm[type]=metricSummary(actual,split.test.map(model.predict),labels)
  }
  return {knn,svm,trainCount:split.train.length,testCount:split.test.length,labels,sampled:split.sampled}
}
