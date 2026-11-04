import { classify } from './knn'

const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0)
const kernel=(a,b,type,gamma=0.5)=>type==='linear'?dot(a,b):type==='polynomial'?Math.pow(1+gamma*dot(a,b),3):Math.exp(-gamma*a.reduce((s,v,i)=>s+(v-b[i])**2,0))
function standardize(train,test){
  const mean=[0,1].map(j=>train.reduce((s,p)=>s+p[['x','y'][j]],0)/train.length)
  const scale=[0,1].map(j=>Math.sqrt(train.reduce((s,p)=>s+(p[['x','y'][j]]-mean[j])**2,0)/train.length)||1)
  const convert=p=>[(p.x-mean[0])/scale[0],(p.y-mean[1])/scale[1]]
  return {train:train.map(convert),test:test.map(convert)}
}
function trainBinarySVM(X,y,type,C=2,gamma=0.5){
  const n=X.length,alpha=Array(n).fill(0),b=0,eps=1e-3
  const K=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>kernel(X[i],X[j],type,gamma)))
  let bias=b,passes=0,iterations=0
  const decision=i=>alpha.reduce((s,a,j)=>s+a*y[j]*K[j][i],bias)
  while(passes<8&&iterations<12000){
    let changed=0
    for(let i=0;i<n;i++){
      const Ei=decision(i)-y[i]
      if(!((y[i]*Ei < -eps && alpha[i]<C-eps)||(y[i]*Ei>eps&&alpha[i]>eps)))continue
      let j=(i+1+Math.floor(Math.random()*(n-1)))%n
      let Ej=decision(j)-y[j], ai=alpha[i], aj=alpha[j], L,H
      if(y[i]!==y[j]){L=Math.max(0,aj-ai);H=Math.min(C,C+aj-ai)}
      else{L=Math.max(0,ai+aj-C);H=Math.min(C,ai+aj)}
      if(H-L<eps)continue
      const eta=2*K[i][j]-K[i][i]-K[j][j]
      if(eta>=0)continue
      let newAj=aj-y[j]*(Ei-Ej)/eta
      newAj=Math.max(L,Math.min(H,newAj))
      if(Math.abs(newAj-aj)<eps)continue
      const newAi=ai+y[i]*y[j]*(aj-newAj)
      const b1=bias-Ei-y[i]*(newAi-ai)*K[i][i]-y[j]*(newAj-aj)*K[i][j]
      const b2=bias-Ej-y[i]*(newAi-ai)*K[i][j]-y[j]*(newAj-aj)*K[j][j]
      bias=newAi>eps&&newAi<C-eps?b1:newAj>eps&&newAj<C-eps?b2:(b1+b2)/2
      alpha[i]=newAi;alpha[j]=newAj;changed++
    }
    passes=changed===0?passes+1:0
    iterations++
  }
  const support=alpha.map((a,i)=>({a,y:y[i],x:X[i]})).filter(v=>v.a>1e-5)
  return point=>support.reduce((s,v)=>s+v.a*v.y*kernel(v.x,point,type,gamma),bias)
}
function trainMulticlassSVM(train,type){
  const {train:X}=standardize(train,[])
  const labels=[...new Set(train.map(p=>p.label))]
  const models=labels.map(label=>({label,score:trainBinarySVM(X,train.map(p=>p.label===label?1:-1),type)}))
  return point=>{
    const normalized=[(point.x-train.reduce((s,p)=>s+p.x,0)/train.length), (point.y-train.reduce((s,p)=>s+p.y,0)/train.length)]
    // Scale query using the same training statistics.
    const mean=[0,1].map(j=>train.reduce((s,p)=>s+p[['x','y'][j]],0)/train.length)
    const scale=[0,1].map(j=>Math.sqrt(train.reduce((s,p)=>s+(p[['x','y'][j]]-mean[j])**2,0)/train.length)||1)
    const z=normalized.map((v,j)=>v/scale[j])
    return models.map(m=>({label:m.label,score:m.score(z)})).sort((a,b)=>b.score-a.score)[0].label
  }
}
function splitData(points){
  const groups=new Map()
  points.forEach((p,i)=>{if(!groups.has(p.label))groups.set(p.label,[]);groups.get(p.label).push({...p,_i:i})})
  const train=[],test=[]
  groups.forEach(group=>group.forEach((p,i)=>(i%4===0?test:train).push(p)))
  return {train,test}
}
export function compareKNNandSVM(points,k=3){
  if(points.length<6)return {error:'At least six labelled samples are required for a comparison.'}
  const {train,test}=splitData(points)
  if(!test.length||!train.length)return {error:'Not enough samples to create a training and test split.'}
  const score=predictions=>({correct:predictions.filter((p,i)=>p===test[i].label).length,total:test.length,accuracy:predictions.filter((p,i)=>p===test[i].label).length/test.length*100})
  const knn=score(test.map(p=>classify(train,p,Math.min(k,train.length),'euclidean').prediction))
  const svm={}
  for(const type of ['linear','polynomial','rbf']){
    const model=trainMulticlassSVM(train,type)
    svm[type]=score(test.map(model))
  }
  return {knn,svm,trainCount:train.length,testCount:test.length,labels:[...new Set(points.map(p=>p.label))]}
}
