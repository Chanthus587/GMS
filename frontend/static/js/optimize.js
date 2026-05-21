let optRunning=false,optStart=0,lastOptData=null,lastAlertTune=null,optAnim=null,optLoading=false,optIterTarget=50,optThinkTimer=null,optThinkIndex=0;

function $o(i){return document.getElementById(i);}

function updateRecallVal(){const v=parseFloat($o('opt-recall').value);$o('opt-recall-val').textContent=(v*100).toFixed(0)+'%';}
function updateFpVal(){const v=parseFloat($o('opt-fp').value);$o('opt-fp-val').textContent=(v*100).toFixed(1)+'%';}
function updateAlertTuneVals(){
  const recall=parseFloat($o('alert-min-recall').value);
  const far=parseFloat($o('alert-target-far').value);
  $o('alert-min-recall-val').textContent=(recall*100).toFixed(0)+'%';
  $o('alert-target-far-val').textContent=(far*100).toFixed(1)+'%';
}

function setPreset(p){
  if(p==='aggressive'){$o('opt-recall').value=0.10;$o('opt-fp').value=0.02;}
  else if(p==='balanced'){$o('opt-recall').value=0.30;$o('opt-fp').value=0.05;}
  else if(p==='sensitive'){$o('opt-recall').value=0.70;$o('opt-fp').value=0.10;}
  updateRecallVal();updateFpVal();
}

function pct(v){return Math.max(0,Math.min(100,(parseFloat(v)||0)*100));}
function setBar(id,value,label){
  const bar=$o(id),val=$o(id+'-val');
  if(bar)bar.style.width=Math.max(0,Math.min(100,value))+'%';
  if(val)val.textContent=label;
}
function setupCanvas(id){
  const cv=$o(id),dpr=Math.min(window.devicePixelRatio||1,2),w=cv.clientWidth||360,h=cv.clientHeight||178;
  cv.width=w*dpr;cv.height=h*dpr;
  const ctx=cv.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  return {ctx,w,h};
}
function drawEmptyViz(){
  for(const id of ['opt-convergence','opt-params']){
    const {ctx,w,h}=setupCanvas(id);
    ctx.fillStyle='rgba(139,148,158,.55)';
    ctx.font='10px Courier New';
    ctx.textAlign='center';
    ctx.fillText('Run optimization to render visual analysis',w/2,h/2);
  }
}
function setLoadingCards(on){
  document.querySelectorAll('.opt-card').forEach(card=>card.classList.toggle('loading',on));
  document.querySelectorAll('.opt-track').forEach(track=>track.classList.toggle('loading',on));
}
function drawLoadingConvergence(t){
  const {ctx,w,h}=setupCanvas('opt-convergence');
  const pad={l:34,r:12,t:12,b:28},plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
  ctx.strokeStyle='rgba(48,54,61,.62)';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=pad.t+i*plotH/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();}
  const scan=(t/18)%(plotW||1);
  const grad=ctx.createLinearGradient(pad.l+scan-80,0,pad.l+scan+20,0);
  grad.addColorStop(0,'rgba(63,185,80,0)');
  grad.addColorStop(.72,'rgba(63,185,80,.38)');
  grad.addColorStop(1,'rgba(63,185,80,.95)');
  ctx.strokeStyle=grad;ctx.lineWidth=2.2;ctx.beginPath();
  for(let i=0;i<90;i++){
    const p=i/89,x=pad.l+p*plotW;
    const decay=Math.exp(-p*2.2),wave=Math.sin(p*13+t/290)*.08+Math.sin(p*31+t/170)*.035;
    const y=pad.t+plotH*(.20+decay*.62+wave);
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.stroke();
  ctx.fillStyle='rgba(188,140,255,.55)';
  for(let i=0;i<22;i++){
    const p=((i*.073+t/3800)%1),x=pad.l+p*plotW;
    const y=pad.t+plotH*(.25+Math.exp(-p*2.5)*.60+Math.sin(i+t/300)*.06);
    ctx.beginPath();ctx.arc(x,y,1.6,0,Math.PI*2);ctx.fill();
  }
  ctx.fillStyle='rgba(139,148,158,.82)';ctx.font='9px Courier New';ctx.textAlign='left';
  ctx.fillText('sampling candidates',pad.l,12);
  ctx.fillStyle='rgba(63,185,80,.95)';ctx.fillText('best path forming',pad.l+100,12);
}
function drawLoadingParams(t){
  const {ctx,w,h}=setupCanvas('opt-params');
  const cx=w*.36,cy=h*.46,r=Math.min(w,h)*.28,cols=['#388BFD','#3FB950','#BC8CFF','#D29922'];
  const vals=Array.from({length:4},(_,i)=>Math.max(.10,.18+.10*Math.sin(t/520+i*1.7)+.07*Math.cos(t/740+i)));
  const total=vals.reduce((sum,v)=>sum+v,0)||1;
  let a=-Math.PI/2+t/900;
  for(let i=0;i<4;i++){
    const next=a+(Math.PI*2)*(vals[i]/total);
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,a,next);ctx.closePath();
    ctx.fillStyle=cols[i];ctx.globalAlpha=.76;ctx.fill();ctx.globalAlpha=1;
    a=next;
  }
  ctx.strokeStyle='rgba(63,185,80,.70)';ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(cx,cy,r+7,0,t/500%(Math.PI*2));ctx.stroke();
  ctx.fillStyle='rgba(13,17,23,.88)';ctx.beginPath();ctx.arc(cx,cy,r*.50,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='rgba(230,237,243,.95)';ctx.font='bold 11px Courier New';ctx.textAlign='center';ctx.fillText('SEARCH',cx,cy+4);
  ctx.font='9px Courier New';ctx.textAlign='left';
  ['w1 gradient','w2 momentum','w3 nis','w4 duration'].forEach((label,i)=>{
    const y=18+i*18,pulse=.42+.34*Math.sin(t/360+i);
    ctx.fillStyle=cols[i];ctx.fillRect(w*.61,y-8,Math.max(10,52*pulse),5);
    ctx.fillStyle='rgba(230,237,243,.82)';ctx.fillText(label,w*.61+60,y-3);
  });
  ctx.fillStyle='rgba(139,148,158,.82)';ctx.font='8px Courier New';ctx.textAlign='center';
  ctx.fillText('testing thresholds and weight combinations',w/2,h-12);
}
function drawLoadingViz(ts=performance.now()){
  if(!optLoading)return;
  const elapsed=(Date.now()-optStart)/1000;
  const progress=Math.min(92,10+elapsed*9);
  const pseudoEval=Math.min(optIterTarget,Math.max(1,Math.floor(progress/92*optIterTarget)));
  $o('opt-progress-bar').style.width=progress.toFixed(1)+'%';
  $o('opt-progress-iter').textContent=`searching ${pseudoEval} / ${optIterTarget} iterations`;
  $o('opt-progress-time').textContent=elapsed.toFixed(1)+'s';
  $o('viz-best-loss').textContent='best searching';
  $o('viz-evals').textContent=pseudoEval;
  $o('viz-f1').textContent='sampling';
  $o('viz-far').textContent='sampling';
  $o('viz-window').textContent='window tuning';
  $o('viz-profile').textContent='optimizing';
  setBar('bar-acc',35+25*Math.sin(ts/520)**2,'...');
  setBar('bar-prec',28+30*Math.sin(ts/650+1)**2,'...');
  setBar('bar-rec',20+25*Math.sin(ts/720+2)**2,'...');
  setBar('bar-far',55+28*Math.sin(ts/600+3)**2,'...');
  drawLoadingConvergence(ts);
  drawLoadingParams(ts);
  optAnim=requestAnimationFrame(drawLoadingViz);
}
function startLoadingViz(iter){
  optIterTarget=iter||50;
  optLoading=true;
  lastOptData=null;
  setLoadingCards(true);
  if(optAnim)cancelAnimationFrame(optAnim);
  optAnim=requestAnimationFrame(drawLoadingViz);
}
function stopLoadingViz(){
  optLoading=false;
  setLoadingCards(false);
  if(optAnim)cancelAnimationFrame(optAnim);
  optAnim=null;
}
function setParamPlaceholders(txt){
  ['p-w1','p-w2','p-w3','p-w4','p-theta','p-alpha','p-beta','p-hyst','p-persist','p-window'].forEach(id=>$o(id).textContent=txt);
}
function setThinkingState(label,running=false){
  const card=$o('think-card'),state=$o('think-state');
  if(card)card.classList.toggle('running',running);
  if(state)state.textContent=label;
}
function setThinkingLines(lines){
  const stream=$o('think-stream');
  if(!stream)return;
  stream.innerHTML=lines.slice(-8).map((item,idx)=>{
    const text=typeof item==='string'?item:item.text;
    const cls=typeof item==='string'?'':(item.cls||'');
    return `<div class="think-line ${idx===lines.slice(-8).length-1?'active ':''}${cls}">${text}</div>`;
  }).join('');
}
function startThinkingTrace(iter,recall,fp){
  if(optThinkTimer)clearInterval(optThinkTimer);
  optThinkIndex=0;
  setThinkingState('thinking',true);
  const lines=[`Reading targets: recall <= ${(recall*100).toFixed(0)}%, FAR <= ${(fp*100).toFixed(1)}%.`];
  setThinkingLines(lines);
  const steps=[
    'Preparing bounded search over w1, w2, w3, w4, theta, alpha, beta, and window.',
    'Testing candidate weights while keeping total weight near 1.0.',
    'Running the GMS model and scoring every node across the full timeline.',
    'Comparing predictions against event ground truth to count TP, FP, FN, and TN.',
    'Penalizing candidates that create false alarms outside event windows.',
    'Checking whether recall stays within the selected mode target.',
    'Keeping the lowest-loss candidate as the current best parameter set.',
    'Refining thresholds so moderate and high instability stay separated.'
  ];
  optThinkTimer=setInterval(()=>{
    const step=steps[optThinkIndex%steps.length];
    lines.push(`Search check: ${step}`);
    setThinkingLines(lines);
    optThinkIndex++;
  },1250);
}
function stopThinkingTrace(label='idle'){
  if(optThinkTimer)clearInterval(optThinkTimer);
  optThinkTimer=null;
  setThinkingState(label,false);
}
function summarizeThinking(data){
  const p=data.best_params,m=data.metrics;
  const weights=[['Gradient',p.w1],['Momentum',p.w2],['NIS',p.w3],['Duration',p.w4]].sort((a,b)=>b[1]-a[1]);
  const farOk=m.far<=parseFloat($o('opt-fp').value);
  const recallOk=m.recall<=parseFloat($o('opt-recall').value);
  stopThinkingTrace('complete');
  setThinkingLines([
    {text:`Best loss reached ${data.best_loss.toFixed(4)} after ${data.iterations_evaluated} evaluations.`,cls:'good'},
    `Main signal emphasis: ${weights[0][0]} (${(weights[0][1]*100).toFixed(0)}%) and ${weights[1][0]} (${(weights[1][1]*100).toFixed(0)}%).`,
    `Threshold choice: alpha ${p.alpha.toFixed(3)} for first alert, beta ${p.beta.toFixed(3)} for high-risk alert.`,
    `Persistence window selected: ${p.window} timesteps for filtering short noise spikes.`,
    {text:`Final metrics: accuracy ${(m.accuracy*100).toFixed(1)}%, F1 ${(m.f1*100).toFixed(1)}%, FAR ${(m.far*100).toFixed(2)}%.`,cls:farOk?'good':'warn'},
    {text:`Mode check: recall target ${recallOk?'met':'exceeded'}, false-alarm target ${farOk?'met':'exceeded'}.`,cls:(recallOk&&farOk)?'good':'warn'}
  ]);
}
function drawConvergence(history){
  const {ctx,w,h}=setupCanvas('opt-convergence');
  if(!history||!history.length)return;
  const pad={l:38,r:28,t:30,b:34},plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
  const last=history[history.length-1];
  const firstBest=Number(history[0].best_loss),finalBest=Number(last.best_loss);
  const bestVals=history.map(x=>Number(x.best_loss)).filter(Number.isFinite);
  const rawVals=history.map(x=>Number(x.loss)).filter(Number.isFinite);
  const maxLoss=Math.max(...rawVals),minBest=Math.min(...bestVals);
  const span=(firstBest-minBest)||1;
  const xAt=i=>pad.l+(history[i].eval-1)/(last.eval-1||1)*plotW;
  const yAtBest=v=>pad.t+(1-(firstBest-v)/span)*plotH;
  const yAtRaw=v=>{
    const clamped=Math.max(minBest,Math.min(maxLoss,v));
    return pad.t+(1-(maxLoss-clamped)/(maxLoss-minBest||1))*plotH;
  };

  ctx.fillStyle='rgba(230,237,243,.92)';
  ctx.font='bold 10px Courier New';
  ctx.textAlign='left';
  ctx.fillText('Did search improve?',pad.l,14);
  ctx.fillStyle='rgba(139,148,158,.8)';
  ctx.font='9px Courier New';
  ctx.fillText('green line going down = better parameters found',pad.l,27);

  ctx.strokeStyle='rgba(48,54,61,.65)';
  ctx.lineWidth=1;
  for(let i=0;i<=3;i++){const y=pad.t+i*plotH/3;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();}

  ctx.fillStyle='rgba(188,140,255,.18)';
  history.forEach((p,i)=>{
    if(i%Math.ceil(history.length/120)!==0)return;
    ctx.beginPath();ctx.arc(xAt(i),yAtRaw(Number(p.loss)),1.5,0,Math.PI*2);ctx.fill();
  });

  ctx.strokeStyle='rgba(63,185,80,.95)';
  ctx.lineWidth=3;
  ctx.beginPath();
  history.forEach((p,i)=>{
    const x=xAt(i),y=yAtBest(Number(p.best_loss));
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  });
  ctx.stroke();

  const startY=yAtBest(firstBest),endY=yAtBest(finalBest),endX=xAt(history.length-1);
  ctx.fillStyle='rgba(56,139,253,.95)';
  ctx.beginPath();ctx.arc(pad.l,startY,4,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='rgba(63,185,80,.95)';
  ctx.beginPath();ctx.arc(endX,endY,5,0,Math.PI*2);ctx.fill();

  const improvement=firstBest-finalBest;
  ctx.fillStyle='rgba(230,237,243,.92)';
  ctx.font='9px Courier New';
  ctx.textAlign='left';
  ctx.fillText(`start ${firstBest.toFixed(4)}`,pad.l+8,startY-6);
  ctx.textAlign='right';
  ctx.fillText(`best ${finalBest.toFixed(4)}`,w-pad.r,endY-8);
  ctx.fillStyle=improvement>0?'rgba(63,185,80,.95)':'rgba(210,153,34,.95)';
  ctx.fillText(improvement>0?`improved by ${improvement.toFixed(4)}`:'no later improvement',w-pad.r,16);

  ctx.fillStyle='rgba(139,148,158,.8)';
  ctx.textAlign='center';
  ctx.fillText('search start',pad.l,pad.t+plotH+20);
  ctx.fillText(`${last.eval} tests`,w-pad.r,pad.t+plotH+20);
  ctx.textAlign='left';
  ctx.fillStyle='rgba(188,140,255,.45)';
  ctx.beginPath();ctx.arc(pad.l,pad.t+plotH+31,2,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='rgba(139,148,158,.85)';
  ctx.fillText('tested candidates',pad.l+8,pad.t+plotH+34);
}
function drawParams(p){
  const {ctx,w,h}=setupCanvas('opt-params');
  if(!p)return;
  const weights=[['w1',p.w1,'#388BFD'],['w2',p.w2,'#3FB950'],['w3',p.w3,'#BC8CFF'],['w4',p.w4,'#D29922']];
  const rawTotal=weights.reduce((sum,[,val])=>sum+Math.max(0,Number(val)||0),0);
  const gap=1-rawTotal;
  const closeToZero=Math.abs(gap)<=0.02;
  const slices=weights.map(([name,val,col])=>[name,Math.max(0,Number(val)||0),col]);
  if(gap>0.02)slices.push(['gap',gap,'rgba(139,148,158,.30)']);
  const total=closeToZero ? (rawTotal||1) : Math.max(1,rawTotal);
  const cx=w*.36,cy=h*.46,r=Math.min(w,h)*.26;
  let a=-Math.PI/2;
  for(const [name,val,col] of slices){
    const next=a+(val/total)*Math.PI*2;
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,a,next);ctx.closePath();
    ctx.fillStyle=col;ctx.globalAlpha=.86;ctx.fill();ctx.globalAlpha=1;
    a=next;
  }
  ctx.fillStyle='rgba(13,17,23,.88)';ctx.beginPath();ctx.arc(cx,cy,r*.48,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='rgba(230,237,243,.95)';ctx.font='bold 12px Courier New';ctx.textAlign='center';ctx.fillText('W',cx,cy+4);
  ctx.font='9px Courier New';ctx.textAlign='left';
  weights.forEach(([name,val,col],i)=>{const y=18+i*18;ctx.fillStyle=col;ctx.fillRect(w*.64,y-8,10,5);ctx.fillStyle='rgba(230,237,243,.85)';ctx.fillText(`${name} ${(Math.max(0,val)*100).toFixed(0)}%`,w*.64+16,y-3);});
  if(gap>0.02){
    const y=90;
    ctx.fillStyle='rgba(139,148,158,.40)';ctx.fillRect(w*.64,y-8,10,5);
    ctx.fillStyle='rgba(139,148,158,.85)';ctx.fillText(`gap ${(gap*100).toFixed(0)}%`,w*.64+16,y-3);
  }else if(rawTotal>1.02){
    ctx.fillStyle='rgba(210,153,34,.9)';
    ctx.fillText(`sum ${(rawTotal*100).toFixed(0)}%`,w*.64+16,87);
  }
  const bars=[['theta',Math.min(1,p.theta/2),'#3FB950',p.theta.toFixed(2)],['alpha',Math.min(1,p.alpha/.5),'#D29922',p.alpha.toFixed(2)],['beta',Math.min(1,p.beta),'#F85149',p.beta.toFixed(2)]];
  bars.forEach(([name,val,col,label],i)=>{const x=18,y=h-42+i*13,bw=w-36;ctx.fillStyle='rgba(48,54,61,.72)';ctx.fillRect(x,y,bw,6);ctx.fillStyle=col;ctx.fillRect(x,y,bw*val,6);ctx.fillStyle='rgba(139,148,158,.9)';ctx.font='8px Courier New';ctx.textAlign='left';ctx.fillText(name,x,y-2);ctx.textAlign='right';ctx.fillText(label,x+bw,y-2);});
}
function updateVisuals(data){
  lastOptData=data;
  const m=data.metrics,p=data.best_params,h=data.history||[];
  drawConvergence(h);
  drawParams(p);
  setBar('bar-acc',pct(m.accuracy),(m.accuracy*100).toFixed(1)+'%');
  setBar('bar-prec',pct(m.precision),(m.precision*100).toFixed(1)+'%');
  setBar('bar-rec',pct(m.recall),(m.recall*100).toFixed(1)+'%');
  setBar('bar-far',100-pct(m.far),(100-m.far*100).toFixed(1)+'% quiet');
  $o('viz-best-loss').textContent='best '+data.best_loss.toFixed(4);
  $o('viz-evals').textContent=data.iterations_evaluated;
  $o('viz-f1').textContent=(m.f1*100).toFixed(1)+'%';
  $o('viz-far').textContent=(m.far*100).toFixed(2)+'%';
  $o('viz-window').textContent='window '+p.window;
  $o('viz-profile').textContent='complete';
}
async function runOptimize(){
  if(optRunning)return;
  optRunning=true;optStart=Date.now();
  $o('btn-opt-start').hidden=true;
  $o('btn-opt-cancel').hidden=false;
  $o('opt-status').className='opt-status running';
  $o('opt-status').textContent='Optimization running...';
  $o('opt-results-card').hidden=true;
  $o('opt-progress-bar').style.width='0%';
  $o('viz-profile').textContent='running';
  setParamPlaceholders('...');
  const recall=parseFloat($o('opt-recall').value);
  const fp=parseFloat($o('opt-fp').value);
  const iter=parseInt($o('opt-iter').value);
  const seed=parseInt($o('opt-seed').value);
  startLoadingViz(iter);
  startThinkingTrace(iter,recall,fp);
  try{
    const resp=await fetch('/api/optimize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({iterations:iter,seed:seed,target_recall:recall,target_fp_rate:fp})});
    const data=await resp.json();
    if(data.success){
      stopLoadingViz();
      const m=data.metrics;
      $o('res-acc').textContent=(m.accuracy*100).toFixed(1)+'%';
      $o('res-prec').textContent=(m.precision*100).toFixed(1)+'%';
      $o('res-recall').className='opt-result-value'+(m.recall<0.15?' good':' bad');
      $o('res-recall').textContent=(m.recall*100).toFixed(1)+'%';
      $o('res-far').className='opt-result-value'+(m.far<0.05?' good':' bad');
      $o('res-far').textContent=(m.far*100).toFixed(2)+'%';
      const p=data.best_params;
      $o('p-w1').textContent=p.w1.toFixed(4);
      $o('p-w2').textContent=p.w2.toFixed(4);
      $o('p-w3').textContent=p.w3.toFixed(4);
      $o('p-w4').textContent=p.w4.toFixed(4);
      $o('p-theta').textContent=p.theta.toFixed(4);
      $o('p-alpha').textContent=p.alpha.toFixed(4);
      $o('p-beta').textContent=p.beta.toFixed(4);
      $o('p-hyst').textContent=p.hysteresis_margin!==undefined?p.hysteresis_margin.toFixed(4):'-';
      $o('p-persist').textContent=p.alert_persistence_frames||'-';
      $o('p-window').textContent=p.window;
      updateVisuals(data);
      summarizeThinking(data);
      $o('opt-results-card').hidden=false;
      $o('opt-status').className='opt-status done';
      $o('opt-status').textContent='Optimization complete!';
    }else{
      stopLoadingViz();
      stopThinkingTrace('error');
      setParamPlaceholders('-');
      $o('opt-status').className='opt-status';
      $o('opt-status').textContent='Error: '+data.error;
    }
  }catch(e){
    stopLoadingViz();
    stopThinkingTrace('error');
    setParamPlaceholders('-');
    $o('opt-status').textContent='Error: '+e.message;
  }
  optRunning=false;
  $o('btn-opt-start').hidden=false;
  $o('btn-opt-cancel').hidden=true;
  $o('opt-progress-bar').style.width='100%';
}
function cancelOptimize(){
  optRunning=false;
  stopLoadingViz();
  stopThinkingTrace('cancelled');
  setThinkingLines([{text:'Optimization cancelled before a final parameter set was selected.',cls:'warn'}]);
  $o('btn-opt-start').hidden=false;
  $o('btn-opt-cancel').hidden=true;
  $o('opt-status').textContent='Cancelled';
}
async function applyOptimizedParams(){
  const fields=[
    ['w1','p-w1'],['w2','p-w2'],['w3','p-w3'],['w4','p-w4'],
    ['theta','p-theta'],['alpha','p-alpha'],['beta','p-beta'],
    ['hysteresis_margin','p-hyst']
  ];
  const p={};
  fields.forEach(([key,id])=>{
    const value=parseFloat($o(id).textContent);
    if(Number.isFinite(value))p[key]=value;
  });
  const windowVal=parseInt($o('p-window').textContent);
  const persist=parseInt($o('p-persist').textContent);
  if(Number.isFinite(windowVal))p.window=windowVal;
  if(Number.isFinite(persist))p.alert_persistence_frames=persist;
  if(!Object.keys(p).length){
    alert('No optimized parameters available yet.');
    return;
  }
  try{
    const alertOnly=!['w1','w2','w3','w4','theta','window'].some(key=>Number.isFinite(p[key]));
    const resp=await fetch('/api/apply_optimized_params',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({params:p})});
    const data=await resp.json();
    if(data.success){
      alert(alertOnly?'Alert policy applied. Opening Alerts module.':'Parameters applied to engine. Go to Dashboard to see results.');
      window.location=alertOnly?'/alerts':'/';
    }
    else alert('Error: '+data.error);
  }catch(e){alert('Error: '+e.message);}
}

function setAlertTuneResults(data){
  lastAlertTune=data;
  const best=data.best,params=best.params,m=best.metrics;
  $o('p-alpha').textContent=params.alpha.toFixed(4);
  $o('p-beta').textContent=params.beta.toFixed(4);
  $o('p-hyst').textContent=params.hysteresis_margin.toFixed(4);
  $o('p-persist').textContent=params.alert_persistence_frames;
  $o('alert-ai-summary').innerHTML=
    `Best alert policy after ${data.evaluations} tests:<br>`+
    `Precision ${(m.precision*100).toFixed(1)}%, recall ${(m.recall*100).toFixed(1)}%, `+
    `FAR ${(m.far*100).toFixed(2)}%, F1 ${(m.f1*100).toFixed(1)}%.<br>`+
    `Use alpha ${params.alpha.toFixed(3)}, beta ${params.beta.toFixed(3)}, `+
    `margin ${params.hysteresis_margin.toFixed(3)}, persistence ${params.alert_persistence_frames} frame(s).`;
  $o('btn-alert-apply').hidden=false;
  setThinkingLines([
    {text:`AI alert tuner tested ${data.evaluations} policies against event ground truth.`,cls:'good'},
    `Best policy: alpha ${params.alpha.toFixed(3)}, beta ${params.beta.toFixed(3)}, margin ${params.hysteresis_margin.toFixed(3)}.`,
    `False alarms: ${m.fp}, true positives: ${m.tp}, false negatives: ${m.fn}.`,
    {text:`FAR ${(m.far*100).toFixed(2)}%, precision ${(m.precision*100).toFixed(1)}%, recall ${(m.recall*100).toFixed(1)}%.`,cls:m.far<=parseFloat($o('alert-target-far').value)?'good':'warn'}
  ]);
}

async function runAlertTune(){
  if(optRunning)return;
  optRunning=true;
  $o('btn-alert-tune').textContent='TUNING...';
  $o('btn-alert-tune').disabled=true;
  $o('btn-alert-apply').hidden=true;
  setThinkingState('alert tuning',true);
  setThinkingLines(['AI alert tuner is testing thresholds, hysteresis margin, and persistence frames.']);
  try{
    const body={
      iterations:450,
      seed:parseInt($o('opt-seed').value)||42,
      min_recall:parseFloat($o('alert-min-recall').value),
      target_far:parseFloat($o('alert-target-far').value),
      apply:false
    };
    const resp=await fetch('/api/alerts/ai_tune',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await resp.json();
    if(data.success){
      setAlertTuneResults(data);
      setThinkingState('complete',false);
    }else{
      setThinkingState('error',false);
      $o('alert-ai-summary').textContent='AI alert tuning failed.';
    }
  }catch(e){
    setThinkingState('error',false);
    $o('alert-ai-summary').textContent='AI alert tuning error: '+e.message;
  }
  optRunning=false;
  $o('btn-alert-tune').disabled=false;
  $o('btn-alert-tune').textContent='TUNE ALERT POLICY';
}

async function applyAlertTune(){
  if(!lastAlertTune)return;
  const params=lastAlertTune.best.params;
  const resp=await fetch('/api/apply_optimized_params',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({params})});
  const data=await resp.json();
  if(data.success){
    const policy=data.alert_policy || {};
    $o('alert-ai-summary').innerHTML+=`<br><strong>Applied to live alert module.</strong> High enters at ${policy.high_enter}, releases at ${policy.high_release}.`;
    $o('btn-alert-apply').hidden=true;
    setTimeout(()=>{window.location='/alerts';},700);
  }else{
    alert('Error: '+data.error);
  }
}

document.addEventListener('DOMContentLoaded',()=>{
  $o('opt-recall').addEventListener('input',updateRecallVal);
  $o('opt-fp').addEventListener('input',updateFpVal);
  $o('alert-min-recall').addEventListener('input',updateAlertTuneVals);
  $o('alert-target-far').addEventListener('input',updateAlertTuneVals);
  updateRecallVal();
  updateFpVal();
  updateAlertTuneVals();
  drawEmptyViz();
});
window.addEventListener('resize',()=>{lastOptData?updateVisuals(lastOptData):drawEmptyViz();});
