const N_NODES=40,T=120,G=10;
let state=null,sn=0,playing=false,noiseOn=false,pend={},ptmr=null;
function $(i){return document.getElementById(i);}
function frameTimeLabel(s,t){
  const idx=Math.max(0,Math.round(Number(t)||0));
  if(s?.time_axis?.[idx])return s.time_axis[idx];
  if(s?.time_label&&idx===s.t)return s.time_label;
  const total=Math.max(1,Number(s?.T)||T);
  const mins=Math.round(idx/total*24*60)%(24*60);
  return String(Math.floor(mins/60)).padStart(2,'0')+':'+String(mins%60).padStart(2,'0');
}
function timeRangeLabel(s,start,end){return `${frameTimeLabel(s,start)}-${frameTimeLabel(s,end)}`;}

// Charts
const lbs=Array.from({length:T},(_,i)=>i);
let gmsC=null,tmpC=null;
if(window.Chart){
  const co={animation:{duration:100},plugins:{legend:{display:false},tooltip:{backgroundColor:'#1C2128',titleColor:'#E6EDF3',bodyColor:'#8B949E',borderColor:'#30363D',borderWidth:1}},scales:{x:{ticks:{color:'#484F58',maxTicksLimit:10,font:{size:7}},grid:{color:'rgba(48,54,61,.4)'},border:{color:'#30363D'}},y:{ticks:{color:'#484F58',font:{size:7}},grid:{color:'rgba(48,54,61,.4)'},border:{color:'#30363D'}}},elements:{point:{radius:0,hoverRadius:4}},responsive:true,maintainAspectRatio:false};
  gmsC=new Chart($('gmsChart').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[
    {data:Array(T).fill(0),borderColor:'#388BFD',backgroundColor:'rgba(56,139,253,.12)',fill:true,borderWidth:2},
    {data:Array(T).fill(0),borderColor:'rgba(139,148,158,.2)',borderWidth:1,fill:false},
    {data:Array(T).fill(.25),borderColor:'rgba(210,153,34,.6)',borderWidth:1,borderDash:[4,4],fill:false},
    {data:Array(T).fill(.60),borderColor:'rgba(248,81,73,.6)',borderWidth:1,borderDash:[4,4],fill:false},
  ]},options:{...co,scales:{...co.scales,y:{...co.scales.y,min:0,max:1.05}}}});
  tmpC=new Chart($('tmpChart').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[
    {data:Array(T).fill(22),borderColor:'#3FB950',backgroundColor:'rgba(63,185,80,.08)',fill:true,borderWidth:2},
    {data:Array(T).fill(26.5),borderColor:'rgba(248,81,73,.6)',borderWidth:1,borderDash:[4,4],fill:false},
  ]},options:co});
}else{
  console.warn('Chart.js did not load; dashboard charts are disabled.');
}

// Canvas map
const canvas=$('sc'),ctx=canvas.getContext('2d');
function rsz(){const w=$('mw');canvas.width=w.clientWidth;canvas.height=w.clientHeight;}
window.addEventListener('resize',()=>{rsz();if(state)dMap(state);});rsz();

function nc(l){return l===2?'#F85149':l===1?'#D29922':'#388BFD';}
function hexToRgba(hex,a){
  const n=parseInt(hex.replace('#',''),16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}
function terrainHeight(x,y){return .45+.28*Math.sin(x*.82+y*.36)+.16*Math.sin(y*1.7)-.11*Math.cos((x-y)*1.05);}
function riverY(x){return 5+1.08*Math.sin(x*.75)+.36*Math.sin(x*1.9);}
function isoScale(){return Math.min(canvas.width/19,canvas.height/10.8);}
function ts(x,y,z=0){
  const sc=isoScale(),tx=sc*.96,ty=sc*.48,cx=canvas.width/2,cy=canvas.height*.51;
  return [cx+(x-y)*tx,cy+(x+y-G)*ty-z*sc*.82];
}
function invIso(sx,sy){
  const sc=isoScale(),tx=sc*.96,ty=sc*.48,cx=canvas.width/2,cy=canvas.height*.51;
  const dx=sx-cx,dy=sy-cy;
  return [(dx/tx+dy/ty+G)/2,(dy/ty-dx/tx+G)/2];
}
function heatAt(s,x,y){
  let v=0;
  for(const n of s.nodes){
    const d=Math.hypot(n.x-x,n.y-y);
    v+=n.gms*Math.exp(-(d*d)/2.7);
  }
  return Math.min(1,v/1.78);
}
function heatCol(v,a=1){
  if(v<.08)return `rgba(56,139,253,${.04*a})`;
  if(v<.5)return `rgba(${Math.round(42+150*v*2)},${Math.round(104+64*v*2)},${Math.round(75-22*v*2)},${(.10+v*.32)*a})`;
  return `rgba(${Math.round(210+38*(v-.5)*2)},${Math.round(153-72*(v-.5)*2)},${Math.round(34+39*(v-.5)*2)},${(.24+v*.42)*a})`;
}
function atmosphere(t=0){
  const p=(t%T)/(T-1);
  if(p<.22)return {name:'Dawn',top:'#0b1a2a',bot:'#23475f',sun:'#ffd18c',moon:0};
  if(p<.58)return {name:'Day',top:'#102b4a',bot:'#285f67',sun:'#ffd889',moon:.18};
  if(p<.78)return {name:'Dusk',top:'#160f27',bot:'#5d3c40',sun:'#ffb071',moon:.25};
  return {name:'Night',top:'#030711',bot:'#0b1c2b',sun:'#5f4b35',moon:.82};
}
function drawSky(s){
  const w=canvas.width,h=canvas.height,atm=atmosphere(s.t);
  const bg=ctx.createLinearGradient(0,0,0,h);
  bg.addColorStop(0,atm.top);bg.addColorStop(.58,'#11304d');bg.addColorStop(1,atm.bot);
  ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
  const p=(s.t%T)/(T-1),sunX=w*(.18+p*.58),sunY=h*(.23-.13*Math.sin(p*Math.PI));
  const sunGlow=ctx.createRadialGradient(sunX,sunY,0,sunX,sunY,90);
  sunGlow.addColorStop(0,'rgba(255,211,139,.38)');sunGlow.addColorStop(1,'rgba(255,211,139,0)');
  ctx.fillStyle=sunGlow;ctx.beginPath();ctx.arc(sunX,sunY,90,0,Math.PI*2);ctx.fill();
  ctx.fillStyle=atm.sun;ctx.beginPath();ctx.arc(sunX,sunY,11,0,Math.PI*2);ctx.fill();
  if(atm.moon>.05){
    const mx=w*.80,my=h*.28,r=13;
    ctx.fillStyle=`rgba(200,218,255,${atm.moon})`;ctx.beginPath();ctx.arc(mx,my,r,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='rgba(5,10,19,.55)';ctx.beginPath();ctx.arc(mx+7,my-1,r*.85,0,Math.PI*2);ctx.fill();
  }
  ctx.fillStyle='rgba(255,220,130,.22)';
  for(let i=0;i<70;i++){const x=(i*97%w),y=(i*53%(h*.48))+34;ctx.fillRect(x,y,1,1);}
}
function drawTile(x,y,s,step){
  const h=terrainHeight(x+step/2,y+step/2),heat=heatAt(s,x+step/2,y+step/2),water=Math.abs((y+step/2)-riverY(x+step/2))<.42;
  let fill=water?`rgba(${32+heat*80},${95+heat*30},${119+heat*28},.82)`:heatCol(heat,.9);
  if(!water&&heat<.10)fill='rgba(42,93,52,.72)';
  const p1=ts(x,y,h),p2=ts(x+step,y,h),p3=ts(x+step,y+step,h),p4=ts(x,y+step,h);
  ctx.beginPath();ctx.moveTo(...p1);ctx.lineTo(...p2);ctx.lineTo(...p3);ctx.lineTo(...p4);ctx.closePath();
  ctx.fillStyle=fill;ctx.fill();
  ctx.strokeStyle=water?'rgba(77,174,216,.28)':'rgba(56,139,253,.18)';
  ctx.lineWidth=.55;ctx.stroke();
}
function drawTerrain(s){
  for(let y=0;y<G;y+=.5){for(let x=0;x<G;x+=.5)drawTile(x,y,s,.5);}
}
function drawEventZones(s){
  const active=new Set(s.active_events||[]);
  for(const ev of s.events||[]){
    const nodes=ev.nodes.map(i=>s.nodes[i]).filter(Boolean);
    if(!nodes.length)continue;
    const cx=nodes.reduce((a,n)=>a+n.x,0)/nodes.length,cy=nodes.reduce((a,n)=>a+n.y,0)/nodes.length;
    const activeNow=active.has(ev.label),pulse=.5+.5*Math.sin(performance.now()/360+cx);
    const [sx,sy]=ts(cx,cy,terrainHeight(cx,cy)+.08),sc=isoScale();
    const r=Math.max(1.15,...nodes.map(n=>Math.hypot(n.x-cx,n.y-cy)+.55))*sc;
    ctx.save();ctx.translate(sx,sy);ctx.scale(1,.48);
    ctx.fillStyle=activeNow?hexToRgba(ev.color,.10+.05*pulse):'rgba(139,148,158,.035)';
    ctx.strokeStyle=activeNow?ev.color:'rgba(139,148,158,.18)';
    ctx.lineWidth=activeNow?2.2:1;
    ctx.beginPath();ctx.arc(0,0,r,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.restore();
    if(activeNow){ctx.fillStyle='rgba(230,237,243,.9)';ctx.font='bold 9px Courier New';ctx.textAlign='center';ctx.fillText(ev.label,sx,sy-r*.50-7);}
  }
}
function drawEdges(s){
  for(let i=0;i<N_NODES;i++){
    for(const j of s.adj[i]||[]){
      if(j<=i)continue;
      const a=s.nodes[i],b=s.nodes[j],ha=terrainHeight(a.x,a.y),hb=terrainHeight(b.x,b.y);
      const [x1,y1]=ts(a.x,a.y,ha+.74),[x2,y2]=ts(b.x,b.y,hb+.74);
      const hot=Math.min(1,(a.gms+b.gms)/1.4);
      ctx.strokeStyle=`rgba(${Math.round(56+154*hot)},${Math.round(139+14*hot)},${Math.round(253-219*hot)},${.14+hot*.34})`;
      ctx.lineWidth=.55+hot*1.25;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
    }
  }
  for(const pe of s.prop_edges||[]){
    const a=s.nodes[pe.src],b=s.nodes[pe.dst],[x1,y1]=ts(a.x,a.y,terrainHeight(a.x,a.y)+.88),[x2,y2]=ts(b.x,b.y,terrainHeight(b.x,b.y)+.88);
    ctx.strokeStyle=`rgba(210,153,34,${.28+pe.strength*.55})`;ctx.lineWidth=1.3+pe.strength*2;
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  }
}
function drawTower(nd){
  const h=terrainHeight(nd.x,nd.y),[bx,by]=ts(nd.x,nd.y,h),height=18+nd.gms*38,col=nc(nd.label),topY=by-height;
  if(nd.label>0){
    const plume=ctx.createRadialGradient(bx,topY,0,bx,topY,34+nd.gms*48);
    const rgb=nd.label===2?'248,81,73':'210,153,34';
    plume.addColorStop(0,`rgba(${rgb},${.22+nd.gms*.28})`);plume.addColorStop(1,`rgba(${rgb},0)`);
    ctx.fillStyle=plume;ctx.beginPath();ctx.arc(bx,topY,34+nd.gms*48,0,Math.PI*2);ctx.fill();
    ctx.fillStyle=`rgba(${rgb},.22)`;
    ctx.beginPath();ctx.moveTo(bx-24-nd.gms*24,by+10);ctx.lineTo(bx,topY-58*nd.gms);ctx.lineTo(bx+24+nd.gms*24,by+10);ctx.closePath();ctx.fill();
  }
  ctx.strokeStyle='rgba(230,237,243,.55)';ctx.lineWidth=1.1;
  ctx.beginPath();ctx.moveTo(bx-5,by);ctx.lineTo(bx,topY);ctx.lineTo(bx+5,by);ctx.stroke();
  const pulse=nd.label===2?.7+.3*Math.sin(performance.now()/120+nd.id):1;
  const r=7+nd.gms*8+(nd.label===2?3*pulse:0);
  if(nd.id===sn){ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.arc(bx,topY,r+6,0,Math.PI*2);ctx.stroke();}
  ctx.fillStyle=col;ctx.beginPath();ctx.arc(bx,topY,r,0,Math.PI*2);ctx.fill();
  ctx.strokeStyle='rgba(255,255,255,.72)';ctx.lineWidth=1;ctx.stroke();
  ctx.fillStyle='rgba(255,255,255,.92)';ctx.font=`bold ${8+nd.gms*2}px Courier New`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(nd.id,bx,topY);
}
function dMap(s){
  const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);drawSky(s);drawTerrain(s);drawEventZones(s);drawEdges(s);
  const sorted=[...s.nodes].sort((a,b)=>(a.x+a.y)-(b.x+b.y));
  for(const nd of sorted)drawTower(nd);
  if(s.high_count>0){ctx.strokeStyle='rgba(248,81,73,.50)';ctx.lineWidth=5;ctx.strokeRect(3,3,w-6,h-6);}
  ctx.fillStyle='rgba(139,148,158,.55)';ctx.font='9px Courier New';ctx.textAlign='left';ctx.textBaseline='top';
  ctx.fillText(`isometric GMS world | N=${N_NODES} | ${s.noise_on?'NOISE ON':'CLEAN DATA'}`,10,8);
}

// Heatmap
const hcv=$('hc'),hx=hcv.getContext('2d');
const riskCv=$('riskTimeline'),riskCtx=riskCv.getContext('2d');
const profileCv=$('profileSpark'),profileCtx=profileCv.getContext('2d');
const healthCv=$('healthRing'),healthCtx=healthCv.getContext('2d');
function dHeat(gf,t){
  hcv.width=hcv.clientWidth||240;hcv.height=hcv.clientHeight||52;const w=hcv.width,h=hcv.height,cw=w/T,ch=h/N_NODES;hx.clearRect(0,0,w,h);
  for(let i=0;i<N_NODES;i++){for(let t2=0;t2<T;t2++){const v=gf[i][t2];hx.fillStyle=v<.3?`rgba(56,139,253,${.12+v*2})`:v<.6?`rgb(${Math.round(56+(210-56)*(v-.3)/.3)},${Math.round(139+(153-139)*(v-.3)/.3)},${Math.round(253+(34-253)*(v-.3)/.3)})`:(`rgb(${Math.round(210+(248-210)*(v-.6)/.4)},${Math.round(153+(81-153)*(v-.6)/.4)},${Math.round(34+(73-34)*(v-.6)/.4)})`);hx.fillRect(t2*cw,(N_NODES-1-i)*ch,Math.ceil(cw)+1,Math.ceil(ch)+1);}}
  hx.strokeStyle='rgba(255,255,255,.8)';hx.lineWidth=1.5;hx.beginPath();hx.moveTo(t*cw,0);hx.lineTo(t*cw,h);hx.stroke();
  hx.strokeStyle='rgba(56,139,253,.6)';hx.lineWidth=1;hx.strokeRect(0,(N_NODES-1-sn)*ch,w,ch);
}
function dRiskTimeline(s){
  riskCv.width=riskCv.clientWidth||600;riskCv.height=riskCv.clientHeight||58;
  const w=riskCv.width,h=riskCv.height,means=[];
  riskCtx.clearRect(0,0,w,h);
  const bg=riskCtx.createLinearGradient(0,0,0,h);bg.addColorStop(0,'rgba(13,17,23,.8)');bg.addColorStop(1,'rgba(7,16,24,.5)');
  riskCtx.fillStyle=bg;riskCtx.fillRect(0,0,w,h);
  for(let t2=0;t2<T;t2++)means.push(s.gms_full.reduce((a,r)=>a+r[t2],0)/N_NODES);
  for(const ev of s.events){const x1=ev.t_start/(T-1)*w,x2=ev.t_end/(T-1)*w;riskCtx.fillStyle=`${ev.color}22`;riskCtx.fillRect(x1,8,x2-x1,h-16);}
  const fill=riskCtx.createLinearGradient(0,0,w,0);fill.addColorStop(0,'rgba(56,139,253,.28)');fill.addColorStop(.55,'rgba(210,153,34,.35)');fill.addColorStop(1,'rgba(248,81,73,.42)');
  riskCtx.beginPath();riskCtx.moveTo(0,h-8);means.forEach((v,i)=>riskCtx.lineTo(i/(T-1)*w,h-8-v*(h-20)));riskCtx.lineTo(w,h-8);riskCtx.closePath();riskCtx.fillStyle=fill;riskCtx.fill();
  riskCtx.strokeStyle='rgba(230,237,243,.74)';riskCtx.lineWidth=1.4;riskCtx.beginPath();means.forEach((v,i)=>{const x=i/(T-1)*w,y=h-8-v*(h-20);if(i===0)riskCtx.moveTo(x,y);else riskCtx.lineTo(x,y);});riskCtx.stroke();
  const x=s.t/(T-1)*w;riskCtx.strokeStyle='white';riskCtx.lineWidth=1.5;riskCtx.beginPath();riskCtx.moveTo(x,3);riskCtx.lineTo(x,h-4);riskCtx.stroke();
  riskCtx.fillStyle='rgba(139,148,158,.9)';riskCtx.font='9px Courier New';riskCtx.fillText('network instability timeline',8,14);
}
function normSeries(arr){
  const mn=Math.min(...arr),mx=Math.max(...arr),d=mx-mn||1;
  return arr.map(v=>(v-mn)/d);
}
function lineSeries(ctx,arr,color,w,h){
  ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.beginPath();
  arr.forEach((v,i)=>{const x=i/(T-1)*w,y=h-8-v*(h-18);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
  ctx.stroke();
}
function dProfile(s){
  profileCv.width=profileCv.clientWidth||220;profileCv.height=profileCv.clientHeight||58;
  const w=profileCv.width,h=profileCv.height;
  profileCtx.clearRect(0,0,w,h);
  profileCtx.fillStyle='rgba(7,16,24,.45)';profileCtx.fillRect(0,0,w,h);
  profileCtx.strokeStyle='rgba(48,54,61,.45)';profileCtx.lineWidth=1;
  for(let y=18;y<h;y+=22){profileCtx.beginPath();profileCtx.moveTo(0,y);profileCtx.lineTo(w,y);profileCtx.stroke();}
  lineSeries(profileCtx,normSeries(s.gms_full[sn]),'#388BFD',w,h);
  lineSeries(profileCtx,normSeries(s.temp_full[sn]),'#3FB950',w,h);
  lineSeries(profileCtx,normSeries(s.grad_full[sn].map(Math.abs)),'#D29922',w,h);
  lineSeries(profileCtx,normSeries(s.zscore_full[sn]),'#BC8CFF',w,h);
  const x=s.t/(T-1)*w;profileCtx.strokeStyle='rgba(255,255,255,.78)';profileCtx.beginPath();profileCtx.moveTo(x,4);profileCtx.lineTo(x,h-4);profileCtx.stroke();
}
function updateEventCards(s){
  const wrap=$('eventCards');if(!wrap)return;
  wrap.innerHTML=(s.events||[]).map((ev,idx)=>{
    const active=s.active_events.includes(ev.label);
    const progress=Math.max(0,Math.min(1,(s.t-ev.t_start)/(ev.t_end-ev.t_start)));
    const current=ev.nodes.reduce((a,id)=>a+(s.nodes[id]?.gms||0),0)/ev.nodes.length;
    return `<div class="event-card ${active?'active':''}" onclick="api('trigger_event',{idx:${idx}})">
      <div class="event-top"><span style="color:${ev.color}">${ev.label}</span><span>${active?'ACTIVE':Math.round(progress*100)+'%'}</span></div>
      <div class="event-meta">nodes ${ev.nodes.join(', ')} | ${ev.time_range || timeRangeLabel(s, ev.t_start, ev.t_end)} | mean ${current.toFixed(2)}</div>
      <div class="event-bar"><span style="width:${progress*100}%;background:${ev.color}"></span></div>
    </div>`;
  }).join('');
}
function drawHealthRing(s){
  healthCv.width=62;healthCv.height=62;
  const stable=Math.max(0,s.N-s.high_count-s.mod_count),mod=s.mod_count,high=s.high_count,total=s.N||1;
  const parts=[
    {v:stable/total,c:'#388BFD'},
    {v:mod/total,c:'#D29922'},
    {v:high/total,c:'#F85149'}
  ];
  const cx=31,cy=31,r=23;
  healthCtx.clearRect(0,0,62,62);
  healthCtx.lineWidth=6;
  healthCtx.strokeStyle='rgba(48,54,61,.72)';
  healthCtx.beginPath();healthCtx.arc(cx,cy,r,0,Math.PI*2);healthCtx.stroke();
  let a=-Math.PI/2;
  for(const p of parts){
    if(p.v<=0)continue;
    healthCtx.strokeStyle=p.c;healthCtx.beginPath();healthCtx.arc(cx,cy,r,a,a+p.v*Math.PI*2);healthCtx.stroke();
    a+=p.v*Math.PI*2;
  }
  const health=Math.max(0,Math.round((stable+.45*mod)/total*100));
  $('healthPct').textContent=health+'%';
}
function metric(m,k){return Math.max(0,Math.min(100,parseFloat(m?.[k])||0));}
function setOptBar(id,ref,gms){
  const r=$(id+'-ref'),g=$(id+'-gms');
  if(!r||!g)return;
  r.style.width=Math.max(0,Math.min(100,ref))+'%';
  g.style.width=Math.max(0,Math.min(100,gms))+'%';
}
function drawOptimizationLens(s){
  const cv=$('optLens');
  if(!cv||!s.perf_gms)return;
  const pg=s.perf_gms,pb=s.perf_base,pz=s.perf_z;
  const best={
    acc:Math.max(metric(pb,'acc'),metric(pz,'acc')),
    prec:Math.max(metric(pb,'prec'),metric(pz,'prec')),
    f1:Math.max(metric(pb,'f1'),metric(pz,'f1')),
    quiet:100-Math.min(metric(pb,'far'),metric(pz,'far'))
  };
  const gms={acc:metric(pg,'acc'),prec:metric(pg,'prec'),f1:metric(pg,'f1'),quiet:100-metric(pg,'far')};
  const score=m=>m.acc*.30+m.prec*.22+m.f1*.28+m.quiet*.20;
  const edge=score(gms)-score(best);
  const edgeEl=$('opt-edge');
  if(edgeEl){
    edgeEl.textContent=(edge>=0?'+':'')+edge.toFixed(1);
    edgeEl.className=edge<0?'neg':'';
  }

  $('opt-acc-val').textContent=gms.acc.toFixed(1)+'%';
  $('opt-f1-val').textContent=gms.f1.toFixed(1)+'%';
  const farSaved=Math.min(metric(pb,'far'),metric(pz,'far'))-metric(pg,'far');
  $('opt-far-val').textContent=(farSaved>=0?'+':'')+farSaved.toFixed(1)+'%';
  setOptBar('opt-acc',best.acc,gms.acc);
  setOptBar('opt-f1',best.f1,gms.f1);
  setOptBar('opt-far',best.quiet,gms.quiet);

  const dpr=Math.min(window.devicePixelRatio||1,2),w=cv.clientWidth||220,h=cv.clientHeight||108;
  cv.width=w*dpr;cv.height=h*dpr;
  const c=cv.getContext('2d');
  c.setTransform(dpr,0,0,dpr,0,0);
  c.clearRect(0,0,w,h);
  const cx=w*.42,cy=h*.52,r=Math.min(w,h)*.36;
  const axes=[
    ['acc','ACC'],
    ['prec','PREC'],
    ['f1','F1'],
    ['quiet','QUIET']
  ];
  c.strokeStyle='rgba(48,54,61,.75)';c.lineWidth=1;
  for(let ring=1;ring<=3;ring++){
    c.beginPath();
    axes.forEach((_,i)=>{
      const a=-Math.PI/2+i*Math.PI*2/axes.length,rr=r*ring/3;
      const x=cx+Math.cos(a)*rr,y=cy+Math.sin(a)*rr;
      if(i===0)c.moveTo(x,y);else c.lineTo(x,y);
    });
    c.closePath();c.stroke();
  }
  axes.forEach(([_,label],i)=>{
    const a=-Math.PI/2+i*Math.PI*2/axes.length;
    c.strokeStyle='rgba(139,148,158,.20)';
    c.beginPath();c.moveTo(cx,cy);c.lineTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r);c.stroke();
    c.fillStyle='rgba(139,148,158,.86)';c.font='8px Courier New';c.textAlign='center';c.textBaseline='middle';
    c.fillText(label,cx+Math.cos(a)*(r+13),cy+Math.sin(a)*(r+10));
  });
  function poly(m,fill,stroke){
    c.beginPath();
    axes.forEach(([k],i)=>{
      const a=-Math.PI/2+i*Math.PI*2/axes.length,rr=r*(m[k]/100);
      const x=cx+Math.cos(a)*rr,y=cy+Math.sin(a)*rr;
      if(i===0)c.moveTo(x,y);else c.lineTo(x,y);
    });
    c.closePath();c.fillStyle=fill;c.strokeStyle=stroke;c.lineWidth=1.5;c.fill();c.stroke();
  }
  poly(best,'rgba(188,140,255,.13)','rgba(188,140,255,.55)');
  poly(gms,'rgba(63,185,80,.18)','rgba(63,185,80,.88)');
  c.fillStyle='rgba(188,140,255,.85)';c.fillRect(w-62,h-24,10,3);
  c.fillStyle='rgba(139,148,158,.9)';c.font='8px Courier New';c.textAlign='left';c.fillText('Best base',w-48,h-22);
  c.fillStyle='rgba(63,185,80,.9)';c.fillRect(w-62,h-12,10,3);
  c.fillStyle='rgba(139,148,158,.9)';c.fillText('GMS',w-48,h-10);
}
function updateMissionRibbon(s,t){
  const mean=s.gms_full.reduce((acc,row)=>acc+row[t],0)/N_NODES;
  const peak=s.nodes.reduce((best,n)=>n.gms>best.gms?n:best,s.nodes[0]);
  $('kpiMean').textContent=mean.toFixed(3);
  $('kpiPeak').textContent='N'+peak.id;
  $('kpiPeakVal').textContent=peak.gms.toFixed(3)+' GMS';
  $('kpiEvent').textContent=s.active_events.length?s.active_events.map(x=>x.replace('Event ','')).join('+'):'None';
  $('kpiNoise').textContent=s.noise_on?'On':'Off';
  $('kpiNoise').style.color=s.noise_on?'var(--amber)':'var(--teal)';
  drawHealthRing(s);
  const banner=$('alertBanner');
  if(s.high_count>0){banner.textContent=`${s.high_count} high-risk node${s.high_count===1?'':'s'} detected at ${frameTimeLabel(s,t)}`;banner.className='alert-banner show';}
  else banner.className='alert-banner';
}

// Build node grid
function selectNode(n){sn=n;if(state)render(state);}
function buildGrid(){const g=$('ngrid');for(let i=0;i<N_NODES;i++){const b=document.createElement('button');b.id=`nb-${i}`;b.className='nb'+(i===0?' sel':'');b.textContent=`${i}`;b.onclick=()=>selectNode(i);g.appendChild(b);}}
buildGrid();

function render(s){
  state=s;const t=s.t,nd=s.nodes[sn];
  noiseOn=s.noise_on;
  $('tdisp').textContent=frameTimeLabel(s,t);$('tl').value=t;
  $('cn1').textContent=sn;$('cn2').textContent=sn;$('mn').textContent=sn;
  playing=s.playing;$('btn-play').textContent=playing?'PAUSE':'PLAY';$('btn-play').className=playing?'pri':'';
  // noise banner
  const nb=$('noise-tgl'),ban=$('noise-banner');
  nb.className=s.noise_on?'tgl on':'tgl';ban.className=s.noise_on?'noise-banner show':'noise-banner';
  // event badges
  ['a','b','c','d'].forEach((k,i)=>{const on=s.active_events.includes(['Event A','Event B','Event C','Event D'][i]);$(`b${k}`).classList.toggle('on',on);});
  updateEventCards(s);
  const meanG=s.gms_full.reduce((acc,row)=>acc+row[t],0)/N_NODES;
  const load=s.nodes.reduce((acc,n)=>acc+n.gms,0)/N_NODES;
  $('hud-mean').textContent=meanG.toFixed(3);$('hud-high').textContent=s.high_count;$('hud-load').textContent=Math.round(load*100)+'%';
  updateMissionRibbon(s,t);
  dMap(s);
  // Charts
  const gd=s.gms_full[sn],gm=lbs.map(t2=>s.gms_full.reduce((a,r)=>a+r[t2],0)/N_NODES);
  if(gmsC){
    gmsC.data.labels=s.time_axis||lbs.map(t2=>frameTimeLabel(s,t2));
    gmsC.data.datasets[0].data=gd;gmsC.data.datasets[1].data=gm;
    gmsC.data.datasets[2].data=Array(T).fill(s.alpha);gmsC.data.datasets[3].data=Array(T).fill(s.beta);
    gmsC.data.datasets[0].pointRadius=gd.map((_,i)=>i===t?5:0);gmsC.update('none');
  }
  const td=s.temp_full[sn];
  if(tmpC){tmpC.data.labels=s.time_axis||lbs.map(t2=>frameTimeLabel(s,t2));tmpC.data.datasets[0].data=td;tmpC.data.datasets[0].pointRadius=td.map((_,i)=>i===t?5:0);tmpC.update('none');}
  dRiskTimeline(s);
  dProfile(s);
  dHeat(s.gms_full,t);
  // Status
  const sb=$('sb');if(nd.label===2){sb.className='status-box s-hi';sb.textContent='HIGH UNSTABLE';}else if(nd.label===1){sb.className='status-box s-mo';sb.textContent='MOD. UNSTABLE';}else{sb.className='status-box s-ok';sb.textContent='STABLE';}
  $('ss').textContent=`N${sn} @ ${frameTimeLabel(s,t)}`;
  $('m-gms').textContent=nd.gms.toFixed(3);$('m-zs').textContent=nd.zscore.toFixed(2);$('m-gr').textContent=(nd.grad>=0?'+':'')+nd.grad.toFixed(2)+' C';$('m-mo').textContent=(nd.mom>=0?'+':'')+nd.mom.toFixed(2)+' C';$('m-du').textContent=nd.dur.toFixed(3);$('m-te').textContent=nd.temp.toFixed(1)+' C';
  // Node buttons
  s.nodes.forEach((n,i)=>{const b=$(`nb-${i}`);if(!b)return;b.className='nb'+(i===sn?' sel':'')+(n.label===2?' high':n.label===1?' mod':'');});
  // Performance table - 3 baselines
  const pg=s.perf_gms,pb=s.perf_base,pz=s.perf_z;
  $('pa-b').textContent=pb.acc+'%';$('pa-z').textContent=pz.acc+'%';$('pa-g').textContent=pg.acc+'%';
  $('pp-b').textContent=pb.prec+'%';$('pp-z').textContent=pz.prec+'%';$('pp-g').textContent=pg.prec+'%';
  $('pr-b').textContent=pb.rec+'%';$('pr-z').textContent=pz.rec+'%';$('pr-g').textContent=pg.rec+'%';
  $('pf-b').textContent=pb.far+'%';$('pf-z').textContent=pz.far+'%';$('pf-g').textContent=pg.far+'%';
  $('p1-b').textContent=pb.f1+'%';$('p1-z').textContent=pz.f1+'%';$('p1-g').textContent=pg.f1+'%';
  drawOptimizationLens(s);
}

function onFrame(d){render(d);}
function onAlert(msg){
  const log=$('alog'),cls={danger:'al-d',warn:'al-w',ok:'al-o',info:'al-i'}[msg.level]||'al-i';
  const ico={danger:'HIGH',warn:'WARN',ok:'OK',info:'INFO'}[msg.level]||'INFO';
  const d=document.createElement('div');d.className=cls;
  d.textContent=`[${msg.time_label||frameTimeLabel(state,msg.t)}] ${ico} ${msg.msg}`;
  log.insertBefore(d,log.firstChild);if(log.children.length>80)log.removeChild(log.lastChild);
}

// Tooltip
function nodeScreenPos(nd){
  const h=terrainHeight(nd.x,nd.y),[bx,by]=ts(nd.x,nd.y,h);
  return [bx,by-(18+nd.gms*38)];
}
canvas.addEventListener('mousemove',e=>{if(!state)return;const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=1e9;for(let i=0;i<N_NODES;i++){const[sx,sy]=nodeScreenPos(state.nodes[i]);const d=Math.hypot(sx-mx,sy-my);if(d<bd){bd=d;best=i;}}const tt=$('tt');if(bd<30&&best>=0){const nd=state.nodes[best];const cls=['Stable','Mod. Unstable','High Unstable'][nd.label];tt.innerHTML=`<div class="tt-t">N${best} - ${cls}</div>GMS: <b>${nd.gms.toFixed(3)}</b><br>Z-Score: <b>${nd.zscore.toFixed(2)}</b><br>Grad: <b>${nd.grad>0?'+':''}${nd.grad.toFixed(2)} C</b><br>M: <b>${nd.mom>0?'+':''}${nd.mom.toFixed(2)}</b><br>NIS: <b>${nd.nis.toFixed(3)}</b><br>Temp: <b>${nd.temp.toFixed(1)} C</b>`;tt.style.display='block';tt.style.left=(mx+14)+'px';tt.style.top=(my-10)+'px';}else tt.style.display='none';});
canvas.addEventListener('click',e=>{if(!state)return;const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=1e9;for(let i=0;i<N_NODES;i++){const[sx,sy]=nodeScreenPos(state.nodes[i]);const d=Math.hypot(sx-mx,sy-my);if(d<bd){bd=d;best=i;}}if(bd<34)selectNode(best);});
canvas.addEventListener('mouseleave',()=>{$('tt').style.display='none';});

async function api(ep,b={}){await fetch(`/api/${ep}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
async function exportData(){
  const resp=await fetch('/export');
  const data=await resp.json();
  alert(data.status==='saved'?`Saved ${data.file}`:`Export failed: ${data.error||'unknown error'}`);
}
function togglePlay(){playing?api('pause'):api('play');}
function toggleNoise(){api('toggle_noise',{on:!noiseOn});}
function setSpd(v){$('spv').textContent='x'+v;api('speed',{speed:Math.max(.04,1.2-v*.058)});}
function upW(k,v){const val=parseFloat((+v).toFixed(2));$(`vl-${k}`).textContent=val.toFixed(2);pend[k]=val;const s=['w1','w2','w3','w4'].reduce((a,k2)=>a+parseFloat($(`sl-${k2}`).value),0);const sw=$('sw');if(Math.abs(s-1)>.06){sw.className='sw show';sw.textContent=`Weights sum to ${s.toFixed(2)}`;}else sw.className='sw';sched();}
function upT(k,v){const val=parseFloat((+v).toFixed(2));const m={theta:'th',alpha:'al',beta:'be'}[k];$(`vl-${m}`).textContent=val.toFixed(2);pend[k]=val;sched();}
function sched(){if(ptmr)clearTimeout(ptmr);ptmr=setTimeout(()=>{api('params',pend);pend={};},400);}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.code==='ArrowRight')api('step',{dir:1});if(e.code==='ArrowLeft')api('step',{dir:-1});if(e.code==='KeyR')api('reset');if(e.code==='KeyN')toggleNoise();if(e.key==='1')api('trigger_event',{idx:0});if(e.key==='2')api('trigger_event',{idx:1});if(e.key==='3')api('trigger_event',{idx:2});if(e.key==='4')api('trigger_event',{idx:3});});
