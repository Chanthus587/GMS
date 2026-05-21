let al=[], levelFilter='all', statusFilter='open', query='', sortOrder='newest';
let hysteresisTimer=null;

function frameTimeLabel(t,total=120){
  const idx=Math.max(0,Math.round(Number(t)||0));
  const mins=Math.round(idx/Math.max(1,total)*24*60)%(24*60);
  return String(Math.floor(mins/60)).padStart(2,'0')+':'+String(mins%60).padStart(2,'0');
}

function norm(a){
  const nodes = Array.isArray(a.nodes) ? a.nodes : (a.node === null || a.node === undefined ? [] : [a.node]);
  return {
    id:a.id || `${a.t || 0}-${a.level || 'info'}-${Math.random()}`,
    msg:a.msg || '',
    level:a.level || 'info',
    t:Number.isFinite(+a.t) ? +a.t : 0,
    time_label:a.time_label || frameTimeLabel(a.t),
    status:a.status || (a.level==='danger'||a.level==='warn' ? 'active' : 'resolved'),
    node:a.node ?? null,
    nodes:nodes.map(Number).filter(Number.isFinite),
    category:a.category || 'system',
    details:a.details || {},
    action:a.action || ''
  };
}

function esc(s){
  return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function setLevel(v){
  levelFilter=v;
  ['all','danger','warn','ok','info'].forEach(k=>{
    document.getElementById('fb-level-'+k).className='fb';
  });
  const cls={all:'all',danger:'hi',warn:'mo',ok:'ok',info:'in'}[v] || 'all';
  document.getElementById('fb-level-'+v).className='fb on-'+cls;
  renderList();
}

function setStatus(v){
  statusFilter=v;
  ['open','active','acknowledged','resolved','all'].forEach(k=>{
    document.getElementById('fb-status-'+k).className='fb';
  });
  document.getElementById('fb-status-'+v).className='fb on-st';
  renderList();
}

function setQuery(v){
  query=v.trim().toLowerCase();
  renderList();
}

function setSort(v){
  sortOrder=v;
  renderList();
}

function visibleAlerts(){
  return al.filter(a=>{
    const levelOk=levelFilter==='all' || a.level===levelFilter;
    const statusOk=statusFilter==='all' || (statusFilter==='open' ? a.status!=='resolved' : a.status===statusFilter);
    const nodeText=(a.nodes || []).map(n=>`N${n}`).join(' ');
    const hay=`${a.msg} ${a.node ?? ''} ${nodeText} ${a.category} ${JSON.stringify(a.details)}`.toLowerCase();
    return levelOk && statusOk && (!query || hay.includes(query));
  }).sort((a,b)=>sortOrder==='oldest' ? a.t-b.t || a.id.localeCompare(b.id) : b.t-a.t || b.id.localeCompare(a.id));
}

function detailHtml(a){
  const d=a.details || {};
  const pairs=[
    ['GMS',d.gms],['Temp',d.temp],['Grad',d.gradient],
    ['Momentum',d.momentum],['Duration',d.duration],['NIS',d.nis]
  ].filter(x=>x[1]!==undefined && x[1]!==null);
  if(!pairs.length)return '';
  return `<div class="details">${pairs.map(([k,v])=>`${k}: <b>${esc(v)}</b>`).join(' | ')}</div>`;
}

function actionHtml(a){
  const canAck=a.status==='active' && (a.level==='danger'||a.level==='warn');
  const canResolve=a.status!=='resolved';
  return `<div class="row-actions">
    ${canAck?`<button data-alert-id="${esc(a.id)}" data-alert-state="acknowledged">Ack</button>`:''}
    ${canResolve?`<button data-alert-id="${esc(a.id)}" data-alert-state="resolved">Resolve</button>`:''}
  </div>`;
}

function renderList(){
  const log=document.getElementById('alf'), items=visibleAlerts();
  if(!items.length){
    log.innerHTML='<div class="empty">No alerts for this view.</div>';
    return;
  }
  const cls={danger:'al-d',warn:'al-w',ok:'al-o',info:'al-i'};
  const ico={danger:'HI',warn:'MO',ok:'OK',info:'IN'};
  log.innerHTML=items.map(a=>{
    const nodeList=(a.nodes || []).length ? a.nodes.map(n=>`N${n}`).join(', ') : '';
    const node=nodeList?`<span class="chip node">${esc(nodeList)}</span>`:'';
    const primaryNode=(a.nodes || [a.node]).find(n=>Number.isFinite(Number(n)));
    return `<div class="ar ${cls[a.level]||'al-i'}" data-map-node="${esc(primaryNode ?? '')}" data-map-time="${esc(a.t)}">
      <span class="ts">${esc(a.time_label)}</span>
      <span class="ic">${ico[a.level]||'IN'}</span>
      <div>
        <div class="mg"><span class="lvl">${esc(a.level.toUpperCase())}</span> ${esc(a.msg)}</div>
        <div class="meta">${node}<span class="chip ${esc(a.status)}">${esc(a.status)}</span><span class="chip">${esc(a.category)}</span></div>
        ${detailHtml(a)}
        ${a.action?`<div class="action">${esc(a.action)}</div>`:''}
      </div>
      ${actionHtml(a)}
    </div>`;
  }).join('');
}

function updateStats(){
  document.getElementById('alertsToday').textContent=al.length;
  document.getElementById('criticalCount').textContent=al.filter(a=>a.level==='danger').length;
  document.getElementById('openAlerts').textContent=al.filter(a=>a.status!=='resolved').length;
}

function refresh(){
  renderList();
  updateStats();
  loadAnalytics();
}

function onAlert(msg){
  al.unshift(norm(msg));
  if(al.length>300)al.pop();
  refresh();
}

async function api(ep,b={}){
  await fetch(`/api/${ep}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
}

async function setAlertState(id,status){
  const r=await fetch(`/api/alerts/${encodeURIComponent(id)}/status`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({status})
  });
  if(r.ok){
    const data=await r.json();
    const idx=al.findIndex(a=>a.id===id);
    if(idx>=0)al[idx]=norm(data.alert);
    refresh();
  }
}

async function clearAlerts(){
  await fetch('/api/alerts/clear',{method:'POST'});
  al=[];
  refresh();
}

function exportAlerts(format){
  window.location.href=`/api/alerts/export?format=${encodeURIComponent(format)}`;
}

function drawTrend(points){
  const canvas=document.getElementById('alertTrend');
  if(!canvas)return;
  const ctx=canvas.getContext('2d');
  const width=canvas.width=canvas.clientWidth || 900;
  const height=canvas.height=64;
  ctx.clearRect(0,0,width,height);
  ctx.fillStyle='rgba(13,17,23,.38)';
  ctx.fillRect(0,0,width,height);
  if(!points || !points.length){
    document.getElementById('trendNote').textContent='No alerts yet';
    return;
  }
  document.getElementById('trendNote').textContent=`${points.length} active time bucket${points.length===1?'':'s'}`;
  const maxCount=Math.max(...points.map(p=>p.count),1);
  const minT=Math.min(...points.map(p=>p.t));
  const maxT=Math.max(...points.map(p=>p.t),minT+1);
  ctx.strokeStyle='rgba(48,54,61,.75)';
  ctx.lineWidth=1;
  for(let y=16;y<height;y+=16){
    ctx.beginPath();
    ctx.moveTo(0,y);
    ctx.lineTo(width,y);
    ctx.stroke();
  }
  ctx.strokeStyle='#388BFD';
  ctx.fillStyle='rgba(56,139,253,.20)';
  ctx.lineWidth=2;
  ctx.beginPath();
  points.forEach((p,idx)=>{
    const x=(p.t-minT)/(maxT-minT)*Math.max(1,width-18)+9;
    const y=height-8-(p.count/maxCount)*(height-18);
    if(idx===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  });
  ctx.stroke();
  points.forEach(p=>{
    const x=(p.t-minT)/(maxT-minT)*Math.max(1,width-18)+9;
    const h=(p.count/maxCount)*(height-18);
    ctx.fillRect(x-3,height-8-h,6,h);
  });
}

function loadAnalytics(){
  fetch('/api/alerts/analytics')
    .then(r=>r.json())
    .then(d=>{
      document.getElementById('alertsToday').textContent=d.alerts_today ?? al.length;
      document.getElementById('criticalCount').textContent=d.critical_count ?? 0;
      document.getElementById('mostUnstable').textContent=`N${d.most_unstable_node ?? 0}`;
      document.getElementById('avgResponse').textContent=`${d.avg_response_frames ?? 0}f`;
      document.getElementById('falseEstimate').textContent=`${d.false_alert_estimate ?? 0}%`;
      document.getElementById('openAlerts').textContent=d.active_count ?? 0;
      updateHysteresisView(d);
      drawTrend(d.trend || []);
    });
}

function updateHysteresisView(d){
  const margin=Number(d.hysteresis_margin ?? 0.05);
  const enter=Number(d.high_enter ?? 0.60);
  const release=Number(d.high_release ?? enter-margin);
  document.getElementById('hysteresisValue').textContent=margin.toFixed(2);
  document.getElementById('suppressedFlickers').textContent=d.suppressed_flickers ?? 0;
  document.getElementById('hysteresisText').textContent=
    `High alert enters at GMS >= ${enter.toFixed(2)} and releases only below ${release.toFixed(2)}.`;
  const input=document.getElementById('hysteresisMargin');
  if(document.activeElement !== input) input.value=margin.toFixed(2);
}

function setHysteresisMargin(value){
  const margin=Number(value);
  document.getElementById('hysteresisValue').textContent=margin.toFixed(2);
  if(hysteresisTimer) clearTimeout(hysteresisTimer);
  hysteresisTimer=setTimeout(()=>{
    fetch('/api/alerts/hysteresis',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({margin})
    }).then(()=>loadAnalytics());
  },300);
}

function bindControls(){
  document.querySelectorAll('[data-level]').forEach(btn=>{
    btn.addEventListener('click',()=>setLevel(btn.dataset.level));
  });
  document.querySelectorAll('[data-status]').forEach(btn=>{
    btn.addEventListener('click',()=>setStatus(btn.dataset.status));
  });
  document.querySelectorAll('[data-api]').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const body=btn.dataset.body ? JSON.parse(btn.dataset.body) : {};
      api(btn.dataset.api,body);
    });
  });
  document.querySelectorAll('[data-export]').forEach(btn=>{
    btn.addEventListener('click',()=>exportAlerts(btn.dataset.export));
  });
  document.getElementById('q').addEventListener('input',e=>setQuery(e.target.value));
  document.getElementById('sortOrder').addEventListener('change',e=>setSort(e.target.value));
  document.getElementById('hysteresisMargin').addEventListener('input',e=>setHysteresisMargin(e.target.value));
  document.getElementById('clearAlerts').addEventListener('click',clearAlerts);
  document.getElementById('alf').addEventListener('click',e=>{
    const btn=e.target.closest('[data-alert-id][data-alert-state]');
    if(btn){
      setAlertState(btn.dataset.alertId,btn.dataset.alertState);
      return;
    }
    const row=e.target.closest('[data-map-node]');
    if(row && row.dataset.mapNode !== ''){
      window.location.href=`/map?node=${encodeURIComponent(row.dataset.mapNode)}&t=${encodeURIComponent(row.dataset.mapTime)}`;
    }
  });
}

function loadAlerts(){
  fetch('/api/alerts')
    .then(r=>r.json())
    .then(d=>{al=d.map(norm);refresh();});
}

function onFrame(d){}

window.onFrame=onFrame;
window.onAlert=onAlert;

bindControls();
loadAlerts();
