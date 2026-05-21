const _es=new EventSource('/stream');
function _gmsFrameTime(d){
  if(d && d.time_label)return d.time_label;
  const total=Math.max(1,Number(d?.T)||120);
  const t=Math.max(0,Number(d?.t)||0);
  const mins=Math.round(t/total*24*60)%(24*60);
  return String(Math.floor(mins/60)).padStart(2,'0')+':'+String(mins%60).padStart(2,'0');
}
_es.onmessage=e=>{
  const msg=JSON.parse(e.data);
  if(msg.type==='frame'){
    const d=msg.data;
    const et=document.getElementById('nav-t');
    if(et)et.textContent=_gmsFrameTime(d);
    const es=document.getElementById('nav-st');
    if(es){
      if(d.high_count>0){es.className='pill p-hi';es.textContent='▲ '+d.high_count+' HIGH';}
      else{es.className='pill p-ok';es.textContent='✓ ALL STABLE';}
    }
    const nn=document.getElementById('nav-noise');
    if(nn){nn.className=d.noise_on?'pill p-ns':'pill p-ok';nn.textContent=d.noise_on?'⚡ NOISE ON':'CLEAN DATA';}
    if(typeof onFrame==='function')onFrame(d);
  }
  if(msg.type==='alert'){if(typeof onAlert==='function')onAlert(msg);}
};
