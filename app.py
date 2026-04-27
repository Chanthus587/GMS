"""
app.py  —  GMS Mission Control  v2  (40-node, 3-baseline, noise, weight sensitivity)
=====================================================================================
Run:   python app.py
Open:  http://localhost:5000
"""

import json, time, threading
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from scipy.spatial.distance import cdist

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════
#  ENGINE  — 40 nodes, 4 events, 3 baselines, noise toggle
# ══════════════════════════════════════════════════════════════════════

class GMSEngine:
    N=40; T=120; G=10.0; RAD=2.8
    Z_THRESH=1.2          # Z-score anomaly threshold
    ABS_THRESH=26.5       # absolute temperature threshold

    EVENTS=[
        dict(nodes=[0,1,2,3,4],   t_start=20,t_end=55,  dT=8.0,label="Event A",color="#388BFD"),
        dict(nodes=[15,16,17,18], t_start=35,t_end=70,  dT=6.0,label="Event B",color="#3FB950"),
        dict(nodes=[8,9,10,11],   t_start=50,t_end=90,  dT=7.0,label="Event C",color="#D29922"),
        dict(nodes=[25,26,27,28], t_start=65,t_end=100, dT=5.5,label="Event D",color="#BC8CFF"),
    ]

    def __init__(self):
        np.random.seed(2024)
        self.w1,self.w2,self.w3,self.w4 = 0.35,0.25,0.20,0.20
        self.theta=1.2; self.window=8; self.alpha=0.25; self.beta=0.60
        self.t=0; self.playing=False; self.speed=0.25
        self.noise_on=False
        self._lock=threading.Lock(); self._thread=None
        self._subs=[]; self._sub_lock=threading.Lock()
        self.alert_history=[]
        self._build()
        self._simulate()
        self._gms()
        self.logs = []

        

    # ── network ──────────────────────────────────────────────────────
    def _build(self):
        pos=np.random.uniform(0.5,self.G-0.5,(self.N,2))
        self.pos=pos
        d=cdist(pos,pos)
        self.adj={i:[j for j in range(self.N) if j!=i and d[i,j]<=self.RAD]
                  for i in range(self.N)}

    # ── simulate clean temperature ────────────────────────────────────
    def _simulate(self):
        t=np.linspace(0,2*np.pi,self.T)
        T=np.zeros((self.N,self.T)); H=np.zeros((self.N,self.T))
        for i in range(self.N):
            e=(self.pos[i,0]+self.pos[i,1])/(2*self.G)
            T[i]=22+6*np.sin(t-0.3)+e*2.5+np.random.normal(0,0.25,self.T)
            H[i]=65-8*np.sin(t)-e*3 +np.random.normal(0,0.50,self.T)
        for ev in self.EVENTS:
            dur=ev['t_end']-ev['t_start']
            for i in ev['nodes']:
                r=np.zeros(self.T)
                r[ev['t_start']:ev['t_end']]=np.linspace(0,ev['dT'],dur)
                r[ev['t_end']:]=ev['dT']*np.exp(-np.arange(self.T-ev['t_end'])/12.)
                T[i]+=r; H[i]-=r*0.8
        self._Temp_clean=T; self.Humid=H

    # ── GMS pipeline ─────────────────────────────────────────────────
    def _n(self,x):
        a,b=x.min(),x.max(); return (x-a)/(b-a+1e-12)

    def _gms(self):
        N,T=self.N,self.T
        # Apply noise if toggled
        np.random.seed(42)
        if self.noise_on:
            self.Temp=self._Temp_clean+np.random.uniform(-0.5,0.5,(N,T))
        else:
            self.Temp=self._Temp_clean.copy()

        # ── Component 1: Spatial Gradient ΔT_ij = T_i - T_j ──────
        G2=np.zeros((N,T))
        for i in range(N):
            nb=self.adj[i]
            if nb: G2[i]=np.array([self.Temp[i]-self.Temp[j] for j in nb]).mean(0)

        # ── Component 2: Temporal Momentum M = ΔT(t)-ΔT(t-1) ─────
        M=np.zeros((N,T)); M[:,1:]=G2[:,1:]-G2[:,:-1]

        # ── Component 3: Duration / Persistence ───────────────────
        D=np.zeros((N,T))
        for i in range(N):
            for t in range(T):
                ws=max(0,t-self.window+1)
                D[i,t]=np.mean(np.abs(G2[i,ws:t+1])>self.theta)

        # ── Component 4: Neighbor Influence Score ─────────────────
        NIS=G2.copy()
        a,b=NIS.min(),NIS.max()
        if b>a: NIS=(NIS-a)/(b-a)

        # ── GMS Composite Score S = w1|ΔT|+w2|M|+w3·NIS+w4·D ─────
        raw=(self.w1*self._n(np.abs(G2))+self.w2*self._n(np.abs(M))
            +self.w3*NIS+self.w4*D)
        gms=np.clip(self._n(raw),0,1)
        lbl=np.zeros((N,T),dtype=int)
        lbl[gms>=self.alpha]=1; lbl[gms>=self.beta]=2

        self.grad=G2; self.mom=M; self.dur=D; self.nis=NIS
        self.gms=gms; self.label=lbl

        # ── Baselines ─────────────────────────────────────────────
        # 1. Absolute threshold
        self.baseline_abs=(self.Temp>self.ABS_THRESH).astype(int)
        # 2. Z-score per-node
        mu=self.Temp.mean(axis=1,keepdims=True)
        sig=self.Temp.std(axis=1,keepdims=True)+1e-9
        self.z_scores=np.abs((self.Temp-mu)/sig)
        self.baseline_z=(self.z_scores>self.Z_THRESH).astype(int)

        # onset times
        self.onset=np.full(N,np.inf)
        for i in range(N):
            ab=np.where(gms[i]>self.alpha)[0]
            if len(ab): self.onset[i]=ab[0]

    # ── ground truth ──────────────────────────────────────────────────
    def _gt(self):
        gt=np.zeros((self.N,self.T),dtype=int)
        for ev in self.EVENTS:
            for i in ev['nodes']: gt[i,ev['t_start']:ev['t_end']]=1
        return gt

    # ── metrics ───────────────────────────────────────────────────────
    def _mets(self,p,gt):
        TP=int(((p==1)&(gt==1)).sum()); TN=int(((p==0)&(gt==0)).sum())
        FP=int(((p==1)&(gt==0)).sum()); FN=int(((p==0)&(gt==1)).sum())
        tot=TP+TN+FP+FN; acc=(TP+TN)/tot if tot else 0
        pr=TP/(TP+FP) if TP+FP else 0; re=TP/(TP+FN) if TP+FN else 0
        fa=FP/(FP+TN) if FP+TN else 0; f1=2*pr*re/(pr+re) if pr+re else 0
        return dict(acc=round(acc*100,1),prec=round(pr*100,1),
                    rec=round(re*100,1),far=round(fa*100,1),f1=round(f1*100,1))

    def _perf(self):
        gt=self._gt()
        pg=self._mets((self.gms>=self.alpha).astype(int),gt)
        pb=self._mets(self.baseline_abs,gt)
        pz=self._mets(self.baseline_z,gt)
        return pg,pb,pz
    def log_step(self, t):
        # 🔴 skip if already logged
        if hasattr(self, "last_logged_t") and self.last_logged_t == t:
            return

        self.last_logged_t = t

        gt = self._gt()

        for i in range(self.N):
            self.logs.append({
                "time": int(t),
                "node": int(i),
                "temp": float(self.Temp[i, t]),
                "gradient": float(self.grad[i, t]),
                "momentum": float(self.mom[i, t]),
                "duration": float(self.dur[i, t]),
                "nis": float(self.nis[i, t]),
                "gms": float(self.gms[i, t]),
                "zscore": float(self.z_scores[i, t]),
                "pred": int(self.gms[i, t] >= self.alpha),
                "truth": int(gt[i, t])
            })
    # ── noise toggle ─────────────────────────────────────────────────
    def toggle_noise(self,on):
        with self._lock:
            self.noise_on=on
            self._gms()
        self._bcast()
        self._alert(f"Noise {'ENABLED (+0.5°C random)' if on else 'DISABLED (clean data)'}","info")

    # ── rerun with new params ─────────────────────────────────────────
    def rerun(self,p):
        with self._lock:
            for k in ['w1','w2','w3','w4','theta','alpha','beta']:
                if k in p: setattr(self,k,float(p[k]))
            self._gms()
        self._bcast()

    # ── frame data payload ────────────────────────────────────────────
    def frame_data(self,t=None):
        if t is None: t=self.t
        t=max(0,min(t,self.T-1))
        ae=[ev['label'] for ev in self.EVENTS if ev['t_start']<=t<ev['t_end']]
        pe=[]
        for i in range(self.N):
            for j in self.adj[i]:
                if j>i:
                    oi,oj=self.onset[i],self.onset[j]
                    if (np.isfinite(oi) and np.isfinite(oj)
                            and t>=min(oi,oj) and abs(oj-oi)<=20):
                        src=i if oi<oj else j; dst=j if oi<oj else i
                        pe.append({'src':int(src),'dst':int(dst),
                                   'strength':float(np.exp(-abs(oj-oi)/10))})
        pg,pb,pz=self._perf()
        return {
            't':int(t),'T':self.T,'playing':self.playing,'noise_on':self.noise_on,
            'high_count':int((self.label[:,t]==2).sum()),
            'mod_count': int((self.label[:,t]==1).sum()),
            'N':self.N,
            'nodes':[{'id':i,'x':float(self.pos[i,0]),'y':float(self.pos[i,1]),
                      'gms':round(float(self.gms[i,t]),4),
                      'label':int(self.label[i,t]),
                      'grad':round(float(self.grad[i,t]),4),
                      'mom':round(float(self.mom[i,t]),4),
                      'dur':round(float(self.dur[i,t]),4),
                      'nis':round(float(self.nis[i,t]),4),
                      'temp':round(float(self.Temp[i,t]),2),
                      'zscore':round(float(self.z_scores[i,t]),3),
                      'onset':int(self.onset[i]) if np.isfinite(self.onset[i]) else None}
                     for i in range(self.N)],
            'adj':{str(i):self.adj[i] for i in range(self.N)},
            'prop_edges':pe,'active_events':ae,
            'gms_full':self.gms.tolist(),
            'temp_full':self.Temp.tolist(),
            'grad_full':self.grad.tolist(),
            'mom_full':self.mom.tolist(),
            'dur_full':self.dur.tolist(),
            'nis_full':self.nis.tolist(),
            'zscore_full':self.z_scores.tolist(),
            'events':self.EVENTS,
            'alpha':self.alpha,'beta':self.beta,
            'weights':{'w1':self.w1,'w2':self.w2,'w3':self.w3,'w4':self.w4},
            'perf_gms':pg,'perf_base':pb,'perf_z':pz,
        }

    # ── SSE pub/sub ───────────────────────────────────────────────────
    def subscribe(self):
        q=[]
        with self._sub_lock: self._subs.append(q)
        return q

    def unsubscribe(self,q):
        with self._sub_lock:
            if q in self._subs: self._subs.remove(q)

    def _bcast(self):
        d=json.dumps({'type':'frame','data':self.frame_data()})
        with self._sub_lock:
            for q in self._subs: q.append(d)

    def _alert(self,msg,level='info'):
        e={'msg':msg,'level':level,'t':int(self.t)}
        self.alert_history.insert(0,e)
        if len(self.alert_history)>300: self.alert_history.pop()
        d=json.dumps({'type':'alert','msg':msg,'level':level,'t':int(self.t)})
        with self._sub_lock:
            for q in self._subs: q.append(d)

    # ── playback ──────────────────────────────────────────────────────
    def play(self):
        with self._lock: self.playing=True
        if self._thread is None or not self._thread.is_alive():
            self._thread=threading.Thread(target=self._loop,daemon=True)
            self._thread.start()

    def pause(self):
        with self._lock: self.playing=False

    def reset(self):
        with self._lock: self.playing=False; self.t=0
        self._bcast()

    def jump(self,t):
        with self._lock: self.t=max(0,min(t,self.T-1))
        self._bcast()

    def step(self,d=1):
        with self._lock: self.t=max(0,min(self.t+d,self.T-1))
        self._bcast()

    def trigger(self,idx):
        ev=self.EVENTS[idx]; self.jump(ev['t_start']); self.play()
        self._alert(f"▶ Triggered {ev['label']} — nodes {ev['nodes']}","warn")

    def _loop(self):
        while True:
            with self._lock:
                if not self.playing: break
                self.t=(self.t+1)%self.T; nt=self.t
            cl=self.label[:,nt]; pl=self.label[:,max(0,nt-1)]
            for i in range(self.N):
                if cl[i]>pl[i]:
                    tag='HIGH UNSTABLE' if cl[i]==2 else 'MOD UNSTABLE'
                    self._alert(f"N{i} → {tag}  GMS={self.gms[i,nt]:.3f}",
                                'danger' if cl[i]==2 else 'warn')
                elif cl[i]<pl[i] and pl[i]>0:
                    self._alert(f"N{i} → STABLE  GMS={self.gms[i,nt]:.3f}","ok")
                    self.log_step(nt)
                    self._bcast()
                    time.sleep(self.speed)            

engine=GMSEngine()

# ══════════════════════════════════════════════════════════════════════
#  SHARED CSS
# ══════════════════════════════════════════════════════════════════════

CSS="""<style>
:root{--bg:#0D1117;--panel:#161B22;--card:#1C2128;--border:#30363D;
  --fg:#E6EDF3;--fg2:#8B949E;--fg3:#484F58;
  --blue:#388BFD;--teal:#3FB950;--amber:#D29922;
  --red:#F85149;--purple:#BC8CFF;--green:#3FB950;
  --r:8px;--rl:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:'Courier New',monospace;font-size:13px}
nav{display:flex;align-items:center;height:52px;background:var(--panel);
    border-bottom:1px solid var(--border);padding:0 16px;
    position:sticky;top:0;z-index:100;gap:0}
.brand{display:flex;align-items:center;gap:9px;padding-right:18px;
       border-right:1px solid var(--border);text-decoration:none}
.bico{width:30px;height:30px;border-radius:7px;background:#0D2240;
      border:1px solid rgba(56,139,253,.5);display:flex;align-items:center;justify-content:center}
.bico svg{width:16px;height:16px}
.bnm{font-size:13px;font-weight:700;letter-spacing:.06em;color:var(--fg)}
.bsb{font-size:9px;color:var(--fg2);letter-spacing:.04em;margin-top:1px}
.nl{display:flex;align-items:center;gap:6px;padding:0 14px;height:100%;
    font-size:11px;font-weight:700;letter-spacing:.04em;color:var(--fg2);
    cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;
    text-decoration:none;white-space:nowrap}
.nl:hover{color:var(--fg);background:rgba(255,255,255,.04)}
.nl.active{color:var(--blue);border-bottom-color:var(--blue)}
.nl svg{width:13px;height:13px;flex-shrink:0}
.nbdg{font-size:9px;font-weight:700;padding:1px 6px;border-radius:999px;
      background:rgba(248,81,73,.18);color:var(--red);border:1px solid rgba(248,81,73,.3)}
.nav-r{display:flex;align-items:center;gap:8px;margin-left:auto;
       padding-left:14px;border-left:1px solid var(--border)}
.pill{font-size:9px;font-weight:700;padding:3px 9px;border-radius:999px;letter-spacing:.05em;white-space:nowrap}
.p-live{background:rgba(63,185,80,.12);color:var(--teal);border:1px solid rgba(63,185,80,.3)}
.p-live::before{content:'';display:inline-block;width:5px;height:5px;background:var(--teal);border-radius:50%;margin-right:4px;animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.p-t{background:rgba(56,139,253,.1);color:var(--blue);border:1px solid rgba(56,139,253,.3)}
.p-hi{background:rgba(248,81,73,.1);color:var(--red);border:1px solid rgba(248,81,73,.3)}
.p-ok{background:rgba(63,185,80,.1);color:var(--teal);border:1px solid rgba(63,185,80,.3)}
.p-ns{background:rgba(210,153,34,.12);color:var(--amber);border:1px solid rgba(210,153,34,.3)}
.sec{font-size:9px;font-weight:700;letter-spacing:.12em;color:var(--fg3);
     text-transform:uppercase;padding-bottom:7px;
     border-bottom:1px solid var(--border);margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--rl);padding:14px}
button{background:var(--card);color:var(--fg);border:1px solid var(--border);border-radius:var(--r);
       padding:5px 13px;cursor:pointer;font-family:inherit;font-size:11px;font-weight:700;
       letter-spacing:.04em;transition:all .15s}
button:hover{background:#21262D;border-color:var(--fg3)}
button.pri{background:var(--blue);color:#fff;border-color:var(--blue)}
button.pri:hover{background:#2979d9}
button.warn-btn{background:rgba(210,153,34,.1);color:var(--amber);border-color:rgba(210,153,34,.4)}
button.warn-btn:hover{background:rgba(210,153,34,.18)}
button.dng{color:var(--red);border-color:rgba(248,81,73,.4)}
button.dng:hover{background:rgba(248,81,73,.08)}
input[type=range]{accent-color:var(--blue);cursor:pointer;width:100%}
.ctrl{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.ctrl label{font-size:10px;color:var(--fg2);min-width:72px}
.ctrl .v{font-size:10px;color:var(--fg);min-width:32px;text-align:right}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.mc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:9px;text-align:center}
.mc .l{font-size:9px;color:var(--fg2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.mc .v{font-size:21px;font-weight:700}
.status-box{border-radius:var(--r);border:1px solid;padding:10px;text-align:center;
            font-size:20px;font-weight:700;transition:all .3s}
.s-ok{color:var(--blue);border-color:rgba(56,139,253,.25);background:rgba(56,139,253,.06)}
.s-mo{color:var(--amber);border-color:rgba(210,153,34,.25);background:rgba(210,153,34,.06)}
.s-hi{color:var(--red);border-color:rgba(248,81,73,.3);background:rgba(248,81,73,.08);animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(248,81,73,.2)}50%{box-shadow:0 0 14px 5px rgba(248,81,73,.1)}}
.alog{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
      padding:8px;overflow-y:auto;font-size:10px;line-height:1.75}
.alog::-webkit-scrollbar{width:4px}.alog::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.al-d{color:var(--red)}.al-w{color:var(--amber)}.al-o{color:var(--teal)}.al-i{color:var(--blue)}
.pt{width:100%;border-collapse:collapse;font-size:10px}
.pt th{text-align:left;font-size:9px;color:var(--fg3);text-transform:uppercase;letter-spacing:.06em;padding:3px 5px;border-bottom:1px solid var(--border)}
.pt td{padding:3px 5px;border-bottom:1px solid rgba(48,54,61,.4)}
.pt .lbl{color:var(--fg2)}
.pt .gv{color:var(--teal);font-weight:700;text-align:right}
.pt .bv{color:var(--fg3);text-align:right}
.pt .zv{color:var(--purple);font-weight:700;text-align:right}
.tt{position:absolute;background:var(--card);border:1px solid var(--border);
    border-radius:var(--r);padding:9px 13px;font-size:11px;
    pointer-events:none;z-index:50;display:none;min-width:160px;line-height:1.9}
.tt-t{font-weight:700;color:var(--blue);margin-bottom:3px}
.toggle-wrap{display:flex;align-items:center;gap:8px;padding:5px 0}
.toggle-wrap label{font-size:11px;color:var(--fg2);cursor:pointer;user-select:none}
.tgl{width:36px;height:20px;border-radius:10px;background:var(--border);
     border:none;cursor:pointer;position:relative;transition:background .2s;padding:0;flex-shrink:0}
.tgl.on{background:var(--amber)}
.tgl::after{content:'';position:absolute;width:14px;height:14px;border-radius:50%;
            background:white;top:3px;left:3px;transition:left .2s}
.tgl.on::after{left:19px}
</style>"""

NAV_UPD="""<script>
const _es=new EventSource('/stream');
_es.onmessage=e=>{
  const msg=JSON.parse(e.data);
  if(msg.type==='frame'){
    const d=msg.data;
    const et=document.getElementById('nav-t');
    if(et)et.textContent='t = '+String(d.t).padStart(3,'0');
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
</script>"""

def navbar(active):
    t=engine.t; hc=int((engine.label[:,t]==2).sum())
    pill_cls='p-hi' if hc>0 else 'p-ok'
    pill_txt=f'▲ {hc} HIGH' if hc>0 else '✓ ALL STABLE'
    noise_cls='p-ns' if engine.noise_on else 'p-ok'
    noise_txt='⚡ NOISE ON' if engine.noise_on else 'CLEAN DATA'
    ac=len([a for a in engine.alert_history if a['level']=='danger'])
    pages=[
        ('/','db','Dashboard','<rect x="1" y="1" width="5" height="5" rx="1" fill="currentColor" opacity=".7"/><rect x="8" y="1" width="5" height="5" rx="1" fill="currentColor"/><rect x="1" y="8" width="5" height="5" rx="1" fill="currentColor" opacity=".4"/><rect x="8" y="8" width="5" height="5" rx="1" fill="currentColor" opacity=".4"/>'),
        ('/map','map','Sensor Map','<circle cx="3" cy="10" r="2" fill="currentColor" opacity=".5"/><circle cx="7" cy="6" r="2" fill="currentColor" opacity=".8"/><circle cx="11" cy="3" r="2" fill="currentColor"/><line x1="3" y1="10" x2="7" y2="6" stroke="currentColor" stroke-width="1.2"/><line x1="7" y1="6" x2="11" y2="3" stroke="currentColor" stroke-width="1.2"/>'),
        ('/analysis','an','Analysis','<polyline points="1,11 4,6 7,8 10,3 13,5" stroke="currentColor" stroke-width="1.5" fill="none"/>'),
        ('/alerts','al','Alerts','<rect x="1" y="2" width="12" height="10" rx="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="4" y1="6" x2="10" y2="6" stroke="currentColor" stroke-width="1"/><line x1="4" y1="8" x2="8" y2="8" stroke="currentColor" stroke-width="1"/>'),
        ('/about','ab','About','<circle cx="7" cy="7" r="5.5" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="7" y1="6" x2="7" y2="10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="7" cy="4" r=".7" fill="currentColor"/>'),
    ]
    links=''.join(
        f'<a href="{href}" class="nl{" active" if pg==active else ""}">'
        f'<svg viewBox="0 0 14 14" fill="none">{ico}</svg>{lbl}'
        f'{"<span class=nbdg>"+str(ac)+"</span>" if pg=="al" and ac>0 else ""}'
        f'</a>'
        for href,pg,lbl,ico in pages)
    return f"""<nav>
  <a href="/" class="brand">
    <div class="bico"><svg viewBox="0 0 16 16" fill="none">
      <circle cx="4" cy="12" r="2.2" fill="#388BFD"/><circle cx="8" cy="7" r="2.2" fill="#D29922"/><circle cx="12" cy="3" r="2.2" fill="#F85149"/>
      <line x1="4" y1="12" x2="8" y2="7" stroke="#388BFD" stroke-width="1.2"/><line x1="8" y1="7" x2="12" y2="3" stroke="#D29922" stroke-width="1.2"/>
    </svg></div>
    <div><div class="bnm">GMS</div><div class="bsb">Mission Control · N=40</div></div>
  </a>
  <div style="display:flex;align-items:stretch;height:100%;flex:1">{links}</div>
  <div class="nav-r">
    <span class="pill p-live">LIVE</span>
    <span class="pill p-t" id="nav-t">t = {t:03d}</span>
    <span class="pill {noise_cls}" id="nav-noise">{noise_txt}</span>
    <span class="pill {pill_cls}" id="nav-st">{pill_txt}</span>
  </div>
</nav>"""

def rp(tmpl,active):
    return render_template_string(
        tmpl.replace('{{CSS}}',CSS)
            .replace('{{NAV}}',navbar(active))
            .replace('{{UPD}}',NAV_UPD))

# ══════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════

DASH=r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>GMS — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
{{CSS}}{{NAV}}
<style>
.body{display:grid;grid-template-columns:220px 1fr 260px;height:calc(100vh - 52px);overflow:hidden}
.left{background:var(--panel);border-right:1px solid var(--border);overflow-y:auto;padding:11px 9px;display:flex;flex-direction:column;gap:9px}
.centre{display:flex;flex-direction:column;overflow:hidden}
.right{background:var(--panel);border-left:1px solid var(--border);overflow-y:auto;padding:11px 9px;display:flex;flex-direction:column;gap:9px}
.playbar{display:flex;align-items:center;gap:9px;padding:7px 14px;background:var(--panel);border-bottom:1px solid var(--border);flex-shrink:0}
.tdisp{font-size:17px;font-weight:700;color:var(--blue);min-width:65px;font-variant-numeric:tabular-nums}
.mw{flex:1;position:relative;background:var(--bg);min-height:0}
#sc{width:100%;height:100%;display:block}
.mbdg{position:absolute;top:8px;left:8px;display:flex;gap:5px;flex-wrap:wrap}
.evb{font-size:9px;font-weight:700;padding:3px 8px;border-radius:4px;opacity:0;transition:opacity .4s;border:1px solid}
.evb.on{opacity:1}
.charts{height:215px;display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--border);flex-shrink:0}
.cb{padding:8px;position:relative}
.ct{font-size:9px;color:var(--fg2);letter-spacing:.06em;text-transform:uppercase;margin-bottom:3px}
.ngrid{display:grid;grid-template-columns:repeat(5,1fr);gap:3px}
.nb{padding:3px 0;border-radius:4px;font-size:9px;font-weight:700;cursor:pointer;border:1px solid var(--border);background:var(--card);color:var(--fg);font-family:inherit;transition:all .15s;text-align:center}
.nb.sel{border-color:var(--blue);color:var(--blue)}
.nb.high{border-color:var(--red)!important;color:var(--red)!important}
.nb.mod{border-color:var(--amber)!important;color:var(--amber)!important}
.evbtn{width:100%;padding:7px 10px;border-radius:var(--r);font-size:10px;font-weight:700;cursor:pointer;border:1px solid;transition:all .2s;font-family:inherit;background:transparent;margin-bottom:4px}
#hc{width:100%;height:80px;display:block;border-radius:4px;margin-top:4px}
.sw{font-size:10px;text-align:center;padding:3px;border-radius:4px;display:none}
.sw.show{display:block;background:rgba(248,81,73,.1);color:var(--red)}
.noise-banner{background:rgba(210,153,34,.08);border:1px solid rgba(210,153,34,.3);border-radius:var(--r);padding:7px 10px;font-size:10px;color:var(--amber);text-align:center;display:none}
.noise-banner.show{display:block}
</style>
</head><body><div class="body">

<!-- LEFT SIDEBAR -->
<div class="left">
  <div class="sec">Simulation Controls</div>
  <div style="display:flex;gap:5px;flex-wrap:wrap">
    <button id="btn-play" onclick="togglePlay()">▶ PLAY</button>
    <button onclick="api('reset')">↺ RESET</button>
    <button onclick="api('step',{dir:-1})">◀</button>
    <button onclick="api('step',{dir:1})">▶</button>
    <button onclick="exportData()" style="width:100%;margin-top:6px">
  ⬇ Export CSV
</button>
<button onclick="fetch('/reset_logs',{method:'POST'}).then(()=>alert('Logs cleared'))">
  Clear Logs
</button>
  </div>
  
  <div class="ctrl"><label>Speed</label><input type="range" min="1" max="20" value="7" step="1" oninput="setSpd(this.value)" id="spsl"><span class="v" id="spv">×7</span></div>

  <div class="sec">Noise Experiment</div>
  <div class="toggle-wrap">
    <button class="tgl" id="noise-tgl" onclick="toggleNoise()"></button>
    <label onclick="toggleNoise()">Inject ±0.5°C Noise</label>
  </div>
  <div class="noise-banner" id="noise-banner">⚡ Noise Active — GMS uses Duration+NIS to stay robust</div>

  <div class="sec">GMS Weights</div>
  <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:6px 8px;font-size:10px;color:var(--purple);margin-bottom:2px">S = w₁|ΔT|+w₂|M|+w₃·NIS+w₄·D</div>
  <div id="sw" class="sw"></div>
  <div class="ctrl"><label style="color:var(--blue)">w₁ Gradient</label><input type="range" min="0.05" max="0.70" step="0.01" value="0.35" id="sl-w1" oninput="upW('w1',this.value)"><span class="v" id="vl-w1">0.35</span></div>
  <div class="ctrl"><label style="color:var(--teal)">w₂ Momentum</label><input type="range" min="0.05" max="0.70" step="0.01" value="0.25" id="sl-w2" oninput="upW('w2',this.value)"><span class="v" id="vl-w2">0.25</span></div>
  <div class="ctrl"><label style="color:var(--purple)">w₃ NIS</label><input type="range" min="0.05" max="0.70" step="0.01" value="0.20" id="sl-w3" oninput="upW('w3',this.value)"><span class="v" id="vl-w3">0.20</span></div>
  <div class="ctrl"><label style="color:var(--amber)">w₄ Duration</label><input type="range" min="0.05" max="0.70" step="0.01" value="0.20" id="sl-w4" oninput="upW('w4',this.value)"><span class="v" id="vl-w4">0.20</span></div>

  <div class="sec">Thresholds</div>
  <div class="ctrl"><label style="color:var(--teal)">θ Theta</label><input type="range" min="0.2" max="3.0" step="0.1" value="1.2" id="sl-th" oninput="upT('theta',this.value)"><span class="v" id="vl-th">1.2</span></div>
  <div class="ctrl"><label style="color:var(--amber)">α Mod</label><input type="range" min="0.10" max="0.50" step="0.01" value="0.25" id="sl-al" oninput="upT('alpha',this.value)"><span class="v" id="vl-al">0.25</span></div>
  <div class="ctrl"><label style="color:var(--red)">β High</label><input type="range" min="0.40" max="0.90" step="0.01" value="0.60" id="sl-be" oninput="upT('beta',this.value)"><span class="v" id="vl-be">0.60</span></div>

  <div class="sec">Node Inspector (N=40)</div>
  <div class="ngrid" id="ngrid"></div>

  <div class="sec" style="margin-top:4px">Trigger Events</div>
  <button class="evbtn" style="color:#388BFD;border-color:rgba(56,139,253,.4)" onclick="api('trigger_event',{idx:0})">▶ Event A — N0–N4</button>
  <button class="evbtn" style="color:#3FB950;border-color:rgba(63,185,80,.4)"  onclick="api('trigger_event',{idx:1})">▶ Event B — N15–N18</button>
  <button class="evbtn" style="color:#D29922;border-color:rgba(210,153,34,.4)" onclick="api('trigger_event',{idx:2})">▶ Event C — N8–N11</button>
  <button class="evbtn" style="color:#BC8CFF;border-color:rgba(188,140,255,.4)" onclick="api('trigger_event',{idx:3})">▶ Event D — N25–N28</button>
</div>

<!-- CENTRE -->
<div class="centre">
  <div class="playbar">
    <div class="tdisp" id="tdisp">t = 000</div>
    <input type="range" id="tl" min="0" max="119" value="0" oninput="api('jump',{t:+this.value})" style="flex:1">
    <span style="font-size:10px;color:var(--fg2)">/ 119 &nbsp;·&nbsp; SPACE=play &nbsp;←→=step &nbsp;N=noise &nbsp;1-4=events</span>
  </div>
  <div class="mw" id="mw">
    <canvas id="sc"></canvas>
    <div class="mbdg">
      <div id="ba" class="evb" style="color:#388BFD;border-color:rgba(56,139,253,.5);background:rgba(56,139,253,.1)">■ A</div>
      <div id="bb" class="evb" style="color:#3FB950;border-color:rgba(63,185,80,.5);background:rgba(63,185,80,.1)">■ B</div>
      <div id="bc" class="evb" style="color:#D29922;border-color:rgba(210,153,34,.5);background:rgba(210,153,34,.1)">■ C</div>
      <div id="bd" class="evb" style="color:#BC8CFF;border-color:rgba(188,140,255,.5);background:rgba(188,140,255,.1)">■ D</div>
    </div>
    <div class="tt" id="tt"></div>
  </div>
  <div class="charts">
    <div class="cb"><div class="ct">GMS — N<span id="cn1">0</span> vs network mean</div><canvas id="gmsChart"></canvas></div>
    <div class="cb"><div class="ct">Temperature — N<span id="cn2">0</span> vs abs baseline (26.5°C)</div><canvas id="tmpChart"></canvas></div>
  </div>
</div>

<!-- RIGHT SIDEBAR -->
<div class="right">
  <div class="sec">Node Status</div>
  <div class="status-box s-ok" id="sb">● STABLE</div>
  <div style="font-size:10px;color:var(--fg2);text-align:center;margin-top:3px" id="ss">N0 @ t=0</div>

  <div class="sec">Live Metrics — N<span id="mn">0</span></div>
  <div class="mg">
    <div class="mc"><div class="l">GMS</div><div class="v" id="m-gms" style="color:var(--blue)">0.000</div></div>
    <div class="mc"><div class="l">Z-Score</div><div class="v" id="m-zs" style="color:var(--purple);font-size:16px">0.00</div></div>
    <div class="mc"><div class="l">Gradient</div><div class="v" id="m-gr" style="color:var(--teal);font-size:16px">0.00°</div></div>
    <div class="mc"><div class="l">Momentum</div><div class="v" id="m-mo" style="color:var(--purple);font-size:16px">0.00°</div></div>
    <div class="mc"><div class="l">Duration</div><div class="v" id="m-du" style="color:var(--amber)">0.000</div></div>
    <div class="mc"><div class="l">Temp °C</div><div class="v" id="m-te" style="color:var(--fg2);font-size:16px">00.0°</div></div>
  </div>

  <div class="sec">Heatmap (All Nodes × Time)</div>
  <canvas id="hc"></canvas>

  <div class="sec">Alert Log</div>
  <div class="alog" id="alog" style="height:130px">
    <div class="al-i">[SYS] GMS v2 online — N=40 nodes, 4 events.</div>
    <div class="al-i">[SYS] Press PLAY or trigger an event.</div>
  </div>

  <div class="sec">Performance: GMS vs 3 Baselines</div>
  <table class="pt">
    <tr><th>Metric</th><th>Threshold</th><th>Z-Score</th><th>GMS</th></tr>
    <tr><td class="lbl">Accuracy</td><td class="bv" id="pa-b">—</td><td class="zv" id="pa-z">—</td><td class="gv" id="pa-g">—</td></tr>
    <tr><td class="lbl">Precision</td><td class="bv" id="pp-b">—</td><td class="zv" id="pp-z">—</td><td class="gv" id="pp-g">—</td></tr>
    <tr><td class="lbl">Recall</td><td class="bv" id="pr-b">—</td><td class="zv" id="pr-z">—</td><td class="gv" id="pr-g">—</td></tr>
    <tr><td class="lbl">FAR</td><td class="bv" id="pf-b">—</td><td class="zv" id="pf-z">—</td><td class="gv" id="pf-g">—</td></tr>
    <tr><td class="lbl">F1</td><td class="bv" id="p1-b">—</td><td class="zv" id="p1-z">—</td><td class="gv" id="p1-g">—</td></tr>
  </table>
  <div style="font-size:9px;color:var(--fg3);margin-top:3px">Thresh · Z-Score (purple) · GMS proposed (green)</div>
</div>
</div>

{{UPD}}
<script>
const N_NODES=40,T=120,G=10;
let state=null,sn=0,playing=false,noiseOn=false,pend={},ptmr=null;
function $(i){return document.getElementById(i);}

// Charts
const co={animation:{duration:100},plugins:{legend:{display:false},tooltip:{backgroundColor:'#1C2128',titleColor:'#E6EDF3',bodyColor:'#8B949E',borderColor:'#30363D',borderWidth:1}},scales:{x:{ticks:{color:'#484F58',maxTicksLimit:10,font:{size:7}},grid:{color:'rgba(48,54,61,.4)'},border:{color:'#30363D'}},y:{ticks:{color:'#484F58',font:{size:7}},grid:{color:'rgba(48,54,61,.4)'},border:{color:'#30363D'}}},elements:{point:{radius:0,hoverRadius:4}},responsive:true,maintainAspectRatio:false};
const lbs=Array.from({length:T},(_,i)=>i);
const gmsC=new Chart($('gmsChart').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[
  {data:Array(T).fill(0),borderColor:'#388BFD',backgroundColor:'rgba(56,139,253,.12)',fill:true,borderWidth:2},
  {data:Array(T).fill(0),borderColor:'rgba(139,148,158,.2)',borderWidth:1,fill:false},
  {data:Array(T).fill(.25),borderColor:'rgba(210,153,34,.6)',borderWidth:1,borderDash:[4,4],fill:false},
  {data:Array(T).fill(.60),borderColor:'rgba(248,81,73,.6)',borderWidth:1,borderDash:[4,4],fill:false},
]},options:{...co,scales:{...co.scales,y:{...co.scales.y,min:0,max:1.05}}}});
const tmpC=new Chart($('tmpChart').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[
  {data:Array(T).fill(22),borderColor:'#3FB950',backgroundColor:'rgba(63,185,80,.08)',fill:true,borderWidth:2},
  {data:Array(T).fill(26.5),borderColor:'rgba(248,81,73,.6)',borderWidth:1,borderDash:[4,4],fill:false},
]},options:co});

// Canvas map
const canvas=$('sc'),ctx=canvas.getContext('2d');
function rsz(){const w=$('mw');canvas.width=w.clientWidth;canvas.height=w.clientHeight;}
window.addEventListener('resize',()=>{rsz();if(state)dMap(state);});rsz();

function ts(x,y){const p=28,cw=canvas.width,ch=canvas.height;return[p+(x/G)*(cw-2*p),p+(1-y/G)*(ch-2*p)];}
function nc(l){return l===2?'#F85149':l===1?'#D29922':'#388BFD';}

function dMap(s){
  const w=canvas.width,h=canvas.height; ctx.clearRect(0,0,w,h);
  // Grid
  ctx.strokeStyle='rgba(48,54,61,.25)';ctx.lineWidth=.4;
  for(let v=0;v<=10;v+=2){const[x0,y0]=ts(v,0),[x1,y1]=ts(v,10);ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();const[a,b]=ts(0,v),[c,d]=ts(10,v);ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,d);ctx.stroke();}
  // Edges
  for(let i=0;i<N_NODES;i++){for(const j of s.adj[i]){if(j<=i)continue;const[xi,yi]=ts(s.nodes[i].x,s.nodes[i].y),[xj,yj]=ts(s.nodes[j].x,s.nodes[j].y);const ht=Math.min(1,(Math.abs(s.nodes[i].grad)+Math.abs(s.nodes[j].grad))/2/6);ctx.strokeStyle=`rgba(${Math.round(13+(248-13)*ht)},${Math.round(17+(81-17)*ht)},${Math.round(23+(73-23)*ht)},${.2+ht*.5})`;ctx.lineWidth=.5+ht*1.8;ctx.beginPath();ctx.moveTo(xi,yi);ctx.lineTo(xj,yj);ctx.stroke();}}
  // Prop arrows
  for(const pe of s.prop_edges){const ni=s.nodes[pe.src],nj=s.nodes[pe.dst];const[xi,yi]=ts(ni.x,ni.y),[xj,yj]=ts(nj.x,nj.y);const dx=xj-xi,dy=yj-yi,len=Math.sqrt(dx*dx+dy*dy)+.001,ux=dx/len,uy=dy/len,al=.25+pe.strength*.6;ctx.strokeStyle=`rgba(210,153,34,${al})`;ctx.lineWidth=.8+pe.strength*1.8;ctx.beginPath();ctx.moveTo(xi+ux*10,yi+uy*10);ctx.lineTo(xj-ux*10,yj-uy*10);ctx.stroke();const ang=Math.atan2(yj-yi,xj-xi),ex=xj-ux*10,ey=yj-uy*10;ctx.fillStyle=`rgba(210,153,34,${al})`;ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-7*Math.cos(ang-.45),ey-7*Math.sin(ang-.45));ctx.lineTo(ex-7*Math.cos(ang+.45),ey-7*Math.sin(ang+.45));ctx.closePath();ctx.fill();}
  // Halos
  for(let i=0;i<N_NODES;i++){const nd=s.nodes[i];if(nd.label<1)continue;const[sx,sy]=ts(nd.x,nd.y),r=12+nd.gms*22,col=nd.label===2?'248,81,73':'210,153,34';const gr=ctx.createRadialGradient(sx,sy,0,sx,sy,r);gr.addColorStop(0,`rgba(${col},.18)`);gr.addColorStop(1,`rgba(${col},0)`);ctx.fillStyle=gr;ctx.beginPath();ctx.arc(sx,sy,r,0,2*Math.PI);ctx.fill();}
  // Nodes
  for(let i=0;i<N_NODES;i++){const nd=s.nodes[i];const[sx,sy]=ts(nd.x,nd.y);const r=6+nd.gms*8,col=nc(nd.label);if(i===sn){ctx.strokeStyle='rgba(255,255,255,.85)';ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(sx,sy,r+4,0,2*Math.PI);ctx.stroke();}ctx.fillStyle=col;ctx.beginPath();ctx.arc(sx,sy,r,0,2*Math.PI);ctx.fill();ctx.fillStyle='rgba(255,255,255,.9)';ctx.font=`bold ${7+nd.gms*2}px Courier New`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(`${i}`,sx,sy);ctx.fillStyle=col;ctx.font='7px Courier New';ctx.fillText(nd.gms.toFixed(2),sx,sy+r+7);}
  ctx.fillStyle='rgba(139,148,158,.4)';ctx.font='9px Courier New';ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText(`GMS · N=${N_NODES} nodes · ${s.noise_on?'⚡ NOISE ON':'CLEAN'}`,8,5);
}

// Heatmap
const hcv=$('hc'),hx=hcv.getContext('2d');
function dHeat(gf,t){
  hcv.width=hcv.clientWidth||240;hcv.height=80;const w=hcv.width,h=hcv.height,cw=w/T,ch=h/N_NODES;hx.clearRect(0,0,w,h);
  for(let i=0;i<N_NODES;i++){for(let t2=0;t2<T;t2++){const v=gf[i][t2];hx.fillStyle=v<.3?`rgba(56,139,253,${.12+v*2})`:v<.6?`rgb(${Math.round(56+(210-56)*(v-.3)/.3)},${Math.round(139+(153-139)*(v-.3)/.3)},${Math.round(253+(34-253)*(v-.3)/.3)})`:(`rgb(${Math.round(210+(248-210)*(v-.6)/.4)},${Math.round(153+(81-153)*(v-.6)/.4)},${Math.round(34+(73-34)*(v-.6)/.4)})`);hx.fillRect(t2*cw,(N_NODES-1-i)*ch,Math.ceil(cw)+1,Math.ceil(ch)+1);}}
  hx.strokeStyle='rgba(255,255,255,.8)';hx.lineWidth=1.5;hx.beginPath();hx.moveTo(t*cw,0);hx.lineTo(t*cw,h);hx.stroke();
  hx.strokeStyle='rgba(56,139,253,.6)';hx.lineWidth=1;hx.strokeRect(0,(N_NODES-1-sn)*ch,w,ch);
}

// Build node grid
function buildGrid(){const g=$('ngrid');for(let i=0;i<N_NODES;i++){const b=document.createElement('button');b.id=`nb-${i}`;b.className='nb'+(i===0?' sel':'');b.textContent=`${i}`;b.onclick=(n=>()=>sn=n)(i);g.appendChild(b);}}
buildGrid();

function render(s){
  state=s;const t=s.t,nd=s.nodes[sn];
  $('tdisp').textContent=`t = ${String(t).padStart(3,'0')}`;$('tl').value=t;
  $('cn1').textContent=sn;$('cn2').textContent=sn;$('mn').textContent=sn;
  playing=s.playing;$('btn-play').textContent=playing?'⏸ PAUSE':'▶ PLAY';$('btn-play').className=playing?'pri':'';
  // noise banner
  const nb=$('noise-tgl'),ban=$('noise-banner');
  nb.className=s.noise_on?'tgl on':'tgl';ban.className=s.noise_on?'noise-banner show':'noise-banner';
  // event badges
  ['a','b','c','d'].forEach((k,i)=>{const on=s.active_events.includes(['Event A','Event B','Event C','Event D'][i]);$(`b${k}`).classList.toggle('on',on);});
  dMap(s);
  // Charts
  const gd=s.gms_full[sn],gm=lbs.map(t2=>s.gms_full.reduce((a,r)=>a+r[t2],0)/N_NODES);
  gmsC.data.datasets[0].data=gd;gmsC.data.datasets[1].data=gm;
  gmsC.data.datasets[2].data=Array(T).fill(s.alpha);gmsC.data.datasets[3].data=Array(T).fill(s.beta);
  gmsC.data.datasets[0].pointRadius=gd.map((_,i)=>i===t?5:0);gmsC.update('none');
  const td=s.temp_full[sn];tmpC.data.datasets[0].data=td;tmpC.data.datasets[0].pointRadius=td.map((_,i)=>i===t?5:0);tmpC.update('none');
  dHeat(s.gms_full,t);
  // Status
  const sb=$('sb');if(nd.label===2){sb.className='status-box s-hi';sb.textContent='▲ HIGH UNSTABLE';}else if(nd.label===1){sb.className='status-box s-mo';sb.textContent='◆ MOD. UNSTABLE';}else{sb.className='status-box s-ok';sb.textContent='● STABLE';}
  $('ss').textContent=`N${sn} @ t=${t}`;
  $('m-gms').textContent=nd.gms.toFixed(3);$('m-zs').textContent=nd.zscore.toFixed(2);$('m-gr').textContent=(nd.grad>=0?'+':'')+nd.grad.toFixed(2)+'°';$('m-mo').textContent=(nd.mom>=0?'+':'')+nd.mom.toFixed(2)+'°';$('m-du').textContent=nd.dur.toFixed(3);$('m-te').textContent=nd.temp.toFixed(1)+'°';
  // Node buttons
  s.nodes.forEach((n,i)=>{const b=$(`nb-${i}`);if(!b)return;b.className='nb'+(i===sn?' sel':'')+(n.label===2?' high':n.label===1?' mod':'');});
  // Performance table - 3 baselines
  const pg=s.perf_gms,pb=s.perf_base,pz=s.perf_z;
  $('pa-b').textContent=pb.acc+'%';$('pa-z').textContent=pz.acc+'%';$('pa-g').textContent=pg.acc+'%';
  $('pp-b').textContent=pb.prec+'%';$('pp-z').textContent=pz.prec+'%';$('pp-g').textContent=pg.prec+'%';
  $('pr-b').textContent=pb.rec+'%';$('pr-z').textContent=pz.rec+'%';$('pr-g').textContent=pg.rec+'%';
  $('pf-b').textContent=pb.far+'%';$('pf-z').textContent=pz.far+'%';$('pf-g').textContent=pg.far+'%';
  $('p1-b').textContent=pb.f1+'%';$('p1-z').textContent=pz.f1+'%';$('p1-g').textContent=pg.f1+'%';
}

function onFrame(d){render(d);}
function onAlert(msg){
  const log=$('alog'),cls={danger:'al-d',warn:'al-w',ok:'al-o',info:'al-i'}[msg.level]||'al-i';
  const ico={danger:'▲',warn:'◆',ok:'✓',info:'·'}[msg.level]||'·';
  const d=document.createElement('div');d.className=cls;
  d.textContent=`[t=${String(msg.t).padStart(3,'0')}] ${ico} ${msg.msg}`;
  log.insertBefore(d,log.firstChild);if(log.children.length>80)log.removeChild(log.lastChild);
}

// Tooltip
canvas.addEventListener('mousemove',e=>{if(!state)return;const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=1e9;for(let i=0;i<N_NODES;i++){const[sx,sy]=ts(state.nodes[i].x,state.nodes[i].y);const d=Math.hypot(sx-mx,sy-my);if(d<bd){bd=d;best=i;}}const tt=$('tt');if(bd<26&&best>=0){const nd=state.nodes[best];const cls=['Stable','Mod. Unstable','High Unstable'][nd.label];tt.innerHTML=`<div class="tt-t">N${best} — ${cls}</div>GMS: <b>${nd.gms.toFixed(3)}</b><br>Z-Score: <b>${nd.zscore.toFixed(2)}</b><br>ΔT: <b>${nd.grad>0?'+':''}${nd.grad.toFixed(2)}°</b><br>M: <b>${nd.mom>0?'+':''}${nd.mom.toFixed(2)}</b><br>NIS: <b>${nd.nis.toFixed(3)}</b><br>Temp: <b>${nd.temp.toFixed(1)}°C</b>`;tt.style.display='block';tt.style.left=(mx+14)+'px';tt.style.top=(my-10)+'px';}else tt.style.display='none';});
canvas.addEventListener('click',e=>{if(!state)return;const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=1e9;for(let i=0;i<N_NODES;i++){const[sx,sy]=ts(state.nodes[i].x,state.nodes[i].y);const d=Math.hypot(sx-mx,sy-my);if(d<bd){bd=d;best=i;}}if(bd<28)sn=best;});
canvas.addEventListener('mouseleave',()=>{$('tt').style.display='none';});

async function api(ep,b={}){await fetch(`/api/${ep}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
function togglePlay(){playing?api('pause'):api('play');}
function toggleNoise(){api('toggle_noise',{on:!noiseOn}).then(()=>{noiseOn=!noiseOn;});}
function setSpd(v){$('spv').textContent='×'+v;api('speed',{speed:Math.max(.04,1.2-v*.058)});}
function upW(k,v){const val=parseFloat((+v).toFixed(2));$(`vl-${k}`).textContent=val.toFixed(2);pend[k]=val;const s=['w1','w2','w3','w4'].reduce((a,k2)=>a+parseFloat($(`sl-${k2}`).value),0);const sw=$('sw');if(Math.abs(s-1)>.06){sw.className='sw show';sw.textContent=`⚠ Weights sum to ${s.toFixed(2)}`;}else sw.className='sw';sched();}
function upT(k,v){const val=parseFloat((+v).toFixed(2));const m={theta:'th',alpha:'al',beta:'be'}[k];$(`vl-${m}`).textContent=val.toFixed(2);pend[k]=val;sched();}
function sched(){if(ptmr)clearTimeout(ptmr);ptmr=setTimeout(()=>{api('params',pend);pend={};},400);}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.code==='ArrowRight')api('step',{dir:1});if(e.code==='ArrowLeft')api('step',{dir:-1});if(e.code==='KeyR')api('reset');if(e.code==='KeyN')toggleNoise();if(e.key==='1')api('trigger_event',{idx:0});if(e.key==='2')api('trigger_event',{idx:1});if(e.key==='3')api('trigger_event',{idx:2});if(e.key==='4')api('trigger_event',{idx:3});});
</script></body></html>"""

# ══════════════════════════════════════════════════════════════════════
#  MAP PAGE
# ══════════════════════════════════════════════════════════════════════

MAP=r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>GMS — Sensor Map</title>
{{CSS}}{{NAV}}
<style>
.mf{position:relative;height:calc(100vh - 52px);background:var(--bg)}
#mc{width:100%;height:100%;display:block}
.ctrls{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);display:flex;gap:8px;align-items:center;background:rgba(22,27,34,.92);border:1px solid var(--border);border-radius:var(--rl);padding:9px 14px;flex-wrap:wrap;justify-content:center}
.leg{position:absolute;top:12px;right:12px;background:rgba(22,27,34,.9);border:1px solid var(--border);border-radius:var(--r);padding:10px 14px;font-size:10px;line-height:2}
.lr{display:flex;align-items:center;gap:7px}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.info{position:absolute;top:12px;left:12px;background:rgba(22,27,34,.9);border:1px solid var(--border);border-radius:var(--r);padding:10px 14px;font-size:10px;min-width:165px;line-height:1.9}
</style>
</head><body><div class="mf">
  <canvas id="mc"></canvas>
  <div class="info">
    <div style="color:var(--fg2);font-size:9px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Selected node</div>
    <div id="in" style="color:var(--blue);font-weight:700;font-size:14px">N0</div>
    <div id="is" style="color:var(--teal)">● Stable</div>
    <div style="margin-top:6px;border-top:1px solid var(--border);padding-top:6px">
      <div>GMS: <b id="ig">—</b></div><div>Z: <b id="iz">—</b></div>
      <div>ΔT: <b id="id">—</b></div><div>Temp: <b id="it">—</b></div>
    </div>
  </div>
  <div class="leg">
    <div style="color:var(--fg2);font-size:9px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px">N = 40 nodes</div>
    <div class="lr"><div class="ld" style="background:var(--blue)"></div>Stable</div>
    <div class="lr"><div class="ld" style="background:var(--amber)"></div>Mod. Unstable</div>
    <div class="lr"><div class="ld" style="background:var(--red)"></div>High Unstable</div>
    <div class="lr" style="margin-top:4px"><div style="width:20px;height:2px;background:var(--amber);margin-right:7px"></div>Propagation</div>
  </div>
  <div class="ctrls">
    <button onclick="api('reset')">↺</button>
    <button id="bp" onclick="tPlay()">▶ PLAY</button>
    <button onclick="api('step',{dir:-1})">◀</button>
    <input type="range" id="tl" min="0" max="119" value="0" style="width:160px" oninput="api('jump',{t:+this.value})">
    <button onclick="api('step',{dir:1})">▶</button>
    <span id="td" style="color:var(--blue);font-weight:700;min-width:58px">t = 000</span>
    <button onclick="api('trigger_event',{idx:0})" style="color:#388BFD">A</button>
    <button onclick="api('trigger_event',{idx:1})" style="color:#3FB950">B</button>
    <button onclick="api('trigger_event',{idx:2})" style="color:#D29922">C</button>
    <button onclick="api('trigger_event',{idx:3})" style="color:#BC8CFF">D</button>
    <button onclick="api('toggle_noise',{on:true})" class="warn-btn" style="font-size:10px">⚡ Noise</button>
    <button onclick="api('toggle_noise',{on:false})" style="font-size:10px">Clear Noise</button>
  </div>
  <div class="tt" id="tt"></div>
</div>
{{UPD}}
<script>
const NN=40,T=120,G=10;let state=null,sel=0,playing=false;
const canvas=document.getElementById('mc'),ctx=canvas.getContext('2d');
function rsz(){canvas.width=canvas.parentElement.clientWidth;canvas.height=canvas.parentElement.clientHeight;}
window.addEventListener('resize',()=>{rsz();if(state)draw(state);});rsz();
function ts(x,y){const p=42,w=canvas.width,h=canvas.height;return[p+(x/G)*(w-2*p),p+(1-y/G)*(h-2*p)];}
function nc(l){return l===2?'#F85149':l===1?'#D29922':'#388BFD';}
function draw(s){
  const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='rgba(48,54,61,.2)';ctx.lineWidth=.4;
  for(let v=0;v<=10;v++){const[x0,y0]=ts(v,0),[x1,y1]=ts(v,10);ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();const[a,b]=ts(0,v),[c,d]=ts(10,v);ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,d);ctx.stroke();}
  for(let i=0;i<NN;i++){for(const j of s.adj[i]){if(j<=i)continue;const[xi,yi]=ts(s.nodes[i].x,s.nodes[i].y),[xj,yj]=ts(s.nodes[j].x,s.nodes[j].y);const ht=Math.min(1,(Math.abs(s.nodes[i].grad)+Math.abs(s.nodes[j].grad))/2/6);ctx.strokeStyle=`rgba(${Math.round(13+(248-13)*ht)},${Math.round(17+(81-17)*ht)},${Math.round(23+(73-23)*ht)},${.15+ht*.45})`;ctx.lineWidth=.5+ht*1.5;ctx.beginPath();ctx.moveTo(xi,yi);ctx.lineTo(xj,yj);ctx.stroke();}}
  for(const pe of s.prop_edges){const ni=s.nodes[pe.src],nj=s.nodes[pe.dst];const[xi,yi]=ts(ni.x,ni.y),[xj,yj]=ts(nj.x,nj.y);const dx=xj-xi,dy=yj-yi,len=Math.sqrt(dx*dx+dy*dy)+.001,ux=dx/len,uy=dy/len,al=.25+pe.strength*.6;ctx.strokeStyle=`rgba(210,153,34,${al})`;ctx.lineWidth=1+pe.strength*2;ctx.beginPath();ctx.moveTo(xi+ux*12,yi+uy*12);ctx.lineTo(xj-ux*12,yj-uy*12);ctx.stroke();const ang=Math.atan2(yj-yi,xj-xi),ex=xj-ux*12,ey=yj-uy*12;ctx.fillStyle=`rgba(210,153,34,${al})`;ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-9*Math.cos(ang-.42),ey-9*Math.sin(ang-.42));ctx.lineTo(ex-9*Math.cos(ang+.42),ey-9*Math.sin(ang+.42));ctx.closePath();ctx.fill();}
  for(let i=0;i<NN;i++){const nd=s.nodes[i];if(nd.label<1)continue;const[sx,sy]=ts(nd.x,nd.y),r=15+nd.gms*32,col=nd.label===2?'248,81,73':'210,153,34';const gr=ctx.createRadialGradient(sx,sy,0,sx,sy,r);gr.addColorStop(0,`rgba(${col},.18)`);gr.addColorStop(1,`rgba(${col},0)`);ctx.fillStyle=gr;ctx.beginPath();ctx.arc(sx,sy,r,0,2*Math.PI);ctx.fill();}
  for(let i=0;i<NN;i++){const nd=s.nodes[i];const[sx,sy]=ts(nd.x,nd.y);const r=8+nd.gms*10,col=nc(nd.label);if(i===sel){ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.arc(sx,sy,r+4,0,2*Math.PI);ctx.stroke();}ctx.fillStyle=col;ctx.beginPath();ctx.arc(sx,sy,r,0,2*Math.PI);ctx.fill();ctx.fillStyle='rgba(255,255,255,.92)';ctx.font=`bold ${8+nd.gms*3}px Courier New`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(`${i}`,sx,sy);ctx.fillStyle=col;ctx.font='8px Courier New';ctx.fillText(nd.gms.toFixed(2),sx,sy+r+9);}
}
function upInfo(s){const nd=s.nodes[sel],col=nc(nd.label);document.getElementById('in').textContent=`N${sel}`;document.getElementById('in').style.color=col;document.getElementById('is').textContent=['● Stable','◆ Mod.','▲ High'][nd.label];document.getElementById('is').style.color=col;document.getElementById('ig').textContent=nd.gms.toFixed(3);document.getElementById('iz').textContent=nd.zscore.toFixed(2);document.getElementById('id').textContent=(nd.grad>0?'+':'')+nd.grad.toFixed(2)+'°';document.getElementById('it').textContent=nd.temp.toFixed(1)+'°C';}
function onFrame(s){state=s;draw(s);upInfo(s);playing=s.playing;document.getElementById('bp').textContent=playing?'⏸ PAUSE':'▶ PLAY';document.getElementById('bp').className=playing?'pri':'';document.getElementById('tl').value=s.t;document.getElementById('td').textContent='t = '+String(s.t).padStart(3,'0');}
canvas.addEventListener('click',e=>{if(!state)return;const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=1e9;for(let i=0;i<NN;i++){const[sx,sy]=ts(state.nodes[i].x,state.nodes[i].y);const d=Math.hypot(sx-mx,sy-my);if(d<bd){bd=d;best=i;}}if(bd<35){sel=best;if(state)draw(state);}});
canvas.addEventListener('mousemove',e=>{if(!state)return;const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=1e9;for(let i=0;i<NN;i++){const[sx,sy]=ts(state.nodes[i].x,state.nodes[i].y);const d=Math.hypot(sx-mx,sy-my);if(d<bd){bd=d;best=i;}}const tt=document.getElementById('tt');if(bd<30&&best>=0){const nd=state.nodes[best];tt.innerHTML=`<div class="tt-t">N${best}</div>GMS: <b>${nd.gms.toFixed(3)}</b><br>Z-Score: <b>${nd.zscore.toFixed(2)}</b><br>ΔT: <b>${nd.grad.toFixed(2)}°</b><br>Temp: <b>${nd.temp.toFixed(1)}°C</b>`;tt.style.display='block';tt.style.left=(mx+14)+'px';tt.style.top=(my-10)+'px';}else tt.style.display='none';});
canvas.addEventListener('mouseleave',()=>{document.getElementById('tt').style.display='none';});
async function api(ep,b={}){await fetch(`/api/${ep}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
function tPlay(){playing?api('pause'):api('play');}
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();tPlay();}if(e.code==='ArrowRight')api('step',{dir:1});if(e.code==='ArrowLeft')api('step',{dir:-1});if(e.code==='KeyR')api('reset');if(e.key==='1')api('trigger_event',{idx:0});if(e.key==='2')api('trigger_event',{idx:1});if(e.key==='3')api('trigger_event',{idx:2});if(e.key==='4')api('trigger_event',{idx:3});});
</script></body></html>"""

# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS PAGE — all paper figures + noise comparison
# ══════════════════════════════════════════════════════════════════════

ANALYSIS=r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>GMS — Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
{{CSS}}{{NAV}}
<style>
.page{padding:16px 20px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px}
.g5{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:14px}
.ns{display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.ns label{font-size:11px;color:var(--fg2)}
.ns select{background:var(--card);color:var(--fg);border:1px solid var(--border);border-radius:var(--r);padding:4px 10px;font-family:inherit;font-size:11px;cursor:pointer}
#hf{width:100%;height:100px;display:block;border-radius:4px}
.noise-info{background:rgba(210,153,34,.06);border:1px solid rgba(210,153,34,.25);border-radius:var(--r);padding:8px 12px;font-size:10px;color:var(--amber);margin-bottom:10px;display:none}
.noise-info.show{display:block}
</style>
</head><body><div class="page">
<div class="ns">
  <label>Node:</label>
  <select id="nsel" onchange="sn=+this.value;if(state)upC(state)">
    <script>for(let i=0;i<40;i++)document.write('<option value="'+i+'">N'+i+'</option>');</script>
  </select>
  <label>t = <b id="td" style="color:var(--blue)">000</b></label>
  <button onclick="api('play')" style="font-size:10px">▶ PLAY</button>
  <button onclick="api('pause')" style="font-size:10px">⏸</button>
  <button onclick="api('reset')" style="font-size:10px">↺</button>
  <button onclick="api('trigger_event',{idx:0})" style="color:#388BFD;font-size:10px">Event A</button>
  <button onclick="api('trigger_event',{idx:1})" style="color:#3FB950;font-size:10px">Event B</button>
  <button onclick="api('trigger_event',{idx:2})" style="color:#D29922;font-size:10px">Event C</button>
  <button onclick="api('trigger_event',{idx:3})" style="color:#BC8CFF;font-size:10px">Event D</button>
  <button onclick="api('toggle_noise',{on:true})" class="warn-btn" style="font-size:10px">⚡ Noise ON</button>
  <button onclick="api('toggle_noise',{on:false})" style="font-size:10px">Noise OFF</button>
</div>
<div class="noise-info" id="noise-info">⚡ Noise active — observe GMS stability vs baseline fluctuation in charts below</div>

<div class="sec">Fig 3 — GMS Component Decomposition  (Node N<span id="cn">0</span>)</div>
<div class="g2" style="margin-bottom:10px">
  <div class="card"><div style="font-size:9px;color:var(--blue);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">a) Spatial Gradient  ΔT_i(t) = T_i − mean(T_j)</div><canvas id="gC" height="90"></canvas></div>
  <div class="card"><div style="font-size:9px;color:var(--teal);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">b) Temporal Momentum  M(t) = ΔT(t) − ΔT(t−1)</div><canvas id="mC" height="90"></canvas></div>
  <div class="card"><div style="font-size:9px;color:var(--amber);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">c) Duration / Persistence  D_i(t)  [noise filter]</div><canvas id="dC" height="90"></canvas></div>
  <div class="card"><div style="font-size:9px;color:var(--purple);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">d) Neighbor Influence Score  NIS_i(t)</div><canvas id="nC" height="90"></canvas></div>
</div>
<div class="card" style="margin-bottom:12px"><div style="font-size:9px;color:var(--red);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">e) GMS Composite Score  S = w₁|ΔT|+w₂|M|+w₃·NIS+w₄·D</div><canvas id="gsC" height="80"></canvas></div>

<div class="sec">Fig 4 — Spatio-Temporal GMS Heatmap (N=40 nodes × 120 time steps)</div>
<div class="card" style="margin-bottom:12px">
  <canvas id="hf"></canvas>
  <div style="display:flex;gap:16px;margin-top:5px;font-size:9px;color:var(--fg2)"><span style="color:var(--blue)">■ Stable</span><span style="color:var(--amber)">■ Mod. Unstable</span><span style="color:var(--red)">■ High Unstable</span><span style="color:var(--fg3)">│ = current t</span></div>
</div>

<div class="sec">Fig 5 — Baseline vs GMS vs Z-Score Detection  (Node N<span id="cn2">0</span>)</div>
<div class="g3" style="margin-bottom:12px">
  <div class="card"><div style="font-size:9px;color:var(--teal);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">Temperature vs absolute threshold (26.5°C)</div><canvas id="tpC" height="95"></canvas></div>
  <div class="card"><div style="font-size:9px;color:var(--purple);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">Z-Score anomaly  |Z| > 1.2</div><canvas id="zC" height="95"></canvas></div>
  <div class="card"><div style="font-size:9px;color:var(--blue);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">GMS Score vs thresholds (α=0.25, β=0.60)</div><canvas id="g2C" height="95"></canvas></div>
</div>

<div class="sec">Performance — GMS vs 3 Baselines (noise: <span id="noise-lbl" style="color:var(--amber)">OFF</span>)</div>
<div class="g5">
  <div class="mc"><div class="l">Accuracy</div><div class="v" id="pa-g" style="color:var(--teal)">—</div><div style="font-size:9px;color:var(--purple)" id="pa-z">Z: —</div><div style="font-size:9px;color:var(--fg3)" id="pa-b">Base: —</div></div>
  <div class="mc"><div class="l">Precision</div><div class="v" id="pp-g" style="color:var(--teal)">—</div><div style="font-size:9px;color:var(--purple)" id="pp-z">Z: —</div><div style="font-size:9px;color:var(--fg3)" id="pp-b">Base: —</div></div>
  <div class="mc"><div class="l">Recall</div><div class="v" id="pr-g" style="color:var(--teal)">—</div><div style="font-size:9px;color:var(--purple)" id="pr-z">Z: —</div><div style="font-size:9px;color:var(--fg3)" id="pr-b">Base: —</div></div>
  <div class="mc"><div class="l">FAR</div><div class="v" id="pf-g" style="color:var(--amber)">—</div><div style="font-size:9px;color:var(--purple)" id="pf-z">Z: —</div><div style="font-size:9px;color:var(--fg3)" id="pf-b">Base: —</div></div>
  <div class="mc"><div class="l">F1 Score</div><div class="v" id="p1-g" style="color:var(--teal)">—</div><div style="font-size:9px;color:var(--purple)" id="p1-z">Z: —</div><div style="font-size:9px;color:var(--fg3)" id="p1-b">Base: —</div></div>
</div>
</div>
{{UPD}}
<script>
const NN=40,T=120;let state=null,sn=0;const lbs=Array.from({length:T},(_,i)=>i);
const co={animation:{duration:80},plugins:{legend:{display:false},tooltip:{backgroundColor:'#1C2128',titleColor:'#E6EDF3',bodyColor:'#8B949E',borderColor:'#30363D',borderWidth:1}},scales:{x:{ticks:{color:'#484F58',maxTicksLimit:10,font:{size:7}},grid:{color:'rgba(48,54,61,.4)'},border:{color:'#30363D'}},y:{ticks:{color:'#484F58',font:{size:7}},grid:{color:'rgba(48,54,61,.4)'},border:{color:'#30363D'}}},elements:{point:{radius:0,hoverRadius:3}},responsive:true,maintainAspectRatio:false};
function mk(id,col,mn,mx){const o={...co,scales:{...co.scales,y:{...co.scales.y}}};if(mn!=null)o.scales.y.min=mn;if(mx!=null)o.scales.y.max=mx;return new Chart(document.getElementById(id).getContext('2d'),{type:'line',data:{labels:lbs,datasets:[{data:Array(T).fill(0),borderColor:col,backgroundColor:col.replace('rgb','rgba').replace(')',`,.1)`),fill:true,borderWidth:1.5}]},options:o});}
const gC=mk('gC','#388BFD'),mC=mk('mC','#3FB950'),dC=mk('dC','#D29922',0,1),nC=mk('nC','#BC8CFF',0,1);
const gsC=new Chart(document.getElementById('gsC').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[{data:Array(T).fill(0),borderColor:'#F85149',backgroundColor:'rgba(248,81,73,.1)',fill:true,borderWidth:2},{data:Array(T).fill(.25),borderColor:'rgba(210,153,34,.6)',borderWidth:1,borderDash:[4,4],fill:false},{data:Array(T).fill(.6),borderColor:'rgba(248,81,73,.5)',borderWidth:1,borderDash:[4,4],fill:false}]},options:{...co,scales:{...co.scales,y:{...co.scales.y,min:0,max:1.05}}}});
const tpC=new Chart(document.getElementById('tpC').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[{data:Array(T).fill(22),borderColor:'#3FB950',backgroundColor:'rgba(63,185,80,.1)',fill:true,borderWidth:1.5},{data:Array(T).fill(26.5),borderColor:'rgba(248,81,73,.6)',borderWidth:1,borderDash:[4,4],fill:false}]},options:co});
const zC=new Chart(document.getElementById('zC').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[{data:Array(T).fill(0),borderColor:'#BC8CFF',backgroundColor:'rgba(188,140,255,.1)',fill:true,borderWidth:1.5},{data:Array(T).fill(1.2),borderColor:'rgba(248,81,73,.6)',borderWidth:1,borderDash:[4,4],fill:false}]},options:{...co,scales:{...co.scales,y:{...co.scales.y,min:0}}}});
const g2C=new Chart(document.getElementById('g2C').getContext('2d'),{type:'line',data:{labels:lbs,datasets:[{data:Array(T).fill(0),borderColor:'#388BFD',backgroundColor:'rgba(56,139,253,.1)',fill:true,borderWidth:1.5},{data:Array(T).fill(.25),borderColor:'rgba(210,153,34,.6)',borderWidth:1,borderDash:[4,4],fill:false},{data:Array(T).fill(.6),borderColor:'rgba(248,81,73,.5)',borderWidth:1,borderDash:[4,4],fill:false}]},options:{...co,scales:{...co.scales,y:{...co.scales.y,min:0,max:1.05}}}});
const hfc=document.getElementById('hf'),hx=hfc.getContext('2d');
function dHeat(gf,t){hfc.width=hfc.clientWidth||900;hfc.height=100;const w=hfc.width,h=hfc.height,cw=w/T,ch=h/NN;hx.clearRect(0,0,w,h);for(let i=0;i<NN;i++){for(let t2=0;t2<T;t2++){const v=gf[i][t2];hx.fillStyle=v<.3?`rgba(56,139,253,${.1+v*2})`:v<.6?`rgb(${Math.round(56+(210-56)*(v-.3)/.3)},${Math.round(139+(153-139)*(v-.3)/.3)},${Math.round(253+(34-253)*(v-.3)/.3)})`:(`rgb(${Math.round(210+(248-210)*(v-.6)/.4)},${Math.round(153+(81-153)*(v-.6)/.4)},${Math.round(34+(73-34)*(v-.6)/.4)})`);hx.fillRect(t2*cw,(NN-1-i)*ch,Math.ceil(cw)+1,Math.ceil(ch)+1);}}hx.strokeStyle='rgba(255,255,255,.85)';hx.lineWidth=2;hx.beginPath();hx.moveTo(t*cw,0);hx.lineTo(t*cw,h);hx.stroke();}
function upC(s){
  const n=sn,t=s.t;
  document.getElementById('cn').textContent=n;document.getElementById('cn2').textContent=n;document.getElementById('td').textContent=String(t).padStart(3,'0');
  document.getElementById('noise-lbl').textContent=s.noise_on?'ON (⚡)':'OFF';
  document.getElementById('noise-info').className=s.noise_on?'noise-info show':'noise-info';
  function sc(c,d){c.data.datasets[0].data=d;c.data.datasets[0].pointRadius=d.map((_,i)=>i===t?4:0);c.update('none');}
  sc(gC,s.grad_full[n]);sc(mC,s.mom_full[n]);sc(dC,s.dur_full[n]);sc(nC,s.nis_full[n]);
  gsC.data.datasets[0].data=s.gms_full[n];gsC.data.datasets[0].pointRadius=s.gms_full[n].map((_,i)=>i===t?5:0);gsC.data.datasets[1].data=Array(T).fill(s.alpha);gsC.data.datasets[2].data=Array(T).fill(s.beta);gsC.update('none');
  sc(tpC,s.temp_full[n]);
  sc(zC,s.zscore_full[n]);
  g2C.data.datasets[0].data=s.gms_full[n];g2C.data.datasets[0].pointRadius=s.gms_full[n].map((_,i)=>i===t?5:0);g2C.data.datasets[1].data=Array(T).fill(s.alpha);g2C.data.datasets[2].data=Array(T).fill(s.beta);g2C.update('none');
  dHeat(s.gms_full,t);
  const pg=s.perf_gms,pb=s.perf_base,pz=s.perf_z;
  document.getElementById('pa-g').textContent=pg.acc+'%';document.getElementById('pa-z').textContent='Z: '+pz.acc+'%';document.getElementById('pa-b').textContent='Base: '+pb.acc+'%';
  document.getElementById('pp-g').textContent=pg.prec+'%';document.getElementById('pp-z').textContent='Z: '+pz.prec+'%';document.getElementById('pp-b').textContent='Base: '+pb.prec+'%';
  document.getElementById('pr-g').textContent=pg.rec+'%';document.getElementById('pr-z').textContent='Z: '+pz.rec+'%';document.getElementById('pr-b').textContent='Base: '+pb.rec+'%';
  document.getElementById('pf-g').textContent=pg.far+'%';document.getElementById('pf-z').textContent='Z: '+pz.far+'%';document.getElementById('pf-b').textContent='Base: '+pb.far+'%';
  document.getElementById('p1-g').textContent=pg.f1+'%';document.getElementById('p1-z').textContent='Z: '+pz.f1+'%';document.getElementById('p1-b').textContent='Base: '+pb.f1+'%';
}
function onFrame(s){state=s;upC(s);}
async function api(ep,b={}){await fetch(`/api/${ep}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
</script></body></html>"""

# ══════════════════════════════════════════════════════════════════════
#  ALERTS PAGE
# ══════════════════════════════════════════════════════════════════════

ALERTS=r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>GMS — Alerts</title>
{{CSS}}{{NAV}}
<style>
.page{padding:16px 20px;max-width:920px;margin:0 auto}
.fbar{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.fb{font-size:10px;font-weight:700;padding:4px 12px;border-radius:999px;border:1px solid var(--border);cursor:pointer;background:var(--card);color:var(--fg2);transition:all .15s;font-family:inherit}
.fb.on-all{border-color:var(--fg2);color:var(--fg)}
.fb.on-hi{border-color:var(--red);color:var(--red);background:rgba(248,81,73,.08)}
.fb.on-mo{border-color:var(--amber);color:var(--amber);background:rgba(210,153,34,.08)}
.fb.on-ok{border-color:var(--teal);color:var(--teal);background:rgba(63,185,80,.08)}
.fb.on-in{border-color:var(--blue);color:var(--blue);background:rgba(56,139,253,.08)}
.alf{background:var(--card);border:1px solid var(--border);border-radius:var(--rl);padding:10px;font-size:11px;line-height:2;min-height:400px;max-height:calc(100vh - 270px);overflow-y:auto}
.alf::-webkit-scrollbar{width:4px}.alf::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.ar{display:flex;gap:10px;padding:4px 6px;border-radius:4px;transition:background .1s;align-items:baseline}
.ar:hover{background:rgba(255,255,255,.04)}
.ar .ts{color:var(--fg3);min-width:55px;font-size:10px}.ar .ic{min-width:14px;font-size:12px;font-weight:700}.ar .mg{flex:1}
.al-d .ic,.al-d .mg{color:var(--red)}.al-w .ic,.al-w .mg{color:var(--amber)}.al-o .ic,.al-o .mg{color:var(--teal)}.al-i .ic,.al-i .mg{color:var(--blue)}
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
</style>
</head><body><div class="page">
<div class="sg">
  <div class="mc"><div class="l">High Unstable</div><div class="v" id="ch" style="color:var(--red)">0</div></div>
  <div class="mc"><div class="l">Mod. Unstable</div><div class="v" id="cm" style="color:var(--amber)">0</div></div>
  <div class="mc"><div class="l">Recoveries</div><div class="v" id="co" style="color:var(--teal)">0</div></div>
  <div class="mc"><div class="l">Total Alerts</div><div class="v" id="ca" style="color:var(--fg)">0</div></div>
</div>
<div class="fbar">
  <span style="font-size:10px;color:var(--fg2)">Filter:</span>
  <button class="fb on-all" id="fb-all"  onclick="setF('all')">All</button>
  <button class="fb"        id="fb-hi"   onclick="setF('danger')">▲ High</button>
  <button class="fb"        id="fb-mo"   onclick="setF('warn')">◆ Moderate</button>
  <button class="fb"        id="fb-ok"   onclick="setF('ok')">✓ Recovered</button>
  <button class="fb"        id="fb-in"   onclick="setF('info')">· System</button>
  <button onclick="al=[];rl();us()" class="dng" style="margin-left:auto;font-size:10px">Clear</button>
  <button onclick="api('play')" style="font-size:10px;margin-left:8px">▶ PLAY</button>
  <button onclick="api('pause')" style="font-size:10px">⏸</button>
  <button onclick="api('trigger_event',{idx:0})" style="color:#388BFD;font-size:10px">A</button>
  <button onclick="api('trigger_event',{idx:1})" style="color:#3FB950;font-size:10px">B</button>
  <button onclick="api('trigger_event',{idx:2})" style="color:#D29922;font-size:10px">C</button>
  <button onclick="api('trigger_event',{idx:3})" style="color:#BC8CFF;font-size:10px">D</button>
  <button onclick="api('toggle_noise',{on:true})" class="warn-btn" style="font-size:10px">⚡ Noise</button>
</div>
<div class="alf" id="alf"><div style="color:var(--fg3);font-size:10px;text-align:center;padding:20px">Waiting — press PLAY or trigger an event</div></div>
</div>
{{UPD}}
<script>
let al=[],filt='all';
function setF(f){
  filt=f;
  const m={all:'all',danger:'hi',warn:'mo',ok:'ok',info:'in'};
  ['all','hi','mo','ok','in'].forEach(k=>{const b=document.getElementById('fb-'+k);if(b)b.className='fb';});
  const b=document.getElementById('fb-'+(m[f]||f));if(b)b.className='fb on-'+(m[f]||f);rl();
}
function rl(){
  const log=document.getElementById('alf');const fi=filt==='all'?al:al.filter(a=>a.level===filt);
  if(!fi.length){log.innerHTML='<div style="color:var(--fg3);font-size:10px;text-align:center;padding:20px">No alerts for this filter.</div>';return;}
  const cls={danger:'al-d',warn:'al-w',ok:'al-o',info:'al-i'},ico={danger:'▲',warn:'◆',ok:'✓',info:'·'};
  log.innerHTML=fi.map(a=>`<div class="ar ${cls[a.level]||'al-i'}"><span class="ts">[t=${String(a.t).padStart(3,'0')}]</span><span class="ic">${ico[a.level]||'·'}</span><span class="mg">${a.msg}</span></div>`).join('');
}
function us(){document.getElementById('ch').textContent=al.filter(a=>a.level==='danger').length;document.getElementById('cm').textContent=al.filter(a=>a.level==='warn').length;document.getElementById('co').textContent=al.filter(a=>a.level==='ok').length;document.getElementById('ca').textContent=al.length;}
function onFrame(d){}
function onAlert(msg){al.unshift({msg:msg.msg,level:msg.level,t:msg.t});if(al.length>300)al.pop();rl();us();}
async function api(ep,b={}){await fetch(`/api/${ep}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
fetch('/api/alerts').then(r=>r.json()).then(d=>{al=d;rl();us();});
</script></body></html>"""

# ══════════════════════════════════════════════════════════════════════
#  ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════

ABOUT=r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>GMS — About</title>
{{CSS}}{{NAV}}
<style>
.page{padding:24px 28px;max-width:860px;margin:0 auto}
.eq-card{background:var(--card);border:1px solid var(--border);border-radius:var(--rl);padding:16px 20px;margin-bottom:14px}
.eq{font-size:15px;color:var(--purple);font-weight:700;margin:6px 0}
.eq-d{font-size:11px;color:var(--fg2);line-height:1.8;margin-top:6px}
.cg{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:14px}
.cc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:12px 14px}
.cn{font-size:11px;font-weight:700;margin-bottom:5px}
.ceq{font-size:13px;color:var(--purple);margin-bottom:5px;font-weight:700}
.cd{font-size:10px;color:var(--fg2);line-height:1.7}
.wt{width:100%;border-collapse:collapse;font-size:11px}
.wt th{text-align:left;color:var(--fg2);font-size:9px;text-transform:uppercase;letter-spacing:.07em;padding:4px 8px;border-bottom:1px solid var(--border)}
.wt td{padding:5px 8px;border-bottom:1px solid rgba(48,54,61,.5)}
.wt tr:last-child td{border-bottom:none}
.wk{color:var(--purple);font-weight:700}
.ab{font-size:12px;color:var(--fg2);line-height:1.9;background:var(--card);border:1px solid var(--border);border-radius:var(--rl);padding:16px 20px;margin-bottom:14px;border-left:3px solid var(--blue)}
.contribution{background:rgba(56,139,253,.06);border:1px solid rgba(56,139,253,.25);border-radius:var(--rl);padding:14px 18px;margin-bottom:14px;font-size:11px;color:var(--fg2);line-height:1.9}
.contribution strong{color:var(--blue)}
</style>
</head><body><div class="page">
<div style="margin-bottom:20px">
  <div style="font-size:18px;font-weight:700;color:var(--fg);margin-bottom:4px">Gradient–Momentum Based Microclimate Instability Detection</div>
  <div style="font-size:12px;color:var(--fg2)">Using Spatial–Temporal Sensor Networks</div>
  <div style="font-size:11px;color:var(--fg3);margin-top:6px">IEEE Conference Paper · Chanthu S, Manpreet Kaur · Lovely Professional University</div>
</div>

<div class="ab">This paper proposes a lightweight and interpretable framework for detecting pre-event microclimate instability using distributed sensor networks. Unlike conventional approaches relying on absolute thresholds or late-stage event detection, the proposed GMS method analyzes spatial gradients, temporal momentum, duration, and neighbor influence — achieving higher accuracy and lower false alarm rates compared to both threshold-based and Z-score statistical baselines. The system is extended to N=40 nodes to validate scalability, maintaining stable O(N) complexity.</div>

<div class="contribution">
  <strong>Framework contribution:</strong> A real-time simulation platform for analyzing microclimate instability in WSNs, supporting interactive experimentation with weight sensitivity, noise robustness testing, and multi-baseline comparison — all demonstrated live in this dashboard.
</div>

<div class="sec">GMS Unified Score (Eq. 7) — core novelty</div>
<div class="eq-card">
  <div class="eq">S_i(t) = w₁·|ΔT_i| + w₂·|M_i| + w₃·NIS_i + w₄·D_i</div>
  <div class="eq-d">
    <b style="color:var(--fg)">Default weights:</b> w₁=0.35 · w₂=0.25 · w₃=0.20 · w₄=0.20 &nbsp;|&nbsp;
    <b style="color:var(--fg)">Classification:</b> &lt;0.25 Stable · 0.25–0.60 Mod · ≥0.60 High<br>
    <b style="color:var(--fg)">Complexity:</b> O(N·k)≈O(N) · Memory: O(k)/node · N=40 nodes validated<br>
    <b style="color:var(--fg)">Baselines compared:</b> Absolute threshold (T>26.5°C) · Z-score (|Z|>1.2) · Proposed GMS
  </div>
</div>

<div class="sec">Four Core Components</div>
<div class="cg">
  <div class="cc"><div class="cn" style="color:var(--blue)">① Spatial Gradient  <span style="font-size:9px;color:var(--fg3)">[Eq. 2–3]</span></div><div class="ceq">ΔT_ij(t) = T_i(t) − T_j(t)</div><div class="cd">Relative temperature difference between neighbors. Approximates ∇T(x,t)·d_ij. Detects localized variation invisible to absolute-value methods.</div></div>
  <div class="cc"><div class="cn" style="color:var(--teal)">② Temporal Momentum  <span style="font-size:9px;color:var(--fg3)">[Eq. 4]</span></div><div class="ceq">M_ij(t) = ΔT(t) − ΔT(t−1)</div><div class="cd">First-order temporal difference ≈ d(ΔT)/dt. Positive momentum = increasing instability → enables early detection before threshold crossings.</div></div>
  <div class="cc"><div class="cn" style="color:var(--amber)">③ Duration / Persistence  <span style="font-size:9px;color:var(--fg3)">[Eq. 5]</span></div><div class="ceq">D_i: counter if |ΔT| &gt; θ</div><div class="cd">Low-pass temporal filter. Transient noise ε(t) dissipates; true instability persists. Demonstrated by noise experiment (toggle on Dashboard).</div></div>
  <div class="cc"><div class="cn" style="color:var(--purple)">④ Neighbor Influence Score  <span style="font-size:9px;color:var(--fg3)">[Eq. 6]</span></div><div class="ceq">NIS_i = (1/|N(i)|) Σ ΔT_ij</div><div class="cd">Mean gradient over neighbors N(i). Models spatial propagation. A node becomes influenced when its neighbors are unstable — captures spread dynamics.</div></div>
</div>

<div class="sec">Scalability & Robustness Notes</div>
<div class="eq-card" style="font-size:11px;color:var(--fg2);line-height:1.9">
  <b style="color:var(--fg)">Scalability:</b> Network extended from 12 to 40 nodes with randomized placement. Performance remains stable, supporting the O(N·k)≈O(N) complexity claim.<br>
  <b style="color:var(--fg)">Noise robustness:</b> Random noise ±0.5°C injected (toggle on Dashboard). GMS remains stable due to Duration (persistence filter) and NIS (spatial averaging). Absolute threshold exhibits increased false alarms under noise.<br>
  <b style="color:var(--fg)">Weight sensitivity:</b> Use the Dashboard sliders to vary w₁–w₄. Performance remains stable across weight variations, demonstrating model robustness.<br>
  <b style="color:var(--fg)">Z-score baseline:</b> Per-node statistical anomaly detector |Z|>1.2. Compared alongside absolute threshold in all metric panels.
</div>

<div class="sec">Parameter Reference</div>
<div class="eq-card">
  <table class="wt">
    <tr><th>Symbol</th><th>Component</th><th>Default</th><th>Role</th></tr>
    <tr><td class="wk" style="color:var(--blue)">w₁</td><td>Spatial Gradient |ΔT|</td><td>0.35</td><td>Primary instability signal</td></tr>
    <tr><td class="wk" style="color:var(--teal)">w₂</td><td>Temporal Momentum |M|</td><td>0.25</td><td>Early trend detection</td></tr>
    <tr><td class="wk" style="color:var(--purple)">w₃</td><td>Neighbor Influence NIS</td><td>0.20</td><td>Propagation modeling</td></tr>
    <tr><td class="wk" style="color:var(--amber)">w₄</td><td>Duration D</td><td>0.20</td><td>Noise filtering</td></tr>
    <tr><td class="wk" style="color:var(--amber)">θ</td><td>Gradient threshold</td><td>1.2 °C</td><td>Minimum gradient for Duration counter</td></tr>
    <tr><td class="wk" style="color:var(--amber)">α</td><td>Mod. threshold</td><td>0.25</td><td>GMS ≥ α → Moderately Unstable</td></tr>
    <tr><td class="wk" style="color:var(--red)">β</td><td>High threshold</td><td>0.60</td><td>GMS ≥ β → Highly Unstable</td></tr>
    <tr><td class="wk" style="color:var(--purple)">Z_thr</td><td>Z-score threshold</td><td>1.2 σ</td><td>Statistical anomaly baseline</td></tr>
  </table>
</div>

<div class="sec">Dataset</div>
<div class="eq-card" style="font-size:11px;color:var(--fg2);line-height:1.9">
  <b style="color:var(--fg)">Source:</b> NASA POWER (T2M + RH2M) &nbsp;|&nbsp;
  <b style="color:var(--fg)">Network:</b> N=40 nodes · 10×10 km · radius=2.8 km · T=120 steps<br>
  <b style="color:var(--fg)">Events:</b> A — N0–N4 · t=20–55 · ΔT=8°C &nbsp;|&nbsp; B — N15–N18 · t=35–70 · ΔT=6°C &nbsp;|&nbsp; C — N8–N11 · t=50–90 · ΔT=7°C &nbsp;|&nbsp; D — N25–N28 · t=65–100 · ΔT=5.5°C
</div>
</div>
{{UPD}}
<script>function onFrame(d){}function onAlert(m){}</script>
</body></html>"""

# ══════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════
@app.route('/reset_logs', methods=['POST'])
def reset_logs():
    engine.logs = []
    return {"status": "logs cleared"}

@app.route('/export')
def export():
    import pandas as pd

    df = pd.DataFrame(engine.logs)

    filename = "gms_noise.csv" if engine.noise_on else "gms_clean.csv"
    file_path = f"outputs/{filename}"

    df.to_csv(file_path, index=False)

    return {"status": "saved", "file": file_path}
@app.route('/')         
def pg_db():     return rp(DASH,'db')
@app.route('/map')      
def pg_map():    return rp(MAP,'map')
@app.route('/analysis') 
def pg_an():     return rp(ANALYSIS,'an')
@app.route('/alerts')   
def pg_al():     return rp(ALERTS,'al')
@app.route('/about')    
def pg_ab():     return rp(ABOUT,'ab')

@app.route('/api/state')
def api_state(): return jsonify(engine.frame_data())
@app.route('/api/alerts')
def api_alerts():return jsonify(engine.alert_history)

@app.route('/api/play',         methods=['POST'])
def api_play():  engine.play();  return jsonify(ok=True)
@app.route('/api/pause',        methods=['POST'])
def api_pause(): engine.pause(); return jsonify(ok=True)
@app.route('/api/reset',        methods=['POST'])
def api_reset(): engine.reset(); return jsonify(ok=True)
@app.route('/api/jump',         methods=['POST'])
def api_jump():  engine.jump(int(request.json.get('t',0))); return jsonify(ok=True)
@app.route('/api/step',         methods=['POST'])
def api_step():  engine.step(int(request.json.get('dir',1))); return jsonify(ok=True)
@app.route('/api/speed',        methods=['POST'])
def api_speed(): engine.speed=float(request.json.get('speed',.25)); return jsonify(ok=True)
@app.route('/api/params',       methods=['POST'])
def api_params():engine.rerun(request.json); return jsonify(ok=True)
@app.route('/api/trigger_event',methods=['POST'])
def api_trig():  engine.trigger(int(request.json.get('idx',0))); return jsonify(ok=True)
@app.route('/api/toggle_noise', methods=['POST'])
def api_noise(): engine.toggle_noise(bool(request.json.get('on',False))); return jsonify(ok=True)

@app.route('/stream')
def stream():
    q=engine.subscribe()
    q.append(json.dumps({'type':'frame','data':engine.frame_data()}))
    def gen():
        try:
            while True:
                if q: yield f"data: {q.pop(0)}\n\n"
                else: yield ": hb\n\n"; time.sleep(0.04)
        except GeneratorExit: engine.unsubscribe(q)
    return Response(gen(),mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

if __name__=='__main__':
    print("\n"+"="*55)
    print("  GMS Mission Control  v2")
    print("  40 nodes · Z-score baseline · Noise toggle")
    print("  http://localhost:5000")
    print("  Ctrl+C to stop")
    print("="*55+"\n")
    app.run(debug=False,threaded=True,port=5000)