const DEFAULT_N = 40;
const DEFAULT_T = 120;
const COLORS = {
  blue: '#388BFD',
  teal: '#3FB950',
  amber: '#D29922',
  red: '#F85149',
  purple: '#BC8CFF',
  fg: '#E6EDF3',
  fg2: '#8B949E',
  fg3: '#484F58',
  border: '#30363D'
};

let state = null;
let selectedNode = 0;
let selectedTime = 0;
let charts = {};
let chartIds = {};
let heatmapMode = 'flat';
let heatmapPlot = {left:0,top:0,width:0,height:0};
let expanded = {type:null, sourceId:null, chart:null, plot:null};

const $ = id => document.getElementById(id);
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const fmt = (value, digits = 3) => Number.isFinite(value) ? Number(value).toFixed(digits) : '--';
function frameTimeLabel(s, time){
  const t = Math.max(0, Math.round(Number(time) || 0));
  if(s?.time_axis?.[t]) return s.time_axis[t];
  if(s?.time_label && t === s.t) return s.time_label;
  const total = Math.max(1, Number(s?.T) || DEFAULT_T);
  const mins = Math.round(t / total * 24 * 60) % (24 * 60);
  return `${String(Math.floor(mins / 60)).padStart(2, '0')}:${String(mins % 60).padStart(2, '0')}`;
}
const labels = length => Array.from({length}, (_, i) => frameTimeLabel(state, i));

function rgba(hex, alpha){
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

function heatColor(value, alpha = 1){
  const v = clamp(value || 0, 0, 1);
  if(v < .3){
    return `rgba(56,139,253,${(.14 + v * 1.55) * alpha})`;
  }
  if(v < .6){
    const p = (v - .3) / .3;
    return `rgba(${Math.round(56 + (210 - 56) * p)},${Math.round(139 + (153 - 139) * p)},${Math.round(253 + (34 - 253) * p)},${alpha})`;
  }
  const p = (v - .6) / .4;
  return `rgba(${Math.round(210 + (248 - 210) * p)},${Math.round(153 + (81 - 153) * p)},${Math.round(34 + (73 - 34) * p)},${alpha})`;
}

function heatRgb(value){
  const v = clamp(value || 0, 0, 1);
  if(v < .3){
    const p = v / .3;
    return [
      Math.round(20 + 36 * p),
      Math.round(56 + 83 * p),
      Math.round(108 + 145 * p)
    ];
  }
  if(v < .6){
    const p = (v - .3) / .3;
    return [
      Math.round(56 + (210 - 56) * p),
      Math.round(139 + (153 - 139) * p),
      Math.round(253 + (34 - 253) * p)
    ];
  }
  const p = (v - .6) / .4;
  return [
    Math.round(210 + (248 - 210) * p),
    Math.round(153 + (81 - 153) * p),
    Math.round(34 + (73 - 34) * p)
  ];
}

function gmsAt(s, nodeFloat, timeFloat){
  const n0 = clamp(Math.floor(nodeFloat), 0, s.N - 1);
  const n1 = clamp(n0 + 1, 0, s.N - 1);
  const t0 = clamp(Math.floor(timeFloat), 0, s.T - 1);
  const t1 = clamp(t0 + 1, 0, s.T - 1);
  const nf = nodeFloat - n0;
  const tf = timeFloat - t0;
  const a = s.gms_full[n0][t0] * (1 - tf) + s.gms_full[n0][t1] * tf;
  const b = s.gms_full[n1][t0] * (1 - tf) + s.gms_full[n1][t1] * tf;
  return a * (1 - nf) + b * nf;
}

function pathThrough(ctx, points){
  if(!points.length) return;
  ctx.moveTo(points[0][0], points[0][1]);
  for(let i = 1; i < points.length - 1; i += 1){
    const midX = (points[i][0] + points[i + 1][0]) / 2;
    const midY = (points[i][1] + points[i + 1][1]) / 2;
    ctx.quadraticCurveTo(points[i][0], points[i][1], midX, midY);
  }
  const last = points[points.length - 1];
  ctx.lineTo(last[0], last[1]);
}

function statusFor(score, alpha, beta){
  if(score >= beta) return {name:'High unstable', className:'high'};
  if(score >= alpha) return {name:'Moderate unstable', className:'moderate'};
  return {name:'Stable', className:'stable'};
}

function chartOptions(yMin = null, yMax = null){
  return {
    animation:{duration:80},
    responsive:true,
    maintainAspectRatio:false,
    interaction:{mode:'nearest',intersect:false},
    plugins:{
      legend:{display:false},
      tooltip:{
        backgroundColor:'#1C2128',
        titleColor:COLORS.fg,
        bodyColor:COLORS.fg2,
        borderColor:COLORS.border,
        borderWidth:1
      }
    },
    scales:{
      x:{
        ticks:{color:COLORS.fg3,maxTicksLimit:10,font:{size:8}},
        grid:{color:'rgba(48,54,61,.38)'},
        border:{color:COLORS.border}
      },
      y:{
        ticks:{color:COLORS.fg3,font:{size:8}},
        grid:{color:'rgba(48,54,61,.38)'},
        border:{color:COLORS.border},
        min:yMin == null ? undefined : yMin,
        max:yMax == null ? undefined : yMax
      }
    },
    elements:{point:{radius:0,hoverRadius:4}}
  };
}

function makeLineChart(id, color, yMin = null, yMax = null){
  if(!window.Chart) return null;
  const canvas = $(id);
  const ctx = canvas.getContext('2d');
  const chart = new Chart(ctx, {
    type:'line',
    data:{
      labels:labels(DEFAULT_T),
      datasets:[{
        data:Array(DEFAULT_T).fill(0),
        borderColor:color,
        backgroundColor:rgba(color, .12),
        fill:true,
        borderWidth:1.8,
        tension:.18
      }]
    },
    options:chartOptions(yMin, yMax)
  });
  chartIds[id] = chart;
  canvas.addEventListener('click', event => handleChartClick(chart, event, id));
  return chart;
}

function makeThresholdChart(id, mainColor, thresholds){
  if(!window.Chart) return null;
  const ctx = $(id).getContext('2d');
  const datasets = [{
    data:Array(DEFAULT_T).fill(0),
    borderColor:mainColor,
    backgroundColor:rgba(mainColor, .12),
    fill:true,
    borderWidth:1.8,
    tension:.18
  }];
  thresholds.forEach(th => {
    datasets.push({
      data:Array(DEFAULT_T).fill(th.value),
      borderColor:th.color,
      borderWidth:1,
      borderDash:[4,4],
      fill:false,
      pointRadius:0
    });
  });
  const chart = new Chart(ctx, {
    type:'line',
    data:{labels:labels(DEFAULT_T),datasets},
    options:chartOptions(thresholds.some(th => th.minZero) ? 0 : null, thresholds.find(th => th.max)?.max || null)
  });
  chartIds[id] = chart;
  $(id).addEventListener('click', event => handleChartClick(chart, event, id));
  return chart;
}

function initCharts(){
  if(window.Chart && !Chart._analysisBackdropRegistered){
    Chart.register({
      id:'analysisBackdrop',
      beforeDatasetsDraw(chart){
        if(!state || !chart.chartArea) return;
        const {ctx, chartArea, scales} = chart;
        if(!scales.x) return;
        ctx.save();
        for(const ev of state.events || []){
          const x1 = scales.x.getPixelForValue(ev.t_start);
          const x2 = scales.x.getPixelForValue(ev.t_end);
          ctx.fillStyle = `${ev.color}14`;
          ctx.fillRect(x1, chartArea.top, x2 - x1, chartArea.bottom - chartArea.top);
        }
        const nowX = scales.x.getPixelForValue(state.t);
        const selectedX = scales.x.getPixelForValue(selectedTime);
        ctx.strokeStyle = 'rgba(255,255,255,.62)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(nowX, chartArea.top);
        ctx.lineTo(nowX, chartArea.bottom);
        ctx.stroke();
        if(selectedTime !== state.t){
          ctx.strokeStyle = 'rgba(56,139,253,.85)';
          ctx.beginPath();
          ctx.moveTo(selectedX, chartArea.top);
          ctx.lineTo(selectedX, chartArea.bottom);
          ctx.stroke();
        }
        ctx.restore();
      }
    });
    Chart._analysisBackdropRegistered = true;
  }
  charts.gradient = makeLineChart('gradient-chart', COLORS.blue);
  charts.momentum = makeLineChart('momentum-chart', COLORS.teal);
  charts.duration = makeLineChart('duration-chart', COLORS.amber, 0, 1);
  charts.nis = makeLineChart('nis-chart', COLORS.purple, 0, 1);
  charts.gms = makeThresholdChart('gms-chart', COLORS.red, [
    {value:.25,color:'rgba(210,153,34,.65)',minZero:true,max:1.05},
    {value:.6,color:'rgba(248,81,73,.58)',minZero:true,max:1.05}
  ]);
  charts.temp = makeThresholdChart('temp-chart', COLORS.teal, [
    {value:26.5,color:'rgba(248,81,73,.62)'}
  ]);
  charts.z = makeThresholdChart('z-chart', COLORS.purple, [
    {value:1.2,color:'rgba(248,81,73,.62)',minZero:true}
  ]);
  charts.gmsCompare = makeThresholdChart('gms-compare-chart', COLORS.blue, [
    {value:.25,color:'rgba(210,153,34,.65)',minZero:true,max:1.05},
    {value:.6,color:'rgba(248,81,73,.58)',minZero:true,max:1.05}
  ]);
}

function handleChartClick(chart, event, chartId){
  if(!state || !chart) return;
  const points = chart.getElementsAtEventForMode(event, 'nearest', {intersect:false}, true);
  if(!points.length) return;
  const time = points[0].index;
  selectSample(selectedNode, time, `Chart point from ${chartId.replace('-chart', '').replace('-', ' ')}`);
}

function getChartTitle(sourceId){
  const canvas = $(sourceId);
  const card = canvas?.closest('.chart-card,.heatmap-card');
  return card?.querySelector('.chart-title,.heatmap-title')?.textContent?.trim() || 'Analysis graph';
}

function ensureExpandLayer(){
  let layer = $('analysis-expand-layer');
  if(layer) return layer;
  layer = document.createElement('div');
  layer.id = 'analysis-expand-layer';
  layer.className = 'analysis-expand-layer';
  layer.innerHTML = `
    <div class="expand-panel" role="dialog" aria-modal="true" aria-labelledby="expand-title">
      <div class="expand-head">
        <h2 id="expand-title"></h2>
        <button type="button" class="expand-close" aria-label="Close expanded graph">&times;</button>
      </div>
      <div class="expand-canvas-wrap">
        <canvas id="expanded-canvas"></canvas>
      </div>
    </div>`;
  document.body.appendChild(layer);
  layer.addEventListener('click', event => {
    if(event.target === layer) closeExpandedGraph();
  });
  layer.querySelector('.expand-close').addEventListener('click', closeExpandedGraph);
  $('expanded-canvas').addEventListener('click', handleExpandedCanvasClick);
  document.addEventListener('keydown', event => {
    if(event.key === 'Escape') closeExpandedGraph();
  });
  return layer;
}

function closeExpandedGraph(){
  const layer = $('analysis-expand-layer');
  if(layer) layer.classList.remove('show');
  if(expanded.chart){
    expanded.chart.destroy();
  }
  expanded = {type:null, sourceId:null, chart:null, plot:null};
}

function cloneChartData(chart){
  return {
    labels:[...chart.data.labels],
    datasets:chart.data.datasets.map(ds => ({
      data:[...ds.data],
      borderColor:ds.borderColor,
      backgroundColor:ds.backgroundColor,
      fill:ds.fill,
      borderWidth:ds.borderWidth,
      borderDash:ds.borderDash ? [...ds.borderDash] : undefined,
      pointRadius:Array.isArray(ds.pointRadius) ? [...ds.pointRadius] : ds.pointRadius,
      tension:ds.tension ?? .18
    }))
  };
}

function openExpandedGraph(sourceId){
  if(!state) return;
  const layer = ensureExpandLayer();
  const title = getChartTitle(sourceId);
  const canvas = $('expanded-canvas');
  $('expand-title').textContent = title;
  if(expanded.chart){
    expanded.chart.destroy();
  }
  expanded = {type:sourceId === 'heatmap-canvas' ? 'heatmap' : 'chart', sourceId, chart:null, plot:null};
  layer.classList.add('show');
  requestAnimationFrame(() => {
    if(expanded.type === 'heatmap'){
      expanded.plot = drawHeatmapTo(canvas, state);
    }else{
      const source = chartIds[sourceId];
      if(!source || !window.Chart) return;
      const y = source.options.scales?.y || {};
      expanded.chart = new Chart(canvas.getContext('2d'), {
        type:'line',
        data:cloneChartData(source),
        options:chartOptions(y.min ?? null, y.max ?? null)
      });
      expanded.chart.options.plugins.legend.display = source.data.datasets.length > 1;
      expanded.chart.update('none');
    }
  });
}

function refreshExpandedGraph(){
  const layer = $('analysis-expand-layer');
  if(!state || !layer?.classList.contains('show') || !expanded.type) return;
  if(expanded.type === 'heatmap'){
    expanded.plot = drawHeatmapTo($('expanded-canvas'), state);
    return;
  }
  const source = chartIds[expanded.sourceId];
  if(source && expanded.chart){
    expanded.chart.data = cloneChartData(source);
    expanded.chart.update('none');
  }
}

function handleExpandedCanvasClick(event){
  if(!state || !expanded.type) return;
  if(expanded.type === 'heatmap'){
    pickHeatmapSample(event, $('expanded-canvas'), expanded.plot, true);
    return;
  }
  if(!expanded.chart) return;
  const points = expanded.chart.getElementsAtEventForMode(event, 'nearest', {intersect:false}, true);
  if(!points.length) return;
  selectSample(selectedNode, points[0].index, `Expanded ${getChartTitle(expanded.sourceId)}`);
}

function bindExpandableGraphs(){
  document.querySelectorAll('.chart-card').forEach(card => {
    card.classList.add('expandable-graph');
    card.addEventListener('click', event => {
      const canvas = card.querySelector('canvas');
      if(canvas) openExpandedGraph(canvas.id);
    });
  });
  const heatmapCard = document.querySelector('.heatmap-card');
  if(heatmapCard){
    heatmapCard.classList.add('expandable-graph');
    heatmapCard.addEventListener('click', event => {
      if(event.target.closest('.heatmap-controls')) return;
      openExpandedGraph('heatmap-canvas');
    });
  }
}

function setChartData(chart, data, time, thresholdValues = []){
  if(!chart) return;
  chart.data.labels = labels(data.length);
  chart.data.datasets[0].data = data;
  chart.data.datasets[0].pointRadius = data.map((_, i) => i === time ? 5 : 0);
  thresholdValues.forEach((value, index) => {
    if(chart.data.datasets[index + 1]){
      chart.data.datasets[index + 1].data = Array(data.length).fill(value);
    }
  });
  chart.update('none');
}

function buildNodeSelect(count){
  const select = $('analysis-node');
  if(select.options.length === count) return;
  select.innerHTML = '';
  for(let i = 0; i < count; i += 1){
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = `N${i}`;
    select.appendChild(option);
  }
  select.value = String(selectedNode);
}

async function api(endpoint, body = {}){
  await fetch(`/api/${endpoint}`, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  });
}

function bindControls(){
  $('analysis-node').addEventListener('change', event => {
    selectSample(Number(event.target.value), selectedTime, 'Node selector');
  });
  $('heatmap-mode').addEventListener('change', event => {
    heatmapMode = event.target.value;
    if(state) drawHeatmap(state);
  });
  document.querySelectorAll('[data-api]').forEach(button => {
    button.addEventListener('click', () => {
      const endpoint = button.dataset.api;
      const body = endpoint === 'step' ? {dir:Number(button.dataset.dir || 1)} : {};
      api(endpoint, body);
    });
  });
  document.querySelectorAll('[data-event]').forEach(button => {
    button.addEventListener('click', () => api('trigger_event', {idx:Number(button.dataset.event)}));
  });
  document.querySelectorAll('[data-noise]').forEach(button => {
    button.addEventListener('click', () => api('toggle_noise', {on:button.dataset.noise === 'on'}));
  });
  $('heatmap-canvas').addEventListener('click', handleHeatmapClick);
  window.addEventListener('resize', () => {
    if(state) drawHeatmap(state);
  });
}

function selectSample(node, time, source = 'Selection'){
  if(!state) return;
  const nMax = (state.N || DEFAULT_N) - 1;
  const tMax = (state.T || DEFAULT_T) - 1;
  selectedNode = clamp(node, 0, nMax);
  selectedTime = clamp(time, 0, tMax);
  $('analysis-node').value = String(selectedNode);
  updateCharts(state);
  drawHeatmap(state);
  updateInsight(state, source);
  refreshExpandedGraph();
  if(selectedTime !== state.t){
    api('jump', {t:selectedTime});
  }
}

function componentReading(value, type){
  const abs = Math.abs(value || 0);
  if(type === 'duration'){
    if(value >= .66) return 'persistent signal';
    if(value >= .25) return 'short run detected';
    return 'not persistent';
  }
  if(type === 'nis'){
    if(value >= .66) return 'strong neighbor support';
    if(value >= .33) return 'some neighbor support';
    return 'localized reading';
  }
  if(abs >= 1.5) return 'strong movement';
  if(abs >= .5) return 'visible movement';
  return 'quiet';
}

function updateInsight(s, source = 'Live frame'){
  const n = selectedNode;
  const t = selectedTime;
  const gms = s.gms_full[n][t];
  const temp = s.temp_full[n][t];
  const grad = s.grad_full[n][t];
  const mom = s.mom_full[n][t];
  const dur = s.dur_full[n][t];
  const nis = s.nis_full[n][t];
  const z = s.zscore_full[n][t];
  const status = statusFor(gms, s.alpha, s.beta);
  const event = (s.events || []).find(ev => ev.nodes.includes(n) && t >= ev.t_start && t < ev.t_end);
  const eventText = event ? `${event.label} is active for this node.` : 'No ground-truth event is active for this node/time.';
  const thresholdText = status.className === 'high'
    ? `The score is above beta (${fmt(s.beta, 2)}), so GMS marks it high.`
    : status.className === 'moderate'
      ? `The score is above alpha (${fmt(s.alpha, 2)}) but below beta (${fmt(s.beta, 2)}), so it is moderate.`
      : `The score is below alpha (${fmt(s.alpha, 2)}), so it remains stable.`;

  const card = $('insight-card');
  card.className = `insight-card ${status.className}`;
  $('insight-kicker').textContent = source;
  $('insight-title').textContent = `N${n} at ${frameTimeLabel(s, t)} - ${status.name}`;
  $('insight-text').textContent = `${thresholdText} Gradient is ${componentReading(grad, 'gradient')}, momentum is ${componentReading(mom, 'momentum')}, persistence is ${componentReading(dur, 'duration')}, and NIS shows ${componentReading(nis, 'nis')}. ${eventText}`;
  $('ix-gms').textContent = fmt(gms);
  $('ix-temp').textContent = `${fmt(temp, 1)} C`;
  $('ix-grad').textContent = `${grad >= 0 ? '+' : ''}${fmt(grad, 2)} C`;
  $('ix-mom').textContent = `${mom >= 0 ? '+' : ''}${fmt(mom, 2)}`;
  $('ix-dur').textContent = fmt(dur);
  $('ix-nis').textContent = fmt(nis);
  $('ix-z').textContent = fmt(z, 2);
  $('ix-label').textContent = status.name;
  $('read-gradient').textContent = componentReading(grad, 'gradient');
  $('read-momentum').textContent = componentReading(mom, 'momentum');
  $('read-duration').textContent = componentReading(dur, 'duration');
  $('read-nis').textContent = componentReading(nis, 'nis');
}

function updateSummary(s){
  const t = s.t;
  const node = s.nodes[selectedNode] || s.nodes[0];
  const peak = s.nodes.reduce((best, item) => item.gms > best.gms ? item : best, s.nodes[0]);
  const currentScore = s.gms_full[selectedNode][selectedTime];
  const currentStatus = statusFor(currentScore, s.alpha, s.beta);
  $('analysis-time').textContent = frameTimeLabel(s, t);
  $('sum-node').textContent = `N${selectedNode}`;
  $('sum-node-state').textContent = `${currentStatus.name} at ${frameTimeLabel(s, selectedTime)}`;
  $('sum-gms').textContent = fmt(currentScore);
  $('sum-threshold').textContent = `alpha ${fmt(s.alpha, 2)} / beta ${fmt(s.beta, 2)}`;
  $('sum-peak').textContent = `N${peak.id}`;
  $('sum-peak-val').textContent = `${fmt(peak.gms)} GMS now`;
  $('sum-event').textContent = s.active_events.length ? s.active_events.map(name => name.replace('Event ', '')).join('+') : 'None';
  $('sum-noise').textContent = s.noise_on ? 'noise on' : 'clean data';
  $('noise-info').className = s.noise_on ? 'analysis-status show' : 'analysis-status';
  if(node){
    $('analysis-node').value = String(selectedNode);
  }
}

function updateCharts(s){
  const n = selectedNode;
  const time = selectedTime;
  setChartData(charts.gradient, s.grad_full[n], time);
  setChartData(charts.momentum, s.mom_full[n], time);
  setChartData(charts.duration, s.dur_full[n], time);
  setChartData(charts.nis, s.nis_full[n], time);
  setChartData(charts.gms, s.gms_full[n], time, [s.alpha, s.beta]);
  setChartData(charts.temp, s.temp_full[n], time, [26.5]);
  setChartData(charts.z, s.zscore_full[n], time, [1.2]);
  setChartData(charts.gmsCompare, s.gms_full[n], time, [s.alpha, s.beta]);
}

function updatePerformance(s){
  const pairs = [
    ['pa', 'acc'],
    ['pp', 'prec'],
    ['pr', 'rec'],
    ['pf', 'far'],
    ['p1', 'f1']
  ];
  pairs.forEach(([prefix, key]) => {
    $(`${prefix}-g`).textContent = `${s.perf_gms[key]}%`;
    $(`${prefix}-z`).textContent = `Z: ${s.perf_z[key]}%`;
    $(`${prefix}-b`).textContent = `Base: ${s.perf_base[key]}%`;
  });
}

function setCanvasScale(canvas){
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, width:rect.width, height:rect.height};
}

function drawAxes(ctx, w, h, left, top, plotW, plotH, s){
  ctx.fillStyle = 'rgba(139,148,158,.82)';
  ctx.font = '9px Courier New';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  [0, 10, 20, 30, 39].forEach(node => {
    if(node >= s.N) return;
    const y = top + (s.N - 1 - node + .5) / s.N * plotH;
    ctx.fillText(`N${node}`, left - 7, y);
  });
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  [0, 30, 60, 90, s.T - 1].forEach(time => {
    const x = left + time / Math.max(1, s.T - 1) * plotW;
    ctx.fillText(frameTimeLabel(s, time), x, top + plotH + 8);
  });
  ctx.strokeStyle = 'rgba(139,148,158,.20)';
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, plotW, plotH);
}

function drawEventBands(ctx, left, top, plotW, plotH, s){
  for(const ev of s.events || []){
    const x1 = left + ev.t_start / Math.max(1, s.T - 1) * plotW;
    const x2 = left + ev.t_end / Math.max(1, s.T - 1) * plotW;
    const grd = ctx.createLinearGradient(x1, top, x2, top);
    grd.addColorStop(0, `${ev.color}00`);
    grd.addColorStop(.5, `${ev.color}22`);
    grd.addColorStop(1, `${ev.color}00`);
    ctx.fillStyle = grd;
    ctx.fillRect(x1, top, x2 - x1, plotH);
  }
}

function drawSmoothHeatmap(ctx, left, top, plotW, plotH, s){
  const step = 2;
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, plotW, plotH);
  ctx.clip();
  for(let y = 0; y < plotH; y += step){
    const nodeFloat = (1 - y / Math.max(1, plotH - 1)) * (s.N - 1);
    for(let x = 0; x < plotW; x += step){
      const timeFloat = x / Math.max(1, plotW - 1) * (s.T - 1);
      const v = gmsAt(s, nodeFloat, timeFloat);
      const [r, g, b] = heatRgb(v);
      const shade = .84 + .16 * Math.sin((x / plotW) * Math.PI);
      ctx.fillStyle = `rgba(${Math.round(r * shade)},${Math.round(g * shade)},${Math.round(b * shade)},${.42 + v * .58})`;
      ctx.fillRect(left + x, top + y, step + 1, step + 1);
    }
  }

  const glowPoints = [];
  for(let node = 0; node < s.N; node += 1){
    for(let t = 0; t < s.T; t += 1){
      const v = s.gms_full[node][t];
      if(v > .56){
        glowPoints.push({node, t, v});
      }
    }
  }
  glowPoints.sort((a, b) => b.v - a.v).slice(0, 55).forEach(point => {
    const x = left + point.t / Math.max(1, s.T - 1) * plotW;
    const y = top + (1 - point.node / Math.max(1, s.N - 1)) * plotH;
    const radius = 18 + point.v * 34;
    const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
    glow.addColorStop(0, `rgba(248,81,73,${.20 + point.v * .24})`);
    glow.addColorStop(.45, `rgba(210,153,34,${.10 + point.v * .12})`);
    glow.addColorStop(1, 'rgba(248,81,73,0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  });

  drawEventBands(ctx, left, top, plotW, plotH, s);
  drawContourLines(ctx, left, top, plotW, plotH, s);
  ctx.restore();
}

function drawContourLines(ctx, left, top, plotW, plotH, s){
  const levels = [
    {v:s.alpha, color:'rgba(210,153,34,.70)', width:1},
    {v:s.beta, color:'rgba(248,81,73,.80)', width:1.4}
  ];
  levels.forEach(level => {
    ctx.strokeStyle = level.color;
    ctx.lineWidth = level.width;
    for(let node = 0; node < s.N; node += 2){
      let drawing = false;
      ctx.beginPath();
      for(let t = 0; t < s.T; t += 1){
        const v = s.gms_full[node][t];
        const x = left + t / Math.max(1, s.T - 1) * plotW;
        const y = top + (1 - node / Math.max(1, s.N - 1)) * plotH;
        if(v >= level.v){
          if(!drawing){
            ctx.moveTo(x, y);
            drawing = true;
          }else{
            ctx.lineTo(x, y);
          }
        }else if(drawing){
          ctx.stroke();
          ctx.beginPath();
          drawing = false;
        }
      }
      if(drawing) ctx.stroke();
    }
  });
}

function drawReliefHeatmap(ctx, left, top, plotW, plotH, s){
  const rowGap = plotH / Math.max(1, s.N - 1);
  const amp = Math.min(42, rowGap * 5.3);
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top - amp - 6, plotW, plotH + amp + 10);
  ctx.clip();
  drawEventBands(ctx, left, top - amp * .55, plotW, plotH + amp * .55, s);

  for(let node = 0; node < s.N; node += 1){
    const baseY = top + (1 - node / Math.max(1, s.N - 1)) * plotH;
    const points = [];
    let rowPeak = 0;
    for(let t = 0; t < s.T; t += 2){
      const v = s.gms_full[node][t];
      rowPeak = Math.max(rowPeak, v);
      const x = left + t / Math.max(1, s.T - 1) * plotW;
      points.push([x, baseY - v * amp]);
    }
    points.push([left + plotW, baseY - s.gms_full[node][s.T - 1] * amp]);

    const fill = ctx.createLinearGradient(0, baseY - amp, 0, baseY + rowGap * .85);
    fill.addColorStop(0, heatColor(rowPeak, .70));
    fill.addColorStop(1, 'rgba(7,16,24,.08)');
    ctx.beginPath();
    ctx.moveTo(left, baseY + rowGap * .56);
    pathThrough(ctx, points);
    ctx.lineTo(left + plotW, baseY + rowGap * .56);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();

    ctx.beginPath();
    pathThrough(ctx, points);
    ctx.strokeStyle = rowPeak >= s.beta ? 'rgba(248,81,73,.86)' : rowPeak >= s.alpha ? 'rgba(210,153,34,.66)' : 'rgba(56,139,253,.34)';
    ctx.lineWidth = rowPeak >= s.beta ? 1.35 : .8;
    ctx.stroke();
  }

  const selectedBaseY = top + (1 - selectedNode / Math.max(1, s.N - 1)) * plotH;
  ctx.strokeStyle = 'rgba(230,237,243,.72)';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(left, selectedBaseY);
  ctx.lineTo(left + plotW, selectedBaseY);
  ctx.stroke();
  ctx.restore();
}

function drawHeatmapTo(canvas, s){
  const {ctx, width:w, height:h} = setCanvasScale(canvas);
  const left = 38;
  const top = heatmapMode === 'relief' ? 48 : 18;
  const bottom = 30;
  const right = 8;
  const plotW = Math.max(1, w - left - right);
  const plotH = Math.max(1, h - top - bottom);
  const plot = {left, top, width:plotW, height:plotH};

  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, 'rgba(13,17,23,.95)');
  bg.addColorStop(1, 'rgba(7,16,24,.92)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if(heatmapMode === 'relief') drawReliefHeatmap(ctx, left, top, plotW, plotH, s);
  else drawSmoothHeatmap(ctx, left, top, plotW, plotH, s);

  const cw = plotW / s.T;
  const ch = plotH / s.N;
  const currentX = left + s.t * cw;
  const selectedX = left + selectedTime * cw;
  const selectedY = top + (s.N - 1 - selectedNode) * ch;

  ctx.strokeStyle = 'rgba(255,255,255,.88)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(currentX, top);
  ctx.lineTo(currentX, top + plotH);
  ctx.stroke();

  ctx.strokeStyle = COLORS.blue;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(left, selectedY, plotW, ch);
  ctx.strokeStyle = 'rgba(230,237,243,.95)';
  ctx.lineWidth = 1.4;
  ctx.strokeRect(selectedX, selectedY, Math.max(cw, 4), Math.max(ch, 4));

  drawAxes(ctx, w, h, left, top, plotW, plotH, s);
  return plot;
}

function drawHeatmap(s){
  heatmapPlot = drawHeatmapTo($('heatmap-canvas'), s);
}

function pickHeatmapSample(event, canvas, plot, fromExpanded = false){
  if(!state) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const {left, top, width, height} = plot || heatmapPlot;
  if(x < left || x > left + width || y < top || y > top + height) return;
  const time = clamp(Math.round((x - left) / width * (state.T - 1)), 0, state.T - 1);
  const node = clamp(Math.round((1 - (y - top) / height) * (state.N - 1)), 0, state.N - 1);
  const source = fromExpanded ? 'Expanded heatmap' : heatmapMode === 'relief' ? '2.5D surface point' : 'Smooth heatmap field';
  selectSample(node, time, source);
}

function handleHeatmapClick(event){
  pickHeatmapSample(event, $('heatmap-canvas'), heatmapPlot);
}

function render(s){
  state = s;
  buildNodeSelect(s.N || DEFAULT_N);
  selectedNode = clamp(selectedNode, 0, (s.N || DEFAULT_N) - 1);
  selectedTime = clamp(selectedTime === null ? s.t : selectedTime, 0, (s.T || DEFAULT_T) - 1);
  updateSummary(s);
  updateCharts(s);
  drawHeatmap(s);
  updateInsight(s, selectedTime === s.t ? 'Live frame' : 'Selected sample');
  updatePerformance(s);
  refreshExpandedGraph();
}

function onFrame(data){
  render(data);
}

function bootAnalysis(){
  buildNodeSelect(DEFAULT_N);
  bindControls();
  initCharts();
  bindExpandableGraphs();
}

if(document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', bootAnalysis, {once:true});
}else{
  bootAnalysis();
}
