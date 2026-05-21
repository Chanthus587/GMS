import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const NN = 40;
const T = 120;
const GRID = 10;
const WORLD_OFFSET = 5;
const canvas = document.getElementById('worldCanvas');
const mini = document.getElementById('miniMap');
const timeline = document.getElementById('networkTimeline');
const mctx = mini.getContext('2d');
const tlctx = timeline.getContext('2d');
const tip = document.getElementById('tip');
const sunUi = document.getElementById('sunUi');
const moonUi = document.getElementById('moonUi');
const alarmVignette = document.getElementById('alarmVignette');
const alarmBanner = document.getElementById('alarmBanner');
const loadNote = document.getElementById('loadNote');

const mapParams = new URLSearchParams(window.location.search);
const requestedNode = Number.parseInt(mapParams.get('node') || '0', 10);
const requestedTime = Number.parseInt(mapParams.get('t') || '', 10);
let state = null;
let selectedNode = Number.isFinite(requestedNode) ? requestedNode : 0;
let initialJumpDone = false;
let playing = false;
let noiseOn = false;
let mode = 'world';
let previousHigh = 0;
let alarmUntil = 0;
const envState = {
  windSpeed: .35,
  windDir: 60,
  pressure: 1,
  intensity: .55
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050914);
scene.fog = new THREE.FogExp2(0x071018, 0.030);

const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false, powerPreference: 'high-performance'});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;

const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 80);
camera.position.set(7.8, 8.2, 9.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 6.2;
controls.maxDistance = 20;
controls.maxPolarAngle = Math.PI * 0.46;
controls.target.set(0, 0.25, 0);

const hemi = new THREE.HemisphereLight(0x8cbcff, 0x183425, 1.35);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffd89b, 2.4);
sun.position.set(-5, 9, 4);
sun.castShadow = false;
scene.add(sun);
const fill = new THREE.PointLight(0x388bfd, 1.8, 18);
fill.position.set(2, 4, 1);
scene.add(fill);

const skyGroup = new THREE.Group();
const terrainGroup = new THREE.Group();
const forestGroup = new THREE.Group();
const towerGroup = new THREE.Group();
const plumeGroup = new THREE.Group();
const heatFootprintGroup = new THREE.Group();
const edgeGroup = new THREE.Group();
const eventGroup = new THREE.Group();
const particleGroup = new THREE.Group();
const environmentGroup = new THREE.Group();
scene.add(skyGroup, terrainGroup, forestGroup, heatFootprintGroup, eventGroup, edgeGroup, plumeGroup, towerGroup, particleGroup, environmentGroup);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const towerPickables = [];
const towers = new Map();
const nodeMaterials = new Map();
const tempObject = new THREE.Object3D();
const clock = new THREE.Clock();
const treeData = [];
const waterRippleData = [];
let canopyMesh = null;
let trunkMesh = null;
let sunDisc = null;
let sunHalo = null;
let moonDisc = null;
let moonHalo = null;
let windLines = null;
let windArrows = null;
let pressureDome = null;
let pressurePulse = null;
let waterSurface = null;
let waterRipples = null;
const windStreams = [];

function $(id){return document.getElementById(id);}
function clamp(v, lo, hi){return Math.max(lo, Math.min(hi, v));}
function frameTimeLabel(s, t){
  const idx = Math.max(0, Math.round(Number(t) || 0));
  if(s?.time_axis?.[idx]) return s.time_axis[idx];
  if(s?.time_label && idx === s.t) return s.time_label;
  const total = Math.max(1, Number(s?.T) || T);
  const mins = Math.round(idx / total * 24 * 60) % (24 * 60);
  return `${String(Math.floor(mins / 60)).padStart(2, '0')}:${String(mins % 60).padStart(2, '0')}`;
}
function terrainHeight(x, y){
  return 0.12 + 0.28 * Math.sin(x * 0.76 + y * 0.31) + 0.16 * Math.sin(y * 1.45) - 0.10 * Math.cos((x - y) * 1.05);
}
function riverY(x){return 5 + 1.15 * Math.sin(x * 0.75) + 0.42 * Math.sin(x * 1.9);}
function toWorld(x, y, extra = 0){
  return new THREE.Vector3(x - WORLD_OFFSET, terrainHeight(x, y) + extra, y - WORLD_OFFSET);
}
function nodeColor(label){
  if(label === 2) return 0xf85149;
  if(label === 1) return 0xd29922;
  return 0x388bfd;
}
function colorCss(label){
  if(label === 2) return '#F85149';
  if(label === 1) return '#D29922';
  return '#388BFD';
}
function atmosphere(t = 0){
  const p = (t % T) / (T - 1);
  if(p < .22) return {name:'Dawn', bg:0x071326, fog:0x0b1a2a, sun:0xffc17a, intensity:2.2};
  if(p < .58) return {name:'Day', bg:0x0d2742, fog:0x143a48, sun:0xffe0ae, intensity:2.7};
  if(p < .78) return {name:'Dusk', bg:0x120b20, fog:0x30192b, sun:0xff9f63, intensity:2.0};
  return {name:'Night', bg:0x030711, fog:0x06101c, sun:0xa9c7ff, intensity:1.1};
}
function pseudoRand(seed){
  const v = Math.sin(seed * 12.9898) * 43758.5453;
  return v - Math.floor(v);
}
function forestDensity(x, y){
  const riverGap = Math.abs(y - riverY(x));
  const centerGap = Math.hypot(x - 5.1, y - 5.0);
  const ridge = .52 + .30 * Math.sin(x * .82 + y * .46) + .18 * Math.cos(y * 1.18);
  const waterPenalty = riverGap < .72 ? .18 : 1;
  const centerClearing = centerGap < 2.2 ? .04 : clamp((centerGap - 2.0) / 2.3, .18, 1);
  const sensorClear = state ? state.nodes.reduce((min, n) => Math.min(min, Math.hypot(n.x - x, n.y - y)), 9) : 9;
  const nodePenalty = sensorClear < .36 ? 0 : sensorClear < .62 ? .45 : 1;
  return clamp(ridge * waterPenalty * centerClearing * nodePenalty, 0, 1);
}
function heatAt(x, y){
  if(!state) return 0;
  let v = 0;
  const windRad = THREE.MathUtils.degToRad(envState.windDir);
  const wx = Math.cos(windRad);
  const wy = Math.sin(windRad);
  const intensity = .62 + envState.intensity * .95;
  const pressureLift = 1.08 - (envState.pressure - 1) * .34;
  for(const n of state.nodes){
    const dx = x - n.x;
    const dy = y - n.y;
    const downwind = dx * wx + dy * wy;
    const crosswind = Math.abs(dx * wy - dy * wx);
    const wake = Math.max(0, downwind) * envState.windSpeed;
    const spread = 1.45 + envState.windSpeed * 2.1 + envState.intensity * .8;
    const distance = crosswind * crosswind / spread + Math.max(0, -downwind) * .92 + wake * .16;
    v += n.gms * intensity * pressureLift * Math.exp(-distance / 2.7);
  }
  return clamp(v / 2.05, 0, 1);
}
function heatColor(value){
  const v = clamp(value, 0, 1);
  if(v < .48){
    return new THREE.Color().setRGB((56 + 154 * v * 2) / 255, (139 + 14 * v * 2) / 255, (253 - 219 * v * 2) / 255);
  }
  return new THREE.Color().setRGB((210 + 38 * (v - .5) * 2) / 255, (153 - 72 * (v - .5) * 2) / 255, (34 + 39 * (v - .5) * 2) / 255);
}

function buildTerrain(){
  const seg = 96;
  const positions = [];
  const colors = [];
  const indices = [];
  for(let z = 0; z <= seg; z++){
    for(let x = 0; x <= seg; x++){
      const gx = x / seg * GRID;
      const gy = z / seg * GRID;
      const h = terrainHeight(gx, gy);
      positions.push(gx - WORLD_OFFSET, h, gy - WORLD_OFFSET);
      const water = Math.abs(gy - riverY(gx)) < .34;
      const c = water ? new THREE.Color(0x245d77) : new THREE.Color(0x234832).lerp(new THREE.Color(0x6b7041), clamp((h + .28) / .82, 0, 1));
      colors.push(c.r, c.g, c.b);
    }
  }
  for(let z = 0; z < seg; z++){
    for(let x = 0; x < seg; x++){
      const a = z * (seg + 1) + x;
      indices.push(a, a + 1, a + seg + 1, a + 1, a + seg + 2, a + seg + 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({vertexColors:true, roughness:.95, metalness:.02});
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'terrain';
  terrainGroup.add(mesh);
  buildRiver();
  buildFieldPatches();
  buildForest();
  buildSkyBodies();
}
function updateTerrainColors(){
  const mesh = terrainGroup.getObjectByName('terrain');
  if(!mesh) return;
  const colors = mesh.geometry.attributes.color;
  const pos = mesh.geometry.attributes.position;
  const c = new THREE.Color();
  for(let i = 0; i < pos.count; i++){
    const gx = pos.getX(i) + WORLD_OFFSET;
    const gy = pos.getZ(i) + WORLD_OFFSET;
    const h = pos.getY(i);
    const heat = heatAt(gx, gy);
    const water = Math.abs(gy - riverY(gx)) < .34;
    if(mode === 'heat'){
      c.copy(heatColor(heat)).lerp(new THREE.Color(0x071018), .18);
    }else if(water){
      c.set(0x245d77).lerp(new THREE.Color(0xd29922), heat * .35);
    }else{
      const density = forestDensity(gx, gy);
      c.set(0x1d4b2f)
        .lerp(new THREE.Color(0x527139), clamp((h + .28) / .82, 0, 1) * .52)
        .lerp(new THREE.Color(0x11331f), density * .38)
        .lerp(new THREE.Color(0xd29922), heat * .34);
    }
    colors.setXYZ(i, c.r, c.g, c.b);
  }
  colors.needsUpdate = true;
  updateForestHeat();
}
function buildRiver(){
  waterRippleData.length = 0;
  const samples = 92;
  const positions = [];
  const colors = [];
  const indices = [];
  const base = [];
  const color = new THREE.Color();

  for(let i = 0; i <= samples; i++){
    const x = i / samples * GRID;
    const y = riverY(x);
    const yp = riverY(Math.min(GRID, x + .08));
    const ym = riverY(Math.max(0, x - .08));
    const tangent = new THREE.Vector2(.16, yp - ym).normalize();
    const normal = new THREE.Vector2(-tangent.y, tangent.x);
    const width = .34 + .08 * Math.sin(x * 1.7) + .04 * Math.sin(x * 4.1);
    const h = terrainHeight(x, y) + .045;
    const leftX = x + normal.x * width;
    const leftY = y + normal.y * width;
    const rightX = x - normal.x * width;
    const rightY = y - normal.y * width;
    const leftH = Math.max(h, terrainHeight(leftX, leftY) + .035);
    const rightH = Math.max(h, terrainHeight(rightX, rightY) + .035);

    positions.push(leftX - WORLD_OFFSET, leftH, leftY - WORLD_OFFSET);
    positions.push(rightX - WORLD_OFFSET, rightH, rightY - WORLD_OFFSET);
    base.push(leftH, rightH);

    color.set(0x245d77).lerp(new THREE.Color(0x58c7dc), .16 + .08 * Math.sin(i * .45));
    colors.push(color.r, color.g, color.b, color.r * .72, color.g * .86, Math.min(1, color.b * 1.18));

    if(i < samples){
      const a = i * 2;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }

    if(i % 4 === 0 && i > 1 && i < samples - 1){
      waterRippleData.push({
        x,
        y,
        h,
        width:width * (.62 + pseudoRand(i + 1.7) * .42),
        normal,
        phase:pseudoRand(i + 6.1) * Math.PI * 2
      });
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.userData.baseY = base;

  waterSurface = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      vertexColors:true,
      transparent:true,
      opacity:.74,
      color:0x9be7f5,
      emissive:0x0d3c52,
      emissiveIntensity:.38,
      roughness:.18,
      metalness:.04,
      side:THREE.DoubleSide
    })
  );
  waterSurface.name = 'waterSurface';
  terrainGroup.add(waterSurface);

  const ripplePositions = [];
  for(const r of waterRippleData){
    ripplePositions.push(
      r.x - r.normal.x * r.width - WORLD_OFFSET, r.h + .028, r.y - r.normal.y * r.width - WORLD_OFFSET,
      r.x + r.normal.x * r.width - WORLD_OFFSET, r.h + .028, r.y + r.normal.y * r.width - WORLD_OFFSET
    );
  }
  const rippleGeo = new THREE.BufferGeometry();
  rippleGeo.setAttribute('position', new THREE.Float32BufferAttribute(ripplePositions, 3));
  waterRipples = new THREE.LineSegments(
    rippleGeo,
    new THREE.LineBasicMaterial({
      color:0xbaf7ff,
      transparent:true,
      opacity:.42,
      blending:THREE.AdditiveBlending,
      depthWrite:false
    })
  );
  terrainGroup.add(waterRipples);
}
function buildFieldPatches(){
  const patches = [
    [1.0,1.3,2.5,1.4,0x315d3a],[6.2,1.0,2.5,1.8,0x426238],
    [1.0,7.0,2.3,1.5,0x2c5a47],[6.4,6.9,2.4,1.7,0x57582f]
  ];
  for(const [x,y,w,h,color] of patches){
    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({color, transparent:true, opacity:.34, side:THREE.DoubleSide, depthWrite:false});
    const patch = new THREE.Mesh(geo, mat);
    patch.rotation.x = -Math.PI / 2;
    patch.position.set(x + w / 2 - WORLD_OFFSET, terrainHeight(x + w / 2, y + h / 2) + .035, y + h / 2 - WORLD_OFFSET);
    terrainGroup.add(patch);
  }
}

function buildForest(){
  forestGroup.clear();
  treeData.length = 0;
  const canopyGeo = new THREE.ConeGeometry(.16, .62, 8);
  const trunkGeo = new THREE.CylinderGeometry(.020, .032, .26, 6);
  const canopyMat = new THREE.MeshStandardMaterial({color:0x2a7d46, emissive:0x062b16, emissiveIntensity:.20, roughness:.90, metalness:0});
  const trunkMat = new THREE.MeshStandardMaterial({color:0x4a3324, roughness:.88, metalness:0});
  const candidates = [];
  for(let gx = .28; gx < GRID; gx += .30){
    for(let gy = .28; gy < GRID; gy += .30){
      const seed = gx * 31.7 + gy * 57.3;
      const jx = (pseudoRand(seed) - .5) * .20;
      const jy = (pseudoRand(seed + 9.1) - .5) * .20;
      const x = gx + jx;
      const y = gy + jy;
      const density = forestDensity(x, y);
      if(density < .34 || pseudoRand(seed + 4.2) > density + .04) continue;
      candidates.push({x, y, seed, density});
    }
  }
  const count = Math.min(280, candidates.length);
  canopyMesh = new THREE.InstancedMesh(canopyGeo, canopyMat, count);
  trunkMesh = new THREE.InstancedMesh(trunkGeo, trunkMat, count);
  canopyMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  trunkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for(let i = 0; i < count; i++){
    const d = candidates[i];
    const h = terrainHeight(d.x, d.y);
    const s = .82 + pseudoRand(d.seed + 2.4) * .58;
    const lean = (pseudoRand(d.seed + 7.7) - .5) * .08;
    const trunkYaw = pseudoRand(d.seed + 11) * Math.PI * 2;
    const canopyYaw = pseudoRand(d.seed + 19) * Math.PI * 2;
    treeData.push({x:d.x, y:d.y, h, scale:s, lean, trunkYaw, canopyYaw, baseDensity:d.density});

    tempObject.position.set(d.x - WORLD_OFFSET, h + .13 * s, d.y - WORLD_OFFSET);
    tempObject.rotation.set(lean, trunkYaw, lean * .5);
    tempObject.scale.setScalar(s);
    tempObject.updateMatrix();
    trunkMesh.setMatrixAt(i, tempObject.matrix);

    tempObject.position.set(d.x - WORLD_OFFSET, h + .43 * s, d.y - WORLD_OFFSET);
    tempObject.rotation.set(lean, canopyYaw, lean * .5);
    tempObject.scale.set(s, s * (1.08 + d.density * .22), s);
    tempObject.updateMatrix();
    canopyMesh.setMatrixAt(i, tempObject.matrix);
  }
  forestGroup.add(trunkMesh, canopyMesh);
}

function updateForestHeat(){
  if(!canopyMesh || !state) return;
  const color = new THREE.Color();
  for(let i = 0; i < treeData.length; i++){
    const d = treeData[i];
    const heat = heatAt(d.x, d.y);
    color.set(0x1f6b3f)
      .lerp(new THREE.Color(0x0f3b24), d.baseDensity * .30)
      .lerp(new THREE.Color(0xd29922), heat * .42)
      .lerp(new THREE.Color(0xf85149), Math.max(0, heat - .65) * .38);
    canopyMesh.setColorAt(i, color);
  }
  canopyMesh.instanceColor.needsUpdate = true;
}

function makeSkySprite(color, opacity, scale){
  const cnv = document.createElement('canvas');
  cnv.width = 256;
  cnv.height = 256;
  const c = cnv.getContext('2d');
  const g = c.createRadialGradient(128, 128, 0, 128, 128, 128);
  g.addColorStop(0, color);
  g.addColorStop(.34, color.replace('1)', `${opacity})`));
  g.addColorStop(1, color.replace('1)', '0)'));
  c.fillStyle = g;
  c.fillRect(0, 0, 256, 256);
  const texture = new THREE.CanvasTexture(cnv);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map:texture,
    transparent:true,
    depthWrite:false,
    depthTest:false,
    blending:THREE.AdditiveBlending
  }));
  sprite.scale.setScalar(scale);
  return sprite;
}

function buildSkyBodies(){
  const sunMat = new THREE.MeshBasicMaterial({color:0xffd28a, transparent:true, depthTest:false});
  const moonMat = new THREE.MeshBasicMaterial({color:0xdce8ff, transparent:true, depthTest:false});
  sunDisc = new THREE.Mesh(new THREE.SphereGeometry(.34, 32, 16), sunMat);
  moonDisc = new THREE.Mesh(new THREE.SphereGeometry(.22, 32, 16), moonMat);
  sunHalo = makeSkySprite('rgba(255,196,103,1)', .34, 3.2);
  moonHalo = makeSkySprite('rgba(176,205,255,1)', .22, 2.1);
  skyGroup.add(sunHalo, moonHalo, sunDisc, moonDisc);
}

function updateSkyBodies(){
  if(!sunDisc || !moonDisc) return;
  const t = state ? state.t : 0;
  const p = (t % T) / (T - 1);
  const sunAngle = p * Math.PI * 2 - Math.PI * .15;
  const moonAngle = sunAngle + Math.PI;
  const sx = Math.cos(sunAngle) * 8.4;
  const sy = 4.2 + Math.sin(sunAngle) * 4.2;
  const mx = Math.cos(moonAngle) * 8.1;
  const my = 4.0 + Math.sin(moonAngle) * 4.0;
  sunDisc.position.set(sx, sy, -6.5);
  sunHalo.position.copy(sunDisc.position);
  moonDisc.position.set(mx, my, -6.8);
  moonHalo.position.copy(moonDisc.position);
  sun.position.set(sx * .7, Math.max(1.2, sy), -2.5);
  const sunVisible = sy > .85;
  const moonVisible = my > .85;
  sunDisc.visible = sunHalo.visible = sunVisible;
  moonDisc.visible = moonHalo.visible = moonVisible;
  const sunAlpha = clamp((sy - .85) / 3.5, .18, 1);
  const moonAlpha = clamp((my - .85) / 3.5, .16, .82);
  sunDisc.material.opacity = sunAlpha;
  moonDisc.material.opacity = moonAlpha;
  sunHalo.material.opacity = sunAlpha;
  moonHalo.material.opacity = moonAlpha;
  updateCelestialUi(p, sy, my, sunAlpha, moonAlpha);
}

function updateCelestialUi(p, sunY, moonY, sunAlpha, moonAlpha){
  const sunX = 14 + p * 72;
  const sunTop = 27 - Math.sin(p * Math.PI) * 15;
  const moonPhase = (p + .5) % 1;
  const moonX = 14 + moonPhase * 72;
  const moonTop = 27 - Math.sin(moonPhase * Math.PI) * 15;
  sunUi.style.left = `${sunX}%`;
  sunUi.style.top = `${sunTop}%`;
  moonUi.style.left = `${moonX}%`;
  moonUi.style.top = `${moonTop}%`;
  sunUi.style.opacity = sunY > .45 ? Math.max(.34, sunAlpha) : 0;
  moonUi.style.opacity = moonY > .45 ? Math.max(.30, moonAlpha) : 0;
}

function makeTextSprite(text, color = '#E6EDF3', size = 72){
  const cnv = document.createElement('canvas');
  cnv.width = 160;
  cnv.height = 88;
  const c = cnv.getContext('2d');
  c.font = `900 ${size}px Courier New`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.lineWidth = 8;
  c.strokeStyle = 'rgba(5,9,20,.82)';
  c.strokeText(text, cnv.width / 2, cnv.height / 2);
  c.fillStyle = color;
  c.fillText(text, cnv.width / 2, cnv.height / 2);
  const texture = new THREE.CanvasTexture(cnv);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map:texture, transparent:true, depthWrite:false}));
  sprite.scale.set(.42, .23, 1);
  return sprite;
}
function buildTowers(nodes){
  towerGroup.clear();
  plumeGroup.clear();
  heatFootprintGroup.clear();
  towerPickables.length = 0;
  towers.clear();
  nodeMaterials.clear();
  for(const n of nodes){
    const root = new THREE.Group();
    root.userData.nodeId = n.id;
    const mastMat = new THREE.MeshStandardMaterial({color:0x8b949e, roughness:.48, metalness:.45});
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(.035, .055, .62, 8), mastMat);
    mast.position.y = .31;
    root.add(mast);

    const beaconMat = new THREE.MeshStandardMaterial({
      color:nodeColor(n.label),
      emissive:nodeColor(n.label),
      emissiveIntensity:.9,
      roughness:.25,
      metalness:.15
    });
    const beacon = new THREE.Mesh(new THREE.SphereGeometry(.16, 24, 16), beaconMat);
    beacon.position.y = .72;
    beacon.userData.nodeId = n.id;
    root.add(beacon);
    towerPickables.push(beacon);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(.26, .012, 8, 48),
      new THREE.MeshBasicMaterial({color:0xf85149, transparent:true, opacity:0, depthWrite:false})
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = .72;
    root.add(ring);

    const label = makeTextSprite(String(n.id));
    label.position.y = 1.05;
    root.add(label);

    const p = toWorld(n.x, n.y, .04);
    root.position.copy(p);
    towerGroup.add(root);

    const plume = new THREE.Mesh(
      new THREE.ConeGeometry(.34, 1.35, 32, 1, true),
      new THREE.MeshBasicMaterial({
        color:nodeColor(n.label),
        transparent:true,
        opacity:0,
        side:THREE.DoubleSide,
        depthWrite:false,
        blending:THREE.AdditiveBlending
      })
    );
    plume.position.copy(toWorld(n.x, n.y, .74));
    plume.rotation.y = n.id * .37;
    plumeGroup.add(plume);

    const footprint = new THREE.Mesh(
      new THREE.CircleGeometry(.45, 48),
      new THREE.MeshBasicMaterial({
        color:nodeColor(n.label),
        transparent:true,
        opacity:0,
        depthWrite:false,
        blending:THREE.AdditiveBlending,
        side:THREE.DoubleSide
      })
    );
    footprint.rotation.x = -Math.PI / 2;
    footprint.position.copy(toWorld(n.x, n.y, .045));
    heatFootprintGroup.add(footprint);

    towers.set(n.id, {root, mast, beacon, beaconMat, ring, label, plume, footprint});
    nodeMaterials.set(n.id, beaconMat);
  }
}
function updateTowers(){
  if(!state) return;
  const now = performance.now();
  const windRad = THREE.MathUtils.degToRad(envState.windDir);
  const plumeTilt = envState.windSpeed * .28;
  const intensityBoost = .74 + envState.intensity * .85;
  for(const n of state.nodes){
    const item = towers.get(n.id);
    if(!item) continue;
    const color = nodeColor(n.label);
    const pulse = .5 + .5 * Math.sin(now / 150 + n.id);
    item.beaconMat.color.setHex(color);
    item.beaconMat.emissive.setHex(color);
    item.beaconMat.emissiveIntensity = n.label === 2 ? 1.75 + pulse * 1.2 : .55 + n.gms * 1.15;
    item.root.position.copy(toWorld(n.x, n.y, .04));
    item.root.scale.setScalar(n.id === selectedNode ? 1.18 : 1);
    item.beacon.scale.setScalar(1 + n.gms * .9 + (n.label === 2 ? pulse * .45 : 0));
    item.mast.scale.y = 1 + n.gms * .85;
    item.ring.material.opacity = n.label === 2 ? .68 - pulse * .28 : n.label === 1 ? .18 : 0;
    item.ring.scale.setScalar(1 + pulse * 1.25 + n.gms * 1.1);
    item.ring.material.color.setHex(color);
    item.plume.position.copy(toWorld(n.x, n.y, .76 + n.gms * .3));
    item.plume.rotation.set(Math.cos(windRad) * plumeTilt, n.id * .37 + now / 4200, -Math.sin(windRad) * plumeTilt);
    item.plume.material.color.setHex(color);
    item.plume.material.opacity = mode === 'network' ? 0 : clamp((n.gms * .38 + (n.label === 2 ? .22 : 0)) * intensityBoost, 0, .78);
    item.plume.scale.set(.55 + n.gms * 1.9 + envState.windSpeed * .42, .5 + n.gms * 1.85 + envState.intensity * .58, .55 + n.gms * 1.9);
    item.footprint.position.copy(toWorld(n.x, n.y, .05));
    item.footprint.material.color.setHex(color);
    item.footprint.material.opacity = mode === 'network' ? 0 : clamp(n.gms * .20 + envState.intensity * .08, 0, .42);
    item.footprint.rotation.z = -windRad;
    item.footprint.scale.set(.8 + n.gms * 3.2 + envState.windSpeed * 1.1, .8 + n.gms * 2.4, 1);
  }
}
function rebuildEdges(){
  edgeGroup.clear();
  if(!state || mode === 'heat') return;
  const lineMat = new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:mode === 'network' ? .92 : .46});
  const positions = [];
  const colors = [];
  const c1 = new THREE.Color();
  const c2 = new THREE.Color();
  for(let i = 0; i < NN; i++){
    for(const j of state.adj[String(i)] || []){
      if(j <= i) continue;
      const a = state.nodes[i];
      const b = state.nodes[j];
      const va = toWorld(a.x, a.y, .44);
      const vb = toWorld(b.x, b.y, .44);
      positions.push(va.x, va.y, va.z, vb.x, vb.y, vb.z);
      const avg = (a.gms + b.gms) / 2;
      c1.setHex(mode === 'network' ? 0x388bfd : nodeColor(a.label)).lerp(new THREE.Color(0xf85149), avg * .25);
      c2.setHex(mode === 'network' ? 0x388bfd : nodeColor(b.label)).lerp(new THREE.Color(0xf85149), avg * .25);
      colors.push(c1.r, c1.g, c1.b, c2.r, c2.g, c2.b);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  edgeGroup.add(new THREE.LineSegments(geo, lineMat));

  for(const pe of state.prop_edges || []){
    const a = state.nodes[pe.src];
    const b = state.nodes[pe.dst];
    const geo2 = new THREE.BufferGeometry().setFromPoints([toWorld(a.x, a.y, .66), toWorld(b.x, b.y, .66)]);
    const mat2 = new THREE.LineBasicMaterial({color:0xd29922, transparent:true, opacity:clamp(.35 + pe.strength * .55, .35, .9)});
    const line = new THREE.Line(geo2, mat2);
    edgeGroup.add(line);
  }
}
function rebuildEvents(){
  eventGroup.clear();
  if(!state || !state.events) return;
  const active = new Set(state.active_events || []);
  for(const ev of state.events){
    const nodes = ev.nodes.map(id => state.nodes[id]).filter(Boolean);
    if(!nodes.length) continue;
    const cx = nodes.reduce((a, n) => a + n.x, 0) / nodes.length;
    const cy = nodes.reduce((a, n) => a + n.y, 0) / nodes.length;
    const radius = Math.max(1.0, ...nodes.map(n => Math.hypot(n.x - cx, n.y - cy) + .55)) * (.86 + envState.intensity * .34);
    const isActive = active.has(ev.label);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(radius, .018, 8, 96),
      new THREE.MeshBasicMaterial({color:new THREE.Color(ev.color || '#D29922'), transparent:true, opacity:isActive ? .62 + envState.intensity * .34 : .12 + envState.intensity * .12, depthWrite:false})
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.copy(toWorld(cx, cy, .09));
    eventGroup.add(ring);
    if(isActive){
      const glow = new THREE.Mesh(
        new THREE.CircleGeometry(radius, 96),
        new THREE.MeshBasicMaterial({color:new THREE.Color(ev.color || '#D29922'), transparent:true, opacity:.06 + envState.intensity * .13, depthWrite:false, side:THREE.DoubleSide})
      );
      glow.rotation.x = -Math.PI / 2;
      glow.position.copy(toWorld(cx, cy, .08));
      eventGroup.add(glow);
    }
  }
}
function buildParticles(){
  const geo = new THREE.BufferGeometry();
  const count = 130;
  const positions = [];
  for(let i = 0; i < count; i++){
    positions.push((Math.random() - .5) * 10, .35 + Math.random() * 2.8, (Math.random() - .5) * 10);
  }
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({color:0xd29922, size:.035, transparent:true, opacity:.0, depthWrite:false});
  const pts = new THREE.Points(geo, mat);
  particleGroup.add(pts);
}

function updateWaterMotion(dt){
  if(!waterSurface) return;
  const now = performance.now() / 1000;
  const windRad = THREE.MathUtils.degToRad(envState.windDir);
  const pressureLift = clamp(1 - envState.pressure, 0, .32);
  const pressureWeight = clamp(envState.pressure - 1, 0, .32);
  const pos = waterSurface.geometry.attributes.position;
  const base = waterSurface.geometry.userData.baseY || [];
  for(let i = 0; i < pos.count; i++){
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const wave = Math.sin(now * (1.35 + envState.windSpeed * 2.2) + x * 2.1 + z * 1.4) * (.012 + envState.windSpeed * .024 + pressureLift * .025);
    pos.setY(i, (base[i] || pos.getY(i)) + wave - pressureWeight * .018);
  }
  pos.needsUpdate = true;
  waterSurface.geometry.computeVertexNormals();
  waterSurface.material.opacity = .62 + envState.windSpeed * .14 + pressureLift * .18;
  waterSurface.material.emissiveIntensity = .26 + envState.windSpeed * .28 + pressureLift * .22;

  if(!waterRipples) return;
  waterRipples.rotation.y = -windRad * .08;
  waterRipples.material.opacity = .22 + envState.windSpeed * .44 + pressureLift * .30;
  const rpos = waterRipples.geometry.attributes.position;
  for(let i = 0; i < waterRippleData.length; i++){
    const r = waterRippleData[i];
    const shimmer = Math.sin(now * (2.4 + envState.windSpeed * 3.1) + r.phase) * (.035 + envState.windSpeed * .05);
    const drift = Math.sin(now * .7 + r.phase) * envState.windSpeed * .06;
    const width = r.width * (.82 + envState.windSpeed * .42 + pressureLift * .28);
    rpos.setXYZ(i * 2,
      r.x - r.normal.x * width + drift - WORLD_OFFSET,
      r.h + .045 + shimmer,
      r.y - r.normal.y * width - WORLD_OFFSET
    );
    rpos.setXYZ(i * 2 + 1,
      r.x + r.normal.x * width + drift - WORLD_OFFSET,
      r.h + .045 - shimmer * .2,
      r.y + r.normal.y * width - WORLD_OFFSET
    );
  }
  rpos.needsUpdate = true;
}

function updateForestMotion(dt){
  if(!canopyMesh) return;
  const now = performance.now() / 1000;
  const windRad = THREE.MathUtils.degToRad(envState.windDir);
  const wx = Math.cos(windRad);
  const wz = Math.sin(windRad);
  const pressureLift = clamp(1 - envState.pressure, 0, .32);
  const swayStrength = envState.windSpeed * (.026 + pressureLift * .035);
  for(let i = 0; i < treeData.length; i++){
    const d = treeData[i];
    const gust = Math.sin(now * (1.3 + envState.windSpeed * 2.6) + d.x * 1.7 + d.y * .9);
    const leanX = d.lean + wx * swayStrength * gust;
    const leanZ = d.lean * .5 + wz * swayStrength * gust;
    tempObject.position.set(d.x - WORLD_OFFSET, d.h + .43 * d.scale, d.y - WORLD_OFFSET);
    tempObject.rotation.set(leanZ, d.canopyYaw + gust * envState.windSpeed * .025, -leanX);
    tempObject.scale.set(d.scale, d.scale * (1.08 + d.baseDensity * .22), d.scale);
    tempObject.updateMatrix();
    canopyMesh.setMatrixAt(i, tempObject.matrix);
  }
  canopyMesh.instanceMatrix.needsUpdate = true;
}

function buildEnvironmentControls(){
  environmentGroup.clear();
  windStreams.length = 0;

  const linePositions = [];
  const rows = 22;
  for(let i = 0; i < rows; i++){
    const z = -5.2 + (i % 11) * 1.04;
    const y = .55 + Math.floor(i / 11) * .62 + pseudoRand(i + 2.1) * .24;
    const x = -5.4 + pseudoRand(i + 4.7) * 10.8;
    const len = .62 + pseudoRand(i + 9.4) * .66;
    windStreams.push({z, y, base:x, len, phase:pseudoRand(i + 14.2) * 100});
    linePositions.push(x, y, z, x + len, y, z);
  }
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  const lineMat = new THREE.LineBasicMaterial({
    color:0x9bdcff,
    transparent:true,
    opacity:.35,
    blending:THREE.AdditiveBlending,
    depthWrite:false
  });
  windLines = new THREE.LineSegments(lineGeo, lineMat);
  environmentGroup.add(windLines);

  const arrowGeo = new THREE.ConeGeometry(.075, .32, 10);
  const arrowMat = new THREE.MeshBasicMaterial({
    color:0xb9ecff,
    transparent:true,
    opacity:.48,
    blending:THREE.AdditiveBlending,
    depthWrite:false
  });
  windArrows = new THREE.InstancedMesh(arrowGeo, arrowMat, rows);
  windArrows.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for(let i = 0; i < rows; i++){
    const s = windStreams[i];
    tempObject.position.set(s.base + s.len + .16, s.y, s.z);
    tempObject.rotation.set(0, 0, -Math.PI / 2);
    tempObject.scale.setScalar(.78 + pseudoRand(i + 31) * .45);
    tempObject.updateMatrix();
    windArrows.setMatrixAt(i, tempObject.matrix);
  }
  environmentGroup.add(windArrows);

  pressureDome = new THREE.Mesh(
    new THREE.SphereGeometry(5.65, 48, 16, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshBasicMaterial({
      color:0x388bfd,
      transparent:true,
      opacity:.10,
      wireframe:true,
      blending:THREE.AdditiveBlending,
      depthWrite:false
    })
  );
  pressureDome.position.y = .03;
  environmentGroup.add(pressureDome);

  pressurePulse = new THREE.Mesh(
    new THREE.TorusGeometry(4.9, .018, 8, 128),
    new THREE.MeshBasicMaterial({
      color:0x3fb950,
      transparent:true,
      opacity:.38,
      blending:THREE.AdditiveBlending,
      depthWrite:false
    })
  );
  pressurePulse.rotation.x = Math.PI / 2;
  pressurePulse.position.y = .18;
  environmentGroup.add(pressurePulse);

  updateEnvironmentVisuals();
}

function updateEnvironmentLabels(){
  const windRead = $('windRead');
  const pressureRead = $('pressureRead');
  const dirRead = $('dirRead');
  const intensityRead = $('intensityRead');
  if(windRead) windRead.textContent = `${Math.round(envState.windSpeed * 100)}%`;
  if(pressureRead) pressureRead.textContent = `${Math.round(101.3 * envState.pressure)} kPa`;
  if(dirRead) dirRead.textContent = `${String(Math.round(envState.windDir)).padStart(3, '0')} deg`;
  if(intensityRead) intensityRead.textContent = `${Math.round(envState.intensity * 100)}%`;
}

function updateEnvironmentVisuals(){
  updateEnvironmentLabels();
  const windRad = THREE.MathUtils.degToRad(envState.windDir);
  if(windLines){
    windLines.rotation.y = -windRad;
    windLines.material.opacity = .12 + envState.windSpeed * .56;
  }
  if(windArrows){
    windArrows.rotation.y = -windRad;
    windArrows.material.opacity = .16 + envState.windSpeed * .55;
    windArrows.visible = envState.windSpeed > .02;
  }
  if(pressureDome){
    const lowPressure = clamp(1.16 - envState.pressure, 0, .38);
    pressureDome.material.color.setHex(envState.pressure < .98 ? 0xd29922 : 0x388bfd);
    pressureDome.material.opacity = .055 + Math.abs(envState.pressure - 1) * .45 + envState.intensity * .035;
    pressureDome.scale.set(1, .78 + envState.pressure * .32 + lowPressure * .35, 1);
  }
  if(pressurePulse){
    pressurePulse.material.color.setHex(envState.pressure < .98 ? 0xd29922 : 0x3fb950);
    pressurePulse.material.opacity = .18 + Math.abs(envState.pressure - 1) * .78 + envState.intensity * .18;
  }
}

function updateEnvironmentMotion(dt){
  if(!windLines || !windArrows) return;
  const now = performance.now() / 1000;
  const pos = windLines.geometry.attributes.position;
  for(let i = 0; i < windStreams.length; i++){
    const s = windStreams[i];
    const x = ((s.base + now * (.55 + envState.windSpeed * 2.8) + s.phase + 5.8) % 11.6) - 5.8;
    const y = s.y + Math.sin(now * 1.8 + i) * .035 * (1 + envState.windSpeed);
    pos.setXYZ(i * 2, x, y, s.z);
    pos.setXYZ(i * 2 + 1, x + s.len * (.55 + envState.windSpeed * 1.05), y, s.z);

    tempObject.position.set(x + s.len + .18, y, s.z);
    tempObject.rotation.set(0, 0, -Math.PI / 2);
    tempObject.scale.setScalar((.72 + envState.windSpeed * .72) * (.86 + pseudoRand(i + 31) * .35));
    tempObject.updateMatrix();
    windArrows.setMatrixAt(i, tempObject.matrix);
  }
  pos.needsUpdate = true;
  windArrows.instanceMatrix.needsUpdate = true;
  if(pressurePulse){
    const pulse = 1 + Math.sin(now * (1.2 + envState.intensity * 2.1)) * (.02 + Math.abs(envState.pressure - 1) * .08);
    pressurePulse.scale.setScalar(pulse);
  }
  if(pressureDome){
    pressureDome.rotation.y += dt * (.05 + envState.windSpeed * .12);
  }
}

function bindEnvironmentControls(){
  const bindings = [
    ['windSpeed', value => envState.windSpeed = Number(value) / 100],
    ['windDir', value => envState.windDir = Number(value)],
    ['pressureCtrl', value => envState.pressure = Number(value) / 100],
    ['intensityCtrl', value => envState.intensity = Number(value) / 100]
  ];
  for(const [id, apply] of bindings){
    const el = $(id);
    if(!el) continue;
    apply(el.value);
    el.addEventListener('input', () => {
      apply(el.value);
      updateEnvironmentVisuals();
      if(state){
        updateTerrainColors();
        updateTowers();
        rebuildEvents();
      }
    });
  }
  updateEnvironmentLabels();
}

function updateParticles(dt){
  const pts = particleGroup.children[0];
  if(!pts) return;
  const environmentalDust = envState.windSpeed > .5 || envState.intensity > .64;
  pts.material.opacity = state && (state.noise_on || environmentalDust) ? .18 + envState.intensity * .24 + envState.windSpeed * .12 : 0;
  if(!state || (!state.noise_on && !environmentalDust)) return;
  const windRad = THREE.MathUtils.degToRad(envState.windDir);
  const wx = Math.cos(windRad);
  const wz = Math.sin(windRad);
  const pos = pts.geometry.attributes.position;
  for(let i = 0; i < pos.count; i++){
    let y = pos.getY(i) + dt * (.15 + envState.intensity * .34 + (i % 7) * .012);
    let x = pos.getX(i) + wx * dt * envState.windSpeed * .46 + Math.sin(performance.now() / 700 + i) * .002;
    let z = pos.getZ(i) + wz * dt * envState.windSpeed * .46;
    if(y > 3.6){
      y = .35;
      x = (Math.random() - .5) * 10;
      z = (Math.random() - .5) * 10;
    }
    if(x > 5.4) x = -5.4;
    if(x < -5.4) x = 5.4;
    if(z > 5.4) z = -5.4;
    if(z < -5.4) z = 5.4;
    pos.setX(i, x);
    pos.setY(i, y);
    pos.setZ(i, z);
  }
  pos.needsUpdate = true;
}
function updateAtmosphere(){
  const atm = atmosphere(state ? state.t : 0);
  scene.background.setHex(atm.bg);
  scene.fog.color.setHex(atm.fog);
  sun.color.setHex(atm.sun);
  sun.intensity = atm.intensity;
  fill.intensity = atm.name === 'Night' ? 2.3 : 1.5;
  updateSkyBodies();
}

function drawMini(){
  if(!state) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = mini.clientWidth;
  const h = mini.clientHeight;
  mini.width = Math.floor(w * dpr);
  mini.height = Math.floor(h * dpr);
  mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  mctx.clearRect(0, 0, w, h);
  mctx.fillStyle = 'rgba(7,16,24,.78)';
  mctx.fillRect(0, 0, w, h);
  mctx.strokeStyle = 'rgba(48,54,61,.65)';
  mctx.strokeRect(6, 6, w - 12, h - 12);
  for(const n of state.nodes){
    mctx.fillStyle = colorCss(n.label);
    mctx.beginPath();
    mctx.arc(6 + n.x / GRID * (w - 12), h - 6 - n.y / GRID * (h - 12), 2.2 + n.gms * 3, 0, Math.PI * 2);
    mctx.fill();
  }
  const sn = state.nodes[selectedNode];
  if(sn){
    mctx.strokeStyle = 'white';
    mctx.lineWidth = 1.5;
    mctx.beginPath();
    mctx.arc(6 + sn.x / GRID * (w - 12), h - 6 - sn.y / GRID * (h - 12), 7, 0, Math.PI * 2);
    mctx.stroke();
  }
}
function drawTimeline(){
  if(!state) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = timeline.clientWidth;
  const h = timeline.clientHeight;
  timeline.width = Math.floor(w * dpr);
  timeline.height = Math.floor(h * dpr);
  tlctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  tlctx.clearRect(0, 0, w, h);
  const g = tlctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, 'rgba(13,17,23,.70)');
  g.addColorStop(1, 'rgba(28,33,40,.35)');
  tlctx.fillStyle = g;
  tlctx.fillRect(0, 0, w, h);
  const means = [];
  for(let t = 0; t < T; t++) means.push(state.gms_full.reduce((a, row) => a + row[t], 0) / NN);
  for(const ev of state.events || []){
    const x1 = ev.t_start / (T - 1) * w;
    const x2 = ev.t_end / (T - 1) * w;
    tlctx.fillStyle = `${ev.color || '#D29922'}22`;
    tlctx.fillRect(x1, 4, x2 - x1, h - 12);
  }
  const fillGrad = tlctx.createLinearGradient(0, 0, w, 0);
  fillGrad.addColorStop(0, 'rgba(56,139,253,.32)');
  fillGrad.addColorStop(.55, 'rgba(210,153,34,.38)');
  fillGrad.addColorStop(1, 'rgba(248,81,73,.45)');
  tlctx.beginPath();
  tlctx.moveTo(0, h - 10);
  means.forEach((v, i) => tlctx.lineTo(i / (T - 1) * w, h - 10 - v * (h - 22)));
  tlctx.lineTo(w, h - 10);
  tlctx.closePath();
  tlctx.fillStyle = fillGrad;
  tlctx.fill();
  tlctx.strokeStyle = 'rgba(230,237,243,.72)';
  tlctx.lineWidth = 1.4;
  tlctx.beginPath();
  means.forEach((v, i) => {
    const x = i / (T - 1) * w;
    const y = h - 10 - v * (h - 22);
    if(i === 0) tlctx.moveTo(x, y); else tlctx.lineTo(x, y);
  });
  tlctx.stroke();
  const x = state.t / (T - 1) * w;
  tlctx.strokeStyle = 'white';
  tlctx.lineWidth = 1.5;
  tlctx.beginPath();
  tlctx.moveTo(x, 2);
  tlctx.lineTo(x, h - 4);
  tlctx.stroke();
  tlctx.fillStyle = 'rgba(139,148,158,.9)';
  tlctx.font = '9px Courier New';
  tlctx.fillText('network instability timeline', 8, 13);
}

function updateInfo(){
  if(!state) return;
  const n = state.nodes[selectedNode] || state.nodes[0];
  const labels = ['Stable', 'Moderate', 'High'];
  const classes = ['stable', 'moderate', 'high'];
  $('nodeName').textContent = `N${n.id}`;
  $('nodeName').style.color = colorCss(n.label);
  $('nodeStatus').textContent = labels[n.label];
  $('nodeStatus').className = `status-badge ${classes[n.label]}`;
  $('gmsVal').textContent = n.gms.toFixed(3);
  $('gmsMeter').style.width = `${Math.round(n.gms * 100)}%`;
  $('tempVal').textContent = `${n.temp.toFixed(1)} C`;
  $('gradVal').textContent = `${n.grad >= 0 ? '+' : ''}${n.grad.toFixed(2)} C`;
  $('momVal').textContent = `${n.mom >= 0 ? '+' : ''}${n.mom.toFixed(2)}`;
  $('zVal').textContent = n.zscore.toFixed(2);
  $('highStat').textContent = state.high_count;
  $('modStat').textContent = state.mod_count;
  $('noiseStat').textContent = state.noise_on ? 'On' : 'Off';
  $('noiseStat').style.color = state.noise_on ? 'var(--amber)' : 'var(--teal)';
  const atm = atmosphere(state.t);
  $('skyStat').textContent = atm.name;
  $('skyStat').style.color = atm.name === 'Night' ? '#BC8CFF' : atm.name === 'Dusk' ? '#D29922' : '#388BFD';
  const active = state.active_events && state.active_events.length ? state.active_events.join(' + ') : 'No active event';
  $('eventChip').textContent = active;
  $('playBtn').textContent = state.playing ? 'Pause' : 'Play';
  $('playBtn').className = state.playing ? 'pri' : '';
  $('timeline').value = state.t;
  $('timePill').textContent = frameTimeLabel(state, state.t);
  $('noiseBtn').className = state.noise_on ? 'warn-btn pri' : 'warn-btn';
}
function updateAlarm(){
  if(!state) return;
  const now = performance.now();
  if(state.high_count > 0 && state.high_count !== previousHigh){
    alarmUntil = now + 1600;
    alarmVignette.classList.remove('show');
    void alarmVignette.offsetWidth;
    alarmVignette.classList.add('show');
  }
  previousHigh = state.high_count;
  if(state.high_count > 0){
    const highNodes = state.nodes.filter(n => n.label === 2).map(n => `N${n.id}`).slice(0, 6).join(', ');
    alarmBanner.textContent = `${state.high_count} high instability tower${state.high_count === 1 ? '' : 's'} active: ${highNodes}`;
    alarmBanner.classList.add('show');
  }else if(now > alarmUntil){
    alarmBanner.classList.remove('show');
    alarmVignette.classList.remove('show');
  }
}
function updateNav(d){
  const nt = document.getElementById('nav-t');
  if(nt) nt.textContent = frameTimeLabel(d, d.t);
  const ns = document.getElementById('nav-st');
  if(ns){
    if(d.high_count > 0){ns.className = 'pill p-hi'; ns.textContent = '! ' + d.high_count + ' HIGH';}
    else{ns.className = 'pill p-ok'; ns.textContent = 'ALL STABLE';}
  }
  const nn = document.getElementById('nav-noise');
  if(nn){nn.className = d.noise_on ? 'pill p-ns' : 'pill p-ok'; nn.textContent = d.noise_on ? 'NOISE ON' : 'CLEAN DATA';}
}
function applyState(next){
  const first = !state;
  state = next;
  selectedNode = Math.max(0, Math.min(selectedNode, (state.N || NN) - 1));
  playing = state.playing;
  noiseOn = state.noise_on;
  if(first && !initialJumpDone && Number.isFinite(requestedTime)){
    initialJumpDone = true;
    api('jump', {t:Math.max(0, Math.min(requestedTime, state.T - 1))});
  }
  if(first) buildTowers(state.nodes);
  updateAtmosphere();
  updateTerrainColors();
  updateTowers();
  rebuildEdges();
  rebuildEvents();
  drawMini();
  drawTimeline();
  updateInfo();
  updateAlarm();
  updateNav(state);
  loadNote.style.display = 'none';
}
function onFrame(d){applyState(d);}
function onAlert(msg){
  if(msg && msg.level === 'danger'){
    alarmUntil = performance.now() + 1800;
    alarmVignette.classList.remove('show');
    void alarmVignette.offsetWidth;
    alarmVignette.classList.add('show');
  }
}
window.onFrame = onFrame;
window.onAlert = onAlert;

async function api(ep, body = {}){
  await fetch(`/api/${ep}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
}
function togglePlay(){playing ? api('pause') : api('play');}
function toggleNoise(){api('toggle_noise', {on:!noiseOn});}
function setMode(next){
  mode = next;
  const ids = {world:'modeWorld', heat:'modeHeat', network:'modeNet'};
  Object.values(ids).forEach(id => $(id).classList.remove('on'));
  $(ids[next]).classList.add('on');
  plumeGroup.visible = next !== 'network';
  heatFootprintGroup.visible = next !== 'network';
  edgeGroup.visible = next !== 'heat';
  if(state){
    updateTerrainColors();
    updateTowers();
    rebuildEdges();
  }
}
function fitWorld(){
  camera.position.set(7.8, 8.2, 9.4);
  controls.target.set(0, .25, 0);
  controls.update();
}
window.api = api;
window.togglePlay = togglePlay;
window.toggleNoise = toggleNoise;
window.setMode = setMode;
window.fitWorld = fitWorld;

function resize(){
  const w = canvas.clientWidth || window.innerWidth;
  const h = canvas.clientHeight || window.innerHeight;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  drawMini();
  drawTimeline();
}
window.addEventListener('resize', resize);

canvas.addEventListener('pointerdown', () => canvas.classList.add('dragging'));
window.addEventListener('pointerup', () => canvas.classList.remove('dragging'));
canvas.addEventListener('pointermove', event => {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(towerPickables, false)[0];
  if(hit && state){
    const id = hit.object.userData.nodeId;
    const n = state.nodes[id];
    tip.style.display = 'block';
    tip.style.left = `${event.clientX - rect.left + 18}px`;
    tip.style.top = `${event.clientY - rect.top + 8}px`;
    tip.innerHTML = `<b style="color:${colorCss(n.label)}">N${id}</b><br>GMS ${n.gms.toFixed(3)}<br>Temp ${n.temp.toFixed(1)} C<br>Gradient ${n.grad.toFixed(2)}`;
  }else{
    tip.style.display = 'none';
  }
});
canvas.addEventListener('click', event => {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(towerPickables, false)[0];
  if(hit && typeof hit.object.userData.nodeId === 'number'){
    selectedNode = hit.object.userData.nodeId;
    updateInfo();
    updateTowers();
    drawMini();
  }
});
canvas.addEventListener('mouseleave', () => tip.style.display = 'none');
document.addEventListener('keydown', event => {
  if(event.target && event.target.tagName === 'INPUT') return;
  if(event.code === 'Space'){event.preventDefault(); togglePlay();}
  if(event.code === 'ArrowRight') api('step', {dir:1});
  if(event.code === 'ArrowLeft') api('step', {dir:-1});
  if(event.code === 'KeyR') api('reset');
  if(event.code === 'KeyN') toggleNoise();
  if(event.key === '1') api('trigger_event', {idx:0});
  if(event.key === '2') api('trigger_event', {idx:1});
  if(event.key === '3') api('trigger_event', {idx:2});
  if(event.key === '4') api('trigger_event', {idx:3});
});

function animate(){
  const dt = clock.getDelta();
  controls.update();
  updateEnvironmentMotion(dt);
  updateWaterMotion(dt);
  updateForestMotion(dt);
  updateParticles(dt);
  if(state){
    updateTowers();
    for(const obj of eventGroup.children){
      if(obj.geometry && obj.geometry.type === 'TorusGeometry'){
        const pulse = 1 + Math.sin(performance.now() / 420 + obj.position.x) * .025;
        obj.scale.setScalar(pulse);
      }
    }
    updateAlarm();
  }
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

buildTerrain();
buildParticles();
buildEnvironmentControls();
bindEnvironmentControls();
resize();
fitWorld();
animate();

fetch('/api/state').then(r => r.json()).then(applyState).catch(() => {
  loadNote.textContent = 'Unable to load world state.';
});
const es = new EventSource('/stream');
es.onmessage = event => {
  const msg = JSON.parse(event.data);
  if(msg.type === 'frame') onFrame(msg.data);
  if(msg.type === 'alert') onAlert(msg);
};
