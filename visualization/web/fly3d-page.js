import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { loadFly } from './fly3d.js';

const STRIDE = 0.7;  // body lengths per stride cycle, roughly a walking fly's
const UNTRAINED = { id: 'fly', name: 'Untrained fly', note: 'MANC circuit, not trained yet' };
const EVENTS = {
  sprint: {
    label: 'Sprint', title: '100 body lengths', topSpeed: 1.25 / 0.16,
    note: 'Sprint: one push-or-coast choice per 0.1 s step.',
    controllers: [
      { id: 'trained', name: 'Trained fly', note: 'MANC circuit, fitted threshold' },
      { id: 'fly', name: 'Untrained fly', note: 'MANC circuit, starting threshold' },
      { id: 'rule', name: 'Rule', note: 'Hand-written baseline' },
      { id: 'random', name: 'Random', note: 'Seeded coin flips' },
    ],
  },
  hurdles: {
    label: 'Hurdles', title: '200 body lengths', topSpeed: 1.25 / 0.16,
    note: 'Hurdles: the jumps are part of the race physics. Hurdles are half a body length tall.',
    controllers: [
      UNTRAINED,
      { id: 'rule', name: 'Rule', note: 'Brakes to a safe speed, then jumps' },
      { id: 'nocoast', name: 'Rule, no brakes', note: 'Jumps too fast and loses speed' },
      { id: 'random', name: 'Random', note: 'Seeded random jumps' },
    ],
  },
  swimming: {
    label: 'Swimming', title: '100 body lengths', topSpeed: 4,
    note: 'Swimming: real fruit flies cannot swim; on water they are trapped by surface tension. This event is a game model.',
    controllers: [
      UNTRAINED,
      { id: 'rule', name: 'Rule', note: 'Full strokes, glides after dive and turn' },
      { id: 'full_strokes', name: 'Full strokes', note: 'Never glides' },
      { id: 'random', name: 'Random', note: 'Seeded random pulls' },
    ],
  },
};
const $ = id => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- Scene: black epoxy bench, a lit white CO2 pad, the fly ----------
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.setClearColor(0x000000, 0);  // the page's bench shows through; no canvas box

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 200);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enablePan = false;
controls.enableZoom = false;  // the wheel scrolls the page; drag still orbits
controls.target.set(0, 0.5, 0);
controls.maxPolarAngle = Math.PI * 0.47;

scene.add(new THREE.HemisphereLight(0xfff8ec, 0x1a1b18, 0.55));
const ring = new THREE.SpotLight(0xfffaf0, 140, 30, Math.PI / 7, 0.55, 1.6);  // microscope ring light
ring.position.set(0.6, 11, 2.2);
ring.castShadow = true;
ring.shadow.mapSize.set(2048, 2048);
ring.shadow.radius = 5;
ring.shadow.bias = -0.0004;
scene.add(ring, ring.target);
const rim = new THREE.DirectionalLight(0xffe2b8, 0.7);
rim.position.set(-6, 3, -5);
scene.add(rim);

function noiseTexture(size, base, speck, count, alpha) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d');
  g.fillStyle = base; g.fillRect(0, 0, size, size);
  for (let i = 0; i < count; i++) {
    g.fillStyle = `rgba(${speck},${Math.random() * alpha})`;
    g.beginPath(); g.arc(Math.random() * size, Math.random() * size, Math.random() * 1.6 + 0.4, 0, Math.PI * 2); g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}

// The surface slides under the fly like a treadmill: one texture tile per body length.
const PAD_RADIUS = 4.6, BODY_LENGTH_MM = 2.5;
const tiles = (2 * PAD_RADIUS) / BODY_LENGTH_MM;
const porous = noiseTexture(256, '#f3f1ea', '120,116,104', 5200, 0.16);
const ripples = noiseTexture(256, '#7fb2c9', '235,248,255', 900, 0.5);
porous.repeat.set(tiles, tiles); ripples.repeat.set(tiles, tiles);
const padTop = new THREE.MeshStandardMaterial({ map: porous, roughness: 0.95 });
const waterTop = new THREE.MeshStandardMaterial({ map: ripples, roughness: 0.15, metalness: 0.05 });
const padSide = new THREE.MeshStandardMaterial({ color: 0xd9d5c6, roughness: 0.9 });
const pad = new THREE.Mesh(new THREE.CylinderGeometry(PAD_RADIUS, PAD_RADIUS, 0.18, 96), [padSide, padTop, padSide]);
pad.position.y = -0.09; pad.receiveShadow = true;
scene.add(pad);

// Hurdles: tape-white bars half a body length tall, placed from the recorded course.
const hurdleMaterial = new THREE.MeshStandardMaterial({ color: 0xf1efe6, roughness: 0.7, transparent: true });
const hurdleInk = new THREE.MeshStandardMaterial({ color: 0x151513, roughness: 0.7, transparent: true });
function makeHurdle() {
  const h = 0.5 * BODY_LENGTH_MM, w = 2.6, group = new THREE.Group();
  const bar = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.22, w), hurdleMaterial);
  bar.position.y = h - 0.11;
  for (const z of [-w / 2, w / 2]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.09, h, 0.09), hurdleInk);
    post.position.set(0, h / 2, z); group.add(post);
  }
  group.add(bar);
  group.traverse(o => { o.castShadow = true; });
  return group;
}
const hurdleGroup = new THREE.Group();
scene.add(hurdleGroup);

// Swimming: an ink line across the dish marks the pool wall where the turn happens.
const wallMarker = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.02, 2 * PAD_RADIUS * 0.9), hurdleInk);
wallMarker.position.y = 0.01;
scene.add(wallMarker);

function setAction(text, state) {
  $('action').textContent = text;
  $('action-tape').dataset.action = state ?? '';
}

let fly;
try {
  fly = await loadFly();
  scene.add(fly.object);
} catch (error) {
  setAction('Fly model failed to load', 'error');
  throw error;
}

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  frameThePad();
}

// Keep the whole pad and its rim inside the canvas at any aspect, so the bench shows all round it.
const VIEW_DIRECTION = new THREE.Vector3(6.4, 5.1, 8.0).normalize();
function frameThePad() {
  // Hurdles need headroom for a full jump (up to one body length, 2.5 mm).
  // Human style stands upright (about one body length tall) and jumps from that height.
  const upright = style === 'human' && eventId !== 'swimming' ? BODY_LENGTH_MM * 0.8 : 0;
  const headroom = (eventId === 'hurdles' ? BODY_LENGTH_MM : 0) + upright;
  controls.target.y = 0.5 + headroom * 0.5;
  const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  // The disc fills about 88% of the canvas width, rim clear of every edge.
  const distance = Math.max(PAD_RADIUS / 0.88 / Math.tan(halfH), (PAD_RADIUS * 0.5 + 1.1 + headroom * 0.75) / Math.tan(halfV));
  camera.position.copy(VIEW_DIRECTION).multiplyScalar(distance).add(controls.target);
}
let payload = null, eventId = 'sprint', race = null, selected = 'rule', style = 'human';
new ResizeObserver(resize).observe(canvas);
resize();

// ---------- State ----------
let records = [], index = 0, playing = false, flying = false, rate = 1;
const watched = new Map();  // `${event}:${controller}` -> furthest distance shown, kept as a record

function setMode(flyMode) {
  flying = flyMode;
  fly.setFlying(flying);
  $('run-mode').setAttribute('aria-checked', String(!flying));
  $('fly-mode').setAttribute('aria-checked', String(flying));
  $('cheat').hidden = !flying;
}
$('run-mode').onclick = () => setMode(false);
$('fly-mode').onclick = () => setMode(true);
setMode(false);

function setStyle(value) {
  style = value;
  fly.setStyle(style);
  $('human-style').setAttribute('aria-checked', String(style === 'human'));
  $('fly-style').setAttribute('aria-checked', String(style === 'fly'));
  $('style-note').hidden = style !== 'human';
  frameThePad();
}
$('human-style').onclick = () => setStyle('human');
$('fly-style').onclick = () => setStyle('fly');
setStyle('human');

function setPlaying(value) {
  playing = value;
  $('play').dataset.playing = String(playing);
  $('play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
  $('play-label').textContent = playing ? 'Pause' : records.length && index >= records.length - 1 ? 'Replay' : 'Play';
}
$('play').onclick = () => {
  if (!records.length) return;
  if (!playing && index >= records.length - 1) index = 0;
  setPlaying(!playing);
};
$('fast').onclick = () => {
  rate = rate === 1 ? 4 : 1;
  $('fast').setAttribute('aria-pressed', String(rate === 4));
};
$('timeline').oninput = event => { index = Number(event.target.value); setPlaying(false); };

addEventListener('keydown', event => {
  if (event.target.closest?.('input, select, textarea')) return;
  if (event.code === 'Space') { event.preventDefault(); $('play').click(); }
  else if (event.key >= '1' && event.key <= '4') selectRun(EVENTS[eventId].controllers[Number(event.key) - 1]?.id);
  else if (event.key === 'f') setMode(!flying);
});

// ---------- Events ----------
function eventData(id) {
  if (!payload) return null;
  if (id === 'sprint') return { distance: 100, dt: 0.1, runs: payload.runs };
  return payload.events?.[id] ?? null;
}

function buildEventTabs() {
  $('events').replaceChildren(...Object.entries(EVENTS).map(([id, { label }]) => {
    const tab = document.createElement('button');
    tab.type = 'button'; tab.className = 'tape-button'; tab.dataset.id = id;
    tab.setAttribute('role', 'radio');
    tab.textContent = label;
    tab.disabled = !eventData(id);
    tab.onclick = () => selectEvent(id);
    return tab;
  }));
}

function selectEvent(id) {
  const data = eventData(id); if (!data) return;
  eventId = id; race = data;
  for (const tab of $('events').children) tab.setAttribute('aria-checked', String(tab.dataset.id === id));
  $('event-title').textContent = EVENTS[id].title;
  $('event-note').textContent = EVENTS[id].note;
  pad.material[1] = id === 'swimming' ? waterTop : padTop;
  hurdleGroup.clear();
  if (id === 'hurdles') hurdleGroup.add(...data.hurdles.map(() => makeHurdle()));
  wallMarker.visible = id === 'swimming';
  frameThePad();
  buildRuler(Math.ceil(Math.max(...Object.values(data.runs).map(run => run.at(-1).simulation_time))));
  buildRack();
  selectRun(id === 'sprint' ? 'trained' : 'rule');
}

// ---------- Rack of vials ----------
const key = id => `${eventId}:${id}`;
function resultOf(id) {
  const last = race.runs[id].at(-1);
  return last.position >= race.distance ? `Finished in ${last.simulation_time.toFixed(1)} s` : `Stopped at ${last.position.toFixed(1)} BL`;
}

function buildRack() {
  $('rack').replaceChildren(...EVENTS[eventId].controllers.filter(({ id }) => race.runs[id]).map(({ id, name, note }) => {
    const vial = document.createElement('button');
    vial.type = 'button'; vial.className = 'vial'; vial.dataset.id = id;
    vial.setAttribute('role', 'radio');
    vial.innerHTML = `<span class="vial-tube"><span class="vial-food"></span><ol class="vial-grad">${[25, 50, 75].map(v => `<li style="bottom:${v}%"><b>${Math.round(race.distance * v / 100)}</b></li>`).join('')}</ol></span>
      <span class="vial-label">${name}<small>${note}</small><small class="vial-result">Not watched yet</small></span>`;
    vial.onclick = () => selectRun(id);
    return vial;
  }));
  updateRack();
}

function updateRack() {
  for (const vial of $('rack').children) {
    const id = vial.dataset.id;
    const shown = watched.get(key(id)) ?? 0;
    const finished = shown > 0 && shown >= race.runs[id].at(-1).position - 1e-6;
    const text = finished ? resultOf(id) : shown > 0 ? `${shown.toFixed(1)} BL so far` : 'Not watched yet';
    const name = EVENTS[eventId].controllers.find(c => c.id === id).name;
    vial.setAttribute('aria-checked', String(id === selected));
    vial.style.setProperty('--fill', String(Math.min(1, shown / race.distance)));
    vial.querySelector('.vial-result').textContent = text;
    vial.setAttribute('aria-label', `${name}: ${text}`);
  }
}

function buildRuler(seconds) {
  const major = seconds > 80 ? 20 : 10, items = [];
  for (let s = 0; s <= seconds; s += 1) {
    const isMajor = s % major === 0;
    items.push(`<li class="${isMajor ? 'major' : ''}" style="left:${(s / seconds) * 100}%">${isMajor ? `<b>${s}</b>` : ''}</li>`);
  }
  $('ticks').innerHTML = items.join('');
  $('ticks').dataset.seconds = String(seconds);
}

function selectRun(id) {
  if (!race?.runs[id]) return;
  selected = id; records = race.runs[id]; index = 0;
  setPlaying(false);
  $('timeline').max = String(records.length - 1);
  $('timeline').value = '0';
  $('timeline').disabled = false;
  $('play').disabled = false;
  updateRack();
}

// The scale bar is measured from the camera each frame, so it is true at the fly's position.
const scaleBar = document.querySelector('.scale-bar i');
const flyPoint = new THREE.Vector3(0, 0.6, 0), near = new THREE.Vector3(), far = new THREE.Vector3();
function bodyLengthOnScreen() {
  near.copy(flyPoint).project(camera);
  far.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(BODY_LENGTH_MM).add(flyPoint).project(camera);
  return (Math.abs(far.x - near.x) / 2) * canvas.clientWidth;
}

// ---------- Playback ----------
function sample(at) {
  if (!records.length) return { position: 0, velocity: 0, action: '', height: 0, time: 0, airborne: false };
  const i = Math.min(records.length - 1, Math.floor(at)), j = Math.min(records.length - 1, i + 1), f = at - i;
  const a = records[i], b = records[j];
  const mix = field => (a[field] ?? 0) + ((b[field] ?? 0) - (a[field] ?? 0)) * f;
  return { position: mix('position'), velocity: mix('velocity'), height: mix('height'), time: mix('simulation_time'), action: a.action, airborne: Boolean(a.airborne) || mix('height') > 0 };
}

// Things on the course sit at (their position - the fly's) body lengths ahead, and fade out at the rim.
function placeOnCourse(object, ahead) {
  const x = ahead * BODY_LENGTH_MM, edge = PAD_RADIUS * 0.82;
  object.position.x = x;
  object.visible = Math.abs(x) < edge;
}

let last = performance.now(), wallTime = 0, lastRackUpdate = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now; wallTime += dt;
  if (playing && records.length) {
    index = Math.min(records.length - 1, index + rate * dt / race.dt);  // playback rate only; the recorded run never changes
    if (index >= records.length - 1) setPlaying(false);
    $('timeline').value = String(Math.floor(index));
  }
  const r = sample(index);
  if (records.length) {
    if (index > 0) watched.set(key(selected), Math.max(watched.get(key(selected)) ?? 0, r.position));
    if (now - lastRackUpdate > 100) { updateRack(); lastRackUpdate = now; }
    const seconds = Number($('ticks').dataset.seconds) || 1;
    $('ruler').style.setProperty('--progress', `${Math.min(100, (r.time / seconds) * 100)}%`);
    if (index > 0) setAction(r.action, r.action); else setAction('Ready', '');
  }
  if (eventId === 'hurdles' && race) race.hurdles.forEach((h, i) => placeOnCourse(hurdleGroup.children[i], h - r.position));
  if (eventId === 'swimming' && race) placeOnCourse(wallMarker, race.pool_length - r.position);
  const effort = Math.min(1, r.velocity / EVENTS[eventId].topSpeed);
  const human = style === 'human';
  const stride = human ? (eventId === 'swimming' ? 1.6 : 1.25) : STRIDE;  // body lengths per cycle
  fly.pose(r.position / stride, flying && !human ? 0 : Math.max(effort, r.velocity > 0.01 ? 0.25 : 0), reducedMotion ? 0 : wallTime, {
    event: eventId,
    airborne: r.airborne,
    lift: flying ? 1.6 + (reducedMotion ? 0 : 0.05 * Math.sin(wallTime * 3)) : r.height * BODY_LENGTH_MM,  // recorded jump height
  });
  porous.offset.x = ripples.offset.x = r.position;  // one tile per body length
  scaleBar.style.width = `${bodyLengthOnScreen().toFixed(1)}px`;
  $('time').textContent = r.time.toFixed(1);
  $('distance').textContent = r.position.toFixed(1);
  $('speed').textContent = r.velocity.toFixed(1);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

try {
  const response = await fetch('/runs.json');
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  payload = await response.json();
  buildEventTabs();
  selectEvent('sprint');
} catch (error) {
  setAction('No runs. Start python -m experiments.serve_view', 'error');
  console.error(error);
}
