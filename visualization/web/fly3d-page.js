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
  steeplechase: {
    label: 'Steeplechase', title: '600 body lengths', topSpeed: 1.25 / 0.16,
    note: 'Steeplechase: 12 barriers, four with a water pit behind, and a stamina budget; jumps and splashes are part of the race physics.',
    controllers: [
      { id: 'trained', name: 'Trained fly', note: 'MANC circuit, evolution-trained' },
      UNTRAINED,
      { id: 'rule', name: 'Rule', note: 'Paces itself, jumps each barrier' },
      { id: 'random', name: 'Random', note: 'Seeded random actions' },
    ],
  },
  archery: {
    label: 'Archery', title: '6 arrows, crosswind', topSpeed: 1, scored: true,
    note: 'Archery: aim against a gusty crosswind, then release; scored in points, so higher is better. There is no real bow: the arrows land where the recorded run says.',
    controllers: [
      { id: 'trained', name: 'Trained fly', note: 'MANC circuit, evolution-trained' },
      UNTRAINED,
      { id: 'rule', name: 'Rule', note: 'Aims off the wind, releases when drawn' },
      { id: 'random', name: 'Random', note: 'Seeded random actions' },
    ],
  },
};
const $ = id => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- Scene: a full-size venue on the black bench, and the fly ----------
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

scene.add(new THREE.HemisphereLight(0xfff8ec, 0x1a1b18, 0.9));
const floodlight = new THREE.DirectionalLight(0xfff4e2, 1.1);  // lights the whole venue; the spot follows the fly
floodlight.position.set(-20, 30, 25);
scene.add(floodlight);
const ring = new THREE.SpotLight(0xfffaf0, 120, 30, Math.PI / 6, 0.6, 1.6);  // follows the fly and casts its shadow
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

// Venues are built at full length, one body length (BL) = 2.5 mm, and the camera follows the
// fly down them. x is distance along the course; the fly runs in the middle lane (z = 0).
const BODY_LENGTH_MM = 2.5, LANE = 3.0, LANES = 5, HALF_WIDTH = (LANE * LANES) / 2;
const ripples = noiseTexture(256, '#5f9fbf', '225,242,255', 1400, 0.45);
const trackGrain = noiseTexture(256, '#a9493a', '60,20,14', 3800, 0.22);
const padSide = new THREE.MeshStandardMaterial({ color: 0xd9d5c6, roughness: 0.9 });
const lineWhite = new THREE.MeshStandardMaterial({ color: 0xf1efe6, roughness: 0.8 });
const venue = new THREE.Group();
scene.add(venue);
scene.fog = new THREE.Fog(0x121311, 40, 95);  // far parts of long courses fade into the bench

function strip(length, width, x, z, material, y = 0.004) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(length, width), material);
  mesh.rotation.x = -Math.PI / 2; mesh.position.set(x, y, z); mesh.receiveShadow = true;
  venue.add(mesh);
  return mesh;
}

function buildTrack(lengthBL) {
  const length = lengthBL * BODY_LENGTH_MM, runOff = 30;
  const grain = trackGrain.clone(); grain.needsUpdate = true;
  grain.repeat.set((length + 2 * runOff) / BODY_LENGTH_MM, (2 * HALF_WIDTH + 3) / BODY_LENGTH_MM);
  strip(length + 2 * runOff, 2 * HALF_WIDTH + 3, length / 2, 0, new THREE.MeshStandardMaterial({ map: grain, roughness: 0.95 }), 0);
  for (let i = 0; i <= LANES; i++) strip(length + 2 * runOff, 0.12, length / 2, -HALF_WIDTH + i * LANE, lineWhite);
  strip(0.25, 2 * HALF_WIDTH, 0, 0, lineWhite);                 // start line
  strip(0.6, 2 * HALF_WIDTH, length, 0, lineWhite);             // finish line
  for (let m = 10; m < lengthBL; m += 10) {                      // a mark every 10 BL on both edges
    for (const z of [-HALF_WIDTH - 0.6, HALF_WIDTH + 0.6]) strip(0.12, 0.8, m * BODY_LENGTH_MM, z, lineWhite);
  }
}

function buildPool(lengthBL) {
  const length = lengthBL * BODY_LENGTH_MM, deck = 4, wall = 0.9;
  const water = ripples.clone(); water.needsUpdate = true;
  water.repeat.set(length / BODY_LENGTH_MM, (2 * HALF_WIDTH) / BODY_LENGTH_MM);
  strip(length, 2 * HALF_WIDTH, length / 2, 0, new THREE.MeshStandardMaterial({ map: water, roughness: 0.12, metalness: 0.05 }), 0);
  const tile = new THREE.MeshStandardMaterial({ color: 0xe9e6dc, roughness: 0.85 });
  const box = (w, d, x, z) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, wall, d), tile);
    mesh.position.set(x, wall / 2 - 0.35, z); mesh.receiveShadow = true; venue.add(mesh);
  };
  box(deck, 2 * HALF_WIDTH + 2 * deck, -deck / 2, 0);              // start wall
  box(deck, 2 * HALF_WIDTH + 2 * deck, length + deck / 2, 0);      // turn wall
  box(length, deck, length / 2, -HALF_WIDTH - deck / 2);
  box(length, deck, length / 2, HALF_WIDTH + deck / 2);
  // Lane ropes: floats in alternating colours between the lanes.
  const floats = [new THREE.MeshStandardMaterial({ color: 0xe8628f }), new THREE.MeshStandardMaterial({ color: 0xf1efe6 }), new THREE.MeshStandardMaterial({ color: 0x5b93d6 })];
  const bead = new THREE.SphereGeometry(0.16, 10, 8);
  for (let i = 1; i < LANES; i++) {
    const z = -HALF_WIDTH + i * LANE;
    for (let x = 0.4, k = 0; x < length; x += 0.4, k++) {
      const mesh = new THREE.Mesh(bead, floats[(Math.floor(k / 6)) % floats.length]);
      mesh.position.set(x, 0.06, z); venue.add(mesh);
    }
  }
}

function buildRange() {
  const grass = noiseTexture(256, '#3f6a35', '20,40,16', 4000, 0.3);
  grass.repeat.set(16, 10);
  strip(70, 40, 12, 0, new THREE.MeshStandardMaterial({ map: grass, roughness: 1 }), 0);
  strip(0.3, 14, 0.8, 0, lineWhite);                            // shooting line
}

// Hurdles: tape-white bars half a body length tall, placed from the recorded course.
const hurdleMaterial = new THREE.MeshStandardMaterial({ color: 0xf1efe6, roughness: 0.7, transparent: true });
const hurdleInk = new THREE.MeshStandardMaterial({ color: 0x151513, roughness: 0.7, transparent: true });
const waterTape = new THREE.MeshStandardMaterial({ color: 0x5b93d6, roughness: 0.6, transparent: true });
const pitMaterial = new THREE.MeshStandardMaterial({ color: 0x7fb2c9, roughness: 0.15, metalness: 0.05 });
function makeHurdle(pitLength = 0, w = LANE * 0.85) {
  const h = 0.5 * BODY_LENGTH_MM, group = new THREE.Group();
  if (pitLength) {
    // Steeplechase water jump: the pit starts right behind the barrier.
    const pit = new THREE.Mesh(new THREE.PlaneGeometry(pitLength * BODY_LENGTH_MM, w), pitMaterial);
    pit.rotation.x = -Math.PI / 2; pit.position.set(pitLength * BODY_LENGTH_MM / 2, 0.012, 0);
    pit.userData.length = pitLength * BODY_LENGTH_MM;
    group.userData.pit = pit;
    group.add(pit);
  }
  const bar = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.22, w), pitLength ? waterTape : hurdleMaterial);
  bar.position.y = h - 0.11;
  for (const z of [-w / 2, w / 2]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.09, h, 0.09), hurdleInk);
    post.position.set(0, h / 2, z); group.add(post);
  }
  group.add(bar);
  group.traverse(o => { o.castShadow = true; });
  return group;
}
// Archery: a ring target down the range. Landings are horizontal offsets in
// target units (one ring = ring_width); TARGET_SCALE turns them into millimetres on the face.
const TARGET_X = 16, TARGET_Y = 1.9, TARGET_RADIUS = 1.6;
let TARGET_SCALE = 1;
function ringTexture(rings) {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d'), colours = ['#f1efe6', '#151513', '#5b93d6', '#e8628f', '#e6c63b'];
  for (let i = 0; i < rings; i++) {
    g.beginPath(); g.arc(256, 256, 256 * (1 - i / rings), 0, Math.PI * 2);
    g.fillStyle = colours[Math.floor(i / 2) % colours.length]; g.fill();
    g.strokeStyle = 'rgba(21,21,19,.6)'; g.lineWidth = 2; g.stroke();
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const target = new THREE.Group();
const targetFace = new THREE.Mesh(new THREE.CircleGeometry(TARGET_RADIUS, 64), new THREE.MeshStandardMaterial({ roughness: 0.8 }));
targetFace.rotation.y = -Math.PI / 2;  // faces the fly, which looks along +x
const targetBack = new THREE.Mesh(new THREE.CylinderGeometry(TARGET_RADIUS * 1.04, TARGET_RADIUS * 1.04, 0.12, 64), padSide);
targetBack.rotation.z = Math.PI / 2; targetBack.position.x = 0.07;
const targetLeg = new THREE.Mesh(new THREE.BoxGeometry(0.1, TARGET_Y, 0.1), hurdleInk);
targetLeg.position.set(0.12, -TARGET_Y / 2, 0);
target.add(targetFace, targetBack, targetLeg);
target.position.set(TARGET_X, TARGET_Y, 0);
target.rotation.y = 0.9;  // turned partly toward the viewer so the rings and hits read
target.traverse(o => { o.castShadow = true; });
const hits = new THREE.Group(); target.add(hits);
const aimDot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), new THREE.MeshStandardMaterial({ color: 0xe8628f, emissive: 0x5a1f33 }));
target.add(aimDot);
scene.add(target);
function placeArrows(landings, aim) {
  while (hits.children.length < landings.length) {
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.9, 8), hurdleInk);
    shaft.rotation.z = Math.PI / 2; shaft.castShadow = true;
    hits.add(shaft);
  }
  hits.children.forEach((arrow, i) => {
    const landing = landings[i];
    arrow.visible = i < landings.length && landing != null;
    if (arrow.visible) arrow.position.set(-0.45, 0, -landing * TARGET_SCALE);
  });
  aimDot.position.set(-0.04, 0, -aim * TARGET_SCALE);
}

function setAction(text, state) {
  $('action').textContent = text;
  $('action-tape').dataset.action = state ?? '';
}

// The fly rides in a holder that moves it along the course and turns it at the pool wall.
const holder = new THREE.Group();
scene.add(holder);
let fly;
try {
  fly = await loadFly();
  holder.add(fly.object);
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

// The camera keeps a fixed view of the area around the fly and slides along with it; dragging
// still orbits. Jumping events and the upright Human style get extra headroom.
const VIEW_DIRECTION = new THREE.Vector3(6.4, 5.1, 8.0).normalize();
const VIEW_RADIUS = 7;
function frameThePad() {
  const upright = style === 'human' && eventId !== 'swimming' ? BODY_LENGTH_MM * 0.8 : 0;
  const jumping = eventId === 'hurdles' || eventId === 'steeplechase';
  const headroom = (jumping ? BODY_LENGTH_MM : 0) + upright;
  const radius = eventId === 'archery' ? TARGET_X * 0.62 : VIEW_RADIUS;
  controls.target.set(focusX(), 0.5 + headroom * 0.5, 0);
  const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const distance = Math.max(radius / Math.tan(halfH), (radius * 0.5 + 1.1 + headroom * 0.75) / Math.tan(halfV));
  camera.position.copy(VIEW_DIRECTION).multiplyScalar(distance).add(controls.target);
}
const focusX = () => eventId === 'archery' ? TARGET_X / 2 : holder.position.x;
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
  venue.clear();
  if (id === 'swimming') buildPool(data.pool_length);
  else if (id === 'archery') buildRange();
  else buildTrack(data.distance);
  if (id === 'hurdles') {
    // Hurdles in every lane, at the recorded course positions.
    for (const h of data.hurdles) for (let lane = 0; lane < LANES; lane++) {
      const hurdle = makeHurdle(); hurdle.position.set(h * BODY_LENGTH_MM, 0, -HALF_WIDTH + (lane + 0.5) * LANE); venue.add(hurdle);
    }
  }
  if (id === 'steeplechase') {
    // Steeplechase barriers span the track; water jumps have the pit right behind.
    data.hurdles.forEach((h, i) => {
      const barrier = makeHurdle(data.water.includes(i) ? data.pit_length : 0, 2 * HALF_WIDTH);
      barrier.position.x = h * BODY_LENGTH_MM; venue.add(barrier);
    });
  }
  target.visible = id === 'archery';
  if (id === 'archery') {
    targetFace.material.map?.dispose();
    targetFace.material.map = ringTexture(data.rings); targetFace.material.needsUpdate = true;
    TARGET_SCALE = TARGET_RADIUS / (data.rings * data.ring_width);
  }
  const scored = EVENTS[id].scored;
  $('distance-label').textContent = scored ? 'Points' : 'Distance';
  $('distance-unit').textContent = scored ? `/ ${data.max_points}` : 'BL';
  $('speed-label').textContent = scored ? 'Crosswind' : 'Speed';
  $('speed-unit').textContent = scored ? '' : 'BL/s';
  frameThePad();
  buildRuler(Math.ceil(Math.max(...Object.values(data.runs).map(run => run.at(-1).simulation_time))));
  buildRack();
  selectRun(['sprint', 'steeplechase', 'archery'].includes(id) ? 'trained' : 'rule');
  index = 0; placeFly(0);
  frameThePad();
}

// ---------- Rack of vials ----------
const key = id => `${eventId}:${id}`;
const goal = () => race.max_points ?? race.distance;
const progressOf = record => EVENTS[eventId].scored ? record.total_points : record.position;
function resultOf(id) {
  const last = race.runs[id].at(-1);
  if (EVENTS[eventId].scored) return `${last.total_points} of ${race.max_points} points`;
  return last.position >= race.distance ? `Finished in ${last.simulation_time.toFixed(1)} s` : `Stopped at ${last.position.toFixed(1)} BL`;
}

function buildRack() {
  $('rack').replaceChildren(...EVENTS[eventId].controllers.filter(({ id }) => race.runs[id]).map(({ id, name, note }) => {
    const vial = document.createElement('button');
    vial.type = 'button'; vial.className = 'vial'; vial.dataset.id = id;
    vial.setAttribute('role', 'radio');
    vial.innerHTML = `<span class="vial-tube"><span class="vial-food"></span><ol class="vial-grad">${[25, 50, 75].map(v => `<li style="bottom:${v}%"><b>${Math.round(goal() * v / 100)}</b></li>`).join('')}</ol></span>
      <span class="vial-label">${name}<small>${note}</small><small class="vial-result">Not watched yet</small></span>`;
    vial.onclick = () => selectRun(id);
    return vial;
  }));
  updateRack();
}

function updateRack() {
  for (const vial of $('rack').children) {
    const id = vial.dataset.id;
    const { progress: shown = 0, done = false } = watched.get(key(id)) ?? {};
    const scored = EVENTS[eventId].scored;
    const text = done ? resultOf(id) : watched.has(key(id)) ? (scored ? `${shown} points so far` : `${shown.toFixed(1)} BL so far`) : 'Not watched yet';
    const name = EVENTS[eventId].controllers.find(c => c.id === id).name;
    vial.setAttribute('aria-checked', String(id === selected));
    vial.style.setProperty('--fill', String(Math.min(1, shown / goal())));
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
  return { position: mix('position'), velocity: mix('velocity'), height: mix('height'), time: mix('simulation_time'), action: a.action, airborne: Boolean(a.airborne) || mix('height') > 0, record: a, i };
}

// Where the fly is on the course. In the pool, the second length comes back the other way.
function placeFly(position) {
  let x = position * BODY_LENGTH_MM, heading = 0;
  if (eventId === 'swimming' && race && position > race.pool_length) {
    x = (2 * race.pool_length - position) * BODY_LENGTH_MM; heading = Math.PI;
  }
  if (eventId === 'archery') x = 0;
  holder.position.x = x;
  holder.rotation.y = heading;
}

// Slide the camera and the lights with the fly, keeping whatever angle the viewer orbited to.
function follow() {
  const dx = focusX() - controls.target.x;
  controls.target.x += dx; camera.position.x += dx;
  ring.position.set(holder.position.x + 0.6, 11, 2.2);
  ring.target.position.set(holder.position.x, 0, 0);
  flyPoint.set(holder.position.x, 0.6, 0);
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
    if (index > 0) {
      const before = watched.get(key(selected)) ?? { progress: 0, done: false };
      watched.set(key(selected), { progress: Math.max(before.progress, progressOf(r.record) ?? 0), done: before.done || index >= records.length - 1 });
    }
    if (now - lastRackUpdate > 100) { updateRack(); lastRackUpdate = now; }
    const seconds = Number($('ticks').dataset.seconds) || 1;
    $('ruler').style.setProperty('--progress', `${Math.min(100, (r.time / seconds) * 100)}%`);
    if (index > 0) setAction(r.action, r.action); else setAction('Ready', '');
  }
  placeFly(r.position);
  follow();
  const archery = eventId === 'archery' && race && records.length;
  if (archery) placeArrows(records.slice(0, r.i + 1).filter(x => x.shot).map(x => x.landing), r.record.aim_offset);
  const effort = Math.min(1, r.velocity / EVENTS[eventId].topSpeed);
  const human = style === 'human';
  const stride = human ? (eventId === 'swimming' ? 1.6 : 1.25) : STRIDE;  // body lengths per cycle
  fly.pose(r.position / stride, flying && !human ? 0 : Math.max(effort, r.velocity > 0.01 ? 0.25 : 0), reducedMotion ? 0 : wallTime, {
    event: eventId,
    airborne: r.airborne,
    draw: archery ? r.record.draw : 0,
    lift: flying ? 1.6 + (reducedMotion ? 0 : 0.05 * Math.sin(wallTime * 3)) : r.height * BODY_LENGTH_MM,  // recorded jump height
  });
  scaleBar.style.width = `${bodyLengthOnScreen().toFixed(1)}px`;
  $('time').textContent = r.time.toFixed(1);
  $('distance').textContent = archery ? String(r.record.total_points) : r.position.toFixed(1);
  $('speed').textContent = archery ? (r.record.wind + 0).toFixed(1).replace(/^-0\.0$/, '0.0') : r.velocity.toFixed(1);
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
