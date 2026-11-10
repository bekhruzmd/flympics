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
      { id: 'trained', name: 'Trained fly', note: 'MANC circuit, evolution-trained; never jumps' },
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
    note: 'Archery: points, higher is better; no real bow, arrows land where the recorded run says. "Sees the wind" senses wind via a FlyVis calibration (a model of real fly motion vision); the eye-to-leg link is modeled.',
    controllers: [
      { id: 'eye', name: 'Sees the wind', note: 'FlyVis eye, evolution-trained' },
      { id: 'trained', name: 'Trained fly', note: 'MANC circuit, evolution-trained' },
      UNTRAINED,
      { id: 'rule', name: 'Rule', note: 'Aims off the wind, releases when drawn' },
      { id: 'random', name: 'Random', note: 'Seeded random actions' },
    ],
  },
  pacing: {
    label: '400 m', title: '400 body lengths, pace yourself', topSpeed: 1.25 / 0.10,
    note: '400 m: every lane starts with a modeled energy budget of 100 units. The energy bars replay the recorded budget.',
    controllers: [
      { id: 'trained', name: 'Trained fly', note: 'MANC circuit, fitted threshold' },
      UNTRAINED,
      { id: 'rule', name: 'Rule', note: 'Paced cruise at 3.3 BL/s' },
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

// x is distance along the course. The selected vial gets the middle lane (z = 0).
const BODY_LENGTH_MM = 2.5, LANE = 3.0;
let trackLaneCount = 5, halfTrackWidth = LANE * trackLaneCount / 2;
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
  grain.repeat.set((length + 2 * runOff) / BODY_LENGTH_MM, (2 * halfTrackWidth + 3) / BODY_LENGTH_MM);
  strip(length + 2 * runOff, 2 * halfTrackWidth + 3, length / 2, 0, new THREE.MeshStandardMaterial({ map: grain, roughness: 0.95 }), 0);
  for (let i = 0; i <= trackLaneCount; i++) strip(length + 2 * runOff, 0.12, length / 2, -halfTrackWidth + i * LANE, lineWhite);
  strip(0.25, 2 * halfTrackWidth, 0, 0, lineWhite);                 // start line
  strip(0.6, 2 * halfTrackWidth, length, 0, lineWhite);             // finish line
  for (let m = 10; m < lengthBL; m += 10) {                      // a mark every 10 BL on both edges
    for (const z of [-halfTrackWidth - 0.6, halfTrackWidth + 0.6]) strip(0.12, 0.8, m * BODY_LENGTH_MM, z, lineWhite);
  }
}

function buildPool(lengthBL) {
  const length = lengthBL * BODY_LENGTH_MM, deck = 4, wall = 0.9;
  const water = ripples.clone(); water.needsUpdate = true;
  water.repeat.set(length / BODY_LENGTH_MM, (2 * halfTrackWidth) / BODY_LENGTH_MM);
  strip(length, 2 * halfTrackWidth, length / 2, 0, new THREE.MeshStandardMaterial({ map: water, roughness: 0.12, metalness: 0.05 }), 0);
  const tile = new THREE.MeshStandardMaterial({ color: 0xe9e6dc, roughness: 0.85 });
  const box = (w, d, x, z) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, wall, d), tile);
    mesh.position.set(x, wall / 2 - 0.35, z); mesh.receiveShadow = true; venue.add(mesh);
  };
  box(deck, 2 * halfTrackWidth + 2 * deck, -deck / 2, 0);              // start wall
  box(deck, 2 * halfTrackWidth + 2 * deck, length + deck / 2, 0);      // turn wall
  box(length, deck, length / 2, -halfTrackWidth - deck / 2);
  box(length, deck, length / 2, halfTrackWidth + deck / 2);
  // Lane ropes: floats in alternating colours between the lanes.
  const floats = [new THREE.MeshStandardMaterial({ color: 0xe8628f }), new THREE.MeshStandardMaterial({ color: 0xf1efe6 }), new THREE.MeshStandardMaterial({ color: 0x5b93d6 })];
  const bead = new THREE.SphereGeometry(0.16, 10, 8);
  const beads = Array.from({ length: floats.length }, () => []);
  for (let i = 1; i < trackLaneCount; i++) {
    const z = -halfTrackWidth + i * LANE;
    for (let x = 0.4, k = 0; x < length; x += 0.4, k++) beads[Math.floor(k / 6) % floats.length].push([x, z]);
  }
  const transform = new THREE.Matrix4();
  beads.forEach((positions, i) => {
    const rope = new THREE.InstancedMesh(bead, floats[i], positions.length);
    positions.forEach(([x, z], k) => rope.setMatrixAt(k, transform.makeTranslation(x, 0.06, z)));
    venue.add(rope);
  });
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
function clearVenue() {
  const sharedMaterials = new Set([lineWhite, hurdleMaterial, hurdleInk, waterTape, pitMaterial]);
  const geometries = new Set(), materials = new Set();
  venue.traverse(object => {
    if (object.isInstancedMesh) object.dispose();
    if (!object.isMesh) return;
    geometries.add(object.geometry);
    if (!sharedMaterials.has(object.material)) materials.add(object.material);
  });
  venue.clear();
  geometries.forEach(geometry => geometry.dispose());
  materials.forEach(material => { material.map?.dispose(); material.dispose(); });
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
// Two lanes, each with a target and its own arrows: the selected fly and its opponent.
const LANE_Z = { main: 0, rival: -5 };
function makeTarget(z) {
  const group = new THREE.Group();
  const face = new THREE.Mesh(new THREE.CircleGeometry(TARGET_RADIUS, 64), new THREE.MeshStandardMaterial({ roughness: 0.8 }));
  face.rotation.y = -Math.PI / 2;  // faces the archers, who look along +x
  const back = new THREE.Mesh(new THREE.CylinderGeometry(TARGET_RADIUS * 1.04, TARGET_RADIUS * 1.04, 0.12, 64), padSide);
  back.rotation.z = Math.PI / 2; back.position.x = 0.07;
  const leg = new THREE.Mesh(new THREE.BoxGeometry(0.1, TARGET_Y, 0.1), hurdleInk);
  leg.position.set(0.12, -TARGET_Y / 2, 0);
  const hits = new THREE.Group();
  const aimDot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshStandardMaterial({ color: 0xe8628f, emissive: 0x5a1f33 }));
  group.add(face, back, leg, hits, aimDot);
  group.position.set(TARGET_X, TARGET_Y, z);
  group.traverse(o => { o.castShadow = true; });
  scene.add(group);
  return { group, face, hits, aimDot, flights: [], shown: 0 };
}
const lanes = { main: makeTarget(LANE_Z.main), rival: makeTarget(LANE_Z.rival) };
const target = lanes.main.group;
const arrowGeometry = new THREE.CylinderGeometry(0.025, 0.025, 0.9, 8);
arrowGeometry.rotateZ(Math.PI / 2);  // arrows point along +x
// Where an arrow lands on a lane's target, in world space. Landing > 0 is to the archer's right (+z).
function landingPoint(lane, landing) {
  return new THREE.Vector3(TARGET_X - 0.45, TARGET_Y, LANE_Z[lane] + landing * TARGET_SCALE);
}
// Recorded landings appear on the target; a newly released arrow first flies there from the bow.
function placeArrows(lane, landings, aim, from) {
  const t = lanes[lane];
  if (landings.length < t.shown) { t.flights = []; }  // scrubbed backwards
  for (let k = t.shown; k < landings.length; k++) {
    if (landings[k] != null && playing) t.flights.push({ k, start: wallTime, from: from.clone() });
  }
  t.shown = landings.length;
  while (t.hits.children.length < landings.length) {
    const arrow = new THREE.Mesh(arrowGeometry, hurdleInk); arrow.castShadow = true; t.hits.add(arrow);
  }
  const flying = new Set();
  t.flights = t.flights.filter(f => {
    const u = (wallTime - f.start) / 0.35;
    if (u >= 1) return false;
    flying.add(f.k);
    const to = landingPoint(lane, landings[f.k]);
    const arrow = t.hits.children[f.k];
    arrow.visible = true;
    arrow.position.copy(f.from).lerp(to, u).sub(t.group.position);
    arrow.position.y += 0.6 * Math.sin(Math.PI * u);  // a little arc
    return true;
  });
  t.hits.children.forEach((arrow, k) => {
    if (flying.has(k)) return;
    arrow.visible = k < landings.length && landings[k] != null;
    if (arrow.visible) arrow.position.copy(landingPoint(lane, landings[k])).sub(t.group.position);
  });
  t.aimDot.visible = aim != null;
  if (aim != null) t.aimDot.position.set(-0.04, 0, aim * TARGET_SCALE);
}

// A bow for each archer: a curved stave and a string the recorded draw pulls back.
function makeBow() {
  const group = new THREE.Group();
  const stave = new THREE.Mesh(new THREE.TorusGeometry(0.95, 0.045, 8, 40, Math.PI), new THREE.MeshStandardMaterial({ color: 0x8f5e17, roughness: 0.6 }));
  stave.rotation.z = -Math.PI / 2;  // bulges toward the target
  const stringMaterial = new THREE.LineBasicMaterial({ color: 0xf1efe6 });
  const string = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.95, 0), new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -0.95, 0)]), stringMaterial);
  const nocked = new THREE.Mesh(arrowGeometry, hurdleInk);
  group.add(stave, string, nocked);
  scene.add(group);
  return { group, string, nocked };
}
const bows = { main: makeBow(), rival: makeBow() };
function placeBow(bow, flyObject, laneZ, draw, show) {
  bow.group.visible = show;
  if (!show) return;
  // Held out in front of the archer at shoulder height.
  const hand = new THREE.Vector3();
  flyObject.getWorldPosition(hand);
  bow.group.position.set(hand.x + 1.25, 1.75, laneZ + 0.15);
  const pull = -0.55 * Math.max(0, Math.min(1, draw));
  const points = bow.string.geometry.attributes.position;
  points.setX(1, pull); points.needsUpdate = true;
  bow.nocked.visible = draw > 0;
  bow.nocked.position.set(pull + 0.45, 0, 0);
}

function setText(id, text) {
  const element = $(id);
  if (element.textContent !== text) element.textContent = text;
}
function setAction(text, state) {
  setText('action', text);
  const tape = $('action-tape');
  if (tape.dataset.action !== (state ?? '')) tape.dataset.action = state ?? '';
}

// The fly rides in a holder that moves it along the course and turns it at the pool wall.
const holder = new THREE.Group();
scene.add(holder);
// Archery's opponent: a second scanned fly in the next lane.
const rivalHolder = new THREE.Group();
rivalHolder.position.z = LANE_Z.rival;
scene.add(rivalHolder);
let fly, rival;
try {
  [fly, rival] = await Promise.all([loadFly(), loadFly()]);
  holder.add(fly.object);
  rivalHolder.add(rival.object);
  rivalHolder.visible = false;
} catch (error) {
  setAction('Fly model failed to load', 'error');
  throw error;
}

// ---------- Drawing quality ----------
// Full quality by default. ?quality=medium|low draws fewer pixels and shadows for a weak
// demo laptop. Only the picture changes: every lane still replays its own recorded run.
const QUALITY = [
  { name: 'high', pixelRatio: 2, shadows: 'all', shadowMap: 2048, soft: true },
  { name: 'medium', pixelRatio: 1.25, shadows: 'main', shadowMap: 1024, soft: false },
  { name: 'low', pixelRatio: 1, shadows: 'none', shadowMap: 1024, soft: false },
];
const pinnedQuality = QUALITY.findIndex(q => q.name === new URLSearchParams(location.search).get('quality'));
const quality = pinnedQuality >= 0 ? pinnedQuality : 0;

function castShadows(flyModel, on) {
  flyModel.object.traverse(o => {
    if (!o.isMesh) return;
    o.userData.castsShadow ??= o.castShadow;  // wings never cast one
    o.castShadow = on && o.userData.castsShadow;
  });
}
function applyQuality() {
  const q = QUALITY[quality];
  renderer.setPixelRatio(Math.min(devicePixelRatio, q.pixelRatio));
  renderer.shadowMap.type = q.soft ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
  ring.castShadow = q.shadows !== 'none';
  if (ring.shadow.mapSize.x !== q.shadowMap) {
    ring.shadow.mapSize.set(q.shadowMap, q.shadowMap);
    ring.shadow.map?.dispose(); ring.shadow.map = null;
  }
  castShadows(fly, q.shadows !== 'none');
  for (const other of [rival, ...reserveFlies.map(slot => slot.fly)]) castShadows(other, q.shadows === 'all');
  document.body.dataset.quality = q.name;
  resize();
}

// Sizes read once per resize, not every frame: reading layout after the frame's DOM
// writes forces the browser to lay out the whole page again.
const view = { w: 0, h: 0, topEdge: 0, topEdgeAt: -Infinity };
function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  if (!w || !h) return;
  view.w = w; view.h = h; view.topEdgeAt = -Infinity;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  frameThePad();
}

// Event camera presets. Orbit remains available during replay.
const RACE_SHOTS = {
  sprint: { direction: [0.40, 0.42, 1], span: 10, height: 1.5 },
  pacing: { direction: [-1, 0.62, 0.48], span: 13, height: 1.7 },
  hurdles: { direction: [0.52, 0.04, 1], span: 10, height: 1.15 },
  steeplechase: { direction: [0.40, 1.05, 0.84], span: 18, height: 1.2 },
  swimming: { direction: [0, 1, 0.002], span: 13, height: 0 },
};
const PHOTO_SHOT = { direction: [0, 0.35, 1], span: 9, height: 1.1 };
let payload = null, eventId = 'sprint', race = null, selected = 'rule', style = 'human';
let records = [], index = 0, playing = false, flying = false, rate = 1;
let raceLanes = [], replayEnd = 0, firstFinish = -1, photoUntil = 0;
let leader = null, cameraX = 0, currentCameraDistance = 24, photoShown = false, snapCamera = true, sheetStep = -1;
const reserveFlies = [];

function cameraLane() {
  return leader?.holder ?? holder;
}

function focusX() {
  if (eventId === 'archery') return TARGET_X * 0.5;
  if (photoShown) return race.distance * BODY_LENGTH_MM;
  return cameraLane().position.x;
}

function frameThePad() {
  if (eventId === 'archery') {
    // Over the archer's shoulder, looking down the range at both targets.
    controls.target.set(focusX(), 1.4, -2.6);
    camera.position.set(-7.5, 4.6, 4.2);
    return;
  }
  const { direction, span, height } = photoShown ? PHOTO_SHOT : RACE_SHOTS[eventId];
  const laneOffsets = raceLanes.map(lane => lane.holder.position.z);
  const centreZ = laneOffsets.length ? (Math.min(...laneOffsets) + Math.max(...laneOffsets)) / 2 : 0;
  cameraX = focusX();
  controls.target.set(cameraX, height, centreZ);
  const halfV = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const halfSpan = Math.max(span, laneOffsets.length * LANE * 0.65);
  const distance = Math.max(halfSpan / (halfV * camera.aspect), (halfSpan * 0.58 + 1.5) / halfV);
  camera.position.set(...direction).normalize().multiplyScalar(distance).add(controls.target);
  currentCameraDistance = distance;
}
new ResizeObserver(resize).observe(canvas);
resize();

const watched = new Map();  // `${event}:${controller}` -> furthest distance shown, kept as a record

function setMode(flyMode) {
  flying = flyMode && eventId === 'archery';
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
  $('play-label').textContent = playing ? 'Pause' : records.length && index >= replayEnd ? 'Replay' : 'Play';
}
$('play').onclick = () => {
  if (!records.length) return;
  if (!playing && index >= replayEnd) {
    index = 0; photoUntil = 0; snapCamera = true;
    $('timeline').value = '0';
  }
  setPlaying(!playing);
};
$('fast').onclick = () => {
  rate = rate === 4 ? 1 : 4;
  $('fast').setAttribute('aria-pressed', String(rate === 4));
};
$('timeline').oninput = event => {
  index = Number(event.target.value);
  photoUntil = 0; snapCamera = true;
  setPlaying(false);
};

for (const button of $('eye-modes').children) {
  button.onclick = () => {
    eyeMode = button.dataset.eye;
    for (const other of $('eye-modes').children) other.setAttribute('aria-checked', String(other === button));
    if (selected === 'eye') { selectRun('eye'); setPlaying(true); }
  };
}

addEventListener('keydown', event => {
  if (event.target.closest?.('input, select, textarea')) return;
  if (event.code === 'Space') { event.preventDefault(); $('play').click(); }
  else if (event.key >= '1' && event.key <= '9') selectRun(controllers()[Number(event.key) - 1]?.id);
  else if (event.key === 'f') setMode(!flying);
  else if (event.key === 'p' || event.key === 'P') togglePresenter();
  else if (presenting && event.key === 'ArrowRight') presenterStep(step + 1);
  else if (presenting && event.key === 'ArrowLeft') presenterStep(step - 1);
  else if (presenting && event.key === 'Escape') togglePresenter();
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
  $('pad').classList.toggle('is-race', id !== 'archery');
  $('flight-modes').hidden = id !== 'archery';
  $('rack-title').textContent = id === 'archery' ? 'Pull a vial to load its run' : 'Pull a vial to choose the main lane';
  if (id !== 'archery') setMode(false);
  clearVenue();
  if (id === 'swimming') buildPool(data.pool_length);
  else if (id === 'archery') buildRange();
  else buildTrack(data.distance);
  if (id === 'hurdles') {
    // Hurdles in every lane, at the recorded course positions.
    for (const h of data.hurdles) for (let lane = 0; lane < trackLaneCount; lane++) {
      const hurdle = makeHurdle(); hurdle.position.set(h * BODY_LENGTH_MM, 0, -halfTrackWidth + (lane + 0.5) * LANE); venue.add(hurdle);
    }
  }
  if (id === 'steeplechase') {
    // Steeplechase barriers span the track; water jumps have the pit right behind.
    data.hurdles.forEach((h, i) => {
      const barrier = makeHurdle(data.water.includes(i) ? data.pit_length : 0, 2 * halfTrackWidth);
      barrier.position.x = h * BODY_LENGTH_MM; venue.add(barrier);
    });
  }
  for (const lane of Object.values(lanes)) {
    lane.group.visible = id === 'archery';
    lane.flights = []; lane.shown = 0;
    if (id === 'archery') {
      lane.face.material.map?.dispose();
      lane.face.material.map = ringTexture(data.rings); lane.face.material.needsUpdate = true;
    }
  }
  if (id === 'archery') TARGET_SCALE = TARGET_RADIUS / (data.rings * data.ring_width);
  rivalHolder.visible = id === 'archery';
  $('rival-pick').hidden = id !== 'archery';
  $('versus-tape').hidden = id !== 'archery';
  $('target-cam').hidden = id !== 'archery';
  if (id !== 'archery') { $('lane-main').hidden = true; $('lane-rival').hidden = true; }
  for (const bow of Object.values(bows)) bow.group.visible = false;
  const scored = EVENTS[id].scored;
  $('distance-label').textContent = scored ? 'Points' : 'Distance';
  $('distance-unit').textContent = scored ? `/ ${data.max_points}` : 'BL';
  $('speed-label').textContent = scored ? 'Crosswind' : 'Speed';
  $('speed-unit').textContent = scored ? '' : 'BL/s';
  buildRack();
  const preferred = id === 'archery' ? 'eye' : ['sprint', 'steeplechase', 'pacing'].includes(id) ? 'trained' : 'rule';
  selectRun(data.runs[runId(preferred)]?.length ? preferred : controllers()[0]?.id);
}

// ---------- Rack of vials ----------
// The "Sees the wind" vial plays one of three recorded runs: eye open, blindfolded or swapped.
let eyeMode = 'eye';
const runId = id => (eventId === 'archery' && id === 'eye' ? eyeMode : id);
const key = id => `${eventId}:${runId(id)}`;
const goal = () => race.max_points ?? race.distance;
const progressOf = record => EVENTS[eventId].scored ? record.total_points : record.position;
function resultOf(id) {
  const last = race.runs[runId(id)].at(-1);
  if (EVENTS[eventId].scored) return `${last.total_points} of ${race.max_points} points`;
  return last.position >= race.distance ? `Finished in ${formatTime(last.simulation_time)} s` : `DNF · ${last.position.toFixed(1)} BL`;
}

function controllers() {
  const known = EVENTS[eventId].controllers;
  if (!race) return [];
  const entries = known.filter(({ id }) => race.runs[runId(id)]?.length);
  if (eventId !== 'archery') {
    for (const [id, run] of Object.entries(race.runs)) {
      if (run.length && !entries.some(c => c.id === id)) entries.push({ id, name: id, note: 'Recorded controller' });
    }
  }
  return entries;
}

function buildRack() {
  $('rack').replaceChildren(...controllers().map(({ id, name, note }) => {
    const vial = document.createElement('button');
    vial.type = 'button'; vial.className = 'vial'; vial.dataset.id = id;
    vial.setAttribute('role', 'radio');
    vial.innerHTML = `<span class="vial-tube"><span class="vial-food"></span><ol class="vial-grad">${[25, 50, 75].map(v => `<li style="bottom:${v}%"><b>${Math.round(goal() * v / 100)}</b></li>`).join('')}</ol></span>
      <span class="vial-label"><span></span><small></small><small class="vial-result">Not watched yet</small></span>`;
    vial.querySelector('.vial-label span').textContent = name;
    vial.querySelector('.vial-label small').textContent = note;
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
    const name = vial.querySelector('.vial-label span').textContent;
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

function syncTimeline() {
  const runs = eventId === 'archery'
    ? [records, race.runs[runId(rivalId)] ?? []]
    : raceLanes.map(lane => lane.run);
  replayEnd = Math.max(...runs.map(run => run.length)) - 1;
  index = Math.min(index, replayEnd);
  $('timeline').max = String(replayEnd);
  $('timeline').value = String(Math.floor(index));
  buildRuler(Math.ceil(Math.max(...runs.map(run => run.at(-1)?.simulation_time ?? 0))));
}

function selectRun(id) {
  if (!race?.runs[runId(id)]?.length) return;
  selected = id; records = race.runs[runId(id)]; index = 0;
  photoUntil = 0; photoShown = false; snapCamera = true;
  $('photo-tape').hidden = true;
  $('eye-modes').hidden = !(eventId === 'archery' && id === 'eye');
  if (eventId === 'archery') {
    fillRivalOptions();
    for (const lane of Object.values(lanes)) { lane.flights = []; lane.shown = 0; }
  }
  buildRaceLanes();
  syncTimeline();
  setPlaying(false);
  $('timeline').disabled = false;
  $('play').disabled = false;
  frameThePad();
  updateRack();
}

// The scale bar is measured from the camera each frame, so it is true at the fly's position.
const scaleBar = document.querySelector('.scale-bar i');
const flyPoint = new THREE.Vector3(0, 0.6, 0), near = new THREE.Vector3(), far = new THREE.Vector3();
function bodyLengthOnScreen() {
  near.copy(flyPoint).project(camera);
  far.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(BODY_LENGTH_MM).add(flyPoint).project(camera);
  return (Math.abs(far.x - near.x) / 2) * view.w;
}

// Lead group: the leader plus any racers within 8 body lengths.
// Hysteresis keeps members from flickering: join at 8, drop out at 10.
const LEAD_GROUP_JOIN = 8;
const LEAD_GROUP_DROP = 10;
const BASE_RING_ANGLE = Math.PI / 6;
const MAX_RING_ANGLE = Math.PI * 0.42;
const MAX_PACK_DISTANCE = 120;

const packDirection = new THREE.Vector3();
const packRight = new THREE.Vector3();
const packUp = new THREE.Vector3();
const packBounds = new THREE.Box3();
const packPoint = new THREE.Vector3();
const packCentre = new THREE.Vector3();
const packOffset = new THREE.Vector3();
const goalTarget = new THREE.Vector3();
const goalRingPos = new THREE.Vector3();
const goalRingTarget = new THREE.Vector3();
const goalFlyPoint = new THREE.Vector3();

function updateLeadGroup() {
  if (!raceLanes.length) return;
  let maxPos = -Infinity;
  for (const lane of raceLanes) {
    const pos = lane.record?.position ?? 0;
    if (pos > maxPos) maxPos = pos;
  }
  for (const lane of raceLanes) {
    const pos = lane.record?.position ?? 0;
    const gap = maxPos - pos;
    // Ties for the lead are always in the group.
    if (gap < 0.001) {
      lane.inLeadGroup = true;
    } else if (snapCamera || !playing) {
      lane.inLeadGroup = gap <= LEAD_GROUP_JOIN;
    } else if (lane.inLeadGroup) {
      if (gap > LEAD_GROUP_DROP) lane.inLeadGroup = false;
    } else {
      if (gap <= LEAD_GROUP_JOIN) lane.inLeadGroup = true;
    }
  }
}

function fitLeadGroup(dt) {
  if (!raceLanes.length) return;
  updateLeadGroup();

  packBounds.makeEmpty();
  let leadCount = 0;
  for (const lane of raceLanes) {
    if (lane.inLeadGroup) {
      leadCount++;
      packPoint.copy(lane.holder.position);
      packPoint.y = (lane.record?.height ?? 0) * BODY_LENGTH_MM + 1.5;
      packBounds.expandByPoint(packPoint);
    }
  }
  if (!leadCount && raceLanes.length) {
    const fallback = leader ?? raceLanes[0];
    fallback.inLeadGroup = true;
    packPoint.copy(fallback.holder.position);
    packPoint.y = (fallback.record?.height ?? 0) * BODY_LENGTH_MM + 1.5;
    packBounds.expandByPoint(packPoint);
  }

  packBounds.getCenter(packCentre);
  packDirection.copy(camera.position).sub(controls.target);
  if (packDirection.lengthSq() < 0.0001) {
    const defaultDir = RACE_SHOTS[eventId]?.direction ?? [0.4, 0.42, 1];
    packDirection.set(...defaultDir);
  }
  packDirection.normalize();
  packRight.crossVectors(camera.up, packDirection).normalize();
  packUp.crossVectors(packDirection, packRight).normalize();

  const vertical = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.78;
  const horizontal = vertical * camera.aspect;
  let distance = 24;
  let maxSpread = 0;

  for (const lane of raceLanes) {
    if (!lane.inLeadGroup) continue;
    packOffset.copy(lane.holder.position).sub(packCentre);
    packOffset.y += (lane.record?.height ?? 0) * BODY_LENGTH_MM + 1.5;
    distance = Math.max(distance, packOffset.dot(packDirection) +
      Math.max((Math.abs(packOffset.dot(packRight)) + 3) / horizontal,
               (Math.abs(packOffset.dot(packUp)) + 4) / vertical));

    const dx = lane.holder.position.x - packCentre.x;
    const dz = lane.holder.position.z - packCentre.z;
    const r = Math.sqrt(dx * dx + dz * dz);
    if (r > maxSpread) maxSpread = r;
  }

  const goalDistance = Math.min(distance, MAX_PACK_DISTANCE);
  goalTarget.copy(packCentre);

  // Spotlight follows the lead group centre and widens when spread out
  goalRingTarget.set(packCentre.x, 0, packCentre.z);
  goalRingPos.set(packCentre.x + 0.6, 11, packCentre.z + 2.2);
  const neededAngle = Math.atan((maxSpread + 2.5) / 11);
  const goalRingAngle = Math.min(MAX_RING_ANGLE, Math.max(BASE_RING_ANGLE, neededAngle));
  goalFlyPoint.set(packCentre.x, 0.6, packCentre.z);

  // Smooth gliding with follow()'s existing damping
  const amount = snapCamera || reducedMotion || !playing ? 1 : Math.min(1, dt * 14);

  if (amount >= 1) {
    controls.target.copy(goalTarget);
    currentCameraDistance = goalDistance;
    camera.position.copy(packDirection).multiplyScalar(goalDistance).add(controls.target);
    ring.target.position.copy(goalRingTarget);
    ring.position.copy(goalRingPos);
    ring.angle = goalRingAngle;
    flyPoint.copy(goalFlyPoint);
  } else {
    controls.target.lerp(goalTarget, amount);
    currentCameraDistance += (goalDistance - currentCameraDistance) * amount;
    camera.position.copy(packDirection).multiplyScalar(currentCameraDistance).add(controls.target);
    ring.target.position.lerp(goalRingTarget, amount);
    ring.position.lerp(goalRingPos, amount);
    ring.angle += (goalRingAngle - ring.angle) * amount;
    flyPoint.lerp(goalFlyPoint, amount);
  }

  cameraX = controls.target.x;
  scene.fog.near = 140;
  scene.fog.far = 190;
}

// ---------- Playback ----------
function follow(dt) {
  if (eventId !== 'archery' && !photoShown) {
    fitLeadGroup(dt);
  } else {
    const goalX = focusX();
    const amount = snapCamera || reducedMotion || !playing ? 1 : Math.min(1, dt * 14);
    cameraX += (goalX - cameraX) * amount;
    const dx = cameraX - controls.target.x;
    controls.target.x += dx; camera.position.x += dx;
    scene.fog.near = 40; scene.fog.far = 95;
    const lit = cameraLane();
    ring.position.set(lit.position.x + 0.6, 11, lit.position.z + 2.2);
    ring.target.position.set(lit.position.x, 0, lit.position.z);
    ring.angle = BASE_RING_ANGLE;
    flyPoint.set(lit.position.x, 0.6, lit.position.z);
  }
  snapCamera = false;
}

// ---------- The eye tag (Archery, "Sees the wind") ----------
function eyeReport(wind) {
  const table = race.eye;
  const interp = w => {
    const xs = table.wind, ys = table.signal, x = Math.min(xs.at(-1), Math.max(xs[0], w));
    let i = 0; while (i < xs.length - 2 && xs[i + 1] < x) i++;
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]);
  };
  const centre = interp(0);
  return interp(eyeMode === 'eye_swapped' ? -wind : wind) - centre;  // > 0: T4/T5 report leftward drift
}
function updateEyeTag(record) {
  const show = eventId === 'archery' && selected === 'eye' && record && race.eye;
  $('eye-tag').hidden = true;  // the eye panel on the right now shows this
  $('eye-panel').hidden = !show;
  if (!show) return;
  const reported = eyeMode === 'eye_blind' ? 0 : eyeReport(record.wind);
  const half = (Math.max(...race.eye.signal) - Math.min(...race.eye.signal)) / 2;
  $('eye-needle').style.left = `${50 + 45 * Math.max(-1, Math.min(1, reported / half))}%`;
  $('eye-note').textContent = eyeMode === 'eye_blind' ? 'Blindfolded: the eye reports no wind'
    : eyeMode === 'eye_swapped' ? "FlyVis motion detectors, with left and right swapped"
    : "What FlyVis's motion detectors report, fed into the leg circuit";
  if (eyeMode === 'eye_blind') { $('eye-tag').textContent = 'Eye blindfolded: it reports no wind'; return; }
  const signal = reported;
  const side = Math.abs(signal) < 0.002 ? 'Eye: hardly any wind' : `Eye sees wind from the ${signal > 0 ? 'right' : 'left'}`;
  $('eye-tag').textContent = eyeMode === 'eye_swapped' ? `${side} (left and right swapped)` : side;
}

// ---------- The circuit panel: the 135 MANC neurons, by role ----------
const circuitCanvas = $('circuit'), circuitContext = circuitCanvas.getContext('2d');
const ROLE_ORDER = [['sensory', 'Sensory'], ['intrinsic', 'Interneurons'], ['motor_extensor', 'Extensor'], ['motor_flexor', 'Flexor']];
const circuitSize = { width: 0, height: 0 };
new ResizeObserver(([entry]) => {
  circuitSize.width = entry.contentRect.width; circuitSize.height = entry.contentRect.height;
  circuitDrawn = null;
}).observe(circuitCanvas);
let circuitDrawn = null, circuitColumns = null, circuitNodes = null;
function drawCircuit(record) {
  const nodes = payload?.circuit?.nodes; if (!nodes) return;
  const { width, height } = circuitSize;
  if (!width || !height) return;
  const drawing = record ?? 'none';
  if (circuitDrawn === drawing) return;  // nothing changed since the last frame
  circuitDrawn = drawing;
  const ratio = Math.min(devicePixelRatio, 2);
  if (circuitCanvas.width !== Math.round(width * ratio)) { circuitCanvas.width = Math.round(width * ratio); circuitCanvas.height = Math.round(height * ratio); }
  const g = circuitContext; g.setTransform(ratio, 0, 0, ratio, 0, 0); g.clearRect(0, 0, width, height);
  const activity = record?.neuron_activity;
  if (circuitNodes !== nodes) {
    circuitNodes = nodes;
    circuitColumns = ROLE_ORDER.map(([role]) => nodes.map((n, i) => (n.role === role ? i : -1)).filter(i => i >= 0));
  }
  const columns = circuitColumns;
  const labelHeight = 16, barHeight = 26, top = labelHeight, bottom = height - barHeight - 6;
  const columnWidth = width / columns.length;
  g.font = `600 ${Math.max(10, Math.round(width / 26))}px Archivo, system-ui, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'top';
  columns.forEach((ids, c) => {
    g.fillStyle = '#a6a89d'; g.fillText(ROLE_ORDER[c][1], columnWidth * (c + 0.5), 0);
    const per = Math.ceil(Math.sqrt(ids.length * (columnWidth - 8) / Math.max(1, bottom - top)));
    const cols = Math.max(1, per), rows = Math.ceil(ids.length / cols);
    const step = Math.min((columnWidth - 8) / cols, (bottom - top) / rows), radius = Math.max(1.5, step * 0.36);
    ids.forEach((id, k) => {
      const x = columnWidth * c + 4 + step * (k % cols + 0.5), y = top + step * (Math.floor(k / cols) + 0.5);
      const a = activity ? Math.max(0, Math.min(1, activity[id])) : 0;
      g.beginPath(); g.arc(x, y, radius, 0, Math.PI * 2);
      g.fillStyle = activity ? `rgba(230, 160, 60, ${0.12 + 0.88 * a})` : 'rgba(166, 168, 157, 0.18)';
      g.fill();
    });
  });
  if (!activity) return;
  const mean = ids => ids.reduce((sum, i) => sum + activity[i], 0) / ids.length;
  const ext = mean(columns[2]), flex = mean(columns[3]);
  [['Extensor', ext], ['Flexor', flex]].forEach(([label, value], k) => {
    const y = bottom + 6 + k * (barHeight / 2);
    const readout = `${label} ${value.toFixed(2)}`;
    const left = Math.max(g.measureText('Extensor -0.00').width + 8, 66);
    const half = Math.max(0, width - left - 4) / 2;
    g.fillStyle = '#2b2e29'; g.fillRect(left, y + 2, half * 2, barHeight / 2 - 4);
    g.fillStyle = value < 0 ? '#5b93d6' : '#f1efe6';
    g.fillRect(left + half, y + 2, half * value, barHeight / 2 - 4);
    g.fillStyle = '#a6a89d'; g.textAlign = 'left'; g.textBaseline = 'middle';
    g.fillText(readout, 0, y + barHeight / 4);
  });
}
function updateCircuitNote(record) {
  const isBrain = Boolean(record?.neuron_activity);
  const name = raceLanes[0]?.name ?? EVENTS[eventId].controllers.find(c => c.id === selected)?.name;
  $('circuit-note').textContent = isBrain
    ? '135 MANC neurons. Bars show signed activity (−1 to 1). Dots show positive activity.'
    : `${name ?? 'This controller'} is a hand-written baseline: no circuit involved`;
}

// ---------- Presenter mode: one key steps through the story ----------
const STORY = [
  { event: 'sprint', run: 'fly', rate: 4, caption: 'Real fly wiring, untrained: it barely gets going.' },
  { event: 'sprint', run: 'trained', rate: 4, caption: 'Same wiring, tuned inputs and outputs: now it runs.' },
  { event: 'hurdles', run: 'rule', rate: 2, caption: 'Hurdles: the jumps are part of the race physics.' },
  { event: 'swimming', run: 'rule', rate: 4, caption: 'Swimming is a game model: real fruit flies cannot swim.' },
  { event: 'archery', run: 'eye', eye: 'eye', rate: 1, caption: 'Archery: the fly senses the wind through a model of its real eyes (FlyVis).' },
  { event: 'archery', run: 'eye', eye: 'eye_blind', rate: 1, caption: 'Blindfolded: same fly, but its eye reports no wind.' },
  { event: 'archery', run: 'eye', eye: 'eye_swapped', rate: 1, caption: 'Left and right swapped: it aims the wrong way.' },
];
let presenting = false, step = 0;
function togglePresenter() {
  presenting = !presenting;
  $('presenter').hidden = !presenting;
  if (presenting) presenterStep(0);
}
function presenterStep(n) {
  step = Math.max(0, Math.min(STORY.length - 1, n));
  const scene = STORY[step];
  if (scene.eye) { eyeMode = scene.eye; for (const b of $('eye-modes').children) b.setAttribute('aria-checked', String(b.dataset.eye === eyeMode)); }
  if (eventId !== scene.event) selectEvent(scene.event);
  selectRun(scene.run);
  rate = scene.rate; $('fast').setAttribute('aria-pressed', String(rate === 4));
  setPlaying(true);
  $('presenter').textContent = `${step + 1} / ${STORY.length}   ${scene.caption}`;
}
$('presenter').onclick = () => presenterStep(step + 1);

// ---------- Archery opponent and target close-up ----------
let rivalId = 'fly';
function fillRivalOptions() {
  const choices = EVENTS.archery.controllers.filter(c => c.id !== selected && race?.runs?.[runId(c.id)]);
  if (!choices.some(c => c.id === rivalId)) rivalId = choices.find(c => c.id === 'fly')?.id ?? choices[0]?.id;
  $('rival').replaceChildren(...choices.map(c => new Option(c.name, c.id, false, c.id === rivalId)));
}
$('rival').onchange = event => {
  rivalId = event.target.value; lanes.rival.flights = []; lanes.rival.shown = 0;
  sheetStep = -1;
  syncTimeline();
  if (index >= replayEnd) setPlaying(false);
};
const targetCam = new THREE.PerspectiveCamera(22, 1, 0.1, 100);
function renderTargetCam() {
  const width = view.w, height = view.h;
  const size = Math.round(Math.min(width, height) * 0.28);
  // Inside the canvas's faded edges, bottom right.
  const x = Math.round(width * 0.9) - size, y = Math.round(height * 0.14);
  targetCam.position.set(TARGET_X - 7, TARGET_Y + 0.4, LANE_Z.main);
  targetCam.lookAt(TARGET_X, TARGET_Y, LANE_Z.main);
  // Name tapes above each target, so the room can tell the two lanes apart.
  const nameOf = id => EVENTS.archery.controllers.find(c => c.id === id)?.name ?? '';
  const eyeSuffix = selected === 'eye' && eyeMode !== 'eye' ? (eyeMode === 'eye_blind' ? ' (blindfolded)' : ' (eye swapped)') : '';
  for (const [lane, id, text] of [['main', 'lane-main', nameOf(selected) + eyeSuffix], ['rival', 'lane-rival', nameOf(rivalId)]]) {
    const p = new THREE.Vector3(TARGET_X, TARGET_Y + TARGET_RADIUS + 0.5, LANE_Z[lane]).project(camera);
    const tag = $(id);
    tag.hidden = false; tag.textContent = text;
    tag.style.left = `${(p.x + 1) / 2 * width}px`; tag.style.top = `${(1 - p.y) / 2 * height}px`;
  }
  renderer.setScissorTest(true);
  renderer.setViewport(x, y, size, size); renderer.setScissor(x, y, size, size);
  renderer.render(scene, targetCam);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, width, height);
  const label = $('target-cam');
  label.style.width = `${size}px`; label.style.left = `${x}px`; label.style.bottom = `${y + size + 4}px`;
}

// ---------- Race lanes and results ----------
const formatTime = time => time.toFixed(race.dt < 0.1 ? 2 : 1);
const finished = record => record.position >= race.distance;
const ordinal = place => {
  const teens = place % 100 >= 11 && place % 100 <= 13;
  return `${place}${teens ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' }[place % 10] ?? 'th')}`;
};

function makeSplash() {
  const count = 18;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  const spray = new THREE.Points(geometry, new THREE.PointsMaterial({
    color: 0xd6efff, size: 0.17, transparent: true, depthWrite: false,
  }));
  spray.frustumCulled = false;
  const ripple = new THREE.Mesh(new THREE.RingGeometry(0.88, 1, 32), new THREE.MeshBasicMaterial({
    color: 0xd6efff, transparent: true, depthWrite: false,
  }));
  ripple.rotation.x = -Math.PI / 2;
  ripple.position.y = 0.03;
  const group = new THREE.Group();
  group.add(ripple, spray);
  group.visible = false;
  scene.add(group);
  return { group, ripple, spray };
}

function paintSplash(lane, step) {
  const { group, ripple, spray } = lane.splash;
  const impact = lane.impacts.findLast(({ frame }) => frame <= step);
  const age = impact ? (step - impact.frame) * race.dt : Infinity;
  group.visible = age < 0.65;
  if (!group.visible) return;
  const u = age / 0.65;
  group.position.set(impact.position * BODY_LENGTH_MM, 0, lane.holder.position.z);
  ripple.scale.setScalar(0.35 + 1.05 * u);
  ripple.material.opacity = 0.85 * (1 - u);
  spray.visible = !reducedMotion;
  spray.material.opacity = 1 - u;
  const points = spray.geometry.attributes.position;
  for (let i = 0; i < points.count; i++) {
    const angle = i * 2.39996;
    const radius = (0.28 + (i % 5) * 0.21) * u;
    points.setXYZ(i, Math.cos(angle) * radius, 0.12 + (0.8 + (i % 4) * 0.3) * Math.sin(Math.PI * u), Math.sin(angle) * radius);
  }
  points.needsUpdate = true;
}

function buildRaceLanes() {
  raceLanes = []; leader = null; firstFinish = -1; sheetStep = -1;
  $('race-tags').replaceChildren();
  $('race-sheet').hidden = true;
  $('race-announcement').textContent = '';
  holder.position.set(0, 0, 0); holder.rotation.y = 0;
  for (const slot of reserveFlies) {
    slot.holder.visible = false;
    if (slot.splash) slot.splash.group.visible = false;
  }
  if (mainSlot.splash) mainSlot.splash.group.visible = false;
  if (eventId === 'archery') return;
  const lineup = controllers();
  const main = lineup.find(c => c.id === selected);
  const finishSteps = [];
  for (const [i, controller] of [main, ...lineup.filter(c => c !== main)].entries()) {
    const { id, name } = controller;
    const slot = i === 0 ? mainSlot : reserveFlies[i - 1];
    const offset = (i % 2 ? -1 : 1) * Math.ceil(i / 2);
    slot.holder.position.z = offset * LANE;
    slot.holder.visible = true;
    const tag = document.createElement('p');
    tag.className = `lane-tag${i ? ' lane-tag-rival' : ''}`;
    tag.textContent = name;
    tag.hidden = true;
    $('race-tags').append(tag);
    const run = race.runs[id];
    if (finished(run.at(-1))) finishSteps.push(run.length - 1);
    const impacts = [];
    if (eventId === 'steeplechase') {
      slot.splash ??= makeSplash();
      run.forEach((record, frame) => {
        if (record.splashes > (run[frame - 1]?.splashes ?? 0)) impacts.push({ frame, position: record.position });
      });
    }
    raceLanes.push({ ...slot, id, name, tag, run, impacts, lane: (trackLaneCount + 1) / 2 + offset, record: null, inLeadGroup: true });
  }
  if (eventId === 'sprint' && finishSteps.length) firstFinish = Math.min(...finishSteps);
}

const mainSlot = { holder, fly };

function markWatched(id, record, done) {
  const before = watched.get(key(id)) ?? { progress: 0, done: false };
  watched.set(key(id), { progress: Math.max(before.progress, progressOf(record)), done: before.done || done });
}

const lerp = (a, b, t) => a + (b - a) * t;

// Where a racer is drawn on the course: swimmers come back along the pool after a turn.
function courseX(record) {
  return eventId === 'swimming' && record.turns > 0 ? 2 * race.pool_length - record.position : record.position;
}

function placeRaceLanes(step, at = step) {
  const stride = style === 'human' ? (eventId === 'swimming' ? 1.6 : 1.25) : STRIDE;
  let time = 0;
  for (const lane of raceLanes) {
    const last = lane.run.length - 1;
    const record = lane.run[Math.min(step, last)];
    time = Math.max(time, record.simulation_time);
    if (lane.record !== record) {
      lane.record = record;
      if (step > 0) markWatched(lane.id, record, step >= last);
    }
    // Draw between this recorded step and the next one, so the flies glide instead of
    // hopping 10-20 times a second. Only the drawing is blended; the readouts, standings
    // and results use the recorded step itself.
    const next = lane.run[Math.min(step + 1, last)];
    const blend = step < last ? Math.min(1, Math.max(0, at - step)) : 0;
    if (lane.drawnAt !== at || lane.style !== style) {
      lane.drawnAt = at;
      lane.style = style;
      const shown = blend < 0.5 ? record : next;
      // At a pool wall, don't blend across the turn; switch direction there instead.
      const sameLeg = (record.turns ?? 0) === (next.turns ?? 0);
      const x = sameLeg ? lerp(courseX(record), courseX(next), blend) : courseX(shown);
      lane.holder.position.x = x * BODY_LENGTH_MM;
      lane.holder.rotation.y = courseX(shown) !== shown.position ? Math.PI : 0;
      lane.fly.setStyle(style);
      const effort = Math.min(1, lerp(record.velocity, next.velocity, blend) / EVENTS[eventId].topSpeed);
      lane.fly.pose(lerp(record.position, next.position, blend) / stride, effort, 0, {
        event: eventId, airborne: shown.airborne, lift: lerp(record.height ?? 0, next.height ?? 0, blend) * BODY_LENGTH_MM,
      });
    }
    if (eventId === 'steeplechase' && sheetStep !== step) paintSplash(lane, step);
  }
  // Keep the camera on the same lane while leaders are tied.
  if (!leader) leader = raceLanes[0];
  for (const lane of raceLanes) {
    if (lane.record.position > leader.record.position) leader = lane;
  }
  return time;
}

const tagPoint = new THREE.Vector3(), cameraDirection = new THREE.Vector3();
function renderRaceTags() {
  const width = view.w, height = view.h;
  const boxes = [];
  camera.getWorldDirection(cameraDirection);
  if (performance.now() - view.topEdgeAt > 1000) {  // the presenter caption can change height
    view.topEdge = Math.max(scaleBar.parentElement.offsetHeight, $('presenter').offsetHeight) + 5;
    view.topEdgeAt = performance.now();
  }
  const topEdge = view.topEdge;
  for (const lane of raceLanes) {
    const tag = lane.tag;
    tagPoint.copy(lane.holder.position);
    tagPoint.y = (lane.record.height ?? 0) * BODY_LENGTH_MM + (eventId === 'swimming' ? 0.4 : 3.5);
    if (eventId === 'swimming') tagPoint.z -= 0.55;
    const inFront = tagPoint.dot(cameraDirection) > camera.position.dot(cameraDirection);
    tagPoint.project(camera);
    const x = (tagPoint.x + 1) * width / 2, y = (1 - tagPoint.y) * height / 2;
    const outside = !inFront || x < 8 || x > width - 8 || y < topEdge || y > height - 8 || tagPoint.z > 1 || camera.position.distanceTo(lane.holder.position) > 185;
    tag.hidden = false;
    tag.classList.toggle('edge-marker', outside);
    const dx = (x - width / 2) * (inFront ? 1 : -1);
    const dy = (y - height / 2) * (inFront ? 1 : -1);
    if (outside) tag.style.setProperty('--pointer-angle', `${Math.atan2(dy, dx)}rad`);
    const sizes = lane.tagSizes ??= {};
    const look = outside ? 'edge' : 'plain';
    if (!sizes[look]?.[0]) sizes[look] = [tag.offsetWidth, tag.offsetHeight];
    const [w, h] = sizes[look];
    const edgeScale = Math.min(
      Math.max(1, width / 2 - w / 2 - 4) / Math.max(0.001, Math.abs(dx)),
      Math.max(1, height / 2 - topEdge - h - 4) / Math.max(0.001, Math.abs(dy)));
    const markerX = outside ? width / 2 + dx * edgeScale : x;
    const markerY = outside ? height / 2 + dy * edgeScale : y;
    const left = Math.max(w / 2 + 2, Math.min(width - w / 2 - 2, markerX));
    const base = Math.max(topEdge + h, Math.min(height - 4, markerY));
    let placed = false;
    for (const dy of [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6]) {
      const bottom = base + dy * (h + 3);
      const box = { l: left - w / 2, r: left + w / 2, t: bottom - h, b: bottom };
      if (box.t < topEdge || box.b > height - 2 ||
          boxes.some(b => box.l < b.r + 3 && box.r > b.l - 3 && box.t < b.b + 2 && box.b > b.t - 2)) continue;
      const placement = `${left.toFixed(1)},${bottom.toFixed(1)}`;
      if (tag.dataset.at !== placement) {
        tag.dataset.at = placement;
        tag.style.left = `${left}px`; tag.style.top = `${bottom}px`;
      }
      boxes.push(box);
      placed = true;
      break;
    }
    tag.hidden = !placed;
  }
}

function sheetRow(entry, place, value) {
  const row = document.createElement('tr');
  row.dataset.controller = entry.id;
  row.dataset.main = String(entry.id === selected);
  const rank = row.insertCell();
  rank.textContent = place;
  const name = document.createElement('th');
  name.scope = 'row'; name.textContent = entry.name;
  row.append(name);
  if (entry.lane) {
    const lane = document.createElement('small');
    lane.className = 'lane-number';
    lane.textContent = `Lane ${entry.lane}${entry.id === selected ? ' · Main' : ''}`;
    name.append(lane);
  }
  if (eventId === 'pacing' && Number.isFinite(entry.record.energy)) {
    const energy = document.createElement('span');
    energy.className = 'energy';
    const meter = document.createElement('meter');
    meter.min = 0; meter.max = 100; meter.value = entry.record.energy;
    meter.setAttribute('aria-label', `${entry.name} energy`);
    energy.append(meter, `${entry.record.energy.toFixed(1)} energy`);
    name.append(energy);
  }
  row.insertCell().textContent = value;
  return row;
}

function renderStandings(step) {
  if (sheetStep === step) return;
  sheetStep = step;
  const archery = eventId === 'archery';
  const final = step >= replayEnd;
  $('race-sheet').hidden = archery && !final;
  if (archery && !final) {
    $('race-announcement').textContent = '';
    return;
  }
  const entries = archery ? [selected, rivalId].map(id => ({
    id,
    name: controllers().find(c => c.id === id)?.name ?? id,
    record: race.runs[runId(id)].at(-1),
  })) : [...raceLanes];
  const metric = entry => archery ? -entry.record.total_points
    : final && finished(entry.record) ? entry.record.simulation_time : -entry.record.position;
  const dnf = entry => final && !archery && !finished(entry.record);
  entries.sort((a, b) => Number(dnf(a)) - Number(dnf(b)) || metric(a) - metric(b));
  let place = 0;
  const rows = entries.map((entry, i) => {
    if (!i || metric(entry) !== metric(entries[i - 1])) place = i + 1;
    const value = archery ? `${entry.record.total_points} / ${race.max_points}`
      : dnf(entry) ? 'DNF'
      : final ? `${formatTime(entry.record.simulation_time)} s`
      : `${entry.record.position.toFixed(2)} BL`;
    return sheetRow(entry, dnf(entry) ? '—' : ordinal(place), value);
  });
  $('race-sheet').dataset.results = String(final);
  $('race-sheet-title').textContent = final ? 'Results' : 'Live standings';
  $('race-measure').textContent = archery ? 'Score' : final ? 'Time' : 'Distance';
  $('race-rows').replaceChildren(...rows);
  $('race-announcement').textContent = final ? 'Results are ready.' : '';
}

let last = performance.now(), wallTime = 0, lastRackUpdate = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now; wallTime += dt;
  if (playing && now >= photoUntil) {
    const next = Math.min(replayEnd, index + rate * dt / race.dt);
    if (firstFinish >= 0 && index < firstFinish && next >= firstFinish) {
      index = firstFinish;
      photoUntil = now + 900;
    } else {
      index = next;
    }
    if (index >= replayEnd) setPlaying(false);
    $('timeline').value = String(Math.floor(index));
  }
  if (!records.length) {
    controls.update(); renderer.render(scene, camera);
    requestAnimationFrame(frame);
    return;
  }

  const step = Math.floor(index);
  const archery = eventId === 'archery';
  let record, time, position, velocity;
  if (archery) {
    const mainStep = Math.min(records.length - 1, step);
    record = records[mainStep];
    const bowAt = lane => bows[lane].group.position.clone().add(new THREE.Vector3(0.5, 0, 0));
    placeArrows('main', records.slice(0, mainStep + 1).filter(x => x.shot).map(x => x.landing), record.aim_offset, bowAt('main'));
    const rivalRun = race.runs[runId(rivalId)] ?? [];
    const k = Math.min(rivalRun.length - 1, step);
    const rivalRecord = rivalRun[k];
    time = Math.max(record.simulation_time, rivalRecord?.simulation_time ?? 0);
    placeArrows('rival', rivalRun.slice(0, k + 1).filter(x => x.shot).map(x => x.landing), rivalRecord?.aim_offset, bowAt('rival'));
    const human = style === 'human';
    placeBow(bows.main, holder, LANE_Z.main, record.draw, human && !flying);
    placeBow(bows.rival, rivalHolder, LANE_Z.rival, rivalRecord?.draw ?? 0, human);
    fly.pose(0, 0, reducedMotion ? 0 : wallTime, { event: 'archery', draw: record.draw, lift: flying ? 1.6 + (reducedMotion ? 0 : 0.05 * Math.sin(wallTime * 3)) : 0 });
    rival.setStyle(style);
    rival.pose(0, 0, reducedMotion ? 0 : wallTime, { event: 'archery', draw: rivalRecord?.draw ?? 0 });
    $('versus').textContent = String(rivalRecord?.total_points ?? 0);
    $('versus-label').textContent = EVENTS.archery.controllers.find(c => c.id === rivalId)?.name ?? 'Opponent';
    if (step > 0) markWatched(selected, record, step >= records.length - 1);
  } else {
    time = placeRaceLanes(step, index);
    record = raceLanes[0].record; position = record.position; velocity = record.velocity;
    const showPhoto = firstFinish >= 0 && step === firstFinish;
    if (photoShown !== showPhoto) {
      photoShown = showPhoto;
      $('photo-tape').hidden = !showPhoto;
      frameThePad();
    }
  }

  if (now - lastRackUpdate > 100) { updateRack(); lastRackUpdate = now; }
  const seconds = Number($('ticks').dataset.seconds) || 1;
  const progress = `${Math.min(100, (time / seconds) * 100).toFixed(2)}%`;
  if ($('ruler').style.getPropertyValue('--progress') !== progress) $('ruler').style.setProperty('--progress', progress);
  const stopped = !archery && step >= records.length - 1;
  if (stopped) setAction(finished(record) ? 'Finished' : 'DNF', '');
  else if (!step) setAction('Ready', '');
  else setAction(record.exhausted && record.action === 'ACCELERATE' ? 'ACC (exhausted)' : record.action, record.action);
  setText('time', formatTime(time));
  setText('distance', archery ? String(record.total_points) : position.toFixed(1));
  setText('speed', archery ? record.wind.toFixed(1).replace(/^-0\.0$/, '0.0') : velocity.toFixed(1));
  updateEyeTag(record);
  drawCircuit(record);
  updateCircuitNote(record);
  renderStandings(step);
  follow(dt);
  controls.update();
  renderer.render(scene, camera);
  const barWidth = `${bodyLengthOnScreen().toFixed(1)}px`;
  if (scaleBar.style.width !== barWidth) scaleBar.style.width = barWidth;
  if (archery) renderTargetCam();
  else renderRaceTags();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

try {
  const response = await fetch('/runs.json');
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  payload = await response.json();
  const runGroups = [payload.runs, ...Object.entries(payload.events ?? {})
    .filter(([id]) => id !== 'archery' && EVENTS[id]).map(([, event]) => event.runs)];
  const maxRacers = Math.max(...runGroups.map(runs => Object.values(runs).filter(run => run.length).length));
  trackLaneCount = Math.max(5, 2 * Math.floor(maxRacers / 2) + 1);
  halfTrackWidth = LANE * trackLaneCount / 2;
  for (let i = 1; i < maxRacers; i++) {
    const extra = await loadFly();
    const slot = { holder: new THREE.Group(), fly: extra };
    extra.setFlying(false);
    slot.holder.add(extra.object);
    scene.add(slot.holder);
    reserveFlies.push(slot);
  }
  applyQuality();
  buildEventTabs();
  selectEvent('sprint');
} catch (error) {
  setAction('No runs. Start python -m experiments.serve_view', 'error');
  console.error(error);
}
