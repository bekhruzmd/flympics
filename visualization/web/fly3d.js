// 3D fruit fly built from the NeuroMechFly body meshes (FlyGym, Apache-2.0).
// The body parts and their rest positions come from micro-CT data; the walking gait and
// wingbeat below are a hand-made animation, not a simulation of fly muscles or joints.
import * as THREE from 'three';
import { STLLoader } from './vendor/three/STLLoader.js';
import { mergeVertices } from './vendor/three/BufferGeometryUtils.js';

const ASSETS = 'assets/fly/';
const DEG = Math.PI / 180;

// Parent of every body part (FlyGym's segment tree). Right-side parts reuse the left mesh, mirrored.
const LEGS = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh'];
const LEG_LINKS = ['coxa', 'trochanterfemur', 'tibia', 'tarsus1', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5'];
const PARENT = {
  c_head: 'c_thorax', c_rostrum: 'c_head', c_haustellum: 'c_rostrum',
  c_abdomen12: 'c_thorax', c_abdomen3: 'c_abdomen12', c_abdomen4: 'c_abdomen3',
  c_abdomen5: 'c_abdomen4', c_abdomen6: 'c_abdomen5',
};
for (const s of ['l', 'r']) {
  Object.assign(PARENT, {
    [`${s}_eye`]: 'c_head', [`${s}_pedicel`]: 'c_head', [`${s}_funiculus`]: `${s}_pedicel`,
    [`${s}_arista`]: `${s}_funiculus`, [`${s}_wing`]: 'c_thorax', [`${s}_haltere`]: 'c_thorax',
  });
}
for (const leg of LEGS) LEG_LINKS.forEach((link, i) => { PARENT[`${leg}_${link}`] = i ? `${leg}_${LEG_LINKS[i - 1]}` : 'c_thorax'; });

// Colours from FlyGym's visuals.yaml.
function colourOf(name) {
  if (name.endsWith('_eye')) return { color: 0xab3520, roughness: 0.35 };
  if (name.endsWith('_wing')) return { color: 0xdde3f0, roughness: 0.15, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false };
  if (name.endsWith('_arista')) return { color: 0x423329 };
  if (name.endsWith('_haltere')) return { color: 0x966e3d };
  if (name === 'c_abdomen6') return { color: 0x4a2a08 };
  if (name.startsWith('c_abdomen')) return { color: 0xb08850 };
  if (name.endsWith('_tibia')) return { color: 0xab7833 };
  if (name.includes('tarsus')) return { color: 0xb5823d };
  if (name.endsWith('_trochanterfemur')) return { color: 0xa06e29 };
  return { color: 0x96631f };
}

function parseYaml(text) {
  // Just enough YAML for FlyGym's rigging and pose files: nested maps, numbers and flow lists.
  const root = {}; const stack = [{ indent: -1, value: root }];
  for (const raw of text.split('\n')) {
    if (!raw.trim() || raw.trim().startsWith('#') || raw.trim().startsWith('-')) continue;
    const indent = raw.length - raw.trimStart().length;
    const [key, ...rest] = raw.trim().split(':');
    const valueText = rest.join(':').trim();
    while (stack.at(-1).indent >= indent) stack.pop();
    const parent = stack.at(-1).value;
    if (!valueText) { parent[key] = {}; stack.push({ indent, value: parent[key] }); }
    else if (valueText.startsWith('[')) parent[key] = JSON.parse(valueText);
    else parent[key] = Number.isNaN(Number(valueText)) ? valueText : Number(valueText);
  }
  return root;
}

// FlyGym's axis names: yaw turns about x, pitch about y, roll about z; joints chain yaw, pitch, roll.
// Right-side yaw and roll axes are flipped so the same angle means the same movement on both sides.
function jointQuaternion(name, yaw = 0, pitch = 0, roll = 0) {
  const flip = name[0] === 'r' ? -1 : 1;
  const q = new THREE.Quaternion();
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), flip * yaw * DEG));
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), pitch * DEG));
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), flip * roll * DEG));
  return q;
}

export async function loadFly() {
  const [rigging, pose] = await Promise.all([
    fetch(`${ASSETS}rigging.yaml`).then(r => r.text()).then(parseYaml),
    fetch(`${ASSETS}neutral_pose.yaml`).then(r => r.text()).then(parseYaml),
  ]);
  // Neutral angles keyed by child part, mirrored from left to right like FlyGym does.
  const neutral = {};
  for (const [dof, angle] of Object.entries(pose.joint_angles)) {
    const [, child, axis] = dof.split('-');
    for (const name of child[0] === 'l' ? [child, `r${child.slice(1)}`] : [child]) {
      (neutral[name] ??= { yaw: 0, pitch: 0, roll: 0 })[axis] = angle;
    }
  }

  const loader = new STLLoader();
  const meshFiles = new Map();
  const names = ['c_thorax', ...Object.keys(PARENT)];
  await Promise.all(names.map(async name => {
    const file = name[0] === 'r' ? `l${name.slice(1)}` : name;
    if (!meshFiles.has(file)) meshFiles.set(file, loader.loadAsync(`${ASSETS}meshes/${file}.stl`));
  }));

  const parts = {};
  for (const name of names) {
    const raw = (await meshFiles.get(name[0] === 'r' ? `l${name.slice(1)}` : name)).clone();
    raw.scale(1000, name[0] === 'r' ? -1000 : 1000, 1000);  // metres -> millimetres, mirror right side
    if (name[0] === 'r') flipWinding(raw);
    raw.deleteAttribute('normal');
    const geometry = mergeVertices(raw, 1e-5);  // shared vertices give smooth shading
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05, ...colourOf(name) }));
    mesh.castShadow = !name.endsWith('_wing');
    const body = new THREE.Group();   // placed by rigging pos/quat (MuJoCo quats are w, x, y, z)
    const joint = new THREE.Group();  // rotated by joint angles
    const { pos, quat } = rigging[name];
    body.position.set(...pos);
    body.quaternion.set(quat[1], quat[2], quat[3], quat[0]).normalize();
    body.add(joint); joint.add(mesh);
    parts[name] = { body, joint, mesh, neutral: neutral[name] ?? { yaw: 0, pitch: 0, roll: 0 } };
  }
  for (const [name, part] of Object.entries(parts)) {
    (name === 'c_thorax' ? null : parts[PARENT[name]].joint)?.add(part.body);
  }
  return new Fly(parts);
}

function flipWinding(geometry) {
  // Mirroring turns triangles inside out; swap two vertices of each to fix the faces.
  const p = geometry.attributes.position;
  for (let i = 0; i < p.count; i += 3) {
    const x = p.getX(i + 1), y = p.getY(i + 1), z = p.getZ(i + 1);
    p.setXYZ(i + 1, p.getX(i + 2), p.getY(i + 2), p.getZ(i + 2));
    p.setXYZ(i + 2, x, y, z);
  }
}

// Tripod gait: front and hind legs on one side step together with the opposite middle leg.
const TRIPOD = { lf: 0, rm: 0, lh: 0, rf: 0.5, lm: 0.5, rh: 0.5 };
const Y_AXIS = new THREE.Vector3(0, 1, 0), X_AXIS = new THREE.Vector3(1, 0, 0);
const tmpQ = new THREE.Quaternion(), tmpV = new THREE.Vector3();

// A direction in the fly's own frame (x forward, y left, z up), `deg` degrees from straight
// down toward forward (negative = backward), with an optional sideways component.
function sag(deg, side = 0) {
  const a = deg * DEG;
  return new THREE.Vector3(Math.sin(a), side, -Math.cos(a)).normalize();
}
const ease = t => (1 - Math.cos(Math.PI * t)) / 2;

// Human-style key poses, from typical human joint-angle ranges (hip about -20..70 deg in a
// sprint stride, knee up to about 115 deg in swing, elbows near 90 deg). Cartoon only.
function strideAngles(p, effort) {
  let thigh, knee;
  if (p < 0.5) { const s = p / 0.5; thigh = -20 + 90 * ease(s); knee = 115 * Math.sin(Math.PI * s); }
  else { const s = (p - 0.5) / 0.5; thigh = 70 - 90 * s; knee = 22 * Math.sin(Math.PI * s); }
  thigh *= effort; knee = 4 + knee * effort;
  return { thigh, shin: thigh - knee };
}

class Fly {
  constructor(parts) {
    this.parts = parts;
    this.root = new THREE.Group();
    this.root.add(parts.c_thorax.body);
    this.root.rotation.x = -Math.PI / 2;  // FlyGym is z-up; three.js is y-up
    this.flying = false;
    this.style = 'fly';
    // Each leg segment's rest direction: from its joint toward the next segment's joint.
    for (const leg of LEGS) LEG_LINKS.forEach((link, i) => {
      const part = parts[`${leg}_${link}`], next = parts[`${leg}_${LEG_LINKS[i + 1]}`];
      part.restDir = next ? next.body.position.clone().normalize() : new THREE.Vector3(0, 0, -1);
    });
    this.pose(0, 0, 0);
  }

  get object() { return this.root; }

  setFlying(flying) {
    this.flying = flying;
    for (const side of ['l', 'r']) this.parts[`${side}_wing`].mesh.visible = flying;
  }

  setStyle(style) { this.style = style; }

  // phase: stride cycles so far; effort: 0 resting .. 1 full stride; time: seconds.
  // opts: { lift, event, airborne }. lift is the recorded height above the ground, in mm.
  pose(phase, effort, time, opts = {}) {
    if (this.style === 'human') return this.poseHuman(phase, effort, time, opts);
    this.parts.c_thorax.joint.quaternion.identity();
    for (const [name, part] of Object.entries(this.parts)) {
      if (name === 'c_thorax') continue;
      const n = part.neutral; let { yaw, pitch, roll } = n;
      const leg = name.slice(0, 2);
      if (leg in TRIPOD) {
        const link = name.slice(3);
        const p = ((phase + TRIPOD[leg]) % 1 + 1) % 1;
        const stance = p < 0.6;                       // 60% of the cycle on the ground
        const u = stance ? p / 0.6 : (p - 0.6) / 0.4;
        const swing = stance ? 1 - 2 * u : -1 + 2 * (0.5 - 0.5 * Math.cos(Math.PI * u));
        const lift = stance ? 0 : Math.sin(Math.PI * u);
        const reach = this.flying ? 0 : effort;
        if (link === 'coxa') pitch += 14 * reach * swing;
        if (link === 'trochanterfemur') pitch += 18 * reach * lift;
        if (link === 'tibia') pitch -= 14 * reach * lift;
        if (this.flying && link === 'trochanterfemur') pitch -= 20;  // legs tucked in flight
        if (this.flying && link === 'tibia') pitch += 40;
      }
      if (this.flying && name.endsWith('_wing')) {
        // Spread the folded wings out to the sides, then beat (illustrative, slowed down).
        const beat = Math.sin(time * 2 * Math.PI * 12);
        pitch += 15 * beat; roll -= 75 + 40 * beat;
      }
      part.joint.quaternion.copy(jointQuaternion(name, yaw, pitch, roll));
    }
    this.parts.c_thorax.body.position.z = 1.3 + (this.flying ? 0 : 0.02 * effort * Math.abs(Math.sin(2 * Math.PI * phase)));
    this.root.position.y = this.flying ? 0.7 + 0.05 * Math.sin(time * 3) : (opts.lift ?? 0);
  }

  // Point one segment along a direction given in the fly's own frame.
  aim(name, dir) {
    const part = this.parts[name];
    part.body.updateWorldMatrix(true, false);
    part.body.getWorldQuaternion(tmpQ);
    const world = tmpV.copy(dir).applyQuaternion(this.root.getWorldQuaternion(new THREE.Quaternion()));
    part.joint.quaternion.setFromUnitVectors(part.restDir, world.applyQuaternion(tmpQ.invert()).normalize());
  }

  limb(leg, upper, lower, end, tip = end) {
    this.aim(`${leg}_coxa`, upper);
    this.aim(`${leg}_trochanterfemur`, upper);
    this.aim(`${leg}_tibia`, lower);
    this.aim(`${leg}_tarsus1`, end);
    for (const t of ['tarsus2', 'tarsus3', 'tarsus4', 'tarsus5']) this.aim(`${leg}_${t}`, tip);
  }

  poseHuman(phase, effort, time, { lift = 0, event = 'sprint', airborne = false, draw = 0 } = {}) {
    const jumpEvent = event === 'hurdles' || event === 'steeplechase';
    const P = this.parts;
    for (const [name, part] of Object.entries(P)) {
      if (!name.match(/^[lr][fmh]_/) && name !== 'c_thorax') {
        const n = part.neutral; part.joint.quaternion.copy(jointQuaternion(name, n.yaw, n.pitch, n.roll));
      }
    }
    for (const side of ['l', 'r']) P[`${side}_wing`].mesh.visible = this.flying;
    P.c_thorax.body.position.z = 1.3;
    const prone = event === 'swimming' || this.flying;
    const p = ((phase % 1) + 1) % 1;

    if (prone) {
      // Front crawl (or a "superhero" glide in Fly mode): body flat, rolling with the stroke.
      const roll = this.flying ? 0 : 14 * Math.sin(2 * Math.PI * p);
      P.c_thorax.joint.quaternion.setFromAxisAngle(X_AXIS, roll * DEG);
      P.c_head.joint.quaternion.setFromAxisAngle(Y_AXIS, -10 * DEG);
      this.root.updateMatrixWorld(true);
      for (const [side, offset, sign] of [['l', 0, 1], ['r', 0.5, -1]]) {
        const phi = this.flying ? 0 : 2 * Math.PI * ((p + offset) % 1);
        // Arm circle: reach forward, pull down and back under the body, recover over the top.
        const upper = new THREE.Vector3(Math.cos(phi), sign * 0.35, -Math.sin(phi)).normalize();
        const recovering = Math.sin(phi) < 0;
        const lower = upper.clone().add(new THREE.Vector3(0, 0, recovering ? -0.5 : -0.15)).normalize();
        this.limb(`${side}f`, upper, lower, lower);
        const kick = this.flying ? 0 : 12 * Math.sin(2 * Math.PI * (3 * p + offset));
        const thigh = sag(-92 + kick, sign * 0.12), shin = sag(-96 - kick * 0.6, sign * 0.1);
        this.limb(`${side}h`, thigh, shin, sag(-120, sign * 0.1));
        const tuck = new THREE.Vector3(-0.85, sign * 0.45, 0.1).normalize();
        this.limb(`${side}m`, tuck, tuck, tuck);
      }
      this.root.position.y = this.flying ? lift : -0.75;  // swimming: body at the waterline
      return;
    }

    // Upright: torso nearly vertical, leaning into the run; head turned to face forward.
    const lean = airborne && jumpEvent ? 40 : 8 + 12 * effort;
    P.c_thorax.joint.quaternion.setFromAxisAngle(Y_AXIS, -(90 - lean) * DEG);
    P.c_head.joint.quaternion.setFromAxisAngle(Y_AXIS, (90 - lean - 10) * DEG);
    this.root.updateMatrixWorld(true);

    if (event === 'archery') {
      // Archer's stance: feet apart, bow arm straight out at the target, draw arm pulled back.
      this.limb('lh', sag(-12, 0.25), sag(-12, 0.22), sag(70, 0.2));
      this.limb('rh', sag(12, -0.25), sag(12, -0.22), sag(90, -0.2));
      this.limb('lf', new THREE.Vector3(1, 0.05, 0.08).normalize(), new THREE.Vector3(1, 0.02, 0.1).normalize(), new THREE.Vector3(1, 0, 0.1).normalize());
      const pull = Math.min(1, draw);
      this.limb('rf', new THREE.Vector3(0.9 - 0.9 * pull, -0.35, 0.15).normalize(), new THREE.Vector3(1 - 1.9 * pull, 0.15, 0.05).normalize(), new THREE.Vector3(1, 0, 0));
    } else if (airborne && jumpEvent) {
      // Hurdle clearance: lead leg straight over the bar, trail leg folded out to the side,
      // opposite arm reaching for the lead foot.
      this.limb('rh', sag(88, -0.05), sag(92, -0.05), sag(150, -0.05));
      this.limb('lh', new THREE.Vector3(-0.15, 0.95, -0.25), new THREE.Vector3(0.65, 0.7, -0.1), new THREE.Vector3(0.3, 0.9, 0.1));
      this.limb('lf', sag(80, 0.15), sag(88, 0.1), sag(92, 0.1));
      this.limb('rf', sag(-40, -0.2), sag(20, -0.2), sag(30, -0.2));
    } else {
      for (const [side, offset, sign] of [['l', 0, 1], ['r', 0.5, -1]]) {
        const { thigh, shin } = strideAngles((p + offset) % 1, effort);
        this.limb(`${side}h`, sag(thigh, sign * 0.08), sag(shin, sign * 0.06), sag(shin + 82, sign * 0.05));
        // Arms swing against the same-side leg, elbows bent near 90 degrees.
        const arm = -0.85 * (thigh - 25 * effort) + 6;
        const bend = 25 + 65 * Math.min(1, effort * 1.5);
        this.limb(`${side}f`, sag(arm, sign * 0.22), sag(arm + bend, sign * 0.12), sag(arm + bend + 10, sign * 0.1));
      }
    }
    for (const [side, sign] of [['l', 1], ['r', -1]]) {
      // Middle legs folded flat against the body, out of the way.
      this.limb(`${side}m`, new THREE.Vector3(-0.35, sign * 0.5, -0.6).normalize(), new THREE.Vector3(0.1, sign * 0.4, -0.9).normalize(), sag(10, sign * 0.3));
    }
    // Plant the lower hind foot on the ground; in a hurdle jump, carry it at the recorded height.
    this.root.position.y = 0;
    this.root.updateMatrixWorld(true);
    let lowest = Infinity;
    for (const leg of ['lh', 'rh']) for (const t of ['tarsus1', 'tarsus3', 'tarsus5']) {
      lowest = Math.min(lowest, P[`${leg}_${t}`].body.getWorldPosition(tmpV).y);
    }
    this.root.position.y = (airborne ? lift : 0) + 0.04 - lowest;
  }
}
