const $ = id => document.getElementById(id);
// Display metadata only; every number on the page comes from /runs.json.
const CONTROLLERS = [
  {key: 'trained', name: 'Trained fly', short: 'TRAINED', kind: 'Connectome', color: '#2563eb', about: 'MANC circuit · motor threshold fitted by explicit training'},
  {key: 'fly', name: 'Untrained fly', short: 'UNTRAINED', kind: 'Connectome', color: '#9b4dca', about: 'Same MANC circuit · original motor threshold'},
  {key: 'rule', name: 'Rule baseline', short: 'RULE', kind: 'Baseline', color: '#d9622b', about: 'Hand-written speed rule · no neural model'},
  {key: 'random', name: 'Random baseline', short: 'RANDOM', kind: 'Baseline', color: '#6b7a73', about: 'Seeded random actions · no neural model'}
];
const POPULATIONS = [
  ['sensory_mean', 'Sensory', 'sensory'],
  ['intrinsic_mean', 'Interneurons', 'intrinsic'],
  ['motor_extensor', 'Extensor motor', 'motor_extensor'],
  ['motor_flexor', 'Flexor motor', 'motor_flexor']
];
const ROLE_LABELS = {sensory: 'Sensory', intrinsic: 'Interneurons', motor_extensor: 'Extensor', motor_flexor: 'Flexor'};
const FLY_SVG = '<svg viewBox="0 0 100 80" aria-hidden="true"><g stroke="#2d342c" stroke-width="3" fill="none"><path d="M44 40 24 17M46 40 18 39M44 40 25 65M61 40 77 15M61 40 84 38M60 40 77 64"/></g><g class="fly-wings" fill="#eef4f1" stroke="#8aa397" stroke-width="1.5" opacity=".92"><ellipse cx="37" cy="26" rx="24" ry="12" transform="rotate(25 37 26)"/><ellipse cx="37" cy="54" rx="24" ry="12" transform="rotate(-25 37 54)"/></g><ellipse cx="44" cy="40" rx="23" ry="13" fill="#a57b4b"/><path d="M33 30v20M42 28v24" stroke="#76522f" stroke-width="3"/><ellipse cx="64" cy="40" rx="14" ry="14" fill="#303b2d"/><circle cx="70" cy="34" r="6" fill="#df704d"/></svg>';

let dataset, endTime = 0, clock = 0, playing = false, previous = null, focus = 'trained', downloadUrl;
let heat = null, heatScale = 'absolute', activeTab = 'heatmap';
const summaries = {}, lanes = {};

function indexAt(rows, time) {
  let lo = 0, hi = rows.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].simulation_time <= time + 1e-9) { found = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return found;
}
const RESET_ROW = {position: 0, velocity: 0, simulation_time: 0, action: 'READY', distance_to_finish: 100};
const rowAt = (key, time) => dataset.runs[key][indexAt(dataset.runs[key], time)] || RESET_ROW;
const isNeural = key => Boolean(dataset.runs[key][0].neuron_activity);
const fmt = (v, digits = 2) => v.toFixed(digits);

function summarize(key) {
  const rows = dataset.runs[key], last = rows.at(-1);
  const finished = last.distance_to_finish <= 0;
  return {
    finished, time: last.simulation_time, distance: last.position,
    topSpeed: Math.max(...rows.map(r => r.velocity)),
    accelShare: rows.filter(r => r.action === 'ACCELERATE').length / rows.length
  };
}

function buildResults() {
  const ranked = [...CONTROLLERS].sort((a, b) => {
    const A = summaries[a.key], B = summaries[b.key];
    if (A.finished !== B.finished) return A.finished ? -1 : 1;
    return A.finished ? A.time - B.time : B.distance - A.distance;
  });
  const container = $('result-cards'); container.replaceChildren();
  CONTROLLERS.forEach((c, n) => {
    const s = summaries[c.key], rank = ranked.indexOf(c) + 1;
    const card = document.createElement('button');
    card.type = 'button'; card.className = 'result-card'; card.dataset.key = c.key;
    card.style.setProperty('--c', c.color);
    card.innerHTML = `
      <div class="rc-top"><span class="rc-name"><i></i>${c.name}</span><span class="rc-kind">${c.kind}</span></div>
      <div class="rc-result"><strong>${s.finished ? `${fmt(s.time, 1)}<small> s</small>` : 'DNF'}</strong><span class="rc-rank">#${rank}</span></div>
      <dl>
        <div><dt>Distance</dt><dd>${fmt(s.distance, 1)} m</dd></div>
        <div><dt>Top speed</dt><dd>${fmt(s.topSpeed)} m/s</dd></div>
        <div><dt>Accelerating</dt><dd>${Math.round(s.accelShare * 100)}% of steps</dd></div>
      </dl>
      <p>${c.about}${s.finished ? '' : ` · stopped at the ${fmt(s.time, 0)} s time limit`}</p>
      <kbd>${n + 1}</kbd>`;
    card.onclick = () => setFocus(c.key);
    container.append(card);
  });
}

function buildLanes() {
  const container = $('lanes'); container.replaceChildren();
  for (const c of CONTROLLERS) {
    const lane = document.createElement('div');
    lane.className = 'lane'; lane.style.setProperty('--c', c.color);
    lane.innerHTML = `
      <button type="button" class="lane-label"><i></i>${c.name}</button>
      <div class="lane-track"><div class="lane-progress"></div><div class="runner">${FLY_SVG}</div><div class="finish-line"></div></div>
      <div class="lane-readout"><span class="lane-dist"></span><span class="lane-speed"></span><span class="lane-action"></span></div>`;
    lane.querySelector('.lane-label').onclick = () => setFocus(c.key);
    lane.querySelector('.lane-track').addEventListener('click', e => {
      // Seek to when this controller is closest to the clicked distance.
      const rect = e.currentTarget.getBoundingClientRect(), target = (e.clientX - rect.left) / rect.width * 100;
      const rows = dataset.runs[c.key];
      let best = 0; rows.forEach((r, i) => { if (Math.abs(r.position - target) < Math.abs(rows[best].position - target)) best = i; });
      setFocus(c.key); seek(rows[best].simulation_time);
    });
    container.append(lane);
    lanes[c.key] = {
      root: lane, runner: lane.querySelector('.runner'), progress: lane.querySelector('.lane-progress'),
      dist: lane.querySelector('.lane-dist'), speed: lane.querySelector('.lane-speed'), action: lane.querySelector('.lane-action')
    };
  }
}

function buildBars() {
  const bars = $('bars'); bars.replaceChildren();
  for (const [field, label, role] of POPULATIONS) {
    const row = document.createElement('div');
    row.className = `bar-row role-${role}`;
    row.innerHTML = `<div class="bar-label"><span>${label}</span><output id="${field}-value">—</output></div><div class="bar-track"><div class="bar-fill" id="${field}"></div></div>`;
    bars.append(row);
  }
}

function setFocus(key) {
  focus = key;
  document.querySelectorAll('.result-card').forEach(card => card.setAttribute('aria-pressed', String(card.dataset.key === key)));
  for (const c of CONTROLLERS) lanes[c.key].root.classList.toggle('is-focus', c.key === key);
  const c = CONTROLLERS.find(x => x.key === key);
  $('circuit-controller').textContent = c.name;
  document.querySelector('.circuit').style.setProperty('--c', c.color);
  const neural = isNeural(key);
  $('circuit-unavailable').hidden = neural;
  document.querySelector('.circuit-body').hidden = !neural;
  document.querySelector('.circuit .tabs').hidden = !neural;
  heat = neural ? buildHeatmap(dataset.runs[key], dataset.circuit.nodes, heatScale) : null;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([JSON.stringify(dataset.runs[key], null, 2)], {type: 'application/json'}));
  const link = $('download');
  link.href = downloadUrl; link.download = `sprint_${key}_seed${dataset.seed}.json`;
  link.textContent = `Download ${c.name.toLowerCase()} JSON`; link.removeAttribute('aria-disabled');
  draw();
}

function seek(time) {
  playing = false;
  clock = Math.max(0, Math.min(endTime, time));
  draw();
}

function drawLanes() {
  for (const c of CONTROLLERS) {
    const row = rowAt(c.key, clock), lane = lanes[c.key], rows = dataset.runs[c.key];
    const done = clock >= rows.at(-1).simulation_time - 1e-9;
    const pct = Math.max(0, Math.min(100, row.position));
    lane.runner.style.left = `${pct}%`;
    lane.progress.style.width = `${pct}%`;
    lane.root.classList.toggle('is-running', playing && !done && row.velocity > 0);
    lane.dist.textContent = `${fmt(row.position, 1)} m`;
    lane.speed.textContent = `${fmt(row.velocity)} m/s`;
    let label = row.action, state = row.action.toLowerCase();
    if (done) [label, state] = row.distance_to_finish <= 0 ? [`FINISH ${fmt(row.simulation_time, 1)} s`, 'finished'] : ['TIME LIMIT', 'timeout'];
    lane.action.textContent = label; lane.action.dataset.state = state;
  }
}

function drawCharts() {
  const series = (field, extra = () => ({})) => CONTROLLERS.map(c => {
    const rows = dataset.runs[c.key], i = indexAt(rows, clock);
    return {
      color: c.color, emphasis: c.key === focus,
      points: [[0, 0], ...rows.map(r => [r.simulation_time, r[field]])],
      marker: i >= 0 ? [rows[i].simulation_time, rows[i][field]] : [0, 0], ...extra(c, rows)
    };
  });
  const topSpeed = Math.max(...CONTROLLERS.map(c => summaries[c.key].topSpeed));
  drawLineChart($('chart-distance'), {series: series('position'), xMax: endTime, yMin: 0, yMax: 100, cursor: clock});
  drawLineChart($('chart-speed'), {series: series('velocity'), xMax: endTime, yMin: 0, yMax: Math.ceil(topSpeed + .5), cursor: clock});
  drawActionChart($('chart-actions'), {xMax: endTime, cursor: clock, lanes: CONTROLLERS.map(c => ({short: c.short, color: c.color, rows: dataset.runs[c.key], emphasis: c.key === focus}))});

  const neural = CONTROLLERS.filter(c => isNeural(c.key));
  const diffs = neural.flatMap(c => dataset.runs[c.key].map(r => r.motor_difference)).concat(neural.map(c => dataset.runs[c.key][0].motor_threshold));
  const positive = diffs.filter(v => v > 0);
  const decoder = neural.map(c => {
    const rows = dataset.runs[c.key], i = indexAt(rows, clock);
    return {color: c.color, emphasis: c.key === focus || !isNeural(focus),
      points: rows.map(r => [r.simulation_time, r.motor_difference]), thresholds: [rows[0].motor_threshold],
      marker: i >= 0 ? [rows[i].simulation_time, rows[i].motor_difference] : null};
  });
  drawLineChart($('chart-decoder'), {series: decoder, xMax: endTime, log: true,
    yMin: Math.floor(Math.log10(Math.min(...positive))), yMax: Math.ceil(Math.log10(Math.max(...positive))), cursor: clock});
}

function drawCircuit() {
  const neural = isNeural(focus), rows = dataset.runs[focus], i = indexAt(rows, clock), row = rows[i];
  $('pop-time').textContent = `${fmt(row?.simulation_time ?? 0, 1)} s`;
  for (const [field] of POPULATIONS) {
    const value = neural ? (row?.[field] ?? null) : null;
    $(field).style.width = `${(value ?? 0) * 100}%`;
    $(`${field}-value`).textContent = value === null ? (neural ? 'reset' : 'N/A') : fmt(value, 4);
  }
  if (neural && row) {
    const above = row.motor_difference > row.motor_threshold;
    $('difference').innerHTML = `${row.motor_difference.toExponential(3)} <span class="badge ${above ? 'up' : 'down'}">${above ? '▲ ACCELERATE' : '▼ COAST'}</span>`;
  } else $('difference').textContent = neural ? '—' : 'N/A';
  $('threshold-note').textContent = neural ? `threshold ${rows[0].motor_threshold.toExponential(3)}` : 'No neural decoder for this baseline';
  if (!neural) return;
  if (activeTab === 'heatmap') drawHeatmap($('heatmap'), heat, i, ROLE_LABELS);
  drawNeural(row, true);
  drawBrain(row, true);
  drawSignalFrame(Math.min(clock, rows.at(-1).simulation_time));
}

function draw() {
  if (!dataset) return;
  drawLanes(); drawCharts(); drawCircuit();
  $('timeline').value = Math.round(clock * 10);
  $('clock').textContent = `${fmt(clock, 1)} / ${fmt(endTime, 1)} s`;
  const over = clock >= endTime - 1e-9;
  $('play').textContent = playing ? '❚❚ Pause' : over ? '↺ Replay' : '▶ Play';
  const leader = CONTROLLERS.map(c => ({c, row: rowAt(c.key, clock)})).sort((a, b) => b.row.position - a.row.position)[0];
  $('status').textContent = clock === 0 ? `Ready · ${CONTROLLERS.length} recorded runs, seed ${dataset.seed}`
    : over ? 'Replay complete' : `${playing ? 'Playing' : 'Paused'} · leading: ${leader.c.name} at ${fmt(leader.row.position, 1)} m`;
}

function tick(now) {
  if (playing && previous !== null) {
    clock = Math.min(endTime, clock + Math.min((now - previous) / 1000, .25) * Number($('speed').value));
    if (clock >= endTime) playing = false;
    draw();
  }
  previous = now; requestAnimationFrame(tick);
}

function selectTab(name) {
  activeTab = name;
  for (const tab of ['heatmap', 'graph', 'anatomy']) {
    $(`tab-${tab}`).setAttribute('aria-selected', String(tab === name));
    $(`panel-${tab}`).hidden = tab !== name;
  }
  draw();
}

// Controls
$('play').onclick = () => { if (clock >= endTime - 1e-9) clock = 0; playing = !playing; previous = null; draw(); };
$('restart').onclick = () => seek(0);
$('timeline').oninput = () => seek(Number($('timeline').value) / 10);
for (const id of ['chart-distance', 'chart-speed', 'chart-actions', 'chart-decoder']) {
  $(id).addEventListener('click', e => { const t = chartTimeAt(e.currentTarget, e.clientX, endTime); if (t !== null) seek(t); });
}
for (const tab of ['heatmap', 'graph', 'anatomy']) $(`tab-${tab}`).onclick = () => selectTab(tab);
$('heatmap-scale').onchange = () => { heatScale = $('heatmap-scale').value; setFocus(focus); };
$('heatmap').addEventListener('click', e => {
  const cell = heat && heatmapCellAt(e.currentTarget, heat, e.clientX, e.clientY);
  if (cell) seek(heat.rows[cell.step].simulation_time);
});
$('heatmap').addEventListener('mousemove', e => {
  const cell = heat && heatmapCellAt(e.currentTarget, heat, e.clientX, e.clientY);
  if (!cell) { e.currentTarget.title = ''; return; }
  const node = dataset.circuit.nodes[cell.neuron], row = heat.rows[cell.step];
  e.currentTarget.title = `${node.cell_type || node.id} · ${ROLE_LABELS[node.role]}\nt = ${fmt(row.simulation_time, 1)} s · activity ${row.neuron_activity[cell.neuron].toFixed(4)}`;
});
window.addEventListener('resize', () => draw());
window.addEventListener('keydown', e => {
  if (!dataset || ['input', 'select', 'textarea'].includes(document.activeElement?.tagName?.toLowerCase())) return;
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  else if (e.code === 'KeyR') { e.preventDefault(); seek(0); }
  else if (e.code === 'ArrowRight' || e.code === 'ArrowLeft') {
    e.preventDefault(); seek(clock + (e.code === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 1 : .2));
  } else if (/^Digit[1-4]$/.test(e.code)) setFocus(CONTROLLERS[Number(e.code.slice(5)) - 1].key);
});

fetch('/runs.json').then(response => { if (!response.ok) throw Error('simulation unavailable'); return response.json(); }).then(data => {
  for (const c of CONTROLLERS) if (!data.runs?.[c.key]?.length) throw Error(`missing "${c.key}" run`);
  if (!data.circuit || !data.runs.trained[0].neuron_activity) throw Error('restart the Python server to load neuron recordings');
  dataset = data;
  endTime = Math.max(...CONTROLLERS.map(c => data.runs[c.key].at(-1).simulation_time));
  for (const c of CONTROLLERS) summaries[c.key] = summarize(c.key);
  $('seed').textContent = `Seed ${data.seed} · ${data.circuit.nodes.length} neurons · ${data.circuit.edges.length.toLocaleString()} connections`;
  $('circuit-size').textContent = `MANC v1.2.1 front-leg circuit · ${data.circuit.nodes.length} neurons · ${data.circuit.edges.length.toLocaleString()} directed connections`;
  $('timeline').max = Math.round(endTime * 10);
  initializeBrain(data.circuit);
  initializeNeural(data.circuit);
  buildResults(); buildLanes(); buildBars();
  for (const id of ['play', 'restart', 'timeline']) $(id).disabled = false;
  setFocus(focus);
  requestAnimationFrame(tick);
}).catch(error => { $('status').textContent = `Could not load runs: ${error.message}. Start python -m experiments.serve_view.`; $('status').classList.add('error'); });
