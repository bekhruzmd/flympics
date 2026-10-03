const $ = id => document.getElementById(id);
const populations = [
  ['sensory_mean', 'Sensory neurons', '#5a93a0'],
  ['intrinsic_mean', 'VNC interneurons', '#9684b2'],
  ['motor_extensor', 'Extensor motor output', '#7d9954'],
  ['motor_flexor', 'Flexor motor output', '#c6a065']
];
for (const [key, label, color] of populations) {
  const row = document.createElement('div'); row.className = 'bar-row';
  row.style.setProperty('--pop-color', color);
  row.innerHTML = `<div class="bar-label"><span>${label}</span><output id="${key}-value">0.000</output></div><div class="bar-track"><div class="bar-fill" id="${key}" style="background:${color}"></div></div>`;
  $('bars').append(row);
}
let dataset, rows, index = -1, playing = false, clock = 0, previous = null, url;
function draw() {
  const row = rows[index] || {position:0,velocity:0,simulation_time:0,action:'READY'};
  $('distance').textContent = row.position.toFixed(2);
  $('velocity').textContent = row.velocity.toFixed(2);
  $('time').textContent = row.simulation_time.toFixed(1);

  const trackPct = Math.max(0, Math.min(100, row.position));
  $('athlete').style.left = `calc(32px + (100% - 64px) * ${trackPct / 100})`;
  $('athlete').classList.toggle('is-running', playing && row.velocity > 0);

  const ended = index === rows.length - 1;
  const actionEl = $('action');
  actionEl.className = '';
  if (ended) {
    if (row.distance_to_finish <= 0) {
      actionEl.textContent = 'FINISHED 🏁';
      actionEl.classList.add('action-finished');
    } else {
      actionEl.textContent = 'TIME LIMIT ⏱';
      actionEl.classList.add('action-timeout');
    }
  } else if (row.action === 'ACCELERATE') {
    actionEl.textContent = '● ACCELERATE';
    actionEl.classList.add('action-accelerate');
  } else if (row.action === 'COAST') {
    actionEl.textContent = '○ COAST';
    actionEl.classList.add('action-coast');
  } else {
    actionEl.textContent = row.action;
    actionEl.classList.add('action-ready');
  }

  $('timeline').value = index + 1;
  $('progress').textContent = `${row.simulation_time.toFixed(1)} / ${rows.at(-1).simulation_time.toFixed(1)} s`;
  const neural = ['fly', 'trained'].includes($('controller').value);
  for (const [key] of populations) {
    const value = neural ? (row[key] ?? 0) : null;
    $(key).style.width = `${(value ?? 0) * 100}%`;
    $(key+'-value').textContent = value === null ? 'N/A' : value.toFixed(3);
  }
  drawNeural(row, neural);
  drawBrain(row, neural);
  drawSignalFrame(Math.min(clock, rows.at(-1).simulation_time));
  const threshold = row.motor_threshold ?? rows[0].motor_threshold;
  $('threshold-note').textContent = neural ? `Accelerate when difference > ${threshold.toPrecision(5)}` : 'No neural decoder for this baseline';
  if (neural) {
    const diff = row.motor_difference ?? 0;
    const isAbove = diff > threshold;
    $('difference').innerHTML = `${diff.toFixed(5)} <span class="diff-badge ${isAbove ? 'diff-above' : 'diff-below'}">${isAbove ? '▲ ACCEL' : '▼ COAST'}</span>`;
  } else {
    $('difference').textContent = 'N/A';
  }
  $('activity-note').textContent = neural ? 'Mean modeled activity (0–1), not measured firing rates. Both motor outputs can be active while their difference commands COAST.' : 'This baseline has no neural model. Neural activity does not apply.';
  $('status').textContent = ended ? (row.distance_to_finish <= 0 ? 'Finished the sprint' : 'Time limit · did not finish') : (playing ? 'Playing simulation' : 'Ready · recorded simulation');
  $('play').textContent = playing ? '❚❚ Pause' : ended ? '↺ Play again' : '▶ Play run';
}
function load() {
  playing = false; index = -1; clock = 0; previous = null;
  rows = dataset.runs[$('controller').value]; $('timeline').max = rows.length;
  if(url) URL.revokeObjectURL(url);
  url = URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)], {type:'application/json'}));
  $('download').href = url; $('download').download = `sprint_${$('controller').value}_seed${dataset.seed}.json`;
  draw();
}
$('play').onclick = () => {if(index===rows.length-1){index=-1;clock=0;} playing=!playing;previous=null;draw();};
$('restart').onclick = () => {playing=false;index=-1;clock=0;draw();};
$('controller').onchange = () => {if(dataset) load();};
document.querySelector('.neuron-inspector').addEventListener('toggle', () => drawSignalFrame(signalTime));
$('timeline').oninput = () => {playing=false;index=Number($('timeline').value)-1;clock=index<0?0:rows[index].simulation_time;draw();};

// Click lane to seek
const trackLane = document.querySelector('.lane');
if (trackLane) {
  trackLane.title = 'Click to scrub along track';
  trackLane.addEventListener('click', e => {
    if (!rows || rows.length === 0) return;
    const rect = trackLane.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const targetDist = pct * 100;
    let bestIdx = 0, bestDiff = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const diff = Math.abs(rows[i].position - targetDist);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIdx = i;
      }
    }
    playing = false;
    index = bestIdx;
    clock = rows[index].simulation_time;
    draw();
  });
}

// Global keyboard shortcuts
window.addEventListener('keydown', e => {
  if (['input', 'select', 'textarea'].includes(document.activeElement?.tagName?.toLowerCase())) return;
  if (e.code === 'Space') {
    e.preventDefault();
    $('play').click();
  } else if (e.code === 'KeyR') {
    e.preventDefault();
    $('restart').click();
  } else if (e.code === 'ArrowRight' && rows) {
    e.preventDefault();
    playing = false;
    index = Math.min(rows.length - 1, index + (e.shiftKey ? 10 : 2));
    clock = index < 0 ? 0 : rows[index].simulation_time;
    draw();
  } else if (e.code === 'ArrowLeft' && rows) {
    e.preventDefault();
    playing = false;
    index = Math.max(-1, index - (e.shiftKey ? 10 : 2));
    clock = index < 0 ? 0 : rows[index].simulation_time;
    draw();
  }
});

function tick(now) {
  if(playing && previous!==null){
    clock += Math.min((now-previous)/1000,.25)*Number($('speed').value);
    while(index+1 < rows.length && rows[index+1].simulation_time <= clock) index++;
    if(index===rows.length-1) playing=false;
    draw();
  }
  previous=now; requestAnimationFrame(tick);
}
fetch('/runs.json').then(response => {if(!response.ok) throw Error('Simulation unavailable');return response.json();}).then(data => {
  if (!data.circuit || !data.runs.trained[0].neuron_activity) throw Error('Restart the Python server to load neuron recordings');
  initializeBrain(data.circuit);
  initializeNeural(data.circuit);
  dataset=data; $('seed').textContent=`Seed ${data.seed} · reproducible run`;
  load();for(const id of ['play','restart','timeline']) $(id).disabled=false;
  requestAnimationFrame(tick);
}).catch(error => {$('status').textContent=`Could not load run: ${error.message}. Start python -m experiments.serve_view.`;});
