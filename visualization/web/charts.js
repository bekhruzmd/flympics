// Canvas charts for recorded runs. Every mark is drawn from a recorded row; nothing is interpolated or smoothed.
const CHART_PAD = {left: 46, right: 14, top: 10, bottom: 24};
const CHART_INK = '#1d2321', CHART_MUTED = '#7c847f', CHART_GRID = '#e6e8e3';

function prepareCanvas(canvas, padLeft = CHART_PAD.left) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  const plot = {x: padLeft, y: CHART_PAD.top, w: width - padLeft - CHART_PAD.right, h: height - CHART_PAD.top - CHART_PAD.bottom};
  canvas._plot = plot;
  return {ctx, plot};
}

function niceStep(span, target) {
  const raw = span / target, power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map(m => m * power).find(step => step >= raw) || raw;
}

function drawTimeAxis(ctx, plot, xMax) {
  const step = niceStep(xMax, Math.max(3, Math.min(8, Math.floor(plot.w / 64))));
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let t = 0; t <= xMax + 1e-9; t += step) {
    const x = plot.x + plot.w * t / xMax;
    ctx.strokeStyle = CHART_GRID; ctx.beginPath(); ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.h); ctx.stroke();
    ctx.fillStyle = CHART_MUTED; ctx.fillText(`${+t.toFixed(1)}s`, x, plot.y + plot.h + 6);
  }
}

function drawCursor(ctx, plot, xMax, cursor) {
  if (cursor == null) return;
  const x = plot.x + plot.w * Math.min(1, cursor / xMax);
  ctx.strokeStyle = CHART_INK; ctx.lineWidth = 1; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(x + .5, plot.y); ctx.lineTo(x + .5, plot.y + plot.h); ctx.stroke();
}

// series: [{color, points:[[t, y]], emphasis, thresholds:[y]}]; log=true maps y through log10 with non-positive values clamped to the floor.
function drawLineChart(canvas, {series, xMax, yMin, yMax, log = false, format = v => v, cursor}) {
  const {ctx, plot} = prepareCanvas(canvas);
  const map = log ? v => Math.log10(Math.max(v, 10 ** yMin)) : v => v;
  const yOf = v => plot.y + plot.h * (1 - (Math.min(yMax, Math.max(yMin, map(v))) - yMin) / (yMax - yMin));
  const xOf = t => plot.x + plot.w * t / xMax;

  drawTimeAxis(ctx, plot, xMax);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  const ticks = [];
  if (log) for (let p = Math.ceil(yMin); p <= yMax; p++) ticks.push(p);
  else { const step = niceStep(yMax - yMin, 4); for (let v = Math.ceil(yMin / step) * step; v <= yMax + 1e-12; v += step) ticks.push(v); }
  for (const tick of ticks) {
    const y = plot.y + plot.h * (1 - (tick - yMin) / (yMax - yMin));
    ctx.strokeStyle = CHART_GRID; ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
    ctx.fillStyle = CHART_MUTED; ctx.fillText(log ? `1e${tick}` : format(tick), plot.x - 6, y);
  }

  ctx.save(); ctx.beginPath(); ctx.rect(plot.x, plot.y - 2, plot.w, plot.h + 4); ctx.clip();
  const ordered = [...series].sort((a, b) => Number(a.emphasis) - Number(b.emphasis));
  for (const s of ordered) {
    ctx.globalAlpha = s.emphasis ? 1 : .45;
    for (const threshold of s.thresholds || []) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.25; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(plot.x, yOf(threshold)); ctx.lineTo(plot.x + plot.w, yOf(threshold)); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.strokeStyle = s.color; ctx.lineWidth = s.emphasis ? 2.25 : 1.5;
    ctx.beginPath();
    s.points.forEach(([t, v], i) => i ? ctx.lineTo(xOf(t), yOf(v)) : ctx.moveTo(xOf(t), yOf(v)));
    ctx.stroke();
    if (s.marker) {
      ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(xOf(s.marker[0]), yOf(s.marker[1]), s.emphasis ? 4 : 3, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.restore(); ctx.globalAlpha = 1;
  drawCursor(ctx, plot, xMax, cursor);
}

// lanes: [{label, color, rows, emphasis}] → one strip per controller; each recorded step is a 0.1 s cell.
function drawActionChart(canvas, {lanes, xMax, cursor}) {
  const {ctx, plot} = prepareCanvas(canvas, 82);
  drawTimeAxis(ctx, plot, xMax);
  const laneH = plot.h / lanes.length;
  lanes.forEach((lane, i) => {
    const top = plot.y + i * laneH + laneH * .2, h = laneH * .6;
    ctx.fillStyle = CHART_MUTED; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(lane.short, plot.x - 6, top + h / 2);
    ctx.globalAlpha = lane.emphasis ? 1 : .55;
    ctx.strokeStyle = lane.color; ctx.lineWidth = 1; ctx.strokeRect(plot.x + .5, top + .5, plot.w - 1, h - 1);
    ctx.fillStyle = lane.color;
    let start = null;
    lane.rows.forEach((row, j) => {
      const on = row.action === 'ACCELERATE';
      if (on && start === null) start = j;
      if ((!on || j === lane.rows.length - 1) && start !== null) {
        const end = on ? j + 1 : j;
        const t0 = start === 0 ? 0 : lane.rows[start - 1].simulation_time, t1 = lane.rows[end - 1].simulation_time;
        ctx.fillRect(plot.x + plot.w * t0 / xMax, top, Math.max(1, plot.w * (t1 - t0) / xMax), h);
        start = null;
      }
    });
    const endX = plot.x + plot.w * lane.rows.at(-1).simulation_time / xMax;
    if (endX < plot.x + plot.w - 2) { ctx.fillStyle = '#fbfbf8'; ctx.fillRect(endX + 1, top - 1, plot.x + plot.w - endX, h + 2); }
    ctx.globalAlpha = 1;
  });
  drawCursor(ctx, plot, xMax, cursor);
}

function chartTimeAt(canvas, clientX, xMax) {
  const plot = canvas._plot; if (!plot) return null;
  const rect = canvas.getBoundingClientRect();
  return Math.max(0, Math.min(xMax, (clientX - rect.left - plot.x) / plot.w * xMax));
}

// Perceptually ordered ramp (dark → bright) for activity values in [0, 1].
const HEAT_STOPS = [[16, 20, 38], [56, 41, 112], [124, 52, 128], [196, 78, 102], [242, 135, 67], [250, 214, 110]];
function heatColor(v) {
  const x = Math.max(0, Math.min(1, v)) * (HEAT_STOPS.length - 1), i = Math.min(HEAT_STOPS.length - 2, Math.floor(x)), f = x - i;
  return HEAT_STOPS[i].map((c, k) => Math.round(c + (HEAT_STOPS[i + 1][k] - c) * f));
}

const HEAT_ROLE_ORDER = ['sensory', 'intrinsic', 'motor_extensor', 'motor_flexor'];
// Returns an offscreen image: one pixel column per recorded step, one pixel row per neuron (grouped by role).
function buildHeatmap(rows, nodes, scale) {
  const order = HEAT_ROLE_ORDER.flatMap(role => nodes.map((node, i) => node.role === role ? i : -1).filter(i => i >= 0));
  const image = document.createElement('canvas');
  image.width = rows.length; image.height = order.length;
  const ctx = image.getContext('2d'), pixels = ctx.createImageData(rows.length, order.length);
  order.forEach((neuron, r) => {
    let lo = 0, hi = 1;
    if (scale === 'row') {
      lo = Infinity; hi = -Infinity;
      for (const row of rows) { lo = Math.min(lo, row.neuron_activity[neuron]); hi = Math.max(hi, row.neuron_activity[neuron]); }
    }
    rows.forEach((row, c) => {
      const v = row.neuron_activity[neuron];
      // 'saturation' maps log10(1 - v) from 1e0 (v = 0, dark) to 1e-4 (v ≥ 0.9999, bright); values are not altered.
      const norm = scale === 'saturation' ? -Math.log10(Math.max(1 - v, 1e-4)) / 4 : hi > lo ? (v - lo) / (hi - lo) : 0;
      const [red, green, blue] = heatColor(norm), p = (r * rows.length + c) * 4;
      pixels.data[p] = red; pixels.data[p + 1] = green; pixels.data[p + 2] = blue; pixels.data[p + 3] = 255;
    });
  });
  ctx.putImageData(pixels, 0, 0);
  const groups = HEAT_ROLE_ORDER.map(role => ({role, count: nodes.filter(n => n.role === role).length}));
  return {image, order, groups, rows};
}

function drawHeatmap(canvas, heat, cursorIndex, labels) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  const gutter = width < 520 ? 92 : 136;
  const plot = {x: gutter, y: 4, w: width - gutter - 6, h: height - 4 - 22};
  canvas._plot = plot;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(heat.image, plot.x, plot.y, plot.w, plot.h);
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  let offset = 0;
  for (const group of heat.groups) {
    const y0 = plot.y + plot.h * offset / heat.order.length, y1 = plot.y + plot.h * (offset + group.count) / heat.order.length;
    ctx.fillStyle = '#aab3c5'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(width < 520 ? labels[group.role] : `${labels[group.role]} (${group.count})`, plot.x - 8, (y0 + y1) / 2);
    if (offset) { ctx.strokeStyle = '#fbfbf8'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(plot.x, y0); ctx.lineTo(plot.x + plot.w, y0); ctx.stroke(); }
    offset += group.count;
  }
  const steps = heat.image.width, last = heat.rows.at(-1).simulation_time, step = niceStep(last, Math.max(3, Math.min(8, Math.floor(plot.w / 64))));
  ctx.fillStyle = '#aab3c5'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let t = 0; t <= last + 1e-9; t += step) ctx.fillText(`${+t.toFixed(1)}s`, plot.x + plot.w * t / last, plot.y + plot.h + 6);
  if (cursorIndex >= 0) {
    const x = plot.x + plot.w * (cursorIndex + 1) / steps;
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.h); ctx.stroke();
  }
}

function heatmapCellAt(canvas, heat, clientX, clientY) {
  const plot = canvas._plot; if (!plot) return null;
  const rect = canvas.getBoundingClientRect(), x = clientX - rect.left - plot.x, y = clientY - rect.top - plot.y;
  if (x < 0 || y < 0 || x >= plot.w || y >= plot.h) return null;
  return {step: Math.floor(x / plot.w * heat.image.width), neuron: heat.order[Math.floor(y / plot.h * heat.order.length)]};
}
