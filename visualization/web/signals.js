// Linear signal travel illustrates edge direction, not measured spike timing.
// Driven by the playback clock: no independent timers or CSS animation loops.
const signalMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let signalTime = 0;
function createSignal(layer, start, end, edge, color, ordinal, bend = 24) {
  const control = [(start[0] + end[0]) / 2, Math.min(start[1], end[1]) - bend];
  const path = brainElement('path', {
    d:`M${start} Q${control} ${end}`, stroke:color, class:'signal-path',
    'stroke-width':1.15, fill:'none'
  });
  path.style.stroke = color;
  const packets = brainElement('g', {class:'signal-packets', 'aria-hidden':'true'});
  const dots = [0,1,2].map(i => {
    const dot = brainElement('circle', {r:i === 0 ? 2.5 : 1.7, fill:i === 0 ? '#f4fcff' : color});
    packets.append(dot);return dot;
  });
  layer.append(path, packets);
  return {path, packets, dots, start, end, control, edge, phase:(ordinal * .61803398875) % 1};
}
function drawSignals(signals, row, available, time) {
  for (const signal of signals) {
    const raw = available ? row?.neuron_activity?.[signal.edge.pre] ?? 0 : 0;
    // Square-root contrast reveals small modeled rates without changing the readout.
    const strength = Math.sqrt(Math.max(0, Math.min(1, raw)));
    signal.path.style.opacity = .08 + strength * .52;
    signal.packets.style.opacity = raw > 0 ? .25 + strength * .75 : 0;
    signal.dots.forEach((dot, i) => {
      const t = signalMotion.matches ? .5 : ((time / 6 + signal.phase - i * .035) % 1 + 1) % 1;
      const u = 1 - t;
      const x = u*u*signal.start[0] + 2*u*t*signal.control[0] + t*t*signal.end[0];
      const y = u*u*signal.start[1] + 2*u*t*signal.control[1] + t*t*signal.end[1];
      dot.style.transform = `translate(${x}px, ${y}px)`;
      dot.style.opacity = signalMotion.matches ? (i === 0 ? 1 : 0) : (1 - i * .3) * Math.min(1, t * 12, (1-t) * 12);
    });
  }
}
function drawSignalFrame(time) {
  signalTime = time;
  drawSignals(brainSignals, brainRow, brainAvailable, time);
  // Avoid drawing the detailed graph while it is collapsed.
  if (document.querySelector('.neuron-inspector').open) {
    drawSignals(neuralSignals, latestNeuralRow, neuralAvailable, time);
  }
}
signalMotion.addEventListener('change', () => drawSignalFrame(signalTime));
