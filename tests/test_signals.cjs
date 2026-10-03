// Run with node --test tests/test_signals.cjs; no frontend dependencies required.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const source = readFileSync('visualization/web/signals.js', 'utf8');
function setup() {
  const media = {matches:false, addEventListener(){}};
  const context = vm.createContext({window:{matchMedia:()=>media}, brainElement:()=>({style:{},append(){}})});
  vm.runInContext(source + `
    var signal = createSignal({append(){}}, [0,0], [100,100], {pre:0,post:1}, '#fff', 0);
    var row = {neuron_activity:[.25,.1]};
  `, context);
  return {context, media, run:code=>vm.runInContext(code, context)};
}
test('packets move source to target and scrubbing restores the exact position', () => {
  const {context,run}=setup();
  run('drawSignals([signal], row, true, 1)');
  const first=context.signal.dots[0].style.transform;
  run('drawSignals([signal], row, true, 2)');
  assert.notEqual(context.signal.dots[0].style.transform,first);
  const paused=context.signal.dots[0].style.transform;
  run('drawSignals([signal], row, true, 2)');
  assert.equal(context.signal.dots[0].style.transform,paused);
  run('drawSignals([signal], row, true, 1)');
  assert.equal(context.signal.dots[0].style.transform,first);
});
test('baseline and reset hide packets; source activity controls contrast', () => {
  const {context,run}=setup();
  run('drawSignals([signal], row, true, 1)');
  assert.ok(context.signal.packets.style.opacity>0);
  const brightness=context.signal.path.style.opacity;
  run('drawSignals([signal], {neuron_activity:[1,0]}, true, 1)');
  assert.ok(context.signal.path.style.opacity>brightness);
  run('drawSignals([signal], row, false, 1)');
  assert.equal(context.signal.packets.style.opacity,0);
  run('drawSignals([signal], undefined, true, 0)');
  assert.equal(context.signal.packets.style.opacity,0);
});
test('reduced motion shows a stationary marker instead of traveling trails', () => {
  const {context,run,media}=setup();media.matches=true;
  run('drawSignals([signal], row, true, 1)');
  const first=context.signal.dots[0].style.transform;
  run('drawSignals([signal], row, true, 3)');
  assert.equal(context.signal.dots[0].style.transform,first);
  assert.equal(context.signal.dots[1].style.opacity,0);
  assert.ok(context.signal.packets.style.opacity>0);
});
