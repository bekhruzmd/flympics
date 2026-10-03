// Fibers are deterministic illustration; only the four MANC groups use run data.
const brainRegions = [
  {id:'vision', label:'Optic lobes', color:'#40d9ec', title:'See the surroundings', description:'The optic lobes process visual information. They provide context for a real fly’s movement, but this sprint has no visual input or optic-lobe simulation.'},
  {id:'navigation', label:'Central complex', color:'#f176df', title:'Orientation & navigation', description:'The central complex supports orientation and navigation in flies. It is shown for context; this straight-line sprint does not model navigation or record activity here.'},
  {id:'descending', label:'Descending pathways', color:'#e6c65c', title:'From brain to movement circuits', description:'Descending neurons carry signals from the brain to the nerve cord. These pathways are illustrative here; the model instead supplies artificial task input directly to selected sensory neurons.'},
  {id:'sensory', label:'Sensory input', color:'#40d9ec', field:'sensory_mean', title:'Read the sprint state', description:'Selected front-leg sensory neurons receive artificial stimulation based on remaining distance and speed. Their brightness follows the recorded mean activity, not measured sensory firing.'},
  {id:'intrinsic', label:'Interneurons', color:'#b18aff', field:'intrinsic_mean', title:'Pass the signal through the circuit', description:'Selected nerve-cord interneurons propagate modeled activity through real MANC connectivity. This circuit links sensory input to motor output; it does not simulate individual leg movements.'},
  {id:'extensor', label:'Extensor output', color:'#b4ee68', field:'motor_extensor', title:'Contribute to the accelerate command', description:'The artificial decoder subtracts flexor mean activity from extensor mean activity. A difference above the fitted threshold commands ACCELERATE; otherwise the athlete coasts.'},
  {id:'flexor', label:'Flexor output', color:'#ffab70', field:'motor_flexor', title:'The other half of the motor readout', description:'Flexor output is read alongside extensor output. Both populations can be active at once. Their difference determines the sprint action; these are not simulated muscle contractions.'}
];
let brainSelected = 'intrinsic', brainRow, brainAvailable = true;
const brainGroups = new Map();
let brainSignals = [];
function brainElement(tag, attributes = {}, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
  if (text) el.textContent = text;
  return el;
}
function initializeBrain(circuit) {
  const svg = document.getElementById('brain-map');
  svg.replaceChildren(); brainGroups.clear(); brainSignals = [];
  let seed = 71;
  const random = () => {seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296;};
  // Draw tangled fibers within elliptical lobes, with no implied measured anatomy.
  function fibers(group, cx, cy, rx, ry, count, palette) {
    for (let n = 0; n < count; n++) {
      const angle = random() * Math.PI * 2, radius = Math.sqrt(random());
      let d = '';
      for (let j = 0; j < 17; j++) {
        const t = angle + j * .23;
        const r = radius * (.83 + .17 * Math.sin(j * 1.6 + n));
        const x = cx + Math.cos(t) * rx * r;
        const y = cy + Math.sin(t) * ry * r;
        d += `${j ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)} `;
      }
      group.append(brainElement('path', {d, stroke:palette[n % palette.length], 'stroke-width': n % 7 ? .65 : 1.1, opacity:.35 + random() * .5}));
    }
  }
  function group(id) {
    const region = brainRegions.find(r => r.id === id);
    const g = brainElement('g', {'data-region':id, role:'button', tabindex:'0', 'aria-label':region.label, 'aria-pressed':'false'});
    g.append(brainElement('title', {}, region.label));
    g.addEventListener('click', () => selectBrainRegion(id));
    g.addEventListener('keydown', e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); selectBrainRegion(id);}});
    svg.append(g); brainGroups.set(id, g); return g;
  }
  const scaffold = brainElement('g', {opacity:'.55', 'aria-hidden':'true'}); svg.append(scaffold);
  fibers(scaffold, 228, 150, 94, 91, 160, ['#365bef','#b349ed','#25c9b3']);
  fibers(scaffold, 372, 150, 94, 91, 160, ['#6f55ee','#eb44a6','#419be5']);
  const vision = group('vision');
  for (const x of [100, 500]) {
    vision.append(brainElement('ellipse', {cx:x, cy:181, rx:73, ry:94, class:'brain-hit'}));
    fibers(vision, x, 181, 73, 94, 210, ['#38d5e7','#569df9','#c9e760','#30d6a2']);
  }
  const navigation = group('navigation');
  navigation.append(brainElement('ellipse', {cx:300, cy:125, rx:92, ry:44, class:'brain-hit'}));
  fibers(navigation, 300, 125, 92, 44, 170, ['#f06be0','#9f64fc','#fa8ad4']);
  const descending = group('descending');
  descending.append(brainElement('path', {d:'M263 173 Q305 231 282 306 M338 173 Q298 231 318 306', stroke:'transparent', 'stroke-width':35}));
  for (let i=0; i<90; i++) {
    const side = i % 2 ? -1 : 1, offset = random() * 38;
    descending.append(brainElement('path', {d:`M${300 + side*(60+offset)} ${153+random()*30} C${300+side*100} 229 ${300-side*28} 220 ${300+side*(8+random()*19)} 308`, stroke:i%3 ? '#c3c853' : '#f1a754', 'stroke-width':'.65', opacity:'.45'}));
  }
  svg.append(brainElement('path', {d:'M48 298 H238 M362 298 H552', stroke:'#27363d', 'stroke-dasharray':'3 5'}));
  svg.append(brainElement('text', {x:300,y:286,'text-anchor':'middle',class:'brain-svg-label'}, 'BRAIN ↑  /  NERVE CORD ↓'));
  const positions = {sensory:[243,338,40,30], intrinsic:[300,374,54,42], extensor:[235,424,43,27], flexor:[365,424,43,27]};
  for (const [id, [x,y,rx,ry]] of Object.entries(positions)) {
    const region = brainRegions.find(r => r.id === id), g = group(id);
    g.append(brainElement('ellipse', {cx:x,cy:y,rx:rx+8,ry:ry+5,class:'brain-hit'}));
    const activity = brainElement('g', {class:'brain-firing'});g.append(activity);
    fibers(activity,x,y,rx,ry,110,[region.color,region.color,'#dceaf0']);
    for (let n=0;n<16;n++) {
      const a=random()*Math.PI*2, r=Math.sqrt(random());
      activity.append(brainElement('circle', {cx:x+Math.cos(a)*rx*r, cy:y+Math.sin(a)*ry*r, r:1.3, fill:region.color}));
    }
    g.append(brainElement('ellipse', {cx:x,cy:y,rx:rx+9,ry:ry+6,class:'brain-ring',stroke:region.color}));
  }
  // Show a readable subset of real edges, balanced across source populations.
  const roleIds = {sensory:'sensory', intrinsic:'intrinsic', motor_extensor:'extensor', motor_flexor:'flexor'};
  const points = circuit.nodes.map((node, i) => {
    const [x,y,rx,ry] = positions[roleIds[node.role]];
    const angle = i * 2.39996323, radius = .3 + .55 * random();
    return [x + Math.cos(angle)*rx*radius, y + Math.sin(angle)*ry*radius];
  });
  const signalLayer = brainElement('g', {class:'brain-connections', 'aria-hidden':'true'});
  svg.append(signalLayer);
  const edgeCounts = {};
  const visibleEdges = [...circuit.edges].sort((a,b) => b.weight-a.weight).filter(edge => {
    const role = circuit.nodes[edge.pre].role;
    edgeCounts[role] = (edgeCounts[role] || 0) + 1;
    return edgeCounts[role] <= 16;
  });
  brainSignals = visibleEdges.map((edge,i) => {
    const region = brainRegions.find(r => r.id === roleIds[circuit.nodes[edge.pre].role]);
    return createSignal(signalLayer, points[edge.pre], points[edge.post], edge, region.color, i, 20 + i%4*8);
  });
  for (const i of new Set(visibleEdges.flatMap(edge => [edge.pre, edge.post]))) {
    const region = brainRegions.find(r => r.id === roleIds[circuit.nodes[i].role]);
    signalLayer.append(brainElement('circle', {cx:points[i][0],cy:points[i][1],r:2.1,fill:region.color,class:'signal-node'}));
  }
  for (const [label,x,y,anchor] of [['OPTIC LOBES',35,68,'start'],['CENTRAL COMPLEX',300,42,'middle'],['SENSORY',140,342,'end'],['INTERNEURONS',402,376,'start'],['EXTENSOR',166,433,'end'],['FLEXOR',433,433,'start']]) svg.append(brainElement('text', {x,y,'text-anchor':anchor,class:'brain-svg-label'},label));
  const picker = document.getElementById('brain-regions');
  picker.replaceChildren();
  const contextGroup = document.createElement('div');
  contextGroup.className = 'region-subgroup';
  contextGroup.innerHTML = '<span class="region-subgroup-title">Brain Context</span>';
  const modeledGroup = document.createElement('div');
  modeledGroup.className = 'region-subgroup';
  modeledGroup.innerHTML = '<span class="region-subgroup-title">Modeled MANC Circuit</span>';

  for (const region of brainRegions) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = region.label;
    button.style.setProperty('--region-color', region.color);
    button.dataset.region = region.id;
    button.onclick = () => selectBrainRegion(region.id);
    if (region.field) {
      modeledGroup.append(button);
    } else {
      contextGroup.append(button);
    }
  }
  picker.append(contextGroup, modeledGroup);
  selectBrainRegion(brainSelected);
}
function selectBrainRegion(id) {
  brainSelected=id;
  for (const [key,group] of brainGroups) {group.classList.toggle('is-selected', key===id);group.setAttribute('aria-pressed', String(key===id));}
  document.querySelectorAll('#brain-regions button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.region===id)));
  const region=brainRegions.find(r=>r.id===id);
  document.getElementById('brain-region-title').textContent=region.title;
  document.getElementById('brain-region-description').textContent=region.description;
  document.getElementById('brain-region-kind').textContent=region.field ? 'MANC · MODELED POPULATION' : 'BRAIN · CONTEXT ONLY';
  document.querySelector('.brain-detail').style.setProperty('--region-color',region.color);
  drawBrain(brainRow,brainAvailable);
}
function drawBrain(row, available) {
  brainRow=row;brainAvailable=available;
  document.getElementById('brain-time').textContent=`${(row?.simulation_time ?? 0).toFixed(1)} s`;
  for (const region of brainRegions.filter(r=>r.field)) {
    const value=available ? (row?.[region.field] ?? 0) : 0;
    const group=brainGroups.get(region.id);
    if (group) group.querySelector('.brain-firing').style.opacity=.12+.88*Math.max(0,Math.min(1,value));
  }
  const selected=brainRegions.find(r=>r.id===brainSelected);
  document.getElementById('brain-region-value').textContent=!selected.field ? 'Not simulated' : !available ? 'N/A' : `${(row?.[selected.field] ?? 0).toFixed(3)} / 1`;
  document.getElementById('brain-state').textContent=!available ? 'Baseline controller · no neural activity recorded.' : !row?.simulation_time ? 'Ready · play or scrub the sprint to see activity.' : `${row.action === 'ACCELERATE' ? 'Accelerating' : 'Coasting'} · ${row.position.toFixed(1)} / 100 m · brightness = population mean (0–1)`;
}
