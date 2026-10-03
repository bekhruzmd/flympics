// Node order matches the Python circuit and each recorded activity vector.
const neuralRoles = {
  sensory: {label: 'Sensory input', color: '#83cddd', x: 34, columns: 3},
  intrinsic: {label: 'VNC interneuron', color: '#c4adea', x: 119, columns: 8},
  motor_extensor: {label: 'Extensor motor output', color: '#b9d987', x: 289, columns: 2},
  motor_flexor: {label: 'Flexor motor output', color: '#dfc480', x: 289, columns: 2}
};
let neuralNodes = [], neuralEdges = [], neuralDots = [], neuralPoints = [], selectedNeuron = 0;
let latestNeuralRow, neuralAvailable = false, edgeLayer;
let neuralSignals = [];
function neuralElement(tag, attributes) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}
function initializeNeural(circuit) {
  neuralNodes = circuit.nodes;
  neuralEdges = circuit.edges;
  const map = document.getElementById('neuron-map');
  const picker = document.getElementById('neuron-select');
  map.replaceChildren(); picker.replaceChildren(); neuralDots = []; neuralPoints = [];
  edgeLayer = neuralElement('g', {}); map.append(edgeLayer);
  const counts = {};
  neuralNodes.forEach((node, index) => {
    const role = neuralRoles[node.role];
    const offset = counts[node.role] || 0; counts[node.role] = offset + 1;
    const x = role.x + (offset % role.columns) * 12;
    const y = (node.role === 'motor_flexor' ? 154 : 30) + Math.floor(offset / role.columns) * 17;
    neuralPoints.push([x, y]);
    const dot = neuralElement('circle', {cx: x, cy: y, r: 3.7, fill: role.color});
    const title = neuralElement('title', {});
    title.textContent = `${node.id} · ${role.label}`; dot.append(title);
    dot.addEventListener('click', () => {picker.value = String(index); selectNeuron(index);});
    dot.addEventListener('mouseenter', () => renderNeuronDetail(index));
    dot.addEventListener('mouseleave', () => renderNeuronDetail(selectedNeuron));
    map.append(dot); neuralDots.push(dot);
    const option = document.createElement('option'); option.value = index;
    option.textContent = `${role.label} · ${node.id}`; picker.append(option);
  });
  picker.onchange = () => selectNeuron(Number(picker.value));
  selectNeuron(0);
}
function selectNeuron(index) {
  selectedNeuron = index;
  neuralDots.forEach((dot, i) => dot.classList.toggle('selected', i === index));
  edgeLayer.replaceChildren(); neuralSignals = [];
  for (const edge of neuralEdges) {
    if (edge.pre !== index && edge.post !== index) continue;
    const [x1, y1] = neuralPoints[edge.pre], [x2, y2] = neuralPoints[edge.post];
    neuralSignals.push(createSignal(edgeLayer, [x1,y1], [x2,y2], edge, neuralRoles[neuralNodes[edge.pre].role].color, neuralSignals.length, 18));
  }
  drawNeural(latestNeuralRow, neuralAvailable);
  drawSignals(neuralSignals, latestNeuralRow, neuralAvailable, signalTime);
}
function renderNeuronDetail(index) {
  const node = neuralNodes[index];
  if (!node) return;
  const values = latestNeuralRow?.neuron_activity;
  const incoming = neuralEdges.filter(edge => edge.post === index).length;
  const outgoing = neuralEdges.filter(edge => edge.pre === index).length;
  document.getElementById('neuron-detail').textContent = !neuralAvailable
    ? 'No neural activity: this baseline uses rules or random actions.'
    : `${node.cell_type || node.id} · ${neuralRoles[node.role].label}. Activity: ${values ? values[index].toFixed(4) : '0.0000 (reset)'}. ${incoming} incoming / ${outgoing} outgoing connections.`;
}
function drawNeural(row, available) {
  latestNeuralRow = row; neuralAvailable = available;
  const values = available ? row?.neuron_activity : null;
  document.querySelector('.neural-inset').classList.toggle('unavailable', !available);
  document.getElementById('neural-time').textContent = `${(row?.simulation_time ?? 0).toFixed(1)} s`;
  neuralDots.forEach((dot, i) => {dot.style.opacity = values ? .15 + .85 * Math.max(0, Math.min(1, values[i])) : .15;});
  renderNeuronDetail(selectedNeuron);
}
