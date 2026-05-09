const form = document.querySelector('#parameter-form');
const runButton = document.querySelector('#run-button');
const statusEl = document.querySelector('#status');
const canvas = document.querySelector('#simulation-canvas');
const ctx = canvas.getContext('2d');
const playButton = document.querySelector('#play-button');
const slider = document.querySelector('#step-slider');
const stepOutput = document.querySelector('#step-output');
const metricsEl = document.querySelector('#metrics');
const frameTitle = document.querySelector('#frame-title');
const speed = document.querySelector('#speed');

let trace = null;
let frameIndex = 0;
let timer = null;

const colors = {
  Support: '#7b8794',
  Oppose: '#d08a18',
  Active: '#b23b2a',
  Jailed: '#2f3a45',
  Security: '#245c73',
};

function formParams() {
  const data = new FormData(form);
  const params = {};
  for (const [key, value] of data.entries()) {
    params[key] = value === 'on' ? true : Number(value);
  }
  params.movement = form.elements.movement.checked;
  return params;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawFrame(frameIndex);
}

function drawGrid(width, height, plot) {
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border');
  ctx.lineWidth = 1;
  const cell = Math.min(plot.w / width, plot.h / height);
  for (let x = 0; x <= width; x += Math.max(1, Math.ceil(width / 20))) {
    const px = plot.x + x * cell;
    ctx.beginPath(); ctx.moveTo(px, plot.y); ctx.lineTo(px, plot.y + height * cell); ctx.stroke();
  }
  for (let y = 0; y <= height; y += Math.max(1, Math.ceil(height / 20))) {
    const py = plot.y + y * cell;
    ctx.beginPath(); ctx.moveTo(plot.x, py); ctx.lineTo(plot.x + width * cell, py); ctx.stroke();
  }
  return cell;
}

function drawFrame(index) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  if (!trace) {
    ctx.fillStyle = '#667085';
    ctx.font = '600 18px IBM Plex Sans, sans-serif';
    ctx.fillText('Run a simulation to draw the agent field.', 28, 44);
    return;
  }

  frameIndex = Math.max(0, Math.min(index, trace.steps.length - 1));
  const frame = trace.steps[frameIndex];
  const params = trace.params;
  const pad = 28;
  const plot = { x: pad, y: pad, w: width - pad * 2, h: height - pad * 2 };
  const cell = drawGrid(params.width, params.height, plot);
  const radius = Math.max(2.2, Math.min(7, cell * 0.36));

  for (const agent of frame.agents) {
    if (agent.x === null || agent.y === null) continue;
    const jitter = agent.type === 'Security' ? 0.64 : 0.36;
    const px = plot.x + (agent.x + jitter) * cell;
    const py = plot.y + (agent.y + jitter) * cell;
    ctx.beginPath();
    ctx.fillStyle = colors[agent.condition] || '#111827';
    if (agent.type === 'Security') {
      ctx.rect(px - radius, py - radius, radius * 2, radius * 2);
    } else {
      ctx.arc(px, py, radius, 0, Math.PI * 2);
    }
    ctx.fill();
    if (agent.flip) {
      ctx.strokeStyle = '#f6f1e8';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  slider.value = String(frameIndex);
  stepOutput.textContent = `${frame.step} / ${trace.steps.length - 1}`;
  frameTitle.textContent = `Step ${frame.step}`;
  renderMetrics(frame);
}

function renderMetrics(frame) {
  const entries = [
    ['Support', frame.counts.Support],
    ['Oppose', frame.counts.Oppose],
    ['Active', frame.counts.Active],
    ['Jailed', frame.counts.Jailed],
    ['Security', frame.counts.Security],
  ];
  metricsEl.innerHTML = entries.map(([label, value]) => `
    <div class="metric"><b>${value}</b><span>${label}</span></div>
  `).join('');
}

async function runSimulation() {
  stopPlayback();
  runButton.disabled = true;
  statusEl.textContent = 'Running core_cpu_mojo…';
  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(formParams()),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'simulation failed');
    trace = payload;
    frameIndex = 0;
    slider.max = String(Math.max(0, trace.steps.length - 1));
    slider.disabled = false;
    playButton.disabled = false;
    statusEl.textContent = `Loaded ${trace.step_count} frames for ${trace.agent_count} agents.`;
    drawFrame(0);
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
}

function play() {
  if (!trace || timer) return;
  playButton.textContent = 'Pause';
  timer = setInterval(() => {
    if (frameIndex >= trace.steps.length - 1) {
      stopPlayback();
      return;
    }
    drawFrame(frameIndex + 1);
  }, Math.max(30, 420 - Number(speed.value) * 18));
}

function stopPlayback() {
  if (timer) clearInterval(timer);
  timer = null;
  playButton.textContent = 'Play';
}

runButton.addEventListener('click', runSimulation);
playButton.addEventListener('click', () => timer ? stopPlayback() : play());
slider.addEventListener('input', () => drawFrame(Number(slider.value)));
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
