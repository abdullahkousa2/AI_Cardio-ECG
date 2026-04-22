const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const results = document.getElementById('results');
const loading = document.getElementById('loading');

const CLASS_COLORS = {
  'Myocardial Infarction': '#e53e3e',
  'History of MI':         '#ed8936',
  'Abnormal Heartbeat':    '#ecc94b',
  'Normal':                '#48bb78',
};

// Drag & drop
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) submitFile(file);
});
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) submitFile(fileInput.files[0]); });

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const which = tab.dataset.tab;
    document.getElementById('originalPanel').style.display = which === 'original' ? '' : 'none';
    document.getElementById('gradcamPanel').style.display  = which === 'gradcam'  ? '' : 'none';
  });
});

async function submitFile(file) {
  loading.style.display = 'flex';
  results.style.display = 'none';

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Prediction failed');
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    loading.style.display = 'none';
  }
}

function renderResults(data) {
  // Badge
  const badge = document.getElementById('diagnosisBadge');
  badge.textContent = data.predicted_class;
  badge.style.color = CLASS_COLORS[data.predicted_class] || '#fff';

  // Confidence
  document.getElementById('confidenceMain').textContent =
    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // Bars
  const barsEl = document.getElementById('classBars');
  barsEl.innerHTML = '';
  const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([cls, prob], i) => {
    const pct = (prob * 100).toFixed(1);
    const isTop = cls === data.predicted_class;
    barsEl.innerHTML += `
      <div class="class-row">
        <div class="class-row-header">
          <span class="class-name">${cls}</span>
          <span class="class-pct" style="color:${isTop ? (CLASS_COLORS[cls] || '#fff') : 'var(--text-muted)'}">${pct}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${isTop ? 'top' : 'other'}"
               style="width:${pct}%; background:${CLASS_COLORS[cls] || 'var(--blue)'}"></div>
        </div>
      </div>`;
  });

  // Images
  document.getElementById('originalImg').src = 'data:image/png;base64,' + data.original_image;
  document.getElementById('gradcamImg').src  = 'data:image/png;base64,' + data.gradcam_image;

  // Reset tabs to original
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="original"]').classList.add('active');
  document.getElementById('originalPanel').style.display = '';
  document.getElementById('gradcamPanel').style.display  = 'none';

  results.style.display = '';
  results.scrollIntoView({ behavior: 'smooth' });
}

function reset() {
  results.style.display = 'none';
  fileInput.value = '';
  document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth' });
}
