const ui = {
  status: document.getElementById("connectionStatus"),
  cameraStatus: document.getElementById("cameraStatus"),
  faceStatus: document.getElementById("faceStatus"),
  fpsStatus: document.getElementById("fpsStatus"),
  sampleStatus: document.getElementById("sampleStatus"),
  qualityStatus: document.getElementById("qualityStatus"),
  calibrationStatus: document.getElementById("calibrationStatus"),
  eyeCrop: document.getElementById("eyeCrop"),
  eyeOverlay: document.getElementById("eyeOverlay"),
  eyePlaceholder: document.getElementById("eyePlaceholder"),
  camera: document.getElementById("cameraInput"),
  method: document.getElementById("methodInput"),
  threshold: document.getElementById("thresholdInput"),
  window: document.getElementById("windowInput"),
  pso: document.getElementById("psoInput"),
  start: document.getElementById("startButton"),
  stop: document.getElementById("stopButton"),
  clearSession: document.getElementById("clearSessionButton"),
  fitCalibration: document.getElementById("fitCalibrationButton"),
  resetCalibration: document.getElementById("resetCalibrationButton"),
  resetAll: document.getElementById("resetAllButton"),
  export: document.getElementById("exportButton"),
  output: document.getElementById("outputStatus"),
  gazeX: document.getElementById("gazeX"),
  gazeY: document.getElementById("gazeY"),
  calGazeX: document.getElementById("calGazeX"),
  calGazeY: document.getElementById("calGazeY"),
  pupilSize: document.getElementById("pupilSize"),
  sampleCount: document.getElementById("sampleCount"),
  duration: document.getElementById("duration"),
  finiteFraction: document.getElementById("finiteFraction"),
  detectionThreshold: document.getElementById("detectionThreshold"),
  saccadeCount: document.getElementById("saccadeCount"),
  fixationCount: document.getElementById("fixationCount"),
  microCount: document.getElementById("microCount"),
  pursuitCount: document.getElementById("pursuitCount"),
  psoCount: document.getElementById("psoCount"),
  qualityIndex: document.getElementById("qualityIndex"),
  samplingCv: document.getElementById("samplingCv"),
  gazePath: document.getElementById("gazePath"),
  pupilValid: document.getElementById("pupilValid"),
  truthBoundary: document.getElementById("truthBoundary"),
  validationRepeats: document.getElementById("validationRepeatsInput"),
  runSyntheticValidation: document.getElementById("runSyntheticValidationButton"),
  clearValidation: document.getElementById("clearValidationButton"),
  macroF1: document.getElementById("macroF1"),
  worstDomain: document.getElementById("worstDomain"),
  stabilityGap: document.getElementById("stabilityGap"),
  liveWarnings: document.getElementById("liveWarnings"),
  domainValidationTable: document.getElementById("domainValidationTable"),
  statsWindow: document.getElementById("statsWindow"),
  sampleRate: document.getElementById("sampleRate"),
  medianDt: document.getElementById("medianDt"),
  saccadeRate: document.getElementById("saccadeRate"),
  fixationDwell: document.getElementById("fixationDwell"),
  pathEfficiency: document.getElementById("pathEfficiency"),
  dispersion: document.getElementById("dispersion"),
  bcea: document.getElementById("bcea"),
  transitionEntropy: document.getElementById("transitionEntropy"),
};

const DEFAULTS = {
  camera: "0",
  method: "ivt",
  threshold: "30",
  window: "10",
  pso: false,
};

const PLOTS = {
  gazeXY: "Waiting for gaze samples",
  gazeTime: "Waiting for gaze time series",
  speedTrace: "Waiting for velocity samples",
  pupilTrace: "Waiting for pupil samples",
  eventRaster: "Waiting for detected events",
  scanpath: "Waiting for fixations",
  directionPolar: "Waiting for saccades",
  amplitudeHist: "Waiting for saccade amplitudes",
  mainSequence: "Waiting for main-sequence points",
  domainValidationPlot: "Run synthetic validation",
};

const state = {
  socket: null,
  latest: null,
  calibrationRangeDeg: 15,
  calibrationActive: false,
  hasSamples: false,
  finiteSampleCount: 0,
  isConnecting: false,
  isLive: false,
  outputConfigured: false,
  validationBusy: false,
  syntheticValidation: null,
};

function fmt(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "--";
}

function pct(value, digits = 0) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "--";
}

function stat(payload, group, key) {
  const value = payload && payload[group] ? payload[group][key] : undefined;
  return Number.isFinite(value) ? value : null;
}

function values(series, key) {
  return (series && Array.isArray(series[key]) ? series[key] : []).filter((v) => Number.isFinite(v));
}

function setChipTone(element, tone = "") {
  element.classList.remove("ok", "warn", "bad");
  if (tone) element.classList.add(tone);
}

function setStatus(message, tone = "") {
  ui.status.textContent = message;
  setChipTone(ui.status, tone);
}

function setOutput(message, tone = "") {
  ui.output.textContent = message;
  ui.output.dataset.tone = tone;
}

function updateButtons() {
  const busy = state.isConnecting || state.isLive;
  ui.start.disabled = busy;
  ui.stop.disabled = !busy;
  ui.clearSession.disabled = !busy && !state.hasSamples;
  ui.fitCalibration.disabled = state.finiteSampleCount < 4;
  ui.resetCalibration.disabled = !state.calibrationActive;
  ui.resetAll.disabled = state.isConnecting;
  ui.export.disabled = !state.outputConfigured || !state.hasSamples;
  ui.runSyntheticValidation.disabled = state.validationBusy;
  ui.clearValidation.disabled = state.validationBusy || !state.syntheticValidation;
  ui.export.title = state.outputConfigured
    ? state.hasSamples
      ? "Export current live session"
      : "Capture samples before export"
    : "Start live-html with --output-dir to enable export";
}

function canvasContext(id) {
  const canvas = document.getElementById(id);
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.font = "12px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "alphabetic";
  return { canvas, ctx, w: rect.width, h: rect.height };
}

function clearPlot(ctx, w, h, { xLabel = "", yLabel = "", empty = "" } = {}) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#eef2f7";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const x = 34 + ((w - 48) * i) / 4;
    const y = 16 + ((h - 42) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, 12);
    ctx.lineTo(x, h - 24);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(34, y);
    ctx.lineTo(w - 12, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#d7dee8";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  ctx.beginPath();
  ctx.moveTo(34, 12);
  ctx.lineTo(34, h - 24);
  ctx.lineTo(w - 12, h - 24);
  ctx.stroke();

  ctx.fillStyle = "#607080";
  if (xLabel) ctx.fillText(xLabel, Math.max(36, w - 86), h - 7);
  if (yLabel) ctx.fillText(yLabel, 8, 16);
  if (empty) {
    ctx.fillStyle = "#8a97a8";
    ctx.textAlign = "center";
    ctx.fillText(empty, w / 2, h / 2);
    ctx.textAlign = "start";
  }
}

function emptyPlot(id, label) {
  const { ctx, w, h } = canvasContext(id);
  clearPlot(ctx, w, h, { empty: label });
}

function resetPlots() {
  Object.entries(PLOTS).forEach(([id, label]) => emptyPlot(id, label));
}

function extent(data, fallback = [-1, 1]) {
  const finite = data.filter((v) => Number.isFinite(v));
  if (!finite.length) return fallback;
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function scale(value, lo, hi, outLo, outHi) {
  if (!Number.isFinite(value)) return null;
  return outLo + ((value - lo) / (hi - lo)) * (outHi - outLo);
}

function drawLine(ctx, xs, ys, x0, x1, y0, y1, w, h) {
  ctx.beginPath();
  let started = false;
  xs.forEach((x, i) => {
    const px = scale(x, x0, x1, 34, w - 12);
    const py = scale(ys[i], y0, y1, h - 24, 12);
    if (px === null || py === null) return;
    if (!started) {
      ctx.moveTo(px, py);
      started = true;
    } else {
      ctx.lineTo(px, py);
    }
  });
  ctx.stroke();
}

function linePlot(id, xs, ys, color, label, yExtra = []) {
  const { ctx, w, h } = canvasContext(id);
  if (!xs.length || !ys.length) {
    clearPlot(ctx, w, h, { empty: PLOTS[id] || "Waiting for samples" });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "time (s)", yLabel: label });
  const [x0, x1] = extent(xs, [0, 1]);
  const [y0, y1] = extent([...ys, ...yExtra], [-1, 1]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  drawLine(ctx, xs, ys, x0, x1, y0, y1, w, h);
  ctx.fillStyle = "#607080";
  ctx.fillText(label, 42, 17);
}

function multiLinePlot(id, xs, series) {
  const { ctx, w, h } = canvasContext(id);
  if (!xs.length) {
    clearPlot(ctx, w, h, { empty: PLOTS[id] || "Waiting for samples" });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "time (s)", yLabel: "deg" });
  const [x0, x1] = extent(xs, [0, 1]);
  const allY = series.flatMap((s) => s.data);
  const [y0, y1] = extent(allY, [-1, 1]);
  series.forEach((entry) => {
    ctx.strokeStyle = entry.color;
    ctx.lineWidth = 1.5;
    drawLine(ctx, xs, entry.data, x0, x1, y0, y1, w, h);
  });
  ctx.fillStyle = "#607080";
  ctx.fillText(series.map((s) => s.label).join(" / "), 42, 17);
}

function drawGazeXY(data) {
  const { ctx, w, h } = canvasContext("gazeXY");
  const xs = values(data.series, "x");
  const ys = values(data.series, "y");
  if (!xs.length || !ys.length) {
    clearPlot(ctx, w, h, { empty: PLOTS.gazeXY });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "x (deg)", yLabel: "y (deg)" });
  const cxs = values(data.series, "calibrated_x");
  const cys = values(data.series, "calibrated_y");
  const [x0, x1] = extent([...xs, ...cxs], [-10, 10]);
  const [y0, y1] = extent([...ys, ...cys], [-10, 10]);
  ctx.strokeStyle = "#0072b2";
  ctx.lineWidth = 1.4;
  drawLine(ctx, xs, ys, x0, x1, y0, y1, w, h);
  if (cxs.length && cys.length) {
    ctx.strokeStyle = "#d55e00";
    drawLine(ctx, cxs, cys, x0, x1, y0, y1, w, h);
  }
  ctx.fillStyle = "#607080";
  ctx.fillText(cxs.length ? "raw blue / calibrated red" : "raw gaze", 42, 17);
}

function drawSpeed(data) {
  const t = values(data.series, "t");
  const speed = values(data.series, "speed");
  const { ctx, w, h } = canvasContext("speedTrace");
  if (!t.length || !speed.length) {
    clearPlot(ctx, w, h, { empty: PLOTS.speedTrace });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "time (s)", yLabel: "deg/s" });
  const threshold = data.method.velocity_threshold_deg_s;
  const [x0, x1] = extent(t, [0, 1]);
  const [y0, y1] = extent([...speed, threshold], [0, 100]);
  const saccades = (data.analysis && data.analysis.saccades) || [];
  ctx.fillStyle = "rgba(230, 159, 0, 0.16)";
  saccades.forEach((s) => {
    const x = scale(s.onset_t, x0, x1, 34, w - 12);
    const x2 = scale(s.offset_t, x0, x1, 34, w - 12);
    if (x === null || x2 === null) return;
    ctx.fillRect(x, 12, Math.max(2, x2 - x), h - 36);
  });
  const ty = scale(threshold, y0, y1, h - 24, 12);
  ctx.strokeStyle = "#d55e00";
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(34, ty);
  ctx.lineTo(w - 12, ty);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#0072b2";
  ctx.lineWidth = 1.8;
  drawLine(ctx, t, speed, x0, x1, y0, y1, w, h);
  ctx.fillStyle = "#607080";
  ctx.fillText(`speed, threshold ${fmt(threshold, 1)} deg/s`, 42, 17);
}

function drawRaster(data) {
  const { ctx, w, h } = canvasContext("eventRaster");
  const report = data.analysis || {};
  const rows = [
    { key: "fixations", label: "fix", color: "#0072b2" },
    { key: "saccades", label: "sac", color: "#e69f00" },
    { key: "microsaccades", label: "micro", color: "#009e73" },
    { key: "psos", label: "pso", color: "#cc79a7" },
  ];
  const hasEvents = rows.some((row) => (report[row.key] || []).length);
  if (!hasEvents) {
    clearPlot(ctx, w, h, { empty: PLOTS.eventRaster });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "time (s)", yLabel: "events" });
  const duration = report.duration_s || 1;
  rows.forEach((row, index) => {
    const y = 32 + index * ((h - 62) / Math.max(1, rows.length - 1));
    ctx.fillStyle = "#607080";
    ctx.fillText(row.label, 6, y + 4);
    (report[row.key] || []).forEach((event) => {
      const x = scale(event.onset_t, 0, duration, 34, w - 12);
      const x2 = scale(event.offset_t, 0, duration, 34, w - 12);
      if (x === null || x2 === null) return;
      ctx.fillStyle = row.color;
      ctx.fillRect(x, y - 8, Math.max(2, x2 - x), 16);
    });
  });
}

function drawScanpath(data) {
  const { ctx, w, h } = canvasContext("scanpath");
  const fix = (data.analysis && data.analysis.fixations) || [];
  if (!fix.length) {
    clearPlot(ctx, w, h, { empty: PLOTS.scanpath });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "x (deg)", yLabel: "y (deg)" });
  const xs = fix.map((f) => f.centroid_x);
  const ys = fix.map((f) => f.centroid_y);
  const [x0, x1] = extent(xs, [-10, 10]);
  const [y0, y1] = extent(ys, [-10, 10]);
  ctx.strokeStyle = "#607080";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  fix.forEach((f, i) => {
    const px = scale(f.centroid_x, x0, x1, 34, w - 12);
    const py = scale(f.centroid_y, y0, y1, h - 24, 12);
    if (px === null || py === null) return;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  fix.forEach((f, i) => {
    const px = scale(f.centroid_x, x0, x1, 34, w - 12);
    const py = scale(f.centroid_y, y0, y1, h - 24, 12);
    if (px === null || py === null) return;
    ctx.fillStyle = "#009e73";
    ctx.beginPath();
    ctx.arc(px, py, 5 + Math.min(10, (f.duration_s || 0) * 20), 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(String(i + 1), px - 3, py + 4);
  });
}

function drawPolar(data) {
  const { ctx, w, h } = canvasContext("directionPolar");
  const saccades = (data.analysis && data.analysis.saccades) || [];
  if (!saccades.length) {
    clearPlot(ctx, w, h, { empty: PLOTS.directionPolar });
    return;
  }
  clearPlot(ctx, w, h, { yLabel: "direction" });
  const bins = new Array(12).fill(0);
  saccades.forEach((s) => {
    const idx = Math.max(0, Math.min(11, Math.floor(((s.direction_deg + 180) / 360) * 12)));
    bins[idx] += 1;
  });
  const cx = w / 2;
  const cy = h / 2 + 4;
  const radiusMax = Math.min(w, h) / 2 - 30;
  ctx.strokeStyle = "#eef2f7";
  [0.33, 0.66, 1].forEach((r) => {
    ctx.beginPath();
    ctx.arc(cx, cy, radiusMax * r, 0, Math.PI * 2);
    ctx.stroke();
  });
  const max = Math.max(1, ...bins);
  bins.forEach((count, i) => {
    const angle = (i / bins.length) * Math.PI * 2 - Math.PI;
    const radius = 14 + (count / max) * radiusMax;
    ctx.strokeStyle = "#009e73";
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.stroke();
  });
  ctx.fillStyle = "#607080";
  ctx.fillText("0° right, +90° up", 42, 17);
}

function drawHistogram(data) {
  const { ctx, w, h } = canvasContext("amplitudeHist");
  const amps = ((data.analysis && data.analysis.saccades) || []).map((s) => s.amplitude_deg);
  if (!amps.length) {
    clearPlot(ctx, w, h, { empty: PLOTS.amplitudeHist });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "amplitude (deg)", yLabel: "count" });
  const bins = 10;
  const [lo, hi] = extent(amps, [0, 1]);
  const counts = new Array(bins).fill(0);
  amps.forEach((amp) => {
    const idx = Math.max(0, Math.min(bins - 1, Math.floor(((amp - lo) / (hi - lo)) * bins)));
    counts[idx] += 1;
  });
  const max = Math.max(1, ...counts);
  counts.forEach((count, i) => {
    const bw = (w - 52) / bins;
    const bh = (count / max) * (h - 46);
    ctx.fillStyle = "#e69f00";
    ctx.fillRect(36 + i * bw, h - 24 - bh, Math.max(2, bw - 3), bh);
  });
}

function drawMainSequence(data) {
  const { ctx, w, h } = canvasContext("mainSequence");
  const saccades = (data.analysis && data.analysis.saccades) || [];
  const amps = saccades.map((s) => s.amplitude_deg);
  const peaks = saccades.map((s) => s.peak_velocity_deg_s);
  if (!amps.length) {
    clearPlot(ctx, w, h, { empty: PLOTS.mainSequence });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "amp (deg)", yLabel: "peak deg/s" });
  const [x0, x1] = extent(amps, [0, 1]);
  const [y0, y1] = extent(peaks, [0, 100]);
  saccades.forEach((s) => {
    const px = scale(s.amplitude_deg, x0, x1, 34, w - 12);
    const py = scale(s.peak_velocity_deg_s, y0, y1, h - 24, 12);
    if (px === null || py === null) return;
    ctx.fillStyle = "#cc79a7";
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawDomainValidation(payload) {
  const { ctx, w, h } = canvasContext("domainValidationPlot");
  const domains = payload?.domains || [];
  if (!domains.length) {
    clearPlot(ctx, w, h, { empty: PLOTS.domainValidationPlot });
    return;
  }
  clearPlot(ctx, w, h, { xLabel: "domain", yLabel: "F1" });
  const plotW = w - 52;
  const plotH = h - 48;
  domains.forEach((domain, i) => {
    const summary = domain.summary || {};
    const f1 = summary.saccade_f1?.mean || 0;
    const ciLow = summary.saccade_f1?.ci_low ?? f1;
    const ciHigh = summary.saccade_f1?.ci_high ?? f1;
    const bw = plotW / domains.length;
    const x = 38 + i * bw + 4;
    const y = 12 + (1 - f1) * plotH;
    const bh = (f1 * plotH);
    ctx.fillStyle = i % 2 ? "#009e73" : "#0072b2";
    ctx.fillRect(x, y, Math.max(10, bw - 10), bh);
    const yLow = 12 + (1 - ciLow) * plotH;
    const yHigh = 12 + (1 - ciHigh) * plotH;
    ctx.strokeStyle = "#18212f";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + (bw - 10) / 2, yHigh);
    ctx.lineTo(x + (bw - 10) / 2, yLow);
    ctx.stroke();
    ctx.fillStyle = "#607080";
    ctx.save();
    ctx.translate(x + 2, h - 29);
    ctx.rotate(-Math.PI / 5);
    ctx.fillText(domain.label || domain.domain, 0, 0);
    ctx.restore();
  });
  ctx.fillStyle = "#607080";
  ctx.fillText("mean F1 with bootstrap interval", 42, 17);
}

function setTableCell(row, text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  row.appendChild(cell);
}

function renderValidation(payload) {
  state.syntheticValidation = payload;
  const cross = payload.cross_domain || {};
  ui.macroF1.textContent = fmt(cross.macro_saccade_f1, 3);
  ui.worstDomain.textContent = cross.worst_domain || "--";
  ui.stabilityGap.textContent = fmt(cross.stability_gap_f1, 3);
  ui.truthBoundary.textContent = payload.truth_boundary || "Synthetic validation has truth labels.";
  const tbody = ui.domainValidationTable.querySelector("tbody");
  tbody.innerHTML = "";
  (payload.domains || []).forEach((domain) => {
    const row = document.createElement("tr");
    const summary = domain.summary || {};
    setTableCell(row, domain.label || domain.domain);
    setTableCell(row, fmt(summary.saccade_f1?.mean, 3));
    setTableCell(row, `${fmt(summary.amplitude_mae_deg?.mean, 2)}°`);
    setTableCell(row, fmt(summary.pupil_valid_fraction?.mean, 3));
    tbody.appendChild(row);
  });
  drawDomainValidation(payload);
  updateButtons();
}

function resetValidation() {
  state.syntheticValidation = null;
  ui.macroF1.textContent = "--";
  ui.worstDomain.textContent = "--";
  ui.stabilityGap.textContent = "--";
  ui.truthBoundary.textContent = "Live webcam diagnostics are plausibility checks, not reference-device truth.";
  const tbody = ui.domainValidationTable.querySelector("tbody");
  tbody.innerHTML = "";
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 4;
  cell.textContent = "Run synthetic validation";
  row.appendChild(cell);
  tbody.appendChild(row);
  emptyPlot("domainValidationPlot", PLOTS.domainValidationPlot);
  updateButtons();
}

function renderDiagnostics(diagnostics = {}) {
  const q = Number(diagnostics.quality_index);
  ui.qualityIndex.textContent = Number.isFinite(q) ? `${fmt(q, 0)}/100` : "--";
  ui.qualityStatus.textContent = Number.isFinite(q) ? `quality: ${fmt(q, 0)}` : "quality: --";
  setChipTone(ui.qualityStatus, q >= 75 ? "ok" : q >= 45 ? "warn" : q > 0 ? "bad" : "");
  ui.samplingCv.textContent = fmt(diagnostics.sampling_interval_cv, 3);
  ui.gazePath.textContent = Number.isFinite(diagnostics.gaze_path_length_deg)
    ? `${fmt(diagnostics.gaze_path_length_deg, 1)}°`
    : "--";
  ui.pupilValid.textContent = fmt(diagnostics.pupil_valid_fraction, 3);
  const warnings = diagnostics.warnings || [];
  ui.liveWarnings.textContent = warnings.length ? warnings.join(", ") : "none";
  if (diagnostics.truth_boundary) {
    ui.truthBoundary.textContent = diagnostics.truth_boundary;
  }
}

function renderStatistics(statistics = {}, method = {}) {
  const windowS = Number(method.rolling_window_s);
  ui.statsWindow.textContent = Number.isFinite(windowS)
    ? `rolling window: ${fmt(windowS, 0)} s`
    : "rolling window: --";
  ui.sampleRate.textContent = fmt(stat(statistics, "recording", "sample_rate_hz"), 1);
  const medianDt = stat(statistics, "recording", "median_interval_s");
  ui.medianDt.textContent = Number.isFinite(medianDt) ? `${fmt(medianDt * 1000, 1)} ms` : "--";
  ui.saccadeRate.textContent = fmt(stat(statistics, "events", "saccade_rate_per_min"), 1);
  ui.fixationDwell.textContent = pct(stat(statistics, "events", "fixation_dwell_fraction"), 0);
  ui.pathEfficiency.textContent = pct(stat(statistics, "gaze", "path_efficiency"), 0);
  const dispersion = stat(statistics, "gaze", "dispersion_deg");
  ui.dispersion.textContent = Number.isFinite(dispersion) ? `${fmt(dispersion, 2)}°` : "--";
  const bcea = stat(statistics, "gaze", "bcea_deg2");
  ui.bcea.textContent = Number.isFinite(bcea) ? `${fmt(bcea, 2)} deg²` : "--";
  ui.transitionEntropy.textContent = fmt(
    stat(statistics, "events", "direction_transition_entropy_bits"),
    2,
  );
}

function calibrationLabel(active = state.calibrationActive) {
  const status = active ? "on" : "off";
  return `calibration: ${status} (±${fmt(state.calibrationRangeDeg, 0)}°)`;
}

function resetDisplay({ keepCalibration = false } = {}) {
  state.latest = null;
  state.hasSamples = false;
  state.finiteSampleCount = 0;
  if (!keepCalibration) state.calibrationActive = false;
  ui.eyeCrop.removeAttribute("src");
  ui.eyeCrop.style.display = "none";
  ui.eyeOverlay.style.display = "none";
  ui.eyePlaceholder.style.display = "grid";
  ui.eyePlaceholder.textContent = "Start camera";
  ui.cameraStatus.textContent = `camera: ${ui.camera.value || "--"}`;
  ui.faceStatus.textContent = "face: --";
  ui.fpsStatus.textContent = "fps: --";
  ui.sampleStatus.textContent = "samples: 0";
  ui.qualityStatus.textContent = "quality: --";
  setChipTone(ui.faceStatus);
  setChipTone(ui.sampleStatus);
  setChipTone(ui.qualityStatus);
  ui.gazeX.textContent = "--";
  ui.gazeY.textContent = "--";
  ui.calGazeX.textContent = "--";
  ui.calGazeY.textContent = "--";
  ui.pupilSize.textContent = "--";
  ui.sampleCount.textContent = "0";
  ui.duration.textContent = "0.00";
  ui.finiteFraction.textContent = "--";
  ui.detectionThreshold.textContent = "--";
  ui.saccadeCount.textContent = "0";
  ui.fixationCount.textContent = "0";
  ui.microCount.textContent = "0";
  ui.pursuitCount.textContent = "0";
  ui.psoCount.textContent = "0";
  ui.qualityIndex.textContent = "--";
  ui.samplingCv.textContent = "--";
  ui.gazePath.textContent = "--";
  ui.pupilValid.textContent = "--";
  ui.liveWarnings.textContent = "--";
  ui.statsWindow.textContent = "rolling window: --";
  ui.sampleRate.textContent = "--";
  ui.medianDt.textContent = "--";
  ui.saccadeRate.textContent = "--";
  ui.fixationDwell.textContent = "--";
  ui.pathEfficiency.textContent = "--";
  ui.dispersion.textContent = "--";
  ui.bcea.textContent = "--";
  ui.transitionEntropy.textContent = "--";
  ui.calibrationStatus.textContent = calibrationLabel();
  setChipTone(ui.calibrationStatus, state.calibrationActive ? "ok" : "");
  resetPlots();
  updateButtons();
}

function resetControls() {
  ui.camera.value = DEFAULTS.camera;
  ui.method.value = DEFAULTS.method;
  ui.threshold.value = DEFAULTS.threshold;
  ui.window.value = DEFAULTS.window;
  ui.pso.checked = DEFAULTS.pso;
}

function render(data) {
  if (data.type === "error") {
    setStatus(data.message || "socket error", "bad");
    updateButtons();
    return;
  }
  state.latest = data;
  state.hasSamples = (data.analysis?.n_samples || 0) > 0;
  const xs = Array.isArray(data.series?.x) ? data.series.x : [];
  const ys = Array.isArray(data.series?.y) ? data.series.y : [];
  state.finiteSampleCount = xs.reduce(
    (count, x, index) => count + (Number.isFinite(x) && Number.isFinite(ys[index]) ? 1 : 0),
    0,
  );
  state.calibrationActive = Boolean(data.calibration?.active);
  state.calibrationRangeDeg = data.calibration?.target_range_deg || state.calibrationRangeDeg;

  if (data.frame?.eye_crop_jpeg) {
    ui.eyeCrop.src = data.frame.eye_crop_jpeg;
    ui.eyeCrop.style.display = "block";
    ui.eyeOverlay.style.display = "block";
    ui.eyePlaceholder.style.display = "none";
  }
  ui.cameraStatus.textContent = `camera: ${ui.camera.value}`;
  const faceDetected = Boolean(data.capture.quality && data.capture.quality.face_detected);
  ui.faceStatus.textContent = faceDetected ? "face: detected" : "face: not detected";
  setChipTone(ui.faceStatus, faceDetected ? "ok" : "warn");
  ui.fpsStatus.textContent = `fps: ${fmt(data.capture.fps_estimate_hz, 1)}`;
  ui.sampleStatus.textContent = `samples: ${data.analysis.n_samples || 0}`;
  setChipTone(ui.sampleStatus, state.hasSamples ? "ok" : "warn");
  ui.gazeX.textContent = fmt(data.capture.gaze.x);
  ui.gazeY.textContent = fmt(data.capture.gaze.y);
  const cal = data.capture.calibrated_gaze || {};
  ui.calGazeX.textContent = fmt(cal.x);
  ui.calGazeY.textContent = fmt(cal.y);
  ui.calibrationStatus.textContent = calibrationLabel();
  setChipTone(ui.calibrationStatus, state.calibrationActive ? "ok" : "");
  const pupil = data.capture.pupil || {};
  ui.pupilSize.textContent = `${fmt(pupil.size, 3)} ${pupil.unit || ""}`;
  ui.sampleCount.textContent = data.analysis.n_samples || 0;
  ui.duration.textContent = fmt(data.analysis.duration_s, 2);
  ui.finiteFraction.textContent = fmt(
    data.diagnostics?.finite_gaze_fraction ?? data.analysis.quality?.finite_sample_fraction,
    3,
  );
  ui.detectionThreshold.textContent = fmt(
    data.analysis.quality?.detection_threshold_deg_s || data.method.velocity_threshold_deg_s,
    1,
  );
  ui.saccadeCount.textContent = data.analysis.n_saccades || 0;
  ui.fixationCount.textContent = data.analysis.n_fixations || 0;
  ui.microCount.textContent = data.analysis.n_microsaccades || 0;
  ui.pursuitCount.textContent = data.analysis.n_smooth_pursuits || 0;
  ui.psoCount.textContent = data.analysis.n_psos || 0;
  renderDiagnostics(data.diagnostics || {});
  renderStatistics(data.statistics || {}, data.method || {});
  if (!faceDetected) {
    setStatus("live: no face detected", "warn");
  } else if ((data.analysis.n_samples || 0) < 3) {
    setStatus("live: collecting samples", "warn");
  } else if ((data.analysis.quality?.finite_sample_fraction ?? 1) < 0.85) {
    setStatus("live: low valid fraction", "warn");
  } else {
    setStatus("live", "ok");
  }

  const t = values(data.series, "t");
  const gazeSeries = [
    { data: values(data.series, "x"), color: "#0072b2", label: "x" },
    { data: values(data.series, "y"), color: "#e69f00", label: "y" },
  ];
  const calibratedX = values(data.series, "calibrated_x");
  const calibratedY = values(data.series, "calibrated_y");
  if (calibratedX.length && calibratedY.length) {
    gazeSeries.push({ data: calibratedX, color: "#d55e00", label: "cal x" });
    gazeSeries.push({ data: calibratedY, color: "#009e73", label: "cal y" });
  }
  multiLinePlot("gazeTime", t, gazeSeries);
  linePlot("pupilTrace", t, values(data.series, "pupil"), "#009e73", "relative pupil");
  drawGazeXY(data);
  drawSpeed(data);
  drawRaster(data);
  drawScanpath(data);
  drawPolar(data);
  drawHistogram(data);
  drawMainSequence(data);
  updateButtons();
}

function liveUrl() {
  const params = new URLSearchParams({
    camera: ui.camera.value,
    method: ui.method.value,
    velocity_threshold: ui.threshold.value,
    include_pso: ui.pso.checked ? "true" : "false",
    window_s: ui.window.value,
  });
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/live?${params.toString()}`;
}

function closeSocket(message = "stopped") {
  if (state.socket) {
    state.socket.close();
    state.socket = null;
  }
  state.isConnecting = false;
  state.isLive = false;
  setStatus(message, "warn");
  updateButtons();
}

function start() {
  closeSocket("connecting");
  state.isConnecting = true;
  setStatus("connecting", "warn");
  updateButtons();
  const socket = new WebSocket(liveUrl());
  state.socket = socket;
  socket.onopen = () => {
    state.isConnecting = false;
    state.isLive = true;
    setStatus("live: waiting for frame", "warn");
    updateButtons();
  };
  socket.onmessage = (event) => {
    try {
      render(JSON.parse(event.data));
    } catch {
      setStatus("socket error: unreadable live payload", "bad");
    }
  };
  socket.onclose = () => {
    if (state.socket === socket) state.socket = null;
    state.isConnecting = false;
    state.isLive = false;
    setStatus("stopped", "warn");
    updateButtons();
  };
  socket.onerror = () => {
    setStatus("socket error: camera permission, hardware, or backend dependency", "bad");
    state.isConnecting = false;
    state.isLive = false;
    updateButtons();
  };
}

function stop() {
  closeSocket("stopped");
}

async function postJson(path, body = null) {
  const response = await fetch(path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { detail: text || response.statusText };
  }
  if (!response.ok) {
    throw new Error(payload.detail || "request failed");
  }
  return payload;
}

async function exportSession() {
  if (!state.outputConfigured) {
    setOutput("output dir not configured; restart with --output-dir", "warn");
    return;
  }
  if (!state.hasSamples) {
    setOutput("no samples captured yet", "warn");
    return;
  }
  try {
    const body = await postJson("/api/export");
    const paths = Object.keys(body.paths || {}).length;
    setOutput(paths ? `exported ${paths} files` : "exported", "ok");
  } catch (error) {
    setOutput(error.message || "export unavailable", "bad");
  }
}

function calibrationTargets() {
  const r = state.calibrationRangeDeg;
  return [
    { x: -r, y: -r },
    { x: r, y: -r },
    { x: -r, y: r },
    { x: r, y: r },
  ];
}

async function fitCalibration() {
  if (!state.latest || state.finiteSampleCount < 4) {
    ui.calibrationStatus.textContent = "calibration: need at least 4 finite gaze samples";
    setChipTone(ui.calibrationStatus, "warn");
    return;
  }
  try {
    const body = await postJson("/api/calibration/fit", { targets: calibrationTargets() });
    state.calibrationActive = Boolean(body.calibration);
    ui.calibrationStatus.textContent = calibrationLabel(true);
    setChipTone(ui.calibrationStatus, "ok");
    updateButtons();
  } catch (error) {
    ui.calibrationStatus.textContent = error.message || "calibration fit failed";
    setChipTone(ui.calibrationStatus, "bad");
  }
}

async function resetCalibration() {
  try {
    await postJson("/api/calibration/reset");
    state.calibrationActive = false;
    ui.calibrationStatus.textContent = calibrationLabel(false);
    setChipTone(ui.calibrationStatus);
    ui.calGazeX.textContent = "--";
    ui.calGazeY.textContent = "--";
    updateButtons();
  } catch (error) {
    ui.calibrationStatus.textContent = error.message || "calibration reset failed";
    setChipTone(ui.calibrationStatus, "bad");
  }
}

async function clearSession() {
  closeSocket("clearing session");
  try {
    const body = await postJson("/api/session/clear");
    state.hasSamples = body.sample_count > 0;
    state.calibrationActive = Boolean(body.calibration_active);
    resetDisplay({ keepCalibration: state.calibrationActive });
    setStatus("session cleared", "ok");
    setOutput(
      state.outputConfigured ? "session cleared; capture samples before export" : "output: memory, export disabled",
      "ok",
    );
  } catch (error) {
    setStatus(error.message || "clear session failed", "bad");
  }
}

async function resetAll() {
  closeSocket("resetting");
  try {
    await postJson("/api/session/reset");
    resetControls();
    state.calibrationActive = false;
    resetDisplay();
    resetValidation();
    setStatus("reset all", "ok");
    setOutput(
      state.outputConfigured ? "reset; capture samples before export" : "output: memory, export disabled",
      "ok",
    );
  } catch (error) {
    setStatus(error.message || "reset failed", "bad");
  }
}

async function runSyntheticValidation() {
  const repetitions = Math.max(1, Math.min(25, Number(ui.validationRepeats.value || 5)));
  ui.validationRepeats.value = String(repetitions);
  state.validationBusy = true;
  setOutput(`running synthetic validation (${repetitions} repeats/domain)`, "warn");
  updateButtons();
  try {
    const params = new URLSearchParams({ repetitions: String(repetitions), first_seed: "0" });
    const payload = await fetch(`/api/validation/synthetic?${params.toString()}`).then((response) => {
      if (!response.ok) throw new Error(`validation failed (${response.status})`);
      return response.json();
    });
    renderValidation(payload);
    setOutput(`synthetic validation complete: ${payload.domain_count} domains`, "ok");
  } catch (error) {
    setOutput(error.message || "synthetic validation failed", "bad");
  } finally {
    state.validationBusy = false;
    updateButtons();
  }
}

async function init() {
  resetPlots();
  resetValidation();
  try {
    const status = await fetch("/api/status").then((r) => r.json());
    ui.camera.value = status.camera_index;
    state.outputConfigured = Boolean(status.output_dir);
    state.calibrationRangeDeg = status.calibration?.target_range_deg || state.calibrationRangeDeg;
    state.calibrationActive = Boolean(status.calibration?.active);
    const sampleCount = status.sample_count || 0;
    resetDisplay({ keepCalibration: state.calibrationActive });
    state.hasSamples = sampleCount > 0;
    ui.sampleStatus.textContent = `samples: ${sampleCount}`;
    ui.sampleCount.textContent = sampleCount;
    setChipTone(ui.sampleStatus, state.hasSamples ? "ok" : "");
    setOutput(
      status.output_dir ? `output: ${status.output_dir}` : "output: memory, export disabled",
      status.output_dir ? "ok" : "warn",
    );
    setStatus("idle");
  } catch {
    setStatus("status unavailable", "bad");
    setOutput("backend status unavailable", "bad");
  }
  updateButtons();
}

ui.start.addEventListener("click", start);
ui.stop.addEventListener("click", stop);
ui.clearSession.addEventListener("click", clearSession);
ui.fitCalibration.addEventListener("click", fitCalibration);
ui.resetCalibration.addEventListener("click", resetCalibration);
ui.resetAll.addEventListener("click", resetAll);
ui.export.addEventListener("click", exportSession);
ui.runSyntheticValidation.addEventListener("click", runSyntheticValidation);
ui.clearValidation.addEventListener("click", resetValidation);
window.addEventListener("resize", () => {
  if (state.latest) render(state.latest);
  else resetPlots();
});
init();
