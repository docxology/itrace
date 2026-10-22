export function protocolGuide(manifest) {
  const list = document.getElementById("protocolGuide");
  const families = new Map();
  for (const trial of manifest.trials) {
    if (!families.has(trial.task_id)) families.set(trial.task_id, new Map());
    const queries = families.get(trial.task_id);
    queries.set(trial.query, trial.options.map(option => `${option.key}: ${option.label}`).join(" · "));
  }
  for (const [task, queries] of families) {
    const item = document.createElement("li");
    const heading = document.createElement("strong");
    heading.textContent = task.replaceAll("_", " ");
    item.append(heading);
    for (const [query, keys] of queries) {
      const rule = document.createElement("p"); rule.textContent = query;
      const choices = document.createElement("p"); choices.className = "quiet"; choices.textContent = keys;
      item.append(rule, choices);
    }
    list.append(item);
  }
}

export function qualityGuidance(status, limits) {
  const quality = status.capture_quality;
  const label = document.getElementById("gazeQuality");
  label.textContent = quality ? `${quality.valid_samples} / ${quality.recent_samples} valid in ${quality.window_s}s` : "Not observed";
  const hint = document.getElementById("readinessHint");
  if (!status.capture_active) hint.textContent = "Start the webcam, then wait for face and finite gaze before calibration.";
  else if (status.state === "post_validation") hint.textContent = "Finish the five post-session checks even if gaze quality declined. Inadequate gaze remains explicit in your saved report.";
  else if (status.target && !quality?.ready_to_calibrate) hint.textContent = `Waiting for at least ${limits.min_samples} fresh finite gaze samples and ${(limits.min_valid_fraction * 100).toFixed(0)}% valid samples. Keep your face framed and well lit.`;
  else if (status.target) hint.textContent = "Fresh gaze samples are available. Follow each dot through calibration and independent validation.";
  else if (status.ready) hint.textContent = "Calibration passed. Keep the window and viewing position unchanged; start the next trial when ready.";
  else hint.textContent = "Trial and rest controls follow the frozen protocol. Escape stops the session.";
}

export function savedSummary(report) {
  const section = document.getElementById("savedSummary");
  const values = document.getElementById("savedValues");
  const count = outcome => report.trials.filter(trial => trial.outcome === outcome).length;
  const answered = role => report.trials.filter(trial => trial.role === role && trial.outcome === "answered").length;
  const items = [
    ["Saved state", report.state],
    ["Practice answers", answered("practice")], ["Scored answers", answered("scored")],
    ["Timeouts", count("timeout")], ["Unattempted", count("unattempted")],
    ["Recorded frames", report.capture.native_rows],
    ["Finite gaze frames", `${report.capture.finite_gaze_rows} / ${report.capture.native_rows}`],
    ["Capture checks", report.capture.integrity_ok ? "Passed declared checks" : `Review: ${report.capture.integrity_defects.join(", ")}`],
    ["Post-session gaze", report.calibration_diagnostics.post_validation_adequate == null ? "Not completed" : report.calibration_diagnostics.post_validation_adequate ? "Adequate for declared threshold" : "Inadequate"],
  ];
  values.replaceChildren(...items.map(([label, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt"); term.textContent = label;
    const detail = document.createElement("dd"); detail.textContent = value;
    row.append(term, detail); return row;
  }));
  section.hidden = false;
}
