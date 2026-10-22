import {afterPaint, rectangle} from "/static/study-api.js";

export async function collectTargets(api, status, update, interrupted) {
  const overlay = document.getElementById("targetOverlay");
  const dot = document.getElementById("targetDot");
  const instruction = document.getElementById("targetInstruction");
  overlay.hidden = false;
  try {
    let current = status;
    while (current.target && !interrupted()) {
      await api.align();
      const target = current.target;
      const position = {x: innerWidth * target.x_fraction, y: innerHeight * target.y_fraction};
      dot.style.left = `${position.x}px`;
      dot.style.top = `${position.y}px`;
      const label = target.phase === "post_validation" ? "Post-session validation" : target.phase === "validation" ? "Independent validation" : "Calibration";
      instruction.textContent = `${label}: look at the dot. Escape stops the session.`;
      const onsetMs = await afterPaint();
      await api.event("target_onset", {
        target_id: target.target_id, rect: rectangle(dot), position,
      }, onsetMs);
      const untilMs = onsetMs + (target.settle_s + target.collect_s + 2 * api.uncertainty + .05) * 1000;
      await new Promise(resolve => setTimeout(resolve, Math.max(0, untilMs - performance.now())));
      if (interrupted()) break;
      await api.event("target_collect", {target_id: target.target_id});
      const body = await api.get("status");
      current = body.status;
      update(body);
    }
  } finally {
    overlay.hidden = true;
  }
}
