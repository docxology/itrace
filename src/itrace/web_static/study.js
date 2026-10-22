import {StudyApi, afterPaint, rectangle} from "/static/study-api.js";
import {collectTargets} from "/static/study-targets.js";

const api = new StudyApi();
const el = id => document.getElementById(id);
let manifest, current, camera, prepared = false, busy = false, cancelled = false;
let activeTrial = null, deadlineTimer = null, imageUrl = null, polling = false;

function message(text, tone = "neutral") {
  el("message").textContent = text;
  el("message").dataset.tone = tone;
}

function render(body) {
  current = body.status;
  camera = body.camera;
  el("phaseLabel").textContent = current.state.replaceAll("_", " ");
  el("outputPath").textContent = current.output_dir;
  el("trialProgress").textContent = `${current.trial_index} / ${current.trial_count}`;
  el("frameCount").textContent = camera.frames_recorded;
  el("cameraState").textContent = camera.error || (camera.active ? "Capturing" : camera.reason);
  el("syncError").textContent = api.uncertainty == null ? "Not measured" : `±${(api.uncertainty * 1000).toFixed(1)} ms`;
  const validated = current.calibration?.heldout_target_count === 5 && current.calibration.heldout_p95_fraction <= manifest.limits.max_validation_error_fraction;
  const postAdequate = current.calibration?.post_validation_adequate;
  el("calibrationState").textContent = typeof postAdequate === "boolean" ? (postAdequate ? "Pre / post checks adequate" : "Post-check inadequate") : validated ? "Held-out check passed" : current.target?.phase || "Not validated";
  el("progressFill").style.width = `${100 * current.trial_index / current.trial_count}%`;
  el("captureButton").disabled = busy || camera.running || !prepared || cancelled;
  el("calibrateButton").disabled = busy || !current.capture_active || !current.target || cancelled;
  el("calibrateButton").textContent = current.state === "post_validation" ? "Finish gaze validation" : "Calibrate and validate";
  const restDone = current.state === "rest" && performance.now() / 1000 + (api.offset || 0) >= current.rest_until_s;
  el("nextButton").disabled = busy || cancelled || (!current.ready && !restDone);
  el("nextButton").textContent = current.state === "rest" && !restDone ? "Rest before next trial" : "Start next trial";
  el("exportButton").textContent = current.state === "complete" ? "Save completed session" : "Save partial session";
  el("exportButton").disabled = busy;
  if (camera.error) message(`Capture stopped: ${camera.error}. Save the partial session.`, "error");
  if (["interrupted", "aborted"].includes(current.state)) {
    cancelled = true;
    clearTimeout(deadlineTimer);
    activeTrial = null;
    document.body.classList.remove("collecting");
    el("targetOverlay").hidden = true;
    lockChoices();
    message(`Session ${current.state}. Save the partial records; scored trials cannot resume.`, "warning");
  }
}
api.onStatus = render;

async function action(callback) {
  if (busy) return;
  busy = true;
  if (current && camera) render({status: current, camera});
  try { await callback(); }
  catch (error) { message(error.message, "error"); }
  finally { busy = false; if (current && camera) render({status: current, camera}); }
}

async function prepare(event) {
  event.preventDefault();
  if (!el("setupForm").reportValidity()) return;
  await action(async () => {
    await document.documentElement.requestFullscreen();
    const entered = performance.now();
    let lastChange = entered, size = `${innerWidth}x${innerHeight}`;
    while (performance.now() - entered < 5000) {
      await afterPaint();
      const nextSize = `${innerWidth}x${innerHeight}`;
      if (nextSize !== size) { size = nextSize; lastChange = performance.now(); }
      if (performance.now() - entered >= 1000 && performance.now() - lastChange >= 300) break;
    }
    if (!document.fullscreenElement || performance.now() - lastChange < 300) throw new Error("Fullscreen geometry did not settle. Try preparation again.");
    const data = new FormData(el("setupForm"));
    await api.post("prepare", {
      participant_id: data.get("participant_id"), device_id: data.get("device_id"),
      display_width_m: Number(data.get("display_width_mm")) / 1000,
      display_height_m: Number(data.get("display_height_mm")) / 1000,
      distance_m: Number(data.get("distance_mm")) / 1000,
      screen_width_px: Number(data.get("screen_width_px")), screen_height_px: Number(data.get("screen_height_px")),
      dpr: devicePixelRatio, viewport_width_px: innerWidth, viewport_height_px: innerHeight,
      browser: navigator.userAgent, client_epoch: api.epoch,
      storage_acknowledged: data.get("storage_acknowledged") === "on",
      geometry_confirmed: data.get("geometry_confirmed") === "on", fullscreen: Boolean(document.fullscreenElement),
    });
    prepared = true;
    el("setupForm").hidden = true;
    el("taskArea").hidden = false;
    await api.align();
    message("Setup saved. Start the webcam and check face / gaze availability before calibration.");
  });
}

function lockChoices() {
  for (const button of el("responses").querySelectorAll("button")) button.disabled = true;
}

async function finishTrial(kind, key, timestamp = performance.now()) {
  const trial = activeTrial;
  if (!trial) return;
  activeTrial = null;
  lockChoices();
  clearTimeout(deadlineTimer);
  try {
    const detail = {trial_id: trial.trial_id};
    if (key != null) detail.key = key;
    const result = await api.event(kind, detail, timestamp);
    document.body.classList.remove("collecting");
    el("stimulus").hidden = true;
    el("query").textContent = "";
    el("responses").replaceChildren();
    el("stageInstruction").hidden = false;
    el("stageInstruction").querySelector("h2").textContent = current.state === "post_validation" ? "Final gaze check" : "Trial recorded";
    el("stageInstruction").querySelector("p").textContent = current.state === "post_validation" ? "Responses are recorded. Finish the gaze validation before saving the completed session." : "Continue when ready. Keep your viewing position unchanged.";
    const feedback = typeof result.practice_correct === "boolean" ? ` Practice feedback: ${result.practice_correct ? "correct" : "incorrect"}.` : "";
    message(`${kind === "timeout" ? "Deadline recorded." : "Response recorded."}${feedback}`);
  } catch (error) {
    message(`Trial acknowledgement failed: ${error.message}. Stop and save the partial session.`, "error");
    await interrupt("response acknowledgement failed");
  }
}

async function startTrial() {
  await action(async () => {
    await api.align();
    const trial = manifest.trials.find(item => item.trial_id === current.next_trial_id);
    if (!trial) throw new Error("The next trial is not in the bound manifest.");
    const response = await fetch(`/api/study/stimulus/${encodeURIComponent(trial.trial_id)}`, {cache: "no-store"});
    if (!response.ok) throw new Error("The bound stimulus could not be loaded.");
    const bytes = await response.arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(x => x.toString(16).padStart(2, "0")).join("");
    if (digest !== trial.stimulus.sha256) throw new Error("Stimulus hash mismatch.");
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = URL.createObjectURL(new Blob([bytes], {type: "image/png"}));
    const image = el("stimulus");
    image.src = imageUrl;
    await image.decode();
    if (image.naturalWidth !== trial.stimulus.width || image.naturalHeight !== trial.stimulus.height) throw new Error("Stimulus dimensions differ from the protocol.");
    await api.post("trial", {trial_id: trial.trial_id});
    document.body.classList.add("collecting");
    el("stageInstruction").hidden = true;
    image.hidden = false;
    el("query").textContent = trial.query;
    el("responses").replaceChildren(...trial.options.map(option => {
      const button = document.createElement("button");
      button.type = "button";
      button.disabled = true;
      const key = document.createElement("kbd"); key.textContent = option.key;
      button.append(key, document.createTextNode(option.label));
      button.addEventListener("click", () => finishTrial("response", option.key));
      return button;
    }));
    window.scrollTo(0, 0);
    const onsetMs = await afterPaint();
    await api.event("trial_onset", {trial_id: trial.trial_id, rect: rectangle(image), presentation: "raf"}, onsetMs);
    activeTrial = trial;
    for (const button of el("responses").querySelectorAll("button")) button.disabled = false;
    deadlineTimer = setTimeout(() => finishTrial("timeout", null), Math.max(0, onsetMs + (trial.deadline_s + 2 * api.uncertainty + .1) * 1000 - performance.now()));
  });
}

async function interrupt(reason) {
  if (!prepared || cancelled || ["complete", "aborted", "interrupted"].includes(current?.state)) return;
  cancelled = true;
  clearTimeout(deadlineTimer);
  activeTrial = null;
  document.body.classList.remove("collecting");
  el("targetOverlay").hidden = true;
  lockChoices();
  try { await api.event("interrupt", {reason}); }
  catch (error) { message(`Interruption could not be acknowledged: ${error.message}`, "error"); }
  try { await api.post("capture-stop"); }
  catch (error) { message(`Capture stop failed: ${error.message}`, "error"); }
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    if (prepared && !cancelled && !["complete", "aborted", "interrupted"].includes(current?.state)) await api.post("heartbeat", {client_epoch: api.epoch});
    render(await api.get("status"));
    if (camera.running && !activeTrial && !busy) {
      const preview = await api.get("preview");
      if (preview.available) {
        el("eyePreview").hidden = false;
        el("eyePreview").src = preview.eye_crop;
        el("faceState").textContent = `${preview.face_detected ? "Face" : "No face"} / ${preview.gaze_valid ? "finite gaze" : "missing gaze"}`;
      }
    }
  } catch (error) { message(`Server unavailable: ${error.message}. Retain the session directory.`, "error"); }
  finally { polling = false; }
}

el("setupForm").addEventListener("submit", prepare);
el("captureButton").addEventListener("click", () => action(async () => {
  await api.post("capture-start"); message("Capture requested. Wait for finite face-valid samples, then calibrate.");
}));
el("calibrateButton").addEventListener("click", () => action(async () => {
  message("Look at each target until it moves. Calibration and held-out validation are separate.");
  await collectTargets(api, current, render, () => cancelled);
  if (current.state === "complete") {
    await api.post("capture-stop");
    const adequate = current.calibration?.post_validation_adequate;
    el("stageInstruction").querySelector("h2").textContent = "Session complete";
    message(adequate ? "Post-session gaze check recorded. Save the completed session." : "Post-session gaze quality was inadequate. Save the records; behavioral responses remain separate.", adequate ? "success" : "warning");
  } else message(current.ready ? "Held-out validation passed. You can start practice." : "Calibration did not establish readiness.", current.ready ? "success" : "warning");
}));
el("nextButton").addEventListener("click", startTrial);
el("abortButton").addEventListener("click", async () => {
  cancelled = true; clearTimeout(deadlineTimer); activeTrial = null;
  el("targetOverlay").hidden = true;
  lockChoices();
  try { await api.post("abort", {reason: "participant stopped the session"}); }
  catch (error) { message(`Stop acknowledgement failed: ${error.message}. Retain the partial records.`, "error"); }
});
el("targetStopButton").addEventListener("click", () => el("abortButton").click());
el("exportButton").addEventListener("click", () => action(async () => {
  const result = await api.post("export");
  el("exportPath").textContent = result.export_dir;
  message("Private export saved. Review completeness, timing and gaze quality in the report.", "success");
}));
document.addEventListener("keydown", event => {
  if (event.key === "Escape") { el("abortButton").click(); return; }
  if (!activeTrial || event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
  const option = activeTrial.options.find(item => item.key.toLowerCase() === event.key.toLowerCase());
  if (option) { event.preventDefault(); finishTrial("response", option.key, performance.now()); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden) interrupt("tab hidden"); });
window.addEventListener("resize", () => interrupt("viewport resized"));
document.addEventListener("fullscreenchange", () => { if (prepared && !document.fullscreenElement) interrupt("fullscreen ended"); });
window.addEventListener("pagehide", () => interrupt("page closed or reloaded"));

try {
  const context = await api.get("context");
  api.token = context.token; manifest = context.manifest; render(context);
  el("studyTitle").textContent = manifest.title;
  el("practiceCount").textContent = manifest.trials.filter(x => x.role === "practice").length;
  el("scoredCount").textContent = manifest.trials.filter(x => x.role === "scored").length;
  if (current.state !== "setup") {
    el("setupForm").hidden = true;
    message("This session belongs to an earlier page. Stop and export it; scored trials cannot resume after reload.", "warning");
  } else message("Private protocol loaded. Complete the measured setup to begin.");
  setInterval(poll, 500);
} catch (error) { message(`Cannot load the study: ${error.message}`, "error"); }
