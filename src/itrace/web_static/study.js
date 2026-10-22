import {StudyApi, afterPaint, rectangle} from "/static/study-api.js";
import {collectTargets} from "/static/study-targets.js";
import {enterFullscreen, refreshGeometry, setupPayload} from "/static/study-setup.js";
import {protocolGuide, qualityGuidance, savedSummary} from "/static/study-guidance.js";

const api = new StudyApi();
const el = id => document.getElementById(id);
let manifest, current, camera, prepared = false, busy = false, cancelled = false;
let activeTrial = null, deadlineTimer = null, imageUrl = null, polling = false;
let restartKey = null, restarting = false, boundGeometry = null;

function message(text, tone = "neutral") {
  el("message").textContent = text;
  el("message").dataset.tone = tone;
}

function render(body) {
  if (api.sessionId && body.status.session_id !== api.sessionId) return;
  const previousState = current?.state;
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
  const sampleReady = current.state === "post_validation" || current.capture_quality?.ready_to_calibrate;
  el("calibrateButton").disabled = busy || !current.capture_active || !current.target || !sampleReady || cancelled;
  el("calibrateButton").textContent = current.state === "post_validation" ? "Finish gaze validation" : "Calibrate and validate";
  const restDone = current.state === "rest" && performance.now() / 1000 + (api.offset || 0) >= current.rest_until_s;
  el("nextButton").disabled = busy || cancelled || (!current.ready && !restDone);
  const restSeconds = Math.max(0, Math.ceil((current.rest_until_s || 0) - performance.now() / 1000 - (api.offset || 0)));
  el("nextButton").textContent = current.state === "rest" && !restDone ? `Rest · ${restSeconds}s remaining` : "Start next trial";
  el("exportButton").textContent = current.state === "complete" ? "Save completed session" : "Save partial session";
  el("exportButton").disabled = busy;
  el("prepareButton").disabled = busy;
  el("restartButton").disabled = busy;
  el("abortButton").disabled = restarting || ["setup", "complete", "aborted", "interrupted"].includes(current.state);
  if (camera.error) message(`Capture stopped: ${camera.error}. Save the partial session.`, "error");
  qualityGuidance(current, manifest.limits);
  if (current.state === "complete") el("recoveryPanel").hidden = false;
  if (["interrupted", "aborted"].includes(current.state)) {
    cancelled = true;
    clearTimeout(deadlineTimer);
    activeTrial = null;
    document.body.classList.remove("collecting");
    el("targetOverlay").hidden = true;
    el("eyePreview").hidden = true;
    lockChoices();
    el("taskArea").hidden = true;
    el("setupForm").hidden = true;
    el("recoveryPanel").hidden = false;
    if (previousState !== current.state) message(`Session ${current.state}. Your records remain private. Save them and start a fresh session below.`, "warning");
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
    const detected = await enterFullscreen();
    await api.post("prepare", setupPayload(api, detected));
    boundGeometry = `${innerWidth}x${innerHeight}x${devicePixelRatio}`;
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
  document.body.classList.remove("collecting");
  el("targetOverlay").hidden = true;
  lockChoices();
  try { await api.post("abort", {reason: "participant stopped the session"}); }
  catch (error) { message(`Stop acknowledgement failed: ${error.message}. Retain the partial records.`, "error"); }
});
el("targetStopButton").addEventListener("click", () => el("abortButton").click());
el("exportButton").addEventListener("click", () => action(async () => {
  const result = await api.post("save");
  el("exportPath").textContent = result.export_dir;
  savedSummary(result.report);
  message("Private export saved. Review completeness, timing and gaze quality in the report.", "success");
}));
el("restartButton").addEventListener("click", () => action(async () => {
  restarting = true;
  prepared = false;
  cancelled = true;
  clearTimeout(deadlineTimer);
  activeTrial = null;
  try {
    if (!["complete", "aborted", "interrupted"].includes(current.state)) await api.post("abort", {reason: "new session requested after page reload"});
    restartKey ??= crypto.randomUUID();
    const result = await api.post("new-session", {request_id: restartKey});
    api.epoch = crypto.randomUUID(); api.syncId = null; api.offset = null; api.uncertainty = null;
    cancelled = false; restartKey = null; boundGeometry = null;
    el("recoveryPanel").hidden = true;
    el("setupForm").hidden = false;
    el("taskArea").hidden = true;
    el("savedSummary").hidden = true;
    el("setupForm").elements.storage_acknowledged.checked = false;
    el("setupForm").elements.geometry_confirmed.checked = false;
    el("exportPath").textContent = `Previous session saved: ${result.saved_export_dir}`;
    refreshGeometry();
    message("Fresh session created. Check your measurements, confirm storage and prepare again. Previous records are saved.", "success");
    el("setupForm").elements.participant_id.focus();
  } finally { restarting = false; }
}));
document.addEventListener("keydown", event => {
  if (event.key === "Escape") { el("abortButton").click(); return; }
  if (!activeTrial || event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
  const option = activeTrial.options.find(item => item.key.toLowerCase() === event.key.toLowerCase());
  if (option) { event.preventDefault(); finishTrial("response", option.key, performance.now()); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden) interrupt("tab hidden"); });
window.addEventListener("resize", () => {
  if (!prepared) refreshGeometry();
  else if (boundGeometry !== `${innerWidth}x${innerHeight}x${devicePixelRatio}`) interrupt("viewport resized");
});
document.addEventListener("fullscreenchange", () => { if (prepared && !document.fullscreenElement) interrupt("fullscreen ended"); });
window.addEventListener("pagehide", () => interrupt("page closed or reloaded"));

try {
  const context = await api.get("context");
  api.token = context.token; api.sessionId = context.status.session_id;
  manifest = context.manifest; render(context);
  protocolGuide(manifest);
  el("setupForm").elements.device_id.value = `webcam-${camera.camera_index}`;
  refreshGeometry();
  el("studyTitle").textContent = manifest.title;
  el("practiceCount").textContent = manifest.trials.filter(x => x.role === "practice").length;
  el("scoredCount").textContent = manifest.trials.filter(x => x.role === "scored").length;
  if (current.state !== "setup") {
    el("setupForm").hidden = true;
    el("recoveryPanel").hidden = false;
    message("This session belongs to an earlier page. Save it and start fresh below; scored trials cannot resume after reload.", "warning");
  } else message("Private protocol loaded. Complete the measured setup to begin.");
  setInterval(poll, 500);
} catch (error) { message(`Cannot load the study: ${error.message}`, "error"); }
