import {afterPaint} from "/static/study-api.js";

export function geometry() {
  const dpr = devicePixelRatio;
  return {
    screen_width_px: Math.round(screen.width * dpr),
    screen_height_px: Math.round(screen.height * dpr),
    dpr, viewport_width_px: innerWidth, viewport_height_px: innerHeight,
  };
}

export function refreshGeometry() {
  const value = geometry();
  const form = document.getElementById("setupForm");
  form.elements.screen_width_px.value = value.screen_width_px;
  form.elements.screen_height_px.value = value.screen_height_px;
  document.getElementById("geometrySummary").textContent =
    `Window ${innerWidth} × ${innerHeight} CSS px · display ${value.screen_width_px} × ${value.screen_height_px} estimated device px · scale ${value.dpr}×`;
  return value;
}

export async function enterFullscreen() {
  if (!document.fullscreenEnabled) throw new Error("Fullscreen is unavailable here. Open this address in a normal browser window and prepare again.");
  if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
  const entered = performance.now();
  let lastChange = entered, size = `${innerWidth}x${innerHeight}`;
  while (performance.now() - entered < 5000) {
    await afterPaint();
    const nextSize = `${innerWidth}x${innerHeight}`;
    if (nextSize !== size) { size = nextSize; lastChange = performance.now(); }
    if (performance.now() - entered >= 1000 && performance.now() - lastChange >= 300) break;
  }
  if (!document.fullscreenElement || performance.now() - lastChange < 300) throw new Error("Fullscreen geometry did not settle. Try preparation again.");
  return refreshGeometry();
}

export function setupPayload(api, detected) {
  const data = new FormData(document.getElementById("setupForm"));
  return {
    participant_id: data.get("participant_id"), device_id: data.get("device_id"),
    display_width_m: Number(data.get("display_width_mm")) / 1000,
    display_height_m: Number(data.get("display_height_mm")) / 1000,
    distance_m: Number(data.get("distance_mm")) / 1000,
    ...detected, browser: navigator.userAgent, client_epoch: api.epoch,
    storage_acknowledged: data.get("storage_acknowledged") === "on",
    geometry_confirmed: data.get("geometry_confirmed") === "on", fullscreen: Boolean(document.fullscreenElement),
  };
}
