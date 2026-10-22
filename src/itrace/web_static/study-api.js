export class StudyApi {
  constructor() {
    this.epoch = crypto.randomUUID();
    this.token = "";
    this.syncId = null;
    this.uncertainty = null;
    this.offset = null;
    this.onStatus = () => {};
  }

  async get(path) {
    const response = await fetch(`/api/study/${path}`, {cache: "no-store"});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Study request failed.");
    return body;
  }

  async post(action, payload = {}) {
    const response = await fetch(`/api/study/${action}`, {
      method: "POST", headers: {"Content-Type": "application/json", "X-iTrace-Control": this.token},
      body: JSON.stringify(payload), keepalive: action === "event" || action === "abort",
    });
    const body = await response.json();
    if (body.status) this.onStatus(body);
    if (!response.ok) throw new Error(body.detail || "Study control failed.");
    return body.result;
  }

  async align() {
    let best = null;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const exchange = await this.post("sync", {client_send_ms: performance.now(), client_epoch: this.epoch});
      const confirmed = await this.post("confirm-sync", {
        sync_id: exchange.sync_id, client_received_ms: performance.now(), client_epoch: this.epoch,
      });
      if (!best || confirmed.error_bound_s < best.error_bound_s) best = confirmed;
    }
    this.syncId = best.sync_id;
    this.uncertainty = best.error_bound_s;
    this.offset = best.offset_s;
    return best;
  }

  async event(kind, detail = {}, browserMs = performance.now()) {
    if (!this.syncId) throw new Error("Clock alignment is required before presentation.");
    return this.post("event", {
      event_id: crypto.randomUUID(), client_epoch: this.epoch, sync_id: this.syncId,
      browser_ms: browserMs, kind, ...detail,
    });
  }
}

export function rectangle(element) {
  const {x, y, width, height} = element.getBoundingClientRect();
  return {x, y, width, height};
}

export function afterPaint() {
  return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}
