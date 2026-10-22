"""Four-timestamp browser-ms to host-monotonic-s exchanges with uncertainty."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from .jsonio import JsonValue, StudyError, fields, number, text
from .model import Limits


@dataclass(frozen=True, slots=True)
class Exchange:
    sync_id: str
    client_epoch: str
    client_send_ms: float
    server_received_s: float
    server_sent_s: float
    client_received_ms: float | None = None
    offset_s: float | None = None
    error_bound_s: float | None = None

    def payload(self) -> dict[str, JsonValue]:
        return {
            "sync_id": self.sync_id,
            "client_epoch": self.client_epoch,
            "client_send_ms": self.client_send_ms,
            "server_received_s": self.server_received_s,
            "server_sent_s": self.server_sent_s,
            "client_received_ms": self.client_received_ms,
            "offset_s": self.offset_s,
            "error_bound_s": self.error_bound_s,
        }


def begin(data: Mapping[str, JsonValue], epoch: str, received: float, sent: float) -> Exchange:
    fields(data, "client_send_ms client_epoch")
    if text(data["client_epoch"], "client_epoch", identifier=True) != epoch:
        raise StudyError("client_epoch", "stale browser epoch")
    return Exchange(
        uuid4().hex, epoch, number(data["client_send_ms"], "client_send_ms"), received, sent
    )


def confirm(data: Mapping[str, JsonValue], exchange: Exchange, limits: Limits) -> Exchange:
    fields(data, "sync_id client_received_ms client_epoch")
    if data["client_epoch"] != exchange.client_epoch or exchange.offset_s is not None:
        raise StudyError("sync_id", "stale epoch or duplicate confirmation")
    received = number(data["client_received_ms"], "client_received_ms")
    network = (received - exchange.client_send_ms) / 1000 - (
        exchange.server_sent_s - exchange.server_received_s
    )
    if network < -1e-9 or network / 2 > limits.max_sync_error_s:
        raise StudyError("sync", "negative RTT or excessive uncertainty")
    offset = (
        (exchange.server_received_s - exchange.client_send_ms / 1000)
        + (exchange.server_sent_s - received / 1000)
    ) / 2
    return Exchange(
        exchange.sync_id,
        exchange.client_epoch,
        exchange.client_send_ms,
        exchange.server_received_s,
        exchange.server_sent_s,
        received,
        offset,
        max(0.0, network / 2),
    )


def event_time(
    data: Mapping[str, JsonValue], exchange: Exchange, now: float, limits: Limits
) -> tuple[float, float]:
    if exchange.offset_s is None or exchange.error_bound_s is None:
        raise StudyError("sync_id", "sync not confirmed")
    if data["client_epoch"] != exchange.client_epoch:
        raise StudyError("client_epoch", "stale browser epoch")
    if now - exchange.server_sent_s > limits.sync_max_age_s:
        raise StudyError("sync_id", "sync expired")
    browser = number(data["browser_ms"], "browser_ms")
    if exchange.client_received_ms is None or browser < exchange.client_received_ms:
        raise StudyError("browser_ms", "event precedes confirmed exchange")
    estimate = browser / 1000 + exchange.offset_s
    if estimate - exchange.error_bound_s > now + 1e-9:
        raise StudyError("browser_ms", "event is in the future")
    if now - estimate > limits.max_event_lag_s + exchange.error_bound_s:
        raise StudyError("browser_ms", "event acknowledgement too late")
    return estimate, exchange.error_bound_s
