from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest

from itrace.capture import CaptureSample, EyeBox, LiveFrameSample, WebcamSource
from itrace.live.capture_service import (
    CaptureBusyError,
    CaptureParameterError,
    CaptureService,
    CaptureStopTimeout,
)
from itrace.types import GazeSample


def sample(index: int, timestamp: float) -> LiveFrameSample:
    return LiveFrameSample(
        CaptureSample(index, timestamp, GazeSample(timestamp, 1.0, 2.0)),
        640,
        480,
        EyeBox(0, 0, 10, 10),
        "",
    )


@dataclass(slots=True)
class Records:
    """Mutable callback sink accumulates observations from the real worker."""

    rows: list[tuple[LiveFrameSample, str, float]] = field(default_factory=list)
    states: list[tuple[bool, str, str | None]] = field(default_factory=list)
    terminal: threading.Event = field(default_factory=threading.Event)

    def frame(self, frame: LiveFrameSample, epoch: str, receipt: float) -> None:
        self.rows.append((frame, epoch, receipt))

    def state(self, active: bool, reason: str, error: str | None) -> None:
        self.states.append((active, reason, error))
        if not active:
            self.terminal.set()


def finished(service: CaptureService, records: Records) -> None:
    assert records.terminal.wait(2.0)
    service.stop()
    json.dumps(service.status(), allow_nan=False)


def test_records_every_native_frame_before_overwriting_preview() -> None:
    # Given: discontinuous native clocks and duplicate frame identifiers.
    records = Records()
    frames = [sample(i, t) for i, t in ((0, 1.0), (1, 2.0), (1, 0.0), (2, 0.0))]
    calls: list[tuple[int, int | None, int]] = []

    def factory(camera: int, limit: int | None) -> Iterable[LiveFrameSample]:
        calls.append((camera, limit, threading.get_ident()))
        return iter(frames)

    service = CaptureService(factory, records.frame, records.state)
    assert not calls
    assert service.latest() is None
    # When: capture runs independently of preview consumption.
    before = time.monotonic()
    epoch = service.start(3, 4)
    finished(service, records)
    # Then: all frames retain their native values, with host receipt metadata.
    assert [(row[0], row[1]) for row in records.rows] == [(f, epoch) for f in frames]
    assert all(before <= row[2] <= time.monotonic() for row in records.rows)
    assert calls == [(3, 4, calls[0][2])]
    assert calls[0][2] != threading.get_ident()
    assert service.latest() is frames[-1]
    assert service.status()["preview_dropped"] == 3
    assert service.status()["frames_recorded"] == 4
    assert records.states == [(True, "active", None), (False, "completed", None)]


def test_blocked_read_keeps_status_responsive_and_timeout_retains_owner() -> None:
    # Given: a real generator paused inside next(), with an observable finally.
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    records = Records()

    def factory(camera: int, limit: int | None) -> Iterator[LiveFrameSample]:
        try:
            entered.set()
            assert release.wait(2.0)
            yield sample(0, 0.0)
        finally:
            closed.set()

    service = CaptureService(factory, records.frame, records.state)
    epoch = service.start(0)
    assert entered.wait(2.0)
    # When: stop reaches its bounded deadline while acquisition remains blocked.
    try:
        before = time.monotonic()
        with pytest.raises(CaptureStopTimeout):
            service.stop(0.01)
        # Then: status/preview are responsive and restart cannot orphan the camera.
        assert time.monotonic() - before < 0.5
        assert service.status()["running"] is True
        assert service.status()["reason"] == "stop_timeout"
        assert service.status()["epoch"] == epoch
        assert service.latest() is None
        with pytest.raises(CaptureBusyError):
            service.start(1)
        assert not closed.is_set()
    finally:
        release.set()
        finished(service, records)
    assert closed.is_set()
    assert len(records.rows) == 1


def test_restart_has_distinct_epoch_and_detached_status() -> None:
    # Given: an exhausted source can safely relinquish ownership.
    records = Records()

    def factory(camera: int, limit: int | None) -> Iterator[LiveFrameSample]:
        yield sample(0, 7.0)

    service = CaptureService(factory, records.frame, records.state)
    first = service.start(0)
    finished(service, records)
    previous = service.status()
    records.terminal.clear()
    # When: a fresh capture starts after full cleanup.
    second = service.start(0)
    finished(service, records)
    # Then: epochs differ and status mutations cannot alter service state.
    assert first != second
    previous["reason"] = "tampered"
    assert service.status()["reason"] == "completed"
    assert [row[1] for row in records.rows] == [first, second]


@pytest.mark.parametrize(
    "failure", ["factory", "source", "frame", "active", "terminal", "close", "system_exit"]
)
def test_foreign_failure_is_explicit_and_cleanup_runs(failure: str) -> None:
    # Given: real code can fail at each worker boundary.
    records, closed = Records(), threading.Event()
    terminal_called = threading.Event()

    def factory(camera: int, limit: int | None) -> Iterable[LiveFrameSample]:
        if failure == "factory":
            raise LookupError("factory fault")

        def frames() -> Iterator[LiveFrameSample]:
            try:
                yield sample(0, 0.0)
                if failure == "source":
                    raise LookupError("source fault")
                if failure == "system_exit":
                    raise SystemExit("source exit")
            finally:
                closed.set()
                if failure == "close":
                    raise LookupError("close fault")

        return frames()

    def frame(value: LiveFrameSample, epoch: str, receipt: float) -> None:
        if failure in ("frame", "close"):
            raise LookupError("frame fault")
        records.frame(value, epoch, receipt)

    def state(active: bool, reason: str, error: str | None) -> None:
        if not active:
            terminal_called.set()
        if (active and failure == "active") or (not active and failure == "terminal"):
            raise LookupError("state fault")
        records.state(active, reason, error)

    service = CaptureService(factory, frame, state)
    # When: the foreign exception escapes its callback or iterator.
    service.start(0)
    assert terminal_called.wait(2.0)
    service.stop()
    # Then: retained typed failure replaces apparent success; recording never skips ahead.
    status = service.status()
    assert status["reason"] == "failed"
    assert ("SystemExit" if failure == "system_exit" else "LookupError") in (status["error"] or "")
    assert status["running"] is False
    if failure in ("frame", "close", "active", "factory"):
        assert service.latest() is None
    assert closed.is_set() == (failure not in ("factory", "active"))
    if failure == "close":
        assert "frame fault" in (status["error"] or "")
        assert "close fault" in (status["error"] or "")


def test_worker_can_request_stop_and_closes_generator() -> None:
    # Given: callback-driven stop must never join the current thread.
    records, closed = Records(), threading.Event()

    def factory(camera: int, limit: int | None) -> Iterator[LiveFrameSample]:
        try:
            yield sample(0, 1.0)
            pytest.fail("capture advanced after stop")
        finally:
            closed.set()

    def frame(value: LiveFrameSample, epoch: str, receipt: float) -> None:
        assert service.latest() is None
        records.frame(value, epoch, receipt)
        service.stop()

    service = CaptureService(factory, frame, records.state)
    # When: the worker requests cooperative shutdown.
    service.start(0)
    finished(service, records)
    # Then: cleanup completes and exactly one record survives.
    assert closed.is_set()
    assert service.status()["reason"] == "stopped"
    assert len(records.rows) == 1


def test_simultaneous_start_invokes_exactly_one_factory() -> None:
    records, release = Records(), threading.Event()
    barrier, calls = threading.Barrier(2), []

    def factory(camera: int, limit: int | None) -> Iterator[LiveFrameSample]:
        calls.append(camera)
        assert release.wait(2.0)
        yield sample(0, 0.0)

    service = CaptureService(factory, records.frame, records.state)

    def attempt() -> str | None:
        barrier.wait(2.0)
        try:
            return service.start(0)
        except CaptureBusyError:
            return None

    try:
        with ThreadPoolExecutor(2) as callers:
            epochs = [
                future.result(2.0) for future in (callers.submit(attempt), callers.submit(attempt))
            ]
        assert sum(epoch is not None for epoch in epochs) == 1
    finally:
        release.set()
        finished(service, records)
    assert calls == [0]


@pytest.mark.parametrize("broken_iterator", [False, True])
def test_iterable_and_iterator_cleanup_remain_owned(broken_iterator: bool) -> None:
    records = Records()
    calls: list[tuple[str, int]] = []

    class Source:
        def __iter__(self) -> Iterator[LiveFrameSample]:
            if broken_iterator:
                raise LookupError("iterator fault")

            def frames() -> Iterator[LiveFrameSample]:
                try:
                    yield sample(0, 0.0)
                    pytest.fail("limit must prevent another read")
                finally:
                    calls.append(("iterator", threading.get_ident()))

            return frames()

        def close(self) -> None:
            calls.append(("source", threading.get_ident()))
            raise LookupError("source close fault")

    service = CaptureService(lambda camera, limit: Source(), records.frame, records.state)
    service.start(0, 1)
    finished(service, records)
    assert [label for label, ident in calls] == (
        ["source"] if broken_iterator else ["iterator", "source"]
    )
    assert all(ident != threading.get_ident() for label, ident in calls)
    assert "source close fault" in (service.status()["error"] or "")
    if broken_iterator:
        assert "iterator fault" in (service.status()["error"] or "")


@pytest.mark.parametrize("value", [True, -1, 1.5, "0", 2**40])
def test_rejects_invalid_camera_before_factory(value: int) -> None:
    service = CaptureService(lambda camera, limit: (), Records().frame, Records().state)
    with pytest.raises(CaptureParameterError):
        service.start(value)
    assert service.status()["epoch"] is None


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1", 2**100])
def test_rejects_invalid_limit_before_factory(value: int) -> None:
    service = CaptureService(lambda camera, limit: (), Records().frame, Records().state)
    with pytest.raises(CaptureParameterError):
        service.start(0, value)


@pytest.mark.parametrize("value", [True, -1.0, float("nan"), float("inf"), 10**1000])
def test_rejects_invalid_stop_timeout(value: float) -> None:
    service = CaptureService(lambda camera, limit: (), Records().frame, Records().state)
    with pytest.raises(CaptureParameterError):
        service.stop(value)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 10**1000])
def test_rejects_invalid_timestamp_origin_before_optional_imports(value: float) -> None:
    with pytest.raises(ValueError, match="timestamp_origin_s"):
        WebcamSource(timestamp_origin_s=value)
