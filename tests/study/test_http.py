import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from itrace.cli import app as cli
from itrace.live.capture_service import CaptureService
from itrace.live.study_connection import BrowserConnection
from itrace.live.study_control import StudyController
from itrace.live.study_routes import install_study_routes
from itrace.study.jsonio import mapping, number

from .helpers import Clock, Harness


def test_http_session_calibration_responses_omission_postcheck_and_export(harness: Harness) -> None:
    capture = CaptureService(lambda camera, limit: iter(()), lambda *args: None, lambda *args: None)
    controller = StudyController(harness.manifest, harness.session, capture, "local-control", 0)
    app = FastAPI()
    install_study_routes(app, controller)
    with TestClient(app) as client:

        def send(action: str, payload: dict) -> dict:
            response = client.post(
                f"/api/study/{action}", json=payload, headers={"X-iTrace-Control": "local-control"}
            )
            assert response.status_code == 200, response.text
            return response.json()

        def target() -> None:
            item = mapping(harness.session.status()["target"], "target")
            x, y = number(item["x_fraction"], "x") * 1000, number(item["y_fraction"], "y") * 800
            send(
                "event",
                harness.event_payload(
                    "target_onset",
                    target_id=item["target_id"],
                    position={"x": x, "y": y},
                    rect={"x": x - 8, "y": y - 8, "width": 16, "height": 16},
                ),
            )
            onset = harness.clock.value
            for index in range(20):
                harness.clock.value = onset + 0.51 + index * 0.9 / 19
                harness.sample((x - 100) / 20, (y - 50) / -15)
            harness.clock.value = onset + 1.51
            send("event", harness.event_payload("target_collect", target_id=item["target_id"]))
            harness.clock.value += 0.01

        send("heartbeat", {"client_epoch": "browser1"})
        assert (
            client.post(
                "/api/study/heartbeat",
                json={"client_epoch": "foreign"},
                headers={"X-iTrace-Control": "local-control"},
            ).status_code
            == 400
        )
        for _ in range(14):
            target()
        assert controller.session.status()["ready"] is True
        for index in range(3):
            send("trial", {"trial_id": f"trial{index}"})
            send(
                "event",
                harness.event_payload(
                    "trial_onset",
                    trial_id=f"trial{index}",
                    rect={"x": 100, "y": 100, "width": 500, "height": 500},
                    presentation="raf",
                ),
            )
            harness.clock.value += 0.5 if index < 2 else 3.1
            event = (
                harness.event_payload("response", trial_id=f"trial{index}", key="a")
                if index < 2
                else harness.event_payload("timeout", trial_id=f"trial{index}")
            )
            send("event", event)
            assert (
                client.post(
                    "/api/study/event", json=event, headers={"X-iTrace-Control": "local-control"}
                ).status_code
                == 400
            )
        assert controller.session.status()["state"] == "post_validation"
        for _ in range(5):
            target()
        exported = send("export", {})["result"]
        assert exported["report"]["state"] == "complete"
        assert exported["report"]["calibration_diagnostics"]["post_validation_adequate"] is True
        assert Path(exported["export_dir"]).is_dir()
        send("capture-stop", {})


def test_browser_connection_expires_only_after_ten_seconds() -> None:
    clock = Clock()
    connection = BrowserConnection(clock)
    assert not connection.expired()
    connection.renew()
    clock.value += 10
    assert not connection.expired()
    clock.value += 0.001
    assert connection.expired()
    connection.renew()
    assert not connection.expired()


@pytest.mark.parametrize("options", [["--host", "0.0.0.0"], ["--port", "0"], ["--camera", "-1"]])
def test_cli_rejects_nonprivate_or_invalid_server_settings(
    tmp_path: Path, options: list[str]
) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "study",
            "--manifest",
            str(tmp_path / "absent.json"),
            "--output-dir",
            str(tmp_path),
            *options,
        ],
    )
    assert result.exit_code == 2
    assert "study failed:" in result.output


def test_cli_report_retains_partial_session_and_refuses_output_collision(
    harness: Harness, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"
    command = ["study-report", str(harness.directory), "--out", str(out)]
    result = CliRunner().invoke(cli, command)
    assert result.exit_code == 0, result.output
    original = out.read_bytes()
    assert json.loads(original)["state"] == "incomplete"
    assert CliRunner().invoke(cli, command).exit_code == 2
    assert out.read_bytes() == original
