import json
import struct
import zlib
from hashlib import sha256
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from itrace.live.study_server import create_study_app
from itrace.study import StudyError


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    def chunk(kind: bytes, content: bytes) -> bytes:
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content))
        )

    pixels = b"\0\xff\xff\xff\x00\x00\x00" * 2
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    (tmp_path / "stimulus.png").write_bytes(image)
    payload = {
        "schema": "itrace.private-study",
        "version": 1,
        "study_id": "server-fixture",
        "title": "Synthetic engineering fixture",
        "limits": {
            "max_sync_error_s": 0.05,
            "sync_max_age_s": 60,
            "max_event_lag_s": 2,
            "max_validation_error_fraction": 0.15,
            "settle_s": 0.5,
            "collect_s": 1,
            "min_samples": 10,
            "min_valid_fraction": 0.6,
        },
        "trials": [
            {
                "trial_id": "trial-1",
                "role": "scored",
                "block_id": "block-1",
                "task_id": "choice",
                "family_id": "scene-1",
                "stimulus": {
                    "path": "stimulus.png",
                    "sha256": sha256(image).hexdigest(),
                    "width": 2,
                    "height": 2,
                },
                "query": "Select the first response",
                "options": [{"key": "1", "label": "First"}, {"key": "2", "label": "Second"}],
                "correct_key": "1",
                "deadline_s": 10,
            }
        ],
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload))
    return path


def test_study_routes_serve_frozen_images_and_no_answer_key(
    manifest_path: Path, tmp_path: Path
) -> None:
    app = create_study_app(manifest_path=manifest_path, output_root=tmp_path / "records")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/").url.path == "/study"
        page = client.get("/study")
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-store"
        assert page.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        for name in ("study.js", "study-api.js", "study-targets.js", "study.css"):
            assert client.get(f"/static/{name}").status_code == 200
        context = client.get("/api/study/context").json()
        assert "correct_key" not in json.dumps(context)
        assert context["status"]["state"] == "setup"
        assert context["camera"]["running"] is False
        original = (tmp_path / "stimulus.png").read_bytes()
        (tmp_path / "stimulus.png").write_bytes(b"changed after binding")
        image = client.get("/api/study/stimulus/trial-1")
        assert image.content == original
        assert image.headers["content-type"] == "image/png"
        assert client.get("/api/study/stimulus/missing").status_code == 400
        assert client.get("/api/study/preview").json() == {"available": False}


@pytest.mark.parametrize(
    ("headers", "action", "body", "expected"),
    [
        ({}, "prepare", "{}", 401),
        ({"X-iTrace-Control": "wrong"}, "prepare", "{}", 401),
        ({"Origin": "https://foreign.invalid"}, "abort", "{}", 403),
        ({"Sec-Fetch-Site": "cross-site"}, "abort", "{}", 403),
        ({"Content-Type": "text/plain"}, "prepare", "{}", 415),
        ({}, "unknown", "{}", 404),
        ({}, "prepare", "{", 400),
        ({}, "prepare", "[]", 400),
        ({}, "prepare", '{"participant_id":1,"participant_id":2}', 400),
        ({}, "export", '{"overwrite":true}', 400),
        ({}, "prepare", " " * 1_048_577, 413),
    ],
    ids=[
        "missing-token",
        "wrong-token",
        "foreign-origin",
        "cross-site",
        "wrong-type",
        "unknown-action",
        "broken-json",
        "array-json",
        "duplicate-json",
        "unexpected-field",
        "body-limit",
    ],
)
def test_control_boundary_rejects_forged_or_malformed_requests(
    manifest_path: Path,
    tmp_path: Path,
    headers: dict[str, str],
    action: str,
    body: str,
    expected: int,
) -> None:
    app = create_study_app(manifest_path=manifest_path, output_root=tmp_path / "records")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/study/context").json()["token"]
        accepted = {"X-iTrace-Control": token, "Content-Type": "application/json"}
        if expected == 401:
            accepted.pop("X-iTrace-Control")
        response = client.post(
            f"/api/study/{action}", headers={**accepted, **headers}, content=body
        )
        assert response.status_code == expected
        assert client.get("/api/study/status").json()["status"]["state"] == "setup"


def test_partial_export_and_abort_are_explicit_and_never_start_camera(
    manifest_path: Path,
    tmp_path: Path,
) -> None:
    app = create_study_app(manifest_path=manifest_path, output_root=tmp_path / "records")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        context = client.get("/api/study/context").json()
        headers = {"X-iTrace-Control": context["token"]}
        first = client.post("/api/study/export", headers=headers, json={})
        assert first.status_code == 200
        result = first.json()["result"]
        assert Path(result["export_dir"]).is_dir()
        second = client.post("/api/study/export", headers=headers, json={})
        assert second.status_code == 200
        assert second.json()["result"]["export_dir"] != result["export_dir"]
        abort = client.post("/api/study/abort", headers=headers, json={"reason": "fixture stop"})
        assert abort.status_code == 200
        assert abort.json()["status"]["state"] == "aborted"
        assert abort.json()["camera"]["running"] is False


def test_study_server_rejects_absent_manifest_before_camera_access(tmp_path: Path) -> None:
    with pytest.raises(StudyError, match=r"manifest|read|file"):
        create_study_app(manifest_path=tmp_path / "missing.json", output_root=tmp_path / "records")


@pytest.mark.parametrize(
    "host",
    [
        "attacker.invalid:8773",
        "127.0.0.1.attacker.invalid",
        "localhost@attacker.invalid",
        "[::1].attacker.invalid",
        "localhost:999999",
    ],
)
def test_study_rejects_untrusted_host_before_exposing_context_or_controls(
    manifest_path: Path, tmp_path: Path, host: str
) -> None:
    app = create_study_app(manifest_path=manifest_path, output_root=tmp_path / "records")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for path in ("/api/study/context", "/study", "/static/study.js"):
            response = client.get(path, headers={"Host": host})
            assert response.status_code == 400
            assert "token" not in response.text
        context = client.get("/api/study/context").json()
        response = client.post(
            "/api/study/abort",
            headers={
                "Host": host,
                "Origin": f"http://{host}",
                "X-iTrace-Control": context["token"],
            },
            json={"reason": "forged host"},
        )
        assert response.status_code == 400
        assert client.get("/api/study/status").json()["status"]["state"] == "setup"


def test_study_server_is_separate_from_historical_manifest_registration() -> None:
    import inspect

    source = inspect.getsource(create_study_app)
    assert "empirical_sessions_manifest" not in source
    assert "LiveState" not in source


def test_study_assets_are_packaged_with_explicit_status_and_keyboard_controls() -> None:
    root = Path(__file__).resolve().parents[1] / "src/itrace/web_static"
    page = (root / "study.html").read_text()
    script = (root / "study.js").read_text()
    assert '<html lang="en">' in page
    assert 'aria-live="polite"' in page
    assert "study.css" in page
    assert "study.js" in page
    assert "visibilitychange" in script
    assert "resize" in script
    assert "Escape" in script
    assert "afterPaint()" in script
    assert "innerHTML" not in script
    assert "correct_key" not in script
