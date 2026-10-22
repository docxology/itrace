from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from itrace.live.study_server import create_study_app
from itrace.live.study_workspace import StudyWorkspace
from itrace.study import StudyError
from itrace.study.jsonio import encoded, mapping, text

from .helpers import Harness, setup_payload


def workspace(harness: Harness, tmp_path: Path) -> StudyWorkspace:
    pytest.importorskip("fastapi")
    app = create_study_app(manifest_path=tmp_path / "study.json", output_root=tmp_path / "sessions")
    return app.state.study_controller


def test_new_session_saves_terminal_records_once_when_request_retried(
    harness: Harness, tmp_path: Path
) -> None:
    from itrace.live.study_control import StudyAction

    owner = workspace(harness, tmp_path)
    original = mapping(owner.status()["status"], "status")
    session_id = text(original["session_id"], "session_id")
    owner.dispatch(StudyAction.ABORT, encoded({"reason": "stopped"}), session_id)
    journal = Path(text(original["output_dir"], "path")) / "journal.jsonl"
    before = journal.read_bytes()
    payload = encoded({"request_id": "retry-1"})
    first = owner.dispatch(StudyAction.NEW_SESSION, payload, session_id)
    second = owner.dispatch(StudyAction.NEW_SESSION, payload, session_id)
    assert first == second
    assert len(list((tmp_path / "sessions").iterdir())) == 2
    assert journal.read_bytes() == before
    result = mapping(first["result"], "result")
    assert (Path(text(result["saved_export_dir"], "export")) / "report.json").is_file()
    fresh = mapping(first["status"], "status")
    assert fresh["session_id"] != session_id
    assert fresh["state"] == "setup"


def test_terminal_save_and_stop_are_idempotent(harness: Harness, tmp_path: Path) -> None:
    from itrace.live.study_control import StudyAction

    owner = workspace(harness, tmp_path)
    status = mapping(owner.status()["status"], "status")
    session_id = text(status["session_id"], "session_id")
    owner.dispatch(StudyAction.ABORT, encoded({"reason": "stopped"}), session_id)
    journal = Path(text(status["output_dir"], "path")) / "journal.jsonl"
    original = journal.read_bytes()
    owner.dispatch(StudyAction.ABORT, encoded({"reason": "stopped"}), session_id)
    first = owner.dispatch(StudyAction.SAVE, b"{}", session_id)
    second = owner.dispatch(StudyAction.SAVE, b"{}", session_id)
    assert first == second
    assert journal.read_bytes() == original
    assert len(list(journal.parent.glob("export-*"))) == 1


def test_new_session_rejects_active_or_stale_controls(harness: Harness, tmp_path: Path) -> None:
    from itrace.live.study_control import StudyAction

    owner = workspace(harness, tmp_path)
    session_id = text(mapping(owner.status()["status"], "status")["session_id"], "id")
    with pytest.raises(StudyError, match="terminal"):
        owner.dispatch(StudyAction.NEW_SESSION, encoded({"request_id": "one"}), session_id)
    with pytest.raises(StudyError, match="session"):
        owner.dispatch(StudyAction.ABORT, encoded({"reason": "stale"}), "other-session")
    assert mapping(owner.status()["status"], "status")["state"] == "setup"


def test_prepare_retry_retains_exact_initial_setup(harness: Harness, tmp_path: Path) -> None:
    from itrace.live.study_control import StudyAction

    owner = workspace(harness, tmp_path)
    status = mapping(owner.status()["status"], "status")
    session_id = text(status["session_id"], "id")
    payload = encoded(setup_payload())
    owner.dispatch(StudyAction.PREPARE, payload, session_id)
    journal = Path(text(status["output_dir"], "path")) / "journal.jsonl"
    original = journal.read_bytes()
    owner.dispatch(StudyAction.PREPARE, payload, session_id)
    assert journal.read_bytes() == original
    changed = setup_payload()
    changed["distance_m"] = 0.7
    with pytest.raises(StudyError):
        owner.dispatch(StudyAction.PREPARE, encoded(changed), session_id)


def test_concurrent_restart_retries_create_only_one_replacement(
    harness: Harness, tmp_path: Path
) -> None:
    from itrace.live.study_control import StudyAction

    owner = workspace(harness, tmp_path)
    session_id = text(mapping(owner.status()["status"], "status")["session_id"], "id")
    owner.dispatch(StudyAction.ABORT, encoded({"reason": "stopped"}), session_id)

    def restart(_: int) -> dict:
        return owner.dispatch(StudyAction.NEW_SESSION, encoded({"request_id": "same"}), session_id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(restart, range(8)))
    assert all(answer == answers[0] for answer in answers)
    assert len(list((tmp_path / "sessions").iterdir())) == 2


@pytest.mark.parametrize("payload", [{}, {"request_id": 1}, {"request_id": "one", "extra": True}])
def test_restart_rejects_bad_identities_without_new_files(
    harness: Harness, tmp_path: Path, payload: dict
) -> None:
    from itrace.live.study_control import StudyAction

    owner = workspace(harness, tmp_path)
    session_id = text(mapping(owner.status()["status"], "status")["session_id"], "id")
    owner.dispatch(StudyAction.ABORT, encoded({"reason": "stopped"}), session_id)
    with pytest.raises(StudyError):
        owner.dispatch(StudyAction.NEW_SESSION, encoded(payload), session_id)
    assert len(list((tmp_path / "sessions").iterdir())) == 1


def test_http_session_binding_rejects_missing_or_stale_identity(
    harness: Harness, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    app = create_study_app(manifest_path=tmp_path / "study.json", output_root=tmp_path / "http")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        context = client.get("/api/study/context").json()
        headers = {"X-iTrace-Control": context["token"]}
        for identity in (None, "stale"):
            chosen = headers if identity is None else {**headers, "X-iTrace-Session": identity}
            result = client.post("/api/study/abort", headers=chosen, json={"reason": "stale"})
            assert result.status_code == 400
        assert client.get("/api/study/status").json()["status"]["state"] == "setup"
