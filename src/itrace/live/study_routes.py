from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from ..study import JsonValue, StudyError
from .capture_service import CaptureBusyError, CaptureParameterError, CaptureStopTimeout
from .study_control import StudyAction, StudyController
from .study_workspace import StudyWorkspace

if TYPE_CHECKING:
    from fastapi import FastAPI


def install_study_routes(app: "FastAPI", controller: StudyController | StudyWorkspace) -> None:
    from fastapi import HTTPException, Request
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool

    @app.get("/api/study/context", response_model=None)
    def context() -> dict[str, JsonValue]:
        return controller.context()

    @app.get("/api/study/status", response_model=None)
    def status() -> dict[str, JsonValue]:
        return controller.status()

    @app.get("/api/study/preview", response_model=None)
    def preview() -> dict[str, JsonValue]:
        return controller.preview()

    @app.get("/api/study/stimulus/{trial_id}", response_model=None)
    def stimulus(trial_id: str) -> Response:
        try:
            return Response(controller.manifest.stimulus_bytes(trial_id), media_type="image/png")
        except StudyError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/study/{action}", response_model=None)
    async def control(action: str, request: Request) -> dict[str, JsonValue]:
        origin = request.headers.get("origin")
        expected = urlsplit(str(request.url))
        if origin is not None and origin != f"{expected.scheme}://{expected.netloc}":
            raise HTTPException(status_code=403, detail="study controls require the same origin")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(status_code=403, detail="cross-site study control is forbidden")
        supplied = request.headers.get("x-itrace-control", "")
        if not controller.authorized(supplied):
            raise HTTPException(status_code=401, detail="study control token is missing or invalid")
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise HTTPException(status_code=415, detail="study controls require application/json")
        try:
            operation = StudyAction(action)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="unknown study action") from error
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > 1_048_576:
                raise HTTPException(status_code=413, detail="study request exceeds 1 MiB")
        payload = bytes(chunks)
        try:
            return await run_in_threadpool(
                controller.dispatch, operation, payload, request.headers.get("x-itrace-session")
            )
        except (StudyError, CaptureBusyError, CaptureParameterError, CaptureStopTimeout) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
