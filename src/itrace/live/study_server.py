import asyncio
import re
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ..capture import LiveFrameSample, WebcamSource, native_stderr
from ..study import StudySession, load_manifest
from ..version import __version__
from .capture_service import CaptureService
from .server import FrameSourceFactory
from .study_control import StudyController
from .study_routes import install_study_routes
from .study_workspace import StudyWorkspace

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_study_app(
    *,
    manifest_path: Path,
    output_root: Path,
    camera_index: int = 0,
    frame_source_factory: FrameSourceFactory | None = None,
    backend_logs: bool = False,
) -> "FastAPI":
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles

    manifest = load_manifest(manifest_path)

    def factory(camera: int, limit: int | None) -> Iterable[LiveFrameSample]:
        with native_stderr(backend_logs):
            source = WebcamSource(camera_index=camera, timestamp_origin_s=0.0)
            yield from source.live_frames(max_frames=limit)

    token = secrets.token_urlsafe(32)

    def build() -> StudyController:
        session = StudySession(manifest, output_root)

        def record(frame: LiveFrameSample, epoch: str, received_s: float) -> None:
            session.capture(frame.capture, epoch, received_s)

        def changed(active: bool, reason: str, error: str | None) -> None:
            epoch = capture.status()["epoch"]
            session.capture_state(active, str(epoch), error or reason)

        capture = CaptureService(frame_source_factory or factory, record, changed)
        return StudyController(manifest, session, capture, token, camera_index)

    controller = StudyWorkspace(build)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from starlette.concurrency import run_in_threadpool

        stopped = asyncio.Event()

        async def watch_connection() -> None:
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=1)
                except asyncio.TimeoutError:
                    await run_in_threadpool(controller.expire_connection)

        watcher = asyncio.create_task(watch_connection())
        try:
            yield
        finally:
            stopped.set()
            await watcher
            await run_in_threadpool(controller.stop)

    app = FastAPI(title="iTrace visual study", version=__version__, lifespan=lifespan)
    app.state.study_controller = controller
    assets = Path(__file__).resolve().parent.parent / "web_static"
    app.mount("/static", StaticFiles(directory=assets), name="static")

    @app.middleware("http")
    async def private_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        hosts = request.headers.getlist("host")
        host_match = (
            re.fullmatch(
                r"(127\.0\.0\.1|localhost|\[::1\])(?::([0-9]{1,5}))?", hosts[0], re.IGNORECASE
            )
            if len(hosts) == 1
            else None
        )
        if host_match is None or (
            host_match[2] is not None and not 1 <= int(host_match[2]) <= 65535
        ):
            return PlainTextResponse("Invalid study host", status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/", response_model=None)
    def index() -> RedirectResponse:
        return RedirectResponse("/study")

    @app.get("/study", response_model=None)
    def study_page() -> FileResponse:
        return FileResponse(assets / "study.html")

    @app.get("/sw.js", include_in_schema=False)
    def service_worker() -> Response:
        return Response("", media_type="text/javascript")

    install_study_routes(app, controller)
    return app


def serve_study(
    *,
    manifest_path: Path,
    output_root: Path,
    camera_index: int = 0,
    host: str = "127.0.0.1",
    port: int = 8766,
    backend_logs: bool = False,
) -> None:
    from ..study import StudyError

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise StudyError("host", "private study server must bind loopback")
    if type(port) is not int or not 1 <= port <= 65535:
        raise StudyError("port", "expected a TCP port from 1 through 65535")
    if type(camera_index) is not int or camera_index < 0:
        raise StudyError("camera", "expected a nonnegative camera index")
    import uvicorn

    app = create_study_app(
        manifest_path=manifest_path,
        output_root=output_root,
        camera_index=camera_index,
        backend_logs=backend_logs,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
