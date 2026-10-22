from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer


def study(
    manifest: Annotated[Path, typer.Option(help="Private visual-study manifest JSON.")],
    output_dir: Annotated[Path, typer.Option(help="Private root for new session directories.")],
    camera: int = 0,
    host: str = "127.0.0.1",
    port: int = 8766,
    backend_logs: bool = False,
) -> None:
    """Open the isolated visual-task dashboard with webcam capture."""
    from .study import StudyError

    try:
        from .live.study_server import serve_study

        serve_study(
            manifest_path=manifest,
            output_root=output_dir,
            camera_index=camera,
            host=host,
            port=port,
            backend_logs=backend_logs,
        )
    except ModuleNotFoundError as error:
        typer.echo(f"study failed: {error}", err=True)
        typer.echo("The dashboard requires the web and capture extras.", err=True)
        raise typer.Exit(2) from error
    except (StudyError, OSError) as error:
        typer.echo(f"study failed: {error}", err=True)
        raise typer.Exit(2) from error


def study_report(session_dir: Path, out: Annotated[Path, typer.Option()]) -> None:
    """Reconstruct a private study report without camera access."""
    from .study import StudyError, reconstruct

    try:
        payload = json.dumps(reconstruct(session_dir), indent=2, allow_nan=False) + "\n"
        with out.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (StudyError, OSError) as error:
        typer.echo(f"study report failed: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"Study report: {out}")
