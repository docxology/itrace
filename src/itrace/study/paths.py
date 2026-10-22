from pathlib import Path

from .jsonio import StudyError


def validate_private_root(output_root: Path) -> Path:
    for component in (output_root, *output_root.parents):
        if component.is_symlink():
            raise StudyError("output_root", "symlink output directory forbidden")
    resolved = output_root.resolve()
    package = Path(__file__).resolve().parent.parent
    repo = next(
        (
            parent
            for parent in (resolved, *resolved.parents)
            if (parent / "pyproject.toml").is_file()
            and (parent / "src/itrace/__init__.py").is_file()
        ),
        None,
    )
    if resolved.is_relative_to(package) or (
        repo is not None and not resolved.is_relative_to(repo / "output/studies")
    ):
        raise StudyError("output_root", "use explicit private storage outside public source")
    return resolved
