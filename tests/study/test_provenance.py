from hashlib import sha256
from pathlib import Path

from itrace.study.provenance import source_bindings


def test_installed_package_binds_actual_code_and_all_browser_assets(tmp_path: Path) -> None:
    package = tmp_path / "site-packages" / "itrace"
    assets = package / "web_static"
    assets.mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = 'test'\n")
    (assets / "study-api.js").write_text("export const clock = 1;\n")
    (assets / "app.css").write_text("body { color: black; }\n")
    first = source_bindings(package)
    assert first["source_files"] == {
        "itrace/__init__.py": sha256(b"VERSION = 'test'\n").hexdigest(),
        "itrace/web_static/app.css": sha256(b"body { color: black; }\n").hexdigest(),
        "itrace/web_static/study-api.js": sha256(b"export const clock = 1;\n").hexdigest(),
    }
    assert first["installation_kind"] == "installed-package"
    (assets / "study-api.js").write_text("export const clock = 2;\n")
    assert source_bindings(package)["source_digest"] != first["source_digest"]
