from pathlib import Path

import pytest

from .helpers import Harness


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.fixture
def ready(harness: Harness) -> Harness:
    harness.ready()
    return harness
