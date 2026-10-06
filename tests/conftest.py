from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        temporary_root = Path(config.rootpath) / ".pytest_cache"
        temporary_root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(temporary_root / ("tmp_" + uuid4().hex))
