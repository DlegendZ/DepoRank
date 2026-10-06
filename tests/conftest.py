from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(config.rootpath) / ".pytest_cache" / ("tmp_" + uuid4().hex))
