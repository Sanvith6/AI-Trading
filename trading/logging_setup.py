"""
Central logging setup — call `setup_logging()` once at every entry point
(Streamlit app, agent loop, backtester, tests) so all module loggers
write to console AND a rotating file. Without this, `getLogger(__name__)`
calls across the package silently go nowhere.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FILE = Path(__file__).parent / "agent.log"
_configured = False


def setup_logging(level: str = "INFO") -> Path:
    global _configured
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _configured:
        return _LOG_FILE
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        fh = RotatingFileHandler(_LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # read-only FS: console logging still works
    _configured = True
    return _LOG_FILE


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    return logging.getLogger(name)
