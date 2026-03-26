"""Structured logging configuration for the AdaptiveBeta pipeline.

Sets up a consistent log format with timestamps, module names, and
log levels. Call setup_logging() at the top of each notebook / script.

Typical usage:
    from src.utils.logging import setup_logging
    setup_logging(level="INFO")
    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging.

    Sets up:
    - Console handler with coloured output (if running in a terminal)
    - Optional file handler writing to log_file
    - ISO 8601 timestamps

    Args:
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to write logs to file.
        name: Logger name (defaults to root logger if None).

    Returns:
        Configured Logger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handlers.append(console)

    # File (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_h.setLevel(log_level)
        file_h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(file_h)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # Remove existing handlers to avoid duplicate output in notebooks
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)

    # Suppress noisy third-party loggers
    for noisy in ["matplotlib", "PIL", "urllib3", "numba"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(
        "Logging configured — level=%s, file=%s, started=%s",
        level,
        log_file or "none",
        datetime.now().isoformat(timespec="seconds"),
    )
    return logger
