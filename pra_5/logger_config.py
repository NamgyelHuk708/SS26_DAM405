"""
logger_config.py  -  Task 1: Configure logging
------------------------------------------------
Creates ONE shared logger called 'ml-service' that writes to:
  1. the console (so you can watch it live in the VS Code terminal)
  2. a rotating file 'service.log' (kept on disk for later analysis)

Every line has a timestamp, the log level and the message.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

# Put service.log next to this file, no matter where the script is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "service.log")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"


def get_logger(level=logging.INFO):
    """Return the configured 'ml-service' logger.

    level: logging.DEBUG in development, logging.INFO in production.
    """
    logger = logging.getLogger("ml-service")
    logger.setLevel(level)

    # Guard: only add handlers once, otherwise every line would be duplicated
    if logger.handlers:
        return logger

    fmt = logging.Formatter(LOG_FORMAT)

    # Handler 1: console output
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    # Handler 2: rotating file - max ~1 MB per file, keep 3 old backups
    # (service.log.1, service.log.2, service.log.3) so the disk never fills up
    fileh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000,
                                backupCount=3, encoding="utf-8")
    fileh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fileh)

    # Don't pass messages up to the root logger (prevents double printing)
    logger.propagate = False
    return logger
