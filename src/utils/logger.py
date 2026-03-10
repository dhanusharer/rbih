from __future__ import annotations
import sys
from pathlib import Path
from loguru import logger

_configured = False

def get_logger(name: str = "mule_hunter", log_dir: str | Path = "logs"):
    global _configured
    if not _configured:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(sys.stderr, level="INFO",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}")
        logger.add(Path(log_dir) / f"{name}.log", rotation="10 MB",
                   level="DEBUG", format="{time} | {level} | {name} | {message}")
        _configured = True
    return logger
