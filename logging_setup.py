"""
Logging setup utility using loguru for steering-evals project.
Configures logging to both terminal and timestamped log files.
"""

import os
import sys
from datetime import datetime
from loguru import logger


def setup_logging(script_name: str, log_level: str = "INFO") -> None:
    """
    Set up logging for a script using loguru.

    Args:
        script_name: Name of the script (used for log file naming)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    # Add terminal handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler with detailed format
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="100 MB",
        retention="1 week",
    )

    logger.info(f"Logging initialized for {script_name}")
    logger.info(f"Log file: {log_file}")

    return logger


def get_logger():
    """Get the configured logger instance."""
    return logger
