#!/usr/bin/env python3
"""
Simple logging setup utility for IPHR evaluation scripts.
"""

import logging
import sys
from typing import Optional

def setup_logging(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        name: Logger name (usually script name)
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (if None, uses root logger)
    
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)
