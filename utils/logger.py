"""
Logging utility for the Climate Risk Integration Platform.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files (default: ~/.logs/climate_risk)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                      datefmt='%H:%M:%S')
    
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir is None:
        log_dir = os.path.expanduser("~/.logs/climate_risk")
    
    try:
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to set up file logging: {e}")
    
    return logger