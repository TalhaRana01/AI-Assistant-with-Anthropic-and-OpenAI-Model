"""Logging configuration utilities.

This module provides centralized logging configuration with support
for different log levels and formats.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    format_string: str | None = None
) -> None:
    """Setup application logging configuration.
    
    Configures logging with the specified level and optional file output.
    Uses a clean format that includes timestamp, level, and message.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Example:
        >>> setup_logging(level="DEBUG")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format if not provided
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


class SensitiveDataFilter(logging.Filter):
    """Filter to prevent sensitive data from being logged.
    
    This filter removes or redacts sensitive information like API keys
    from log messages.
    """
    
    SENSITIVE_PATTERNS = [
        "sk-",  # OpenAI keys
        "sk-ant-",  # Anthropic keys
        "api_key",
        "apikey",
        "token",
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to remove sensitive data.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record, False to block it
        """
        message = record.getMessage().lower()
        
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                # Redact the sensitive part
                record.msg = "[REDACTED - Sensitive data filtered]"
                record.args = ()
                break
        
        return True