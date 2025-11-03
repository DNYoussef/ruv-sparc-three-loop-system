"""
Comprehensive Logging Configuration Template

Provides production-ready logging configuration with:
- Multiple log levels and handlers
- Structured logging (JSON format)
- Performance tracking
- Error tracking with context
- Log rotation and archival
- Distributed tracing support

Usage:
    from logging_config import setup_logging, get_logger

    setup_logging(level='DEBUG', enable_json=True)
    logger = get_logger(__name__)
    logger.info('Application started', extra={'user_id': 123})
"""

import logging
import logging.handlers
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import os


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_obj.update(record.extra_data)

        # Add from record.__dict__ (extra kwargs)
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'lineno', 'module', 'msecs', 'message',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'extra_data', 'getMessage', 'levelno']:
                try:
                    log_obj[key] = value
                except (TypeError, ValueError):
                    pass

        return json.dumps(log_obj)


class ContextFilter(logging.Filter):
    """Add contextual information to log records"""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class PerformanceLogger:
    """Track performance metrics in logs"""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f'Starting: {self.operation}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type:
            self.logger.error(
                f'Failed: {self.operation}',
                extra={
                    'operation': self.operation,
                    'duration_seconds': duration,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                },
                exc_info=True
            )
        else:
            self.logger.info(
                f'Completed: {self.operation}',
                extra={
                    'operation': self.operation,
                    'duration_seconds': duration
                }
            )


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    enable_json: bool = False,
    enable_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Configure application-wide logging

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_json: Use JSON formatter for structured logging
        enable_console: Enable console output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        context: Additional context to add to all log records
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d) - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        root_logger.addFilter(context_filter)

    # Log initial message
    root_logger.info(
        'Logging configured',
        extra={
            'level': level,
            'json_logging': enable_json,
            'log_file': log_file
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_function_call(func):
    """Decorator to log function calls with arguments and results"""
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        logger.debug(
            f'Calling {func.__name__}',
            extra={
                'function': func.__name__,
                'args': str(args)[:100],  # Truncate long args
                'kwargs': str(kwargs)[:100]
            }
        )

        try:
            result = func(*args, **kwargs)
            logger.debug(
                f'Completed {func.__name__}',
                extra={
                    'function': func.__name__,
                    'result': str(result)[:100]
                }
            )
            return result
        except Exception as e:
            logger.exception(
                f'Error in {func.__name__}',
                extra={
                    'function': func.__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    return wrapper


# Example usage
if __name__ == '__main__':
    # Setup logging with JSON format
    setup_logging(
        level='DEBUG',
        log_file='app.log',
        enable_json=True,
        context={
            'app_name': 'my_application',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'version': '1.0.0'
        }
    )

    logger = get_logger(__name__)

    # Basic logging
    logger.debug('Debug message')
    logger.info('Info message', extra={'user_id': 123, 'action': 'login'})
    logger.warning('Warning message')

    # Performance tracking
    with PerformanceLogger(logger, 'database_query'):
        # Simulate some work
        import time
        time.sleep(0.1)

    # Function call logging
    @log_function_call
    def example_function(x, y):
        return x + y

    result = example_function(5, 3)

    # Error logging
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception('Division by zero error', extra={'operation': 'divide'})

    logger.info('Application shutdown')
