import logging
import logging.handlers


def setup_logger():
    """
    Sets up a logger for the provided script. The logger writes logs to a rotating file handler and the console.
    """
    logger = logging.getLogger()

    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            filename=f"logger/log_files/app_log.log",
            maxBytes=1024 * 1024 * 10,
            backupCount=5,
        )
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger