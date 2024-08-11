import logging
import logging.handlers
import os


def setup_logger(script_name):
    logger = logging.getLogger()

    if not logger.handlers:
        log_filename = f"{os.path.splitext(script_name)[0]}.log"
        handler = logging.handlers.RotatingFileHandler(
            filename=f"logger/log_files/{log_filename}", maxBytes=1024 * 1024 * 10, backupCount=5
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
