import logging
import os
import time

from backend.config import LOG_FOLDER


os.makedirs(
    LOG_FOLDER,
    exist_ok=True
)


LOG_FILE = os.path.join(
    LOG_FOLDER,
    "app.log"
)


logging.basicConfig(
    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),

    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),

        logging.StreamHandler()
    ]
)


logger = logging.getLogger(
    "ai-learning-mentor"
)


def start_timer():

    return time.perf_counter()


def elapsed_ms(
    start_time
):

    return round(
        (
            time.perf_counter()
            -
            start_time
        )
        * 1000,
        2
    )