import logging
from datetime import datetime
from logging import LogRecord
from typing import Mapping
from tomlkit import dumps
from jax import Array


def to_dict(d: Mapping) -> dict:
    if isinstance(d, Mapping):
        return {k: to_dict(v) for k, v in d.items()}
    elif isinstance(d, Array):
        return d.tolist()
    else:
        return d


class TOMLFormatter(logging.Formatter):
    def __init__(self, datefmt: str | None = None, time_stamp=True) -> None:
        super().__init__(datefmt=datefmt)
        self.time_stamp = time_stamp

    def format(self, record: LogRecord) -> str:
        if not isinstance(record.msg, Mapping):
            raise TypeError("Log message must be a Mapping, e.g a Dict")

        data = {
            "name": record.name,
            "level": record.levelname,
        }
        if self.time_stamp:
            data["time"] = datetime.fromtimestamp(record.created)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            data["exc_text"] = record.exc_text
        if record.stack_info:
            data["stack_info"] = self.formatStack(record.stack_info)

        data["msg"] = to_dict(record.msg)
        return dumps({"log": [data]})  # so that we can keep appending log
