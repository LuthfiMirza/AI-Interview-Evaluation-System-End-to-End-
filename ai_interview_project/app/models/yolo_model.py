"""YOLOv8 wrapper for vision-based cheating detection."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, List

from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


class YoloDetector:
    """Thin wrapper around Ultralytics YOLOv8 for reusability."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        LOGGER.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)

    def predict(self, source: str | int, **kwargs: Any) -> List[Any]:
        LOGGER.debug("Running YOLO inference on %s", source)
        return self.model.predict(source=source, **kwargs)


@lru_cache(maxsize=1)
def get_detector(model_path: str = "yolov8n.pt") -> YoloDetector:
    """Return a cached detector instance."""
    return YoloDetector(model_path=model_path)
