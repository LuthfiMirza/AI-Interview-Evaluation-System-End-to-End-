"""Vision utilities for non-verbal cue analysis and cheating detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
from app.models.yolo_model import YoloDetector, get_detector

LOGGER = logging.getLogger(__name__)

try:
    import mediapipe as mp

    _MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
except Exception:  # noqa: BLE001
    _MP_FACE_MESH = None
    LOGGER.warning("Mediapipe not available; gaze estimation will be degraded.")


@dataclass
class VisionMetrics:
    eye_contact_ratio: float = 0.0
    phone_detected: bool = False
    multi_person: bool = False
    cheating_score: float = 0.0

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "eye_contact_ratio": round(self.eye_contact_ratio, 3),
            "phone_detected": self.phone_detected,
            "multi_person": self.multi_person,
            "cheating_score": round(self.cheating_score, 3),
        }


def _is_forward_gaze(face_landmarks) -> bool:
    if face_landmarks is None:
        return False
    try:
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        nose_tip = face_landmarks.landmark[1]
    except IndexError:
        return False
    eye_center_x = (left_eye.x + right_eye.x) / 2
    deviation = abs(nose_tip.x - eye_center_x)
    return deviation < 0.03


def analyze_video(
    video_path: str,
    detector: Optional[YoloDetector] = None,
    sample_rate: int = 5,
) -> Dict[str, float | bool]:
    """Analyze video frames and compute cheating-related metrics."""
    detector = detector or get_detector()
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    frame_index = 0
    phone_frames = 0
    multi_person_detected = False
    forward_frames = 0
    face_samples = 0

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_index += 1

            if frame_index % sample_rate != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.predict(rgb_frame, verbose=False)
            if not results:
                continue

            prediction = results[0]
            boxes = prediction.boxes
            class_names = prediction.names

            if boxes is not None and boxes.cls is not None:
                detected_classes = [class_names[int(cls_id)] for cls_id in boxes.cls]
                persons = sum(1 for name in detected_classes if name == "person")
                phones = sum(
                    1
                    for name in detected_classes
                    if name in {"cell phone", "cellphone", "mobile phone", "phone"}
                )
                if persons > 1:
                    multi_person_detected = True
                if phones > 0:
                    phone_frames += 1

            if _MP_FACE_MESH is not None:
                face_results = _MP_FACE_MESH.process(rgb_frame)
                if face_results.multi_face_landmarks:
                    face_samples += 1
                    if len(face_results.multi_face_landmarks) > 1:
                        multi_person_detected = True
                    if _is_forward_gaze(face_results.multi_face_landmarks[0]):
                        forward_frames += 1

    finally:
        capture.release()

    if face_samples == 0:
        eye_contact_ratio = 0.5  # fallback when face tracking unavailable
    else:
        eye_contact_ratio = forward_frames / face_samples

    phone_detected = phone_frames > 0
    cheating_score = min(
        1.0,
        (1.0 - eye_contact_ratio) * 0.5
        + (0.3 if phone_detected else 0.0)
        + (0.2 if multi_person_detected else 0.0),
    )

    metrics = VisionMetrics(
        eye_contact_ratio=eye_contact_ratio,
        phone_detected=phone_detected,
        multi_person=multi_person_detected,
        cheating_score=cheating_score,
    )

    LOGGER.info("Vision metrics computed: %s", metrics.to_dict())
    return metrics.to_dict()
