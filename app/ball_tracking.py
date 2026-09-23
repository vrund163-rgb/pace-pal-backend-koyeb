"""
Real YOLOv8 ball detection, running via ONNX Runtime instead of the full
torch/ultralytics stack. Same trained model (COCO-pretrained yolov8n,
"sports ball" class), same detection quality -- but ONNX Runtime's CPU
inference has a much smaller memory footprint than importing torch, which
is what makes this fit a genuinely free, no-card 512MB host.

The model file (app/models/yolov8n.onnx) was exported once from the
standard ultralytics yolov8n.pt checkpoint via `model.export(format='onnx')`
-- that export step needs ultralytics+torch, but only as a one-time offline
step, never at request time. The deployed app never imports torch.

Decoding the raw ONNX output (manual box/score extraction + NMS) is more
code than calling `model.track()`, but it's what lets us drop torch as a
runtime dependency entirely.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8n.onnx")
SPORTS_BALL_CLASS_ID = 32
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


@dataclass
class TrackedPoint:
    frame: int
    t: float
    x: float
    y: float
    confidence: float


@dataclass
class TrackingResult:
    frames_total: int
    frames_detected: int
    trajectory: list[TrackedPoint] = field(default_factory=list)
    note: str = ""

    @property
    def detection_rate(self) -> float:
        return (self.frames_detected / self.frames_total * 100) if self.frames_total else 0.0


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, scale


def _detect_sports_ball(session: ort.InferenceSession, frame: np.ndarray) -> tuple[float, float, float] | None:
    """Returns (center_x, center_y, confidence) in original-frame pixel coords, or None."""
    blob, scale_x, scale_y = _preprocess(frame)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]  # shape (1, 84, 8400)

    preds = output[0].T  # (8400, 84): 4 box coords + 80 class scores
    class_scores = preds[:, 4:]
    ball_scores = class_scores[:, SPORTS_BALL_CLASS_ID]

    mask = ball_scores >= CONF_THRESHOLD
    if not np.any(mask):
        return None

    boxes = preds[mask, :4]
    scores = ball_scores[mask]

    # boxes are (cx, cy, w, h) in the 640x640 letterboxed space -> convert to xyxy for NMS
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(
        xyxy.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD
    )
    if len(indices) == 0:
        return None

    best_i = max(indices.flatten(), key=lambda i: scores[i])
    cx, cy = boxes[best_i, 0], boxes[best_i, 1]
    # undo the letterbox scale back to original frame pixels
    orig_x, orig_y = cx / scale_x, cy / scale_y
    return float(orig_x), float(orig_y), float(scores[best_i])


def track_ball(video_path: str, max_frames: int = 600) -> TrackingResult:
    session = _get_session()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return TrackingResult(frames_total=0, frames_detected=0, note="Could not open video for tracking.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    trajectory: list[TrackedPoint] = []
    frame_idx = 0
    detected_count = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        det = _detect_sports_ball(session, frame)
        if det is not None:
            cx, cy, conf = det
            trajectory.append(TrackedPoint(frame=frame_idx, t=frame_idx / fps, x=cx, y=cy, confidence=round(conf, 3)))
            detected_count += 1

        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        note = "No frames could be read from this video."
    elif detected_count == 0:
        note = (
            "No ball was detected in any frame by the trained detector. This may mean the ball wasn't "
            "visible/distinguishable, or the footage doesn't resemble what the model was trained on."
        )
    else:
        note = (
            f"Ball detected in {detected_count}/{frame_idx} frames using a trained YOLOv8 detector "
            "(generic 'sports ball' class, not cricket-specific). Pixel-space trajectory only -- "
            "converting to real pace, line, and length requires camera calibration, not implemented yet."
        )

    return TrackingResult(frames_total=frame_idx, frames_detected=detected_count, trajectory=trajectory, note=note)
