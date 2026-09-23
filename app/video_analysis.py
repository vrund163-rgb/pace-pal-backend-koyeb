"""
Real, working video analysis -- no invented numbers.

This module actually opens the uploaded video with OpenCV and computes real
metrics: frame rate, resolution, duration, frame count, and rough brightness /
motion-blur / camera-stability heuristics sampled from real frames.

Ball detection and tracking are NOT implemented here yet (see README) --
that's the next phase of work, not something this module pretends to do.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass
class QualityReadout:
    score: int
    motion_blur: str
    lighting: str
    stability: str
    note: str


def read_metadata(path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file -- it may be corrupt or an unsupported codec.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(fps=fps, frame_count=frame_count, width=width, height=height, duration_seconds=duration)


def assess_quality(path: str, meta: VideoMetadata, sample_frames: int = 12) -> QualityReadout:
    """
    Sample frames spread through the video and compute real, simple heuristics:
    - brightness (mean pixel intensity) -> lighting
    - Laplacian variance (sharpness) -> motion blur
    - frame-to-frame mean absolute difference of the whole frame -> camera stability
      (a crude proxy: large uniform shifts across the whole frame suggest camera
      movement rather than just the ball moving, since the ball is a small object)

    These are real measurements of the actual video, not guesses -- but they are
    coarse heuristics, not the calibrated confidence model the full spec describes.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened() or meta.frame_count == 0:
        cap.release()
        return QualityReadout(
            score=0, motion_blur="high", lighting="poor", stability="unstable",
            note="Could not read frames from this video to assess quality.",
        )

    indices = np.linspace(0, max(meta.frame_count - 1, 0), num=min(sample_frames, meta.frame_count), dtype=int)
    brightness_vals = []
    sharpness_vals = []
    frame_diffs = []
    prev_gray = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray)))
        sharpness_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if prev_gray is not None and prev_gray.shape == gray.shape:
            frame_diffs.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
        prev_gray = gray

    cap.release()

    avg_brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.0
    avg_sharpness = float(np.mean(sharpness_vals)) if sharpness_vals else 0.0
    avg_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0

    lighting = "good" if avg_brightness >= 90 else "moderate" if avg_brightness >= 50 else "poor"
    motion_blur = "low" if avg_sharpness >= 150 else "moderate" if avg_sharpness >= 60 else "high"
    stability = "stable" if avg_diff <= 18 else "minor_movement" if avg_diff <= 35 else "unstable"

    score = 100
    if lighting == "moderate":
        score -= 15
    elif lighting == "poor":
        score -= 35
    if motion_blur == "moderate":
        score -= 15
    elif motion_blur == "high":
        score -= 30
    if stability == "minor_movement":
        score -= 10
    elif stability == "unstable":
        score -= 25
    if meta.fps and meta.fps < 30:
        score -= 15
    score = max(0, min(100, score))

    notes = []
    if lighting != "good":
        notes.append(f"lighting is {lighting}")
    if motion_blur != "low":
        notes.append(f"{motion_blur} motion blur detected")
    if stability != "stable":
        notes.append("camera movement detected")
    if meta.fps and meta.fps < 30:
        notes.append(f"frame rate ({meta.fps:.0f}fps) is low for precise pace analysis")
    note = ("Good recording overall." if not notes else "Issues found: " + "; ".join(notes) + ".")

    return QualityReadout(score=score, motion_blur=motion_blur, lighting=lighting, stability=stability, note=note)
