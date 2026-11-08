"""
Data models for the Pace Pal backend.

These mirror src/types/index.ts in the frontend exactly -- this file IS the
API contract. If you change a field here, update the frontend type too, and
vice versa. Keeping them in lockstep is what lets the frontend swap sample
data for real API responses without touching any UI code.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
import uuid
import time

ConfidenceLevel = Literal["high", "medium", "low", "insufficient"]


def confidence_level_from_score(score: Optional[float]) -> ConfidenceLevel:
    if score is None:
        return "insufficient"
    if score >= 85:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


class Measurement(BaseModel):
    value: Optional[float] = None
    unit: str
    rangeLow: Optional[float] = None
    rangeHigh: Optional[float] = None
    confidence: Optional[float] = None
    confidenceLevel: ConfidenceLevel = "insufficient"
    reason: str = ""


class CalibrationState(BaseModel):
    status: Literal["not_started", "in_progress", "good", "acceptable", "poor"] = "not_started"
    stumpsDetected: bool = False
    pitchDetected: bool = False
    groundPlaneDetected: bool = False
    pitchLengthMetres: Optional[float] = None
    cameraQuality: int = 0
    cameraQualityNote: str = ""
    homography: Optional[list[float]] = None  # flattened 3x3, row-major, once calibrated


class VideoQuality(BaseModel):
    score: int = 0
    fps: float = 0
    resolution: str = ""
    motionBlur: Literal["low", "moderate", "high"] = "moderate"
    lighting: Literal["good", "moderate", "poor"] = "moderate"
    stability: Literal["stable", "minor_movement", "unstable"] = "stable"
    ballVisibility: int = 0
    note: str = ""


class TrackedPointModel(BaseModel):
    frame: int
    t: float
    x: float
    y: float
    confidence: float


class TrackingPreview(BaseModel):
    framesTotal: int = 0
    framesDetected: int = 0
    detectionRatePct: float = 0
    trajectory: list[TrackedPointModel] = Field(default_factory=list)
    note: str = ""


class CalibrationPoints(BaseModel):
    bowlerLeft: tuple[float, float]
    bowlerRight: tuple[float, float]
    batterLeft: tuple[float, float]
    batterRight: tuple[float, float]


class CalibratedResult(BaseModel):
    lineM: Optional[float] = None  # lateral offset from centreline, metres
    lengthM: Optional[float] = None  # distance from batter-end stumps, metres
    bounceFrame: Optional[int] = None
    note: str = ""


PROCESSING_STAGES = [
    "Video uploaded",
    "Reading video metadata",
    "Assessing video quality",
    "Camera calibration",
    "Delivery detection",
    "Ball tracking",
    "Results ready",
]


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    createdAt: float = Field(default_factory=time.time)
    name: str = "New session"
    calibration: CalibrationState = Field(default_factory=CalibrationState)
    videoQuality: VideoQuality = Field(default_factory=VideoQuality)
    status: Literal["draft", "uploaded", "processing", "complete", "failed"] = "draft"
    processingStage: int = 0
    processingStageLabel: str = PROCESSING_STAGES[0]
    deliveries: list = Field(default_factory=list)
    trackingPreview: Optional[TrackingPreview] = None
    calibratedResult: Optional[CalibratedResult] = None
    isSampleData: bool = False
    note: str = ""
    error: Optional[str] = None
