from __future__ import annotations
import os
import shutil
import time
import traceback
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import Session, CalibrationState, VideoQuality, TrackingPreview, TrackedPointModel, PROCESSING_STAGES
from . import storage_db
from . import video_analysis
from . import ball_tracking

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage", "videos")
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="Pace Pal API")

# CORS: allow your Netlify site (and local dev) to call this API.
# Replace "*" with your actual Netlify URL once you have it, for a tighter setup.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/sessions")
def api_list_sessions():
    return [s.model_dump() for s in storage_db.list_sessions()]


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    session = storage_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


@app.post("/api/sessions/upload")
async def upload_session(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session = Session(name=file.filename or "Uploaded session", status="uploaded", isSampleData=False)

    video_path = os.path.join(STORAGE_DIR, f"{session.id}_{file.filename}")
    with open(video_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    storage_db.save_session(session)
    background_tasks.add_task(process_session, session.id, video_path)

    return {"sessionId": session.id}


def _set_stage(session: Session, stage_index: int) -> None:
    session.processingStage = stage_index
    session.processingStageLabel = PROCESSING_STAGES[stage_index]
    storage_db.save_session(session)


def process_session(session_id: str, video_path: str) -> None:
    """
    The real processing pipeline. Stages that are implemented do real work on
    the actual uploaded video. Stages that aren't implemented yet say so
    explicitly in the result rather than inventing numbers.
    """
    session = storage_db.get_session(session_id)
    if session is None:
        return

    try:
        session.status = "processing"
        _set_stage(session, 0)  # Video uploaded

        _set_stage(session, 1)  # Reading video metadata
        meta = video_analysis.read_metadata(video_path)

        _set_stage(session, 2)  # Assessing video quality
        quality = video_analysis.assess_quality(video_path, meta)

        session.videoQuality = VideoQuality(
            score=quality.score,
            fps=round(meta.fps, 2),
            resolution=f"{meta.width}x{meta.height}",
            motionBlur=quality.motion_blur,
            lighting=quality.lighting,
            stability=quality.stability,
            ballVisibility=0,
            note=quality.note,
        )
        storage_db.save_session(session)

        _set_stage(session, 3)  # Camera calibration
        session.calibration = CalibrationState(
            status="not_started",
            stumpsDetected=False,
            pitchDetected=False,
            groundPlaneDetected=False,
            pitchLengthMetres=None,
            cameraQuality=0,
            cameraQualityNote="Automatic calibration (stump/pitch detection) is not implemented yet in this backend.",
        )
        storage_db.save_session(session)

        _set_stage(session, 4)  # Delivery detection
        # Not implemented yet -- deliberately left as no deliveries, since
        # splitting a session into distinct deliveries requires calibration
        # and rally/gap detection we haven't built.

        _set_stage(session, 5)  # Ball tracking
        result = ball_tracking.track_ball(video_path)
        session.trackingPreview = TrackingPreview(
            framesTotal=result.frames_total,
            framesDetected=result.frames_detected,
            detectionRatePct=round(result.detection_rate, 1),
            trajectory=[
                TrackedPointModel(frame=p.frame, t=round(p.t, 3), x=p.x, y=p.y, confidence=round(p.confidence, 3))
                for p in result.trajectory
            ],
            note=result.note,
        )
        storage_db.save_session(session)

        session.deliveries = []

        _set_stage(session, 6)  # Results ready
        session.status = "complete"
        session.note = (
            "Video metadata and quality were read from your real upload, and the ball was tracked "
            "frame-by-frame with a trained YOLOv8 detector via ONNX Runtime (see trackingPreview). Calibration and "
            "delivery-level pace/swing/line/length are not implemented yet, so no delivery data is "
            "shown -- that requires converting pixel positions to real-world measurements, which "
            "needs a calibration step this backend doesn't have yet."
        )
        storage_db.save_session(session)

    except Exception as exc:  # noqa: BLE001
        session.status = "failed"
        session.error = str(exc)
        session.note = "Processing failed."
        storage_db.save_session(session)
        traceback.print_exc()
