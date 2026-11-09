from __future__ import annotations
import os
import glob
import shutil
import time
import traceback
import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    Session, CalibrationState, VideoQuality, TrackingPreview, TrackedPointModel,
    CalibrationPoints, CalibratedResult, PROCESSING_STAGES,
)
from . import storage_db
from . import video_analysis
from . import ball_tracking
from . import homography
from . import bounce

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


def _find_video_path(session_id: str):
    matches = glob.glob(os.path.join(STORAGE_DIR, f"{session_id}_*"))
    return matches[0] if matches else None


@app.get("/api/sessions/{session_id}/frame")
def api_get_frame(session_id: str):
    video_path = _find_video_path(session_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found for this session")
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=500, detail="Could not read a frame from this video")
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/sessions/{session_id}/calibrate")
def api_calibrate_session(session_id: str, points: CalibrationPoints):
    session = storage_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.trackingPreview or not session.trackingPreview.trajectory:
        raise HTTPException(status_code=400, detail="No tracked trajectory available to calibrate against yet")

    image_points = [points.bowlerLeft, points.bowlerRight, points.batterLeft, points.batterRight]
    H = homography.compute_homography(image_points, points.mode)

    traj = session.trackingPreview.trajectory
    est = bounce.find_bounce([p.frame for p in traj], [p.x for p in traj], [p.y for p in traj])

    session.calibration.homography = H.flatten().tolist()
    session.calibration.status = "good"
    session.calibration.pitchLengthMetres = homography.PITCH_LENGTH_M

    if not est.ok:
        session.calibratedResult = CalibratedResult(
            bounceConfidence="none", calibrationMode=points.mode,
            note=(
                "The pitch calibration worked, but the bounce couldn't be found in the tracked path, so no line "
                "or length is shown rather than a guess. " + est.reason + " A clearer recording with the ball "
                "detected in more frames usually fixes this."
            ),
        )
        storage_db.save_session(session)
        return session.model_dump()

    wx, wy = homography.pixel_to_world(H, (est.x, est.y))
    length_m = homography.PITCH_LENGTH_M - wy

    caveat = (
        " The bounce is found as the corner in the ball's path where it stops dropping and starts rising. "
        "In simulated tests with clean tracking and clicks a few pixels off, line was typically within about "
        "10 cm and length within about 1 m (it tended to read a little short on fuller balls); real footage "
        "with missed or wrong detections will be less accurate. "
        "Pace is not available -- it needs the ball's real height in the air, which one camera can't recover."
    )
    if points.mode == "stumps":
        caveat = (
            " WARNING: you calibrated from the stumps, which is very sensitive to click accuracy (the far "
            "stumps are only a few pixels wide in the video). In simulation, length could be off by several "
            "metres. Recalibrate using the crease corners if the pitch has crease lines." + caveat
        )
    if est.confidence == "clear":
        note = "Bounce found clearly in the tracked path." + caveat
    else:
        note = ("LOW CONFIDENCE: the corner in the ball's path was only faintly visible, so treat these numbers "
                "with caution." + caveat)

    session.calibratedResult = CalibratedResult(
        lineM=round(wx, 3), lengthM=round(length_m, 2),
        bounceFrame=round(est.frame, 1), bounceX=round(est.x, 1), bounceY=round(est.y, 1),
        bounceConfidence=est.confidence, calibrationMode=points.mode, note=note,
    )
    storage_db.save_session(session)
    return session.model_dump()


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
        def _on_progress(pct: int) -> None:
            session.processingProgress = pct
            storage_db.save_session(session)

        result = ball_tracking.track_ball(video_path, progress_cb=_on_progress)
        session.processingProgress = None
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
            "Video metadata and quality were read from your real upload, and the ball was tracked with a "
            "trained YOLOv8 detector, checking some frames per second (see trackingPreview). Line and length "
            "become available once you calibrate the pitch. Pace and full delivery analysis (swing, turn) are "
            "not implemented: they need the ball's position in the air, which one camera can't provide."
        )
        storage_db.save_session(session)

    except Exception as exc:  # noqa: BLE001
        session.status = "failed"
        session.error = str(exc)
        session.note = "Processing failed."
        storage_db.save_session(session)
        traceback.print_exc()
