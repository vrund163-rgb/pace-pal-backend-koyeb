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
    CalibrationPoints, CalibratedResult, SeedRequest, TrimRequest, BounceRequest, ManualBounce, PaceResult,
    PROCESSING_STAGES,
)
from . import storage_db
from . import video_analysis
from . import ball_tracking
from . import homography
from . import bounce
from . import seed_tracker
from . import pace as pace_mod

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
def api_get_frame(session_id: str, index: int = 0):
    video_path = _find_video_path(session_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found for this session")
    if index < 0:
        raise HTTPException(status_code=400, detail="Frame index can't be negative")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=404, detail=f"Could not read frame {index} from this video")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def _image_size(session: Session) -> tuple[int, int]:
    try:
        w, h = session.videoQuality.resolution.lower().split("x")
        return int(w), int(h)
    except Exception:  # noqa: BLE001
        return 0, 0


def _build_results(session: Session, H, mode: str) -> tuple[CalibratedResult, PaceResult]:
    """Work out line/length (and a rough pace) from the tracked path and the pitch calibration H
    (H maps image pixels -> pitch metres)."""
    import numpy as np

    traj = session.trackingPreview.trajectory
    manual = session.manualBounce

    if manual is not None:
        est_ok, est_conf = True, "manual"
        b_frame, b_x, b_y = float(manual.frame), manual.x, manual.y
        reason = ""
    else:
        est = bounce.find_bounce([p.frame for p in traj], [p.x for p in traj], [p.y for p in traj])
        est_ok, est_conf = est.ok, est.confidence
        b_frame, b_x, b_y = est.frame, est.x, est.y
        reason = est.reason

    if not est_ok:
        calibrated = CalibratedResult(
            bounceConfidence="none", calibrationMode=mode,
            note=(
                "The pitch calibration worked, but the bounce couldn't be found in the tracked path, so no line "
                "or length is shown rather than a guess. " + reason + " You can mark the bounce yourself "
                "(click the ball where it hits the ground), or track more of the flight."
            ),
        )
        return calibrated, PaceResult(ok=False, note="Pace needs the bounce, which wasn't found. Mark the bounce yourself to continue.")

    wx, wy = homography.pixel_to_world(H, (b_x, b_y))
    length_m = homography.PITCH_LENGTH_M - wy

    caveat = (
        " In simulated tests with clean tracking and clicks a few pixels off, line was typically within about "
        "10 cm and length within about 1 m (it tended to read a little short on fuller balls); real footage "
        "with missed or wrong detections will be less accurate."
    )
    if mode == "stumps":
        caveat = (
            " WARNING: you calibrated from the stumps, which is very sensitive to click accuracy (the far "
            "stumps are only a few pixels wide in the video). In simulation, length could be off by several "
            "metres. Recalibrate using the crease corners if the pitch has crease lines." + caveat
        )
    if est_conf == "manual":
        note = "You marked where the ball bounced, so line and length come straight from that point." + caveat
    elif est_conf == "clear":
        note = ("Bounce found automatically as the corner in the ball's path where it stops dropping and starts "
                "rising. If it looks wrong, click the bounce yourself." + caveat)
    else:
        note = ("LOW CONFIDENCE: the corner in the ball's path was only faintly visible, so treat these numbers "
                "with caution (or click the bounce yourself)." + caveat)

    calibrated = CalibratedResult(
        lineM=round(wx, 3), lengthM=round(length_m, 2),
        bounceFrame=round(b_frame, 1), bounceX=round(b_x, 1), bounceY=round(b_y, 1),
        bounceConfidence=est_conf, calibrationMode=mode, note=note,
    )

    # ---- rough pace ----
    w, h = _image_size(session)
    fps = session.videoQuality.fps
    if mode != "creases":
        pace = PaceResult(ok=False, note="Pace needs the crease-corner calibration (the stump calibration is too imprecise for it).")
    elif not (w and h and fps):
        pace = PaceResult(ok=False, note="Pace needs the video's size and frame rate, which weren't recorded for this upload. Re-upload the clip.")
    else:
        try:
            est_p = pace_mod.estimate_pace(
                np.linalg.inv(np.array(H, dtype=float)), w, h,
                [p.frame for p in traj], [p.x for p in traj], [p.y for p in traj],
                b_frame, (b_x, b_y), fps,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            est_p = pace_mod.PaceEstimate(False, reason=f"Pace calculation failed: {exc}")
        if est_p.ok:
            fnote = ("estimated from the calibration geometry" if est_p.f_source == "geometry"
                     else "assumed (typical phone camera), because the geometry gave an implausible value")
            pace = PaceResult(
                ok=True, kmh=round(est_p.kmh, 0), lowKmh=round(est_p.low_kmh, 0), highKmh=round(est_p.high_kmh, 0),
                rmsPx=round(est_p.rms_px, 2), nPoints=est_p.n_points, focalPx=round(est_p.f_px, 0), focalSource=est_p.f_source,
                note=(
                    "A ROUGH estimate of the ball's speed at the start of the tracked path, found by fitting the ball's "
                    "flight (gravity plus a straight run along the pitch) to the tracked pixels, anchored at the bounce on "
                    "the calibrated ground. It ignores air resistance (the true release speed is usually a few km/h higher) "
                    f"and swing. The camera's focal length was {fnote}. In simulations with realistic click and tracking "
                    "error the true speed was inside this range about 90% of the time, but the range is wide: from "
                    "behind the bowler a single camera can't pin speed down closely. Use it to tell a quick ball from a "
                    "slow one, not to compare deliveries a few km/h apart."
                ),
            )
        else:
            pace = PaceResult(ok=False, kmh=None, rmsPx=est_p.rms_px, nPoints=est_p.n_points, note=est_p.reason)
    return calibrated, pace


@app.post("/api/sessions/{session_id}/calibrate")
def api_calibrate_session(session_id: str, points: CalibrationPoints):
    session = storage_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.trackingPreview or not session.trackingPreview.trajectory:
        raise HTTPException(status_code=400, detail="No tracked trajectory available to calibrate against yet")

    image_points = [points.bowlerLeft, points.bowlerRight, points.batterLeft, points.batterRight]
    H = homography.compute_homography(image_points, points.mode)

    session.calibration.homography = H.flatten().tolist()
    session.calibration.status = "good"
    session.calibration.pitchLengthMetres = homography.PITCH_LENGTH_M
    session.calibratedResult, session.paceResult = _build_results(session, H, points.mode)
    storage_db.save_session(session)
    return session.model_dump()


def _rebuild_results(session: Session) -> None:
    """Re-work line/length/pace after the tracked path or the bounce changed (if the pitch is already calibrated)."""
    import numpy as np
    hflat = session.calibration.homography
    if hflat and len(hflat) == 9 and session.trackingPreview and session.trackingPreview.trajectory:
        mode = session.calibratedResult.calibrationMode if session.calibratedResult else "creases"
        session.calibratedResult, session.paceResult = _build_results(session, np.array(hflat).reshape(3, 3), mode)
    else:
        session.calibratedResult, session.paceResult = None, None


@app.post("/api/sessions/{session_id}/trim")
def api_trim(session_id: str, req: TrimRequest):
    """Keep only the part of the tracked path between two frames (drops stray points before/after the real flight)."""
    session = storage_db.get_session(session_id)
    if not session or not session.trackingPreview:
        raise HTTPException(status_code=404, detail="Session or tracked path not found")
    full = session.trackingPreview.fullTrajectory or session.trackingPreview.trajectory
    keep = [p for p in full if req.startFrame <= p.frame <= req.endFrame]
    if len(keep) < 6:
        raise HTTPException(status_code=400, detail="Keep at least 6 tracked points.")
    session.trackingPreview.fullTrajectory = full
    session.trackingPreview.trajectory = keep
    session.trackingPreview.framesDetected = len(keep)
    if session.manualBounce and not (req.startFrame <= session.manualBounce.frame <= req.endFrame):
        session.manualBounce = None
    _rebuild_results(session)
    storage_db.save_session(session)
    return session.model_dump()


@app.post("/api/sessions/{session_id}/set-bounce")
def api_set_bounce(session_id: str, req: BounceRequest):
    """The person clicked where the ball hits the ground (or asked to go back to automatic detection)."""
    session = storage_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if req.clear:
        session.manualBounce = None
    else:
        if req.frame is None or req.x is None or req.y is None:
            raise HTTPException(status_code=400, detail="Provide frame, x and y (or clear=true)")
        session.manualBounce = ManualBounce(frame=req.frame, x=req.x, y=req.y)
    _rebuild_results(session)
    storage_db.save_session(session)
    return session.model_dump()


@app.post("/api/sessions/{session_id}/track-from-point")
def api_track_from_point(session_id: str, req: SeedRequest, background_tasks: BackgroundTasks):
    """The person clicked the ball on one frame; follow it from there (runs in the background)."""
    session = storage_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    video_path = _find_video_path(session_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found for this session")
    if session.status == "processing":
        raise HTTPException(status_code=409, detail="This session is still processing; wait for it to finish.")
    fc = session.videoQuality.frameCount
    if req.frame < 0 or (fc and req.frame >= fc):
        raise HTTPException(status_code=400, detail=f"Frame must be between 0 and {max(fc - 1, 0)}")

    session.status = "processing"
    session.processingStageLabel = "Tracking from your click"
    session.processingProgress = 0
    session.error = None
    storage_db.save_session(session)
    background_tasks.add_task(process_seeded_tracking, session_id, video_path, req)
    return {"ok": True}


def process_seeded_tracking(session_id: str, video_path: str, req: SeedRequest) -> None:
    session = storage_db.get_session(session_id)
    if session is None:
        return
    try:
        def _on_progress(pct: int) -> None:
            session.processingProgress = pct
            storage_db.save_session(session)

        result, seed = seed_tracker.track_from_seed(
            video_path, req.frame, req.x, req.y, req.direction, progress_cb=_on_progress,
        )
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
        session.trackingPreview.fullTrajectory = list(session.trackingPreview.trajectory)
        session.manualBounce = None   # a new path invalidates any bounce marked on the old one
        _rebuild_results(session)
        session.processingProgress = None
        session.processingStageLabel = PROCESSING_STAGES[-1]
        session.status = "complete"
        storage_db.save_session(session)
    except Exception as exc:  # noqa: BLE001
        session.status = "failed"
        session.error = str(exc)
        session.processingProgress = None
        storage_db.save_session(session)
        traceback.print_exc()


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
            frameCount=meta.frame_count,
            durationSeconds=round(meta.duration_seconds, 2),
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
        session.trackingPreview.fullTrajectory = list(session.trackingPreview.trajectory)
        storage_db.save_session(session)

        session.deliveries = []

        _set_stage(session, 6)  # Results ready
        session.status = "complete"
        session.note = (
            "Video metadata and quality were read from your real upload, and the ball was tracked with a "
            "trained YOLOv8 detector, checking some frames per second (see trackingPreview). Line and length "
            "become available once you calibrate the pitch, along with a rough pace estimate. Swing and turn "
            "are not measured."
        )
        storage_db.save_session(session)

    except Exception as exc:  # noqa: BLE001
        session.status = "failed"
        session.error = str(exc)
        session.note = "Processing failed."
        storage_db.save_session(session)
        traceback.print_exc()
