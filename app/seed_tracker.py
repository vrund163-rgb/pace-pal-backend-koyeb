"""
Track the ball from a point the USER clicked, instead of relying on an AI model to find it.

Why: a generic "sports ball" detector often can't see a small, fast cricket ball at all. A person can
spot it instantly. So the user scrubs to any frame where they can see the ball and clicks it; this module
learns what the ball looks like (colour and size) from that click and follows it frame by frame, both
forward and backward in time.

How it follows the ball, each frame:
  1. Predict where the ball should be (last position + its velocity).
  2. Look only in a window around that prediction.
  3. Candidates = pixels that are BOTH (a) close to the ball's colour and (b) changed since the previous
     frame (the ball moves). Group them into blobs and keep blobs of a plausible size.
  4. Pick the blob closest to the prediction (and the best colour match). Update position, velocity, size.
  5. If nothing plausible is found, coast on the prediction for a few frames; if still nothing, stop.

This works on every frame at the video's real frame rate (cheap, unlike the AI model), so a 120 fps clip
gives a dense path. It is NOT magic: it can lose the ball behind the batter or the bowler's arm, in
heavy motion blur, or follow a similar-looking moving object by mistake. The UI draws the result on the
video so the person can see whether it followed the right thing, and re-click if not.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np

from .ball_tracking import TrackedPoint, TrackingResult


@dataclass
class Seed:
    x: float
    y: float
    area: float
    lab: np.ndarray  # mean ball colour in Lab space
    snapped: bool  # True if we found a blob under the click; False if we fell back to defaults


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def snap_seed(frame: np.ndarray, x: float, y: float) -> Seed:
    """Find the ball-sized blob under the click and measure its colour and size."""
    h, w = frame.shape[:2]
    R = 22
    x0, x1 = max(0, int(x) - R), min(w, int(x) + R + 1)
    y0, y1 = max(0, int(y) - R), min(h, int(y) + R + 1)
    patch = frame[y0:y1, x0:x1]
    lab = _lab(patch)

    ring = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]], axis=0)
    bg = np.median(ring, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)

    d8 = np.clip(dist * 255.0 / max(float(dist.max()), 1.0), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((d8 > thr) & (dist > 14)).astype(np.uint8)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cx, cy = x - x0, y - y0
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3 or area > 500:
            continue
        # distance from the click to the nearest pixel of this blob
        ys, xs = np.nonzero(labels == i)
        d = float(np.min(np.hypot(xs - cx, ys - cy)))
        if d <= 7 and (best is None or d < best[0]):
            best = (d, i, area)
    if best is not None:
        _, i, area = best
        ys, xs = np.nonzero(labels == i)
        mean_lab = lab[ys, xs].mean(axis=0)
        return Seed(float(xs.mean() + x0), float(ys.mean() + y0), float(area), mean_lab, True)

    # Fallback: no distinct blob. Use the colour right at the click and a default small size.
    xi, yi = int(round(x)), int(round(y))
    small = _lab(frame[max(0, yi - 2): yi + 3, max(0, xi - 2): xi + 3]).reshape(-1, 3)
    return Seed(float(x), float(y), 12.0, small.mean(axis=0), False)


def _search_step(frame, prev_frame, pred, half, ball_lab, area_ref, color_tol):
    """Look for the ball near `pred`. Returns (x, y, area, score, color_dist) or None."""
    h, w = frame.shape[:2]
    px, py = pred
    x0, x1 = max(0, int(px - half)), min(w, int(px + half))
    y0, y1 = max(0, int(py - half)), min(h, int(py + half))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = frame[y0:y1, x0:x1]
    proi = prev_frame[y0:y1, x0:x1]

    cdist = np.linalg.norm(_lab(roi) - ball_lab, axis=2)
    moved = cv2.absdiff(roi, proi).max(axis=2) > 14
    cand = (cdist < color_tol) & moved
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    best = None
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < max(2.0, 0.25 * area_ref) or area > 10.0 * area_ref + 12:
            continue
        cx, cy = cents[i][0] + x0, cents[i][1] + y0
        d_pred = math.hypot(cx - px, cy - py) / half
        if d_pred > 1.0:
            continue
        mask_i = labels == i
        col = float(cdist[mask_i].mean())
        score = d_pred + 0.5 * (col / color_tol)
        if best is None or score < best[3]:
            best = (float(cx), float(cy), area, score, col)
    return best


def _iter_forward(cap, start_idx, count):
    """Yield (frame_index, frame) for the `count` frames AFTER start_idx."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx + 1)
    for k in range(1, count + 1):
        ok, f = cap.read()
        if not ok:
            return
        yield start_idx + k, f


def _iter_backward(cap, start_idx, count, chunk=24):
    """Yield (frame_index, frame) for the `count` frames BEFORE start_idx, newest first, reading in small
    chunks so memory stays low."""
    remaining = min(count, start_idx)
    hi = start_idx  # exclusive upper bound of the next chunk
    while remaining > 0:
        lo = max(0, hi - min(chunk, remaining))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        frames = []
        for _ in range(hi - lo):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        for off in range(len(frames) - 1, -1, -1):
            yield lo + off, frames[off]
        remaining -= (hi - lo)
        hi = lo
        if hi <= 0:
            return


def _follow(frames_iter, first_frame, seed: Seed, fps: float, max_missed: int, half0: float, progress):
    """Follow the ball through the frames from `frames_iter`, starting from the seed frame."""
    pos = np.array([seed.x, seed.y], dtype=float)
    vel = np.zeros(2)
    have_vel = False
    area_ref = max(seed.area, 6.0)
    ball_lab = seed.lab.copy()
    color_tol = 42.0
    prev = first_frame
    missed = 0
    out = []  # (frame_idx, x, y, confidence)
    examined = 0

    for idx, frame in frames_iter:
        examined += 1
        pred = pos + vel
        speed = float(np.hypot(*vel))
        half = max(half0, 2.2 * speed + 10) if not have_vel else max(18.0, 2.2 * speed + 4 * math.sqrt(area_ref) + 8)
        found = _search_step(frame, prev, pred, half, ball_lab, area_ref, color_tol)

        if found is not None:
            x, y, area, score, col = found
            new = np.array([x, y])
            step = new - pos
            vel = step if not have_vel else 0.6 * step + 0.4 * vel
            have_vel = True
            pos = new
            area_ref = 0.7 * area_ref + 0.3 * min(max(area, 4.0), 4 * area_ref)
            missed = 0
            out.append((idx, x, y, float(np.clip(1.0 - score / 1.5, 0.05, 1.0))))
        else:
            missed += 1
            if missed > max_missed:
                break
            pos = pred  # coast on the prediction
            vel = vel * 0.95

        h, w = frame.shape[:2]
        if not (-5 <= pos[0] <= w + 5 and -5 <= pos[1] <= h + 5):
            break
        prev = frame
        if progress:
            progress(examined)
    return out, examined


def track_from_seed(
    video_path: str,
    seed_frame: int,
    click_x: float,
    click_y: float,
    direction: str = "both",
    max_seconds_each_way: float = 1.5,
    progress_cb=None,
) -> tuple[TrackingResult, Seed]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total and not (0 <= seed_frame < total):
        raise ValueError(f"Frame {seed_frame} is outside the video (0 to {total - 1}).")

    cap.set(cv2.CAP_PROP_POS_FRAMES, seed_frame)
    ok, first = cap.read()
    if not ok:
        raise ValueError("Could not read the frame you clicked on.")
    H, W = first.shape[:2]
    seed = snap_seed(first, click_x, click_y)

    each_way = max(8, int(max_seconds_each_way * fps))
    max_missed = max(4, int(fps * 0.06))
    # Fast ball, few frames per second => big jumps between frames. Scale the first search window with fps.
    half0 = 0.08 * max(W, H) * float(np.clip(120.0 / fps, 1.0, 4.0))

    jobs = []
    if direction in ("both", "forward"):
        jobs.append(("forward", 1))
    if direction in ("both", "backward"):
        jobs.append(("backward", -1))
    est_total = each_way * len(jobs)
    done = [0]

    def prog_factory(offset_holder):
        def _p(n):
            if progress_cb and n % 8 == 0:
                progress_cb(min(99, int((offset_holder[0] + n) / max(est_total, 1) * 100)))
        return _p

    points: dict[int, tuple[float, float, float]] = {seed_frame: (seed.x, seed.y, 1.0)}
    examined_total = 0
    for name, sign in jobs:
        it = _iter_forward(cap, seed_frame, each_way) if sign > 0 else _iter_backward(cap, seed_frame, each_way)
        res, examined = _follow(it, first, seed, fps, max_missed, half0, prog_factory(done))
        done[0] += examined
        examined_total += examined
        for idx, x, y, c in res:
            points[idx] = (x, y, c)
    cap.release()

    traj = [TrackedPoint(frame=i, t=i / fps, x=v[0], y=v[1], confidence=round(v[2], 3)) for i, v in sorted(points.items())]
    examined_total += 1  # the clicked frame itself

    snap_note = "" if seed.snapped else " (Note: no clear ball-sized blob was found under your click, so default size/colour were used -- if the path looks wrong, click the centre of the ball more precisely or on a frame where it stands out more.)"
    note = (
        f"Tracked from your click on frame {seed_frame}: the ball was followed in {len(traj)} of "
        f"{examined_total} frames examined (every frame, going {'both ways' if len(jobs) == 2 else jobs[0][0]}). "
        "It follows the colour and motion of what you clicked, so it can lose the ball behind a person or in heavy "
        "blur, or follow a similar-looking moving object by mistake -- check the drawn path and re-click if it "
        "looks wrong." + snap_note
    )
    return TrackingResult(frames_total=examined_total, frames_detected=len(traj), trajectory=traj, note=note), seed
