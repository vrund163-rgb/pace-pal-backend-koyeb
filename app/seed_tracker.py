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
    contrast: float = 0.0  # how different the ball is from its immediate surroundings (Lab distance)


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
        contrast = float(np.linalg.norm(mean_lab - bg))
        return Seed(float(xs.mean() + x0), float(ys.mean() + y0), float(area), mean_lab, True, contrast)

    # Fallback: no distinct blob. Use the colour right at the click and a default small size.
    xi, yi = int(round(x)), int(round(y))
    small = _lab(frame[max(0, yi - 2): yi + 3, max(0, xi - 2): xi + 3]).reshape(-1, 3)
    return Seed(float(x), float(y), 12.0, small.mean(axis=0), False, 0.0)


def _global_shift(prev_small: np.ndarray, cur_small: np.ndarray, scale: float) -> tuple[float, float]:
    """How far the whole picture shifted between two frames (camera shake), in full-resolution pixels."""
    try:
        (dx, dy), resp = cv2.phaseCorrelate(prev_small, cur_small)
    except cv2.error:
        return 0.0, 0.0
    if resp < 0.05 or abs(dx) > 12 or abs(dy) > 12:  # not a clean small shift: ignore
        return 0.0, 0.0
    dx, dy = dx * scale, dy * scale
    if math.hypot(dx, dy) < 1.0:                      # sub-pixel wobble: treat as no shake
        return 0.0, 0.0
    return dx, dy


def _candidates(frame, prev_frame, shift, pred, half, ball_lab, area_ref, color_tol):
    """Blobs near `pred` that look like the ball (colour) and have moved. Returns a list of dicts."""
    h, w = frame.shape[:2]
    px, py = pred
    x0, x1 = max(0, int(px - half)), min(w, int(px + half))
    y0, y1 = max(0, int(py - half)), min(h, int(py + half))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    roi = frame[y0:y1, x0:x1]
    # compare with the previous frame, lined up for any camera shake
    sx, sy = int(round(shift[0])), int(round(shift[1]))
    px0, px1, py0, py1 = x0 - sx, x1 - sx, y0 - sy, y1 - sy
    if px0 < 0 or py0 < 0 or px1 > w or py1 > h:
        proi = prev_frame[y0:y1, x0:x1]
    else:
        proi = prev_frame[py0:py1, px0:px1]

    lab_roi = _lab(roi)
    cdist = np.linalg.norm(lab_roi - ball_lab, axis=2)
    moved = cv2.absdiff(roi, proi).max(axis=2) > 14
    cand = (cdist < color_tol) & moved
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < max(2.0, 0.25 * area_ref) or area > 10.0 * area_ref + 12:
            continue
        m = (labels == i)
        # contrast against the ring of pixels just around the blob
        dil = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        ring = dil & ~cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        contrast = float(np.linalg.norm(lab_roi[m].mean(axis=0) - lab_roi[ring].mean(axis=0))) if ring.any() else 0.0
        out.append({
            "x": float(cents[i][0] + x0), "y": float(cents[i][1] + y0), "area": area,
            "col": float(cdist[m].mean()), "contrast": contrast,
        })
    return out


def _quad_predict(hist, idx):
    """Predict the ball's position at frame `idx` from its recent positions, allowing for gravity's curve."""
    pts = hist[-6:]
    f = np.array([q[0] for q in pts], float)
    xs = np.array([q[1] for q in pts], float)
    ys = np.array([q[2] for q in pts], float)
    tt = f - f[-1]
    deg = 2 if len(pts) >= 5 else 1
    cx, cy = np.polyfit(tt, xs, deg), np.polyfit(tt, ys, deg)
    dt = idx - f[-1]
    resid = float(np.sqrt(np.mean((np.polyval(cx, tt) - xs) ** 2 + (np.polyval(cy, tt) - ys) ** 2)))
    return (float(np.polyval(cx, dt)), float(np.polyval(cy, dt))), resid


def _angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return math.degrees(math.acos(float(np.clip(a @ b / (na * nb), -1, 1))))


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


def _follow(frames_iter, first_frame, seed_idx: int, seed: Seed, fps: float, max_missed: int, half0: float, progress):
    """Follow the ball through the frames from `frames_iter`, starting from the seed frame.

    Every new position must make physical sense: close to where the recent path says the ball should be,
    moving at a similar speed in a similar direction, and standing out from its surroundings like the ball did
    at the click. Anything else is ignored, and after a few frames with nothing sensible we stop rather than
    wander onto something else."""
    ball_lab = seed.lab.copy()
    area_ref = max(seed.area, 6.0)
    color_tol = 38.0
    hist = [(seed_idx, seed.x, seed.y)]          # accepted positions (frame, x, y)
    out = []                                      # (frame, x, y, confidence)
    prev = first_frame
    sc = 240.0 / max(first_frame.shape[:2])
    small = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32)
    prev_small = small(first_frame)
    missed = 0
    examined = 0
    min_contrast = max(8.0, 0.35 * seed.contrast) if seed.snapped and seed.contrast > 0 else 0.0

    for idx, frame in frames_iter:
        examined += 1
        cur_small = small(frame)
        shift = _global_shift(prev_small, cur_small, 1.0 / sc)

        if len(hist) >= 3:
            pred, resid = _quad_predict(hist, idx)
        else:
            last = hist[-1]
            if len(hist) == 2:
                gap = hist[1][0] - hist[0][0]
                vx, vy = (hist[1][1] - hist[0][1]) / gap, (hist[1][2] - hist[0][2]) / gap
            else:
                vx = vy = 0.0
            pred, resid = (last[1] + vx * (idx - last[0]), last[2] + vy * (idx - last[0])), 0.0

        if len(hist) >= 2:
            gap = hist[-1][0] - hist[-2][0]
            speed = math.hypot(hist[-1][1] - hist[-2][1], hist[-1][2] - hist[-2][2]) / max(abs(gap), 1)
        else:
            speed = 0.0

        if len(hist) >= 3:
            turn = 0.6 * max(first_frame.shape[:2]) / max(fps, 1.0)   # how far a bounce can kick the ball off its path per frame
            gate = max(3.5, 0.28 * speed) + 2.0 * resid + 0.35 * speed * missed + 1.5 * missed        # sideways: tight
            gate_y = max(3.5, 0.28 * speed, turn) + 2.0 * resid + 0.9 * max(speed, 1.5 * turn) * missed  # vertical: allows a bounce
            half = max(gate_y * 1.2, gate * 1.6, 14.0)
        else:
            gate = gate_y = half0
            half = half0

        best = None
        for cnd in _candidates(frame, prev, shift, pred, half, ball_lab, area_ref, color_tol):
            ex, ey = abs(cnd["x"] - pred[0]) / gate, abs(cnd["y"] - pred[1]) / gate_y
            if ex > 1.0 or ey > 1.0:
                continue
            d = math.hypot(ex, ey) * gate
            if min_contrast and cnd["contrast"] < min_contrast:
                continue
            last = hist[-1]
            step = np.array([cnd["x"] - last[1], cnd["y"] - last[2]])
            gap = max(abs(idx - last[0]), 1)
            if len(hist) == 2:
                # the first moves must head the same way (only checkable when the steps are big enough to measure)
                prev_step = np.array([hist[-1][1] - hist[-2][1], hist[-1][2] - hist[-2][2]])
                a_step = step * np.sign(idx - last[0])
                b_step = prev_step * np.sign(hist[-1][0] - hist[-2][0])
                if np.linalg.norm(a_step) > 5 and np.linalg.norm(b_step) > 5 and _angle_deg(a_step, b_step) > 60:
                    continue
            score = d / max(gate, 1e-6) + 0.5 * (cnd["col"] / color_tol)
            if best is None or score < best[0]:
                best = (score, cnd)

        if best is not None:
            score, cnd = best
            hist.append((idx, cnd["x"], cnd["y"]))
            area_ref = 0.7 * area_ref + 0.3 * min(max(cnd["area"], 4.0), 4 * area_ref)
            missed = 0
            out.append((idx, cnd["x"], cnd["y"], float(np.clip(1.0 - score / 2.0, 0.05, 1.0))))
            pos = (cnd["x"], cnd["y"])
        else:
            missed += 1
            if missed > max_missed:
                break
            pos = pred

        h, w = frame.shape[:2]
        if not (-5 <= pos[0] <= w + 5 and -5 <= pos[1] <= h + 5):
            break
        prev, prev_small = frame, cur_small
        if progress:
            progress(examined)
    return out, examined


def clean_path(points: dict, seed_frame: int, fps: float):
    """Drop points that don't sit on the smooth path, and keep only the unbroken run containing the seed.

    Returns (cleaned dict, number removed)."""
    if len(points) < 6:
        return points, 0
    frames = np.array(sorted(points))
    removed = 0
    for _ in range(3):
        xs = np.array([points[f][0] for f in frames]); ys = np.array([points[f][1] for f in frames])
        res = np.zeros(len(frames))
        for i, f in enumerate(frames):
            nb = [j for j in range(max(0, i - 4), min(len(frames), i + 5)) if j != i]
            if len(nb) < 4:
                continue
            tt = frames[nb] - f
            deg = 2 if len(nb) >= 6 else 1
            px = np.polyval(np.polyfit(tt, xs[nb], deg), 0.0); py = np.polyval(np.polyfit(tt, ys[nb], deg), 0.0)
            res[i] = math.hypot(px - xs[i], py - ys[i])
        mad = float(np.median(res[res > 0])) if np.any(res > 0) else 0.0
        limit = max(4.0, 4.0 * mad)
        bad = res > limit
        bad[np.where(frames == seed_frame)[0]] = False       # the user's own click is trusted
        if not bad.any():
            break
        removed += int(bad.sum())
        frames = frames[~bad]
    # keep the unbroken run (no big gaps) containing the seed
    gaps = np.diff(frames)
    big = max(6, int(0.1 * fps))
    run_start = 0
    runs = []
    for i, g in enumerate(gaps):
        if g > big:
            runs.append((run_start, i)); run_start = i + 1
    runs.append((run_start, len(frames) - 1))
    keep = frames
    for a, b in runs:
        if frames[a] <= seed_frame <= frames[b]:
            keep = frames[a:b + 1]
            break
    removed += len(frames) - len(keep)
    return {int(f): points[int(f)] for f in keep}, removed


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
    max_missed = max(4, int(fps * 0.035))
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
        res, examined = _follow(it, first, seed_frame, seed, fps, max_missed, half0, prog_factory(done))
        done[0] += examined
        examined_total += examined
        for idx, x, y, c in res:
            points[idx] = (x, y, c)
    cap.release()

    points, n_removed = clean_path(points, seed_frame, fps)
    static_object = False
    if len(points) > 6:
        arr = np.array([[v[0], v[1]] for v in points.values()])
        extent = float(np.max(np.hypot(arr[:, None, 0] - arr[None, :, 0], arr[:, None, 1] - arr[None, :, 1])))
        if extent < 12.0:                      # never moved more than a few pixels: a static object, not a ball
            static_object = True
            points = {seed_frame: points.get(seed_frame, (seed.x, seed.y, 1.0))}
    traj = [TrackedPoint(frame=i, t=i / fps, x=v[0], y=v[1], confidence=round(v[2], 3)) for i, v in sorted(points.items())]
    examined_total += 1  # the clicked frame itself

    if static_object:
        note = ("The thing you clicked doesn't move during the clip, so it isn't the ball. Pick the ball on a "
                "frame where it is in flight (it should be moving).")
        return TrackingResult(frames_total=examined_total, frames_detected=len(traj), trajectory=traj, note=note), seed

    snap_note = "" if seed.snapped else " (Note: no clear ball-sized blob was found under your click, so default size/colour were used -- if the path looks wrong, click the centre of the ball more precisely or on a frame where it stands out more.)"
    note = (
        f"Tracked from your click on frame {seed_frame}: the ball was followed in {len(traj)} of "
        f"{examined_total} frames examined (every frame, going {'both ways' if len(jobs) == 2 else jobs[0][0]}"
        f"{f'; {n_removed} stray points that did not fit the ball path were removed' if n_removed else ''}). "
        "It follows the colour and motion of what you clicked, so it can lose the ball behind a person or in heavy "
        "blur, or follow a similar-looking moving object by mistake -- check the drawn path and re-click if it "
        "looks wrong." + snap_note
    )
    return TrackingResult(frames_total=examined_total, frames_detected=len(traj), trajectory=traj, note=note), seed
