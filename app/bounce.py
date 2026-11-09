"""
Finding the bounce from a tracked ball path (pixel coordinates).

Why not just "the lowest point in the image"? With the camera behind the bowler, the ball travels AWAY
from the camera, so it appears lowest near the START of its flight, not at the bounce.

What the image actually shows: before the bounce the ball is dropping toward the ground (which pushes
its image position down the frame while it also moves away), and after the bounce it is rising (which
pushes it up the frame). So the vertical image speed changes suddenly at the bounce -- a corner in the
path. We find that corner by fitting two straight-line segments that meet at a point (a "hinge") to the
vertical image position over time, and picking the hinge time that fits best.

This gives a bounce position between frames (not just at a detected frame), and a measure of how clear
the corner is. If the corner isn't clear, we say so instead of guessing.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MIN_POINTS = 5


@dataclass
class BounceEstimate:
    ok: bool
    confidence: str  # "clear" | "weak" | "none"
    frame: float | None = None
    x: float | None = None
    y: float | None = None
    clarity: float | None = None  # 0..1: how much better two segments fit than one straight line
    reason: str = ""


def _remove_outliers(pts: np.ndarray) -> np.ndarray:
    """pts: rows of (frame, x, y), sorted by frame. Drops detections that jump far from the path."""
    if len(pts) < 5:
        return pts
    f, x, y = pts[:, 0], pts[:, 1], pts[:, 2]
    # speed in pixels per frame between consecutive detections
    step = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(f), 1)
    med = np.median(step)
    if med <= 0:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        jumps = []
        if i > 0:
            jumps.append(step[i - 1])
        if i < len(pts) - 1:
            jumps.append(step[i])
        if jumps and min(jumps) > 4.0 * med:  # far from BOTH neighbours
            keep[i] = False
    return pts[keep]


def find_bounce(frames, xs, ys) -> BounceEstimate:
    pts = np.array(sorted(zip(frames, xs, ys)), dtype=float)
    pts = _remove_outliers(pts)
    if len(pts) < MIN_POINTS:
        return BounceEstimate(False, "none", reason=f"Only {len(pts)} usable tracked points; need at least {MIN_POINTS} to see the bounce.")

    t, x, y = pts[:, 0], pts[:, 1], pts[:, 2]

    # Baseline: one straight line.
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sse_line = float(np.sum((y - A @ coef) ** 2))

    best = None
    # Candidate hinge times: between the 2nd and 2nd-last detection, in half-frame steps.
    for b in np.arange(t[1], t[-2] + 0.01, 0.5):
        before = (t <= b).astype(float)
        after = (t > b).astype(float)
        D = np.vstack([np.ones_like(t), (t - b) * before, (t - b) * after]).T
        c, *_ = np.linalg.lstsq(D, y, rcond=None)
        sse = float(np.sum((y - D @ c) ** 2))
        if best is None or sse < best[0]:
            best = (sse, b, c)

    sse_h, b, c = best
    y_b, s1, s2 = c
    clarity = 0.0 if sse_line <= 1e-9 else max(0.0, 1.0 - sse_h / sse_line)

    # Physically the slope must DROP at a bounce (ball starts rising up the frame).
    if not (s2 < s1):
        return BounceEstimate(False, "none", clarity=clarity,
                              reason="The tracked path has no corner of the kind a bounce makes (the ball does not start rising).")

    # x position at the bounce time from a straight-line fit of x over time.
    cx, *_ = np.linalg.lstsq(A, x, rcond=None)
    x_b = float(cx[0] + cx[1] * b)

    conf = "clear" if clarity >= 0.5 else "weak"
    return BounceEstimate(True, conf, frame=float(b), x=x_b, y=float(y_b), clarity=round(clarity, 3))
