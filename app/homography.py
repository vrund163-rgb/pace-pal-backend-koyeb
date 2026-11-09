"""
Ground-plane calibration via a 4-point homography.

Computes a standard photogrammetry transform (a planar homography) from 4
user-marked points -- the base of the stumps at both ends -- whose REAL-WORLD
positions are fixed and known (standard cricket pitch dimensions). Any point
on the ground plane (like a bounce location) can then be converted from
pixel coordinates to real-world metres.

What this can NOT do: convert a point that is up in the air (the release
point, or the ball mid-flight) to a real-world position, because a single 2D
image has no depth information off the ground plane. Applying this to an
airborne point would silently give a wrong number, so it is only ever used
on the estimated bounce point.
"""

from __future__ import annotations
import cv2
import numpy as np

PITCH_LENGTH_M = 20.12
HALF_STUMP_WIDTH_M = 0.1143

HALF_CREASE_WIDTH_M = 1.32  # the bowling crease is 2.64 m long, centred on the middle stump

# Order of the 4 clicked points: bowler-left, bowler-right, batter-left, batter-right.
# x = lateral offset from centreline (m), y = distance from bowler-end stumps (m).
# The bowling crease is the line the stumps stand on, so it sits at the same y as the stumps.
#
# "creases": the OUTER ENDS of the bowling crease (where it meets the return crease). Recommended: these
#            points are 2.64 m apart, so a click a few pixels off barely matters.
# "stumps":  the base of the outer stumps (only 0.23 m apart). Far less accurate: at the far end the two
#            stumps are only a few pixels apart in the video, so small click errors cause large length errors
#            (simulated: several metres). Use only if the pitch has no crease lines.
WORLD_POINTS_BY_MODE = {
    "creases": np.float32([
        [-HALF_CREASE_WIDTH_M, 0.0], [HALF_CREASE_WIDTH_M, 0.0],
        [-HALF_CREASE_WIDTH_M, PITCH_LENGTH_M], [HALF_CREASE_WIDTH_M, PITCH_LENGTH_M],
    ]),
    "stumps": np.float32([
        [-HALF_STUMP_WIDTH_M, 0.0], [HALF_STUMP_WIDTH_M, 0.0],
        [-HALF_STUMP_WIDTH_M, PITCH_LENGTH_M], [HALF_STUMP_WIDTH_M, PITCH_LENGTH_M],
    ]),
}


def compute_homography(image_points, mode="creases"):
    if len(image_points) != 4:
        raise ValueError("Calibration requires exactly 4 points.")
    if mode not in WORLD_POINTS_BY_MODE:
        raise ValueError(f"Unknown calibration mode: {mode}")
    return cv2.getPerspectiveTransform(np.float32(image_points), WORLD_POINTS_BY_MODE[mode])


def pixel_to_world(H, point):
    """Only valid for points on the ground plane (e.g. a bounce)."""
    vec = np.array([point[0], point[1], 1.0])
    res = H @ vec
    res = res / res[2]
    return float(res[0]), float(res[1])
