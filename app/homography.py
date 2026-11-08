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

# Order: bowler-left, bowler-right, batter-left, batter-right.
# x = lateral offset from centreline (m), y = distance from bowler-end stumps (m).
WORLD_POINTS = np.float32([
    [-HALF_STUMP_WIDTH_M, 0.0],
    [HALF_STUMP_WIDTH_M, 0.0],
    [-HALF_STUMP_WIDTH_M, PITCH_LENGTH_M],
    [HALF_STUMP_WIDTH_M, PITCH_LENGTH_M],
])


def compute_homography(image_points):
    if len(image_points) != 4:
        raise ValueError("Calibration requires exactly 4 points.")
    return cv2.getPerspectiveTransform(np.float32(image_points), WORLD_POINTS)


def pixel_to_world(H, point):
    """Only valid for points on the ground plane (e.g. a bounce)."""
    vec = np.array([point[0], point[1], 1.0])
    res = H @ vec
    res = res / res[2]
    return float(res[0]), float(res[1])
