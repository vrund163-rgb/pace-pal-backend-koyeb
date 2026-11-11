"""
Recover a simple camera model (focal length, position, orientation) from the pitch calibration.

The calibration gives a homography H that maps ground-plane points (metres) to image pixels. A camera looking
at a flat plane with known real dimensions is enough to work out where the camera is and how it is pointed, if we
assume the picture centre is the optical centre and pixels are square (true for ordinary phone video). The focal
length is estimated from the geometry itself (two perpendicularity conditions); if that estimate is unstable or
implausible we fall back to a typical phone-camera value and say so.

World frame: x = across the pitch, y = along the pitch from the bowler-end stumps, z = height above the ground
(metres), with z pointing up toward the camera's side of the ground.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    f: float          # focal length in pixels
    cx: float
    cy: float
    R: np.ndarray     # 3x3, columns are the world x, y, z axes expressed in camera coordinates
    t: np.ndarray     # translation: camera coords = R @ world + t
    f_source: str     # "geometry" or "prior"

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, P: np.ndarray) -> np.ndarray:
        """P: (n,3) world points -> (n,2) pixels."""
        Pc = P @ self.R.T + self.t
        return np.stack([self.cx + self.f * Pc[:, 0] / Pc[:, 2], self.cy + self.f * Pc[:, 1] / Pc[:, 2]], axis=1)


def recover_camera(H_world_to_image: np.ndarray, width: int, height: int, prior_ratio: float = 0.75) -> Camera:
    cx, cy = width / 2.0, height / 2.0
    long_side = float(max(width, height))
    Hc = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ H_world_to_image
    Hc = Hc / np.linalg.norm(Hc[:, 0])
    h1, h2, h3 = Hc[:, 0], Hc[:, 1], Hc[:, 2]

    # Unknown s = 1/f^2. Two conditions: r1 . r2 = 0 and |r1| = |r2|.
    A = np.array([h1[0] * h2[0] + h1[1] * h2[1], (h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)])
    b = np.array([-h1[2] * h2[2], -(h1[2] ** 2 - h2[2] ** 2)])
    f = None
    source = "prior"
    denom = float(A @ A)
    if denom > 1e-12:
        s = float(A @ b) / denom
        if s > 0:
            f_est = 1.0 / np.sqrt(s)
            if 0.35 * long_side <= f_est <= 2.0 * long_side:
                f, source = f_est, "geometry"
    if f is None:
        f = prior_ratio * long_side

    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
    a1, a2, a3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(a1) + np.linalg.norm(a2))
    r1, r2, t = lam * a1, lam * a2, lam * a3
    if t[2] < 0:                      # the plane must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)       # make it a clean rotation
    R = U @ Vt
    # z must point UP, i.e. the camera must be above the ground: camera height = -r3 . t > 0
    if -(R[:, 2] @ t) < 0:
        R[:, 2] = -R[:, 2]
    return Camera(f=float(f), cx=cx, cy=cy, R=R, t=t, f_source=source)
