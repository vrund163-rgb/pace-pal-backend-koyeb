"""
Pace from one camera, using physics.

What is known: the camera (recovered from the pitch calibration), the frame rate, gravity (9.81 m/s^2, straight
down), and that the ball touches the ground at the bounce. A ball in flight follows a simple arc: constant
speed along the ground plus a gravity-bent vertical motion. Each tracked pixel is a line of sight from the
camera; we find the single arc, ending at the bounce on the ground, whose projection best matches ALL the
tracked pixels. Gravity supplies the real-world scale, so the arc's speed comes out in metres per second.

What it assumes / ignores (so the number is approximate):
  * air resistance is ignored (a real ball slows a little in flight, so the true release speed is typically a few
    km/h higher than what this reports as the speed at the start of the tracked path);
  * swing and seam movement are ignored (the ball is assumed to travel in a straight line along the ground);
  * the camera model (focal length etc.) comes from the four clicked calibration points, so click error and lens
    distortion feed into the result. We estimate that spread by re-running the fit with the calibration clicks and
    tracked points jiggled by realistic amounts, and report a RANGE, not a single exact number.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .camera import Camera, recover_camera

G = 9.81
KMH = 3.6
RANGE_WIDEN = 1.4          # from simulation: widens the re-fit spread so it covers the true speed ~90% of the time
MAX_RMS_PX = 4.0
MAX_RELATIVE_HALF_WIDTH = 0.45


def _residuals(theta, cam: Camera, tau_obs: np.ndarray, uv_obs: np.ndarray, tb_prior_sigma: float, fps: float):
    Xb, Yb, vx, vy, vz, tb_off = theta
    tau = tau_obs - tb_off / fps          # seconds relative to the (adjusted) bounce time; <= 0 before the bounce
    P = np.stack([Xb + vx * tau, Yb + vy * tau, vz * tau - 0.5 * G * tau ** 2], axis=1)
    uv = cam.project(P)
    r = (uv - uv_obs).ravel()
    return np.concatenate([r, [tb_off / tb_prior_sigma]])


def _lm(theta0, cam, tau_obs, uv_obs, fps, iters=40):
    theta = np.array(theta0, dtype=float)
    lam = 1e-2
    f = lambda th: _residuals(th, cam, tau_obs, uv_obs, 0.7, fps)
    r = f(theta)
    cost = float(r @ r)
    steps = np.array([0.02, 0.02, 0.05, 0.05, 0.05, 0.02])
    for _ in range(iters):
        J = np.empty((len(r), len(theta)))
        for j in range(len(theta)):
            d = np.zeros(len(theta)); d[j] = steps[j]
            J[:, j] = (f(theta + d) - f(theta - d)) / (2 * steps[j])
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _try in range(8):
            try:
                delta = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            new = theta + delta
            r_new = f(new)
            c_new = float(r_new @ r_new)
            if np.isfinite(c_new) and c_new < cost:
                theta, r, cost = new, r_new, c_new
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        if not improved or np.linalg.norm(delta) < 1e-6:
            break
    return theta, cost


@dataclass
class FitResult:
    speed_ms: float          # speed at the first tracked point
    rms_px: float            # how well the arc matches the tracked pixels
    theta: np.ndarray
    n_obs: int


def fit_arc(cam: Camera, frames, uv, bounce_frame: float, fps: float, ground_xy_init, theta0=None) -> FitResult | None:
    frames = np.asarray(frames, float)
    uv = np.asarray(uv, float)
    keep = frames <= bounce_frame + 0.5
    frames, uv = frames[keep], uv[keep]
    if len(frames) < 8:
        return None
    tau_obs = (frames - bounce_frame) / fps
    best = None
    if theta0 is not None:                       # warm start: just polish from a known good answer
        t0 = np.array(theta0, float)
        t0[0], t0[1] = ground_xy_init
        best = _lm(t0, cam, tau_obs, uv, fps)
    else:
        for sign in (+1, -1):
            for speed in (22.0, 32.0, 40.0):
                for vz in (-6.0, 0.0):
                    th0 = [ground_xy_init[0], ground_xy_init[1], 0.0, sign * speed, vz, 0.0]
                    th, cost = _lm(th0, cam, tau_obs, uv, fps)
                    if best is None or cost < best[1]:
                        best = (th, cost)
    th, cost = best
    # Robustness: a couple of stray tracked points can drag a least-squares fit a long way. Drop observations that
    # sit far from the fitted arc and refit.
    for _ in range(2):
        r = _residuals(th, cam, tau_obs, uv, 0.7, fps)[:-1].reshape(-1, 2)
        d = np.hypot(r[:, 0], r[:, 1])
        keep = d < max(2.5, 3.0 * float(np.median(d)))
        if keep.all() or keep.sum() < 8:
            break
        frames, tau_obs, uv = frames[keep], tau_obs[keep], uv[keep]
        th, cost = _lm(th, cam, tau_obs, uv, fps)
    Xb, Yb, vx, vy, vz, off = th
    tau0 = tau_obs[0] - off / fps
    v0 = np.array([vx, vy, vz - G * tau0])           # velocity at the first tracked point
    n = len(frames)
    rms = float(np.sqrt(max(cost - (off / 0.7) ** 2, 0.0) / (2 * n)))
    return FitResult(float(np.linalg.norm(v0)), rms, th, n)


@dataclass
class PaceEstimate:
    ok: bool
    kmh: float | None = None
    low_kmh: float | None = None
    high_kmh: float | None = None
    rms_px: float | None = None
    n_points: int = 0
    f_px: float | None = None
    f_source: str = ""
    reason: str = ""


def estimate_pace(
    H_world_to_image: np.ndarray,
    img_w: int,
    img_h: int,
    frames,
    xs,
    ys,
    bounce_frame: float,
    bounce_px,
    fps: float,
    n_boot: int = 40,
    click_sigma_px: float = 2.5,
    track_sigma_px: float = 0.8,
    seed: int = 0,
) -> PaceEstimate:
    from . import homography as hg

    frames = np.asarray(frames, float)
    uv = np.stack([np.asarray(xs, float), np.asarray(ys, float)], axis=1)
    pre = frames <= bounce_frame + 0.5
    if pre.sum() < 8:
        return PaceEstimate(False, n_points=int(pre.sum()),
                            reason="Fewer than 8 tracked points before the bounce, which isn't enough to fit the ball's arc.")

    Hinv = np.linalg.inv(H_world_to_image)

    def ground_at(Hw2i, px):
        v = np.linalg.inv(Hw2i) @ np.array([px[0], px[1], 1.0])
        v = v / v[2]
        return v[0], v[1]

    cam = recover_camera(H_world_to_image, img_w, img_h)
    base = fit_arc(cam, frames, uv, bounce_frame, fps, ground_at(H_world_to_image, bounce_px))
    if base is None:
        return PaceEstimate(False, reason="Could not fit an arc to the tracked path.")

    # Re-run with realistic jiggles of the clicks and the tracked pixels to see how much the answer moves.
    rng = np.random.default_rng(seed)
    corners_world = hg.WORLD_POINTS_BY_MODE["creases"]
    corners_img = np.array([(H_world_to_image @ np.array([x, y, 1.0])) for x, y in corners_world])
    corners_img = corners_img[:, :2] / corners_img[:, 2:3]
    speeds = []
    import cv2
    for _ in range(n_boot):
        c2 = corners_img + rng.normal(0, click_sigma_px, corners_img.shape)
        Hn_i2w = cv2.getPerspectiveTransform(c2.astype(np.float32), corners_world.astype(np.float32))
        Hn_w2i = np.linalg.inv(Hn_i2w)
        cam2 = recover_camera(Hn_w2i, img_w, img_h)
        uv2 = uv + rng.normal(0, max(track_sigma_px, min(base.rms_px, 4.0)), uv.shape)
        bf2 = bounce_frame + rng.normal(0, 0.5)
        r = fit_arc(cam2, frames, uv2, bf2, fps, ground_at(Hn_w2i, np.array(bounce_px) + rng.normal(0, track_sigma_px, 2)), theta0=base.theta)
        if r is not None and np.isfinite(r.speed_ms):
            speeds.append(r.speed_ms * KMH)
    if len(speeds) < max(10, n_boot // 2):
        return PaceEstimate(False, n_points=base.n_obs, reason="The fit was unstable when the calibration was nudged slightly.")
    lo, hi = np.percentile(speeds, [5, 95])
    kmh = float(base.speed_ms * KMH)
    half = max((hi - lo) / 2.0, 0.05 * kmh) * RANGE_WIDEN
    est = PaceEstimate(
        True, kmh=kmh, low_kmh=max(float(kmh - half), 15.0), high_kmh=float(kmh + half), rms_px=base.rms_px,
        n_points=base.n_obs, f_px=cam.f, f_source=cam.f_source,
    )
    if base.rms_px > MAX_RMS_PX:
        return PaceEstimate(False, kmh=kmh, rms_px=base.rms_px, n_points=base.n_obs, f_px=cam.f, f_source=cam.f_source,
                            reason=f"The fitted arc doesn't match the tracked path well (average miss {base.rms_px:.1f} px), so the speed isn't trustworthy.")
    if half / max(kmh, 1.0) > MAX_RELATIVE_HALF_WIDTH or not (20.0 <= kmh <= 200.0):
        return PaceEstimate(False, kmh=kmh, low_kmh=est.low_kmh, high_kmh=est.high_kmh, rms_px=base.rms_px, n_points=base.n_obs,
                            f_px=cam.f, f_source=cam.f_source,
                            reason="The speed is too uncertain from this recording (the answer moves too much when the calibration and tracking are nudged slightly).")
    return est
