"""Measure camera motion in a clip: per-frame global move (pan/tilt), zoom and roll.

The camera's motion shows up as the whole background shifting together, so we
track many small background features between consecutive frames and fit one
"similarity" transform (shift + scale + rotation) to them. RANSAC ignores the
features that move on their own (the climber, the anchor's hands, birds).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class FrameMotion:
    dx: float      # px/frame the IMAGE content moves right (camera moves left)
    dy: float      # px/frame the image content moves down (camera tilts up)
    scale: float   # >1 = content grows = camera pushes in / zooms in
    rot: float     # degrees/frame, image rotates counter-clockwise
    inliers: int


def read_frames(path: str, start: int = 0, count: int | None = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start < 0:
        start = max(0, n + start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while count is None or len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps, n


def frame_count_and_fps(path: str):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n, fps


def pair_motion(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> FrameMotion:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(ga, maxCorners=1500, qualityLevel=0.005,
                                  minDistance=7, mask=mask)
    if pts is None or len(pts) < 12:
        return FrameMotion(0.0, 0.0, 1.0, 0.0, 0)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, winSize=(21, 21), maxLevel=4)
    good = st.reshape(-1) == 1
    p0, p1 = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
    if len(p0) < 12:
        return FrameMotion(0.0, 0.0, 1.0, 0.0, 0)
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                         ransacReprojThreshold=1.5, maxIters=3000)
    if M is None:
        return FrameMotion(0.0, 0.0, 1.0, 0.0, 0)
    s = math.hypot(M[0, 0], M[1, 0])
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    # express the shift as the motion of the frame centre, so zoom doesn't leak into pan
    h, w = ga.shape
    cx, cy = w / 2, h / 2
    ncx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ncy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    return FrameMotion(float(ncx - cx), float(ncy - cy), float(s), float(rot), int(inl.sum()))


def clip_motion(frames: list[np.ndarray]) -> list[FrameMotion]:
    return [pair_motion(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]


@dataclass
class MotionSummary:
    fps: float
    width: int
    height: int
    frames_used: int
    # mean velocity, per SECOND
    pan_px_s: float          # + = camera moving/panning RIGHT
    tilt_px_s: float         # + = camera tilting/moving UP
    zoom_per_s: float        # >1 = pushing in
    roll_deg_s: float
    # how the speed is changing over the window (per second, per second)
    pan_accel: float
    tilt_accel: float
    zoom_accel: float
    confidence: float        # 0..1, share of frames with a solid fit

    def as_dict(self):
        return asdict(self)


def summarize(motions: list[FrameMotion], fps: float, w: int, h: int) -> MotionSummary:
    if not motions:
        return MotionSummary(fps, w, h, 0, 0, 0, 1, 0, 0, 0, 0, 0.0)
    dx = np.array([m.dx for m in motions]); dy = np.array([m.dy for m in motions])
    sc = np.array([m.scale for m in motions]); rt = np.array([m.rot for m in motions])
    ok = np.array([m.inliers >= 25 for m in motions])
    t = np.arange(len(motions)) / fps

    def slope(v):
        return float(np.polyfit(t, v, 1)[0]) if len(v) >= 3 else 0.0
    # camera moves opposite to the image content
    pan = -dx * fps; tilt = dy * fps
    zoom_log = np.log(sc) * fps
    return MotionSummary(
        fps=fps, width=w, height=h, frames_used=len(motions),
        pan_px_s=float(np.median(pan)), tilt_px_s=float(np.median(tilt)),
        zoom_per_s=float(math.exp(np.median(zoom_log))), roll_deg_s=float(np.median(rt) * fps),
        pan_accel=slope(pan), tilt_accel=slope(tilt), zoom_accel=slope(zoom_log),
        confidence=float(ok.mean()))


def analyze(path: str, seconds: float = 1.0, where: str = "tail") -> MotionSummary:
    n, fps = frame_count_and_fps(path)
    k = max(3, int(round(seconds * fps)) + 1)
    start = -k if where == "tail" else 0
    frames, fps, _ = read_frames(path, start=start, count=k)
    h, w = frames[0].shape[:2]
    return summarize(clip_motion(frames), fps, w, h)


# ---------- turning numbers into words a video model understands ----------

def _speed_word(frac_per_s: float) -> str:
    a = abs(frac_per_s)
    if a < 0.02: return "very slow"
    if a < 0.06: return "slow"
    if a < 0.15: return "moderate"
    if a < 0.30: return "fast"
    return "very fast"


def _trend_word(vel: float, accel: float) -> str:
    if abs(vel) < 1e-9:
        return "steady"
    r = abs(accel) / abs(vel)          # relative change of speed per second
    if r < 0.35:
        return "steady"
    return "accelerating" if accel * vel > 0 else "slowing"


def _trend_phrase(vel: float, accel: float) -> str:
    t = _trend_word(vel, accel)
    return {"steady": ", steady", "accelerating": ", speeding up", "slowing": ", slowing down"}[t]


def describe(m: MotionSummary) -> dict:
    parts, still = [], True
    pf = m.pan_px_s / m.width; tf = m.tilt_px_s / m.height
    zr = math.log(m.zoom_per_s) if m.zoom_per_s > 0 else 0.0
    if abs(pf) >= 0.008:
        still = False
        parts.append(f"{_speed_word(pf)} pan {'right' if pf > 0 else 'left'}"
                     f"{_trend_phrase(m.pan_px_s, m.pan_accel)} (about {abs(pf)*100:.0f}% of the frame width per second)")
    if abs(tf) >= 0.008:
        still = False
        parts.append(f"{_speed_word(tf)} tilt {'up' if tf > 0 else 'down'} "
                     f"(about {abs(tf)*100:.0f}% of the frame height per second)")
    if abs(zr) >= 0.01:
        still = False
        w = "push in" if zr > 0 else "pull back"
        parts.append(f"{_speed_word(zr)} {w}{_trend_phrase(zr, m.zoom_accel)} "
                     f"(about {abs(zr)*100:.0f}% per second)")
    if abs(m.roll_deg_s) >= 1.0:
        still = False
        parts.append(f"gentle roll {'counter-clockwise' if m.roll_deg_s > 0 else 'clockwise'}")
    if still:
        text = ("Locked-off static camera, exactly as at the end of the previous shot. "
                "No camera movement.")
    else:
        text = ("The camera is already moving when the shot begins and keeps the same move: "
                + ", ".join(parts) + ". Continue this exact motion without any pause, "
                "jolt, restart or change of direction in the first second.")
    return {"static": still, "camera_prompt": text, "low_confidence": m.confidence < 0.6}
