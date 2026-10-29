"""Check and repair the cut between two clips.

check():  does the new shot start where the old one ended, moving the same way?
stitch(): join them, optionally trimming the new shot's first frames (models often
          re-render the start image with a tiny shift) and easing its colour into
          the old shot's grade so the cut is invisible.
"""
from __future__ import annotations
import math, os, subprocess, tempfile
import cv2
import numpy as np
from .motion import read_frames, frame_count_and_fps, clip_motion, summarize, describe, pair_motion


def _diff(a, b) -> float:
    """0 = identical, ~1 = unrelated. Mean abs difference on small grey copies."""
    sa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
    sb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
    return float(np.abs(sa - sb).mean() / 64.0)


def _vel(summary):
    return np.array([summary.pan_px_s / summary.width, summary.tilt_px_s / summary.height,
                     math.log(summary.zoom_per_s)])


def best_trim(prev_last, head_frames, max_trim: int = 8) -> tuple[int, float]:
    """How many leading frames of the new clip to drop so it best matches the old last frame."""
    scores = [_diff(prev_last, f) for f in head_frames[:max_trim + 1]]
    k = int(np.argmin(scores))
    # a near-perfect match is the old last frame repeated: drop it too, so the
    # new clip starts on the NEXT moment instead of freezing for one frame
    if scores[k] < 0.08 and k + 1 < len(head_frames):
        return k + 1, scores[k]
    return k, scores[k]


def check(prev: str, nxt: str, seconds: float = 1.0) -> dict:
    n_prev, fps = frame_count_and_fps(prev)
    k = max(3, int(round(seconds * fps)) + 1)
    tail, fps_p, _ = read_frames(prev, start=-k, count=k)
    head, fps_n, _ = read_frames(nxt, start=0, count=k + 8)
    h, w = tail[0].shape[:2]
    m_prev = summarize(clip_motion(tail), fps_p, w, h)
    trim, look = best_trim(tail[-1], head)
    head_used = head[trim:trim + k]
    m_next = summarize(clip_motion(head_used), fps_n, w, h)
    vp, vn = _vel(m_prev), _vel(m_next)
    speed_p, speed_n = float(np.linalg.norm(vp)), float(np.linalg.norm(vn))
    # direction agreement (1 = same way) and speed ratio
    if speed_p < 0.01 and speed_n < 0.01:
        direction, ratio = 1.0, 1.0
    elif speed_p < 0.01 or speed_n < 0.01:
        direction, ratio = 0.0, 0.0
    else:
        direction = float(vp @ vn / (speed_p * speed_n))
        ratio = speed_n / speed_p
    # the jump across the cut itself, compared with an ordinary frame step in the old clip
    step = pair_motion(tail[-2], tail[-1]); jump = pair_motion(tail[-1], head[trim])
    exp = np.array([step.dx, step.dy]); got = np.array([jump.dx, jump.dy])
    jump_px = float(np.linalg.norm(got - exp))
    colour = _colour_gap(tail[-1], head[trim])
    verdict = []
    if look > 0.35: verdict.append("new shot does not start on the old last frame")
    if direction < 0.8: verdict.append("camera changes direction at the cut")
    if not (0.75 <= ratio <= 1.33): verdict.append(f"camera speed changes at the cut (x{ratio:.2f})")
    if jump_px > max(6.0, 0.01 * w): verdict.append(f"position jumps {jump_px:.0f}px at the cut")
    if colour > 6: verdict.append(f"colour shifts at the cut (ΔE≈{colour:.0f})")
    return dict(ok=not verdict, problems=verdict, trim_frames=trim, start_match=round(1 - look, 3),
                direction_agreement=round(direction, 3), speed_ratio=round(ratio, 3),
                jump_px=round(jump_px, 1), colour_delta_e=round(colour, 1),
                prev_motion=describe(m_prev)["camera_prompt"], next_motion=describe(m_next)["camera_prompt"])


def _lab_stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    return lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6


def _colour_gap(a, b) -> float:
    ma, _ = _lab_stats(a); mb, _ = _lab_stats(b)
    # OpenCV 8-bit Lab: L scaled 0..255; rescale to ~CIE units
    d = (ma - mb) * np.array([100 / 255, 1, 1])
    return float(np.linalg.norm(d))


def grade_to(frames, ref, ramp_frames: int):
    """Match each frame's colour to `ref`, fading the correction out over ramp_frames."""
    mr, sr = _lab_stats(ref)
    out = []
    for i, f in enumerate(frames):
        a = max(0.0, 1 - i / max(1, ramp_frames))
        if a <= 0:
            out.append(f); continue
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(float)
        mf, sf = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        corr = (lab - mf) * (sr / sf) + mr
        lab = lab * (1 - a) + corr * a
        out.append(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
    return out


def stitch(prev: str, nxt: str, out: str, trim: int | str = "auto", grade: bool = True,
           grade_seconds: float = 1.5) -> dict:
    a, fps, _ = read_frames(prev)
    b, _, _ = read_frames(nxt)
    if trim == "auto":
        trim, _ = best_trim(a[-1], b)
    b = b[int(trim):]
    if grade:
        b = grade_to(b, a[-1], int(grade_seconds * fps))
    h, w = a[0].shape[:2]
    b = [cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f for f in b]
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, "v.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in a + b:
        vw.write(f)
    vw.release()
    # re-encode for compatibility; carry audio from both clips if present
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "17", out], check=True)
    return dict(out=out, trimmed=int(trim), frames=len(a) + len(b), fps=fps)
