"""Bridge two existing shots with a generated camera move, and hide both cuts.

Shot A ends with the drone flying fast one way; shot B starts with it drifting
another way. A single start frame and a single end frame are not enough for a
video model to join them: one picture shows WHERE the camera is, not HOW it is
moving in 3D. So the bridge is generated from real motion on both sides:

  * the last `tail` frames of A are fed in as a video guide at the start,
  * the first `head` frames of B (from `head_start`) as a video guide at the end,
  * all three pieces are cropped to ONE frame size the model can render
    (LTX needs multiples of 32: 1280x704), so the geometry never changes at a cut.

The model rarely puts guide frames exactly where it was asked, so the cut points
are not assumed: `find_cuts` measures every candidate and picks, on each side,
the frame where the jump across the cut is closest to an ordinary frame step.
"""
from __future__ import annotations
import copy, json, os, subprocess
import cv2
import numpy as np
from .motion import read_frames, frame_count_and_fps

DEFAULT_SIZE = (1280, 704)


# ---------- geometry: one crop for everything ----------

def fit_crop(frame: np.ndarray, size=DEFAULT_SIZE) -> np.ndarray:
    """Scale to cover `size`, then centre-crop. Same rule for every piece."""
    w, h = size
    fh, fw = frame.shape[:2]
    s = max(w / fw, h / fh)
    if abs(s - 1) > 1e-6:
        frame = cv2.resize(frame, (round(fw * s), round(fh * s)), interpolation=cv2.INTER_LANCZOS4)
    fh, fw = frame.shape[:2]
    x, y = (fw - w) // 2, (fh - h) // 2
    return frame[y:y + h, x:x + w]


def _ffmpeg_crop(src_w: int, src_h: int, size=DEFAULT_SIZE) -> str:
    """The same rule as fit_crop, as an ffmpeg filter (for the final assembly)."""
    w, h = size
    s = max(w / src_w, h / src_h)
    sw, sh = round(src_w * s), round(src_h * s)
    return f"scale={sw}:{sh}:flags=lanczos,crop={w}:{h}:{(sw - w) // 2}:{(sh - h) // 2}"


def _write(frames, path, fps):
    h, w = frames[0].shape[:2]
    tmp = path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                    "-crf", "8", "-pix_fmt", "yuv420p", path], check=True)
    os.remove(tmp)


def _valid_len(n: int) -> int:
    """LTX wants 8n+1 frames."""
    return max(1, ((n - 1) // 8) * 8 + 1)


# ---------- 1. prepare the two motion guides ----------

def prepare_bridge(prev: str, nxt: str, out_dir: str, tail: int = 25, head: int = 9,
                   head_start: int = 0, seconds: float = 4.0, size=DEFAULT_SIZE,
                   prev_end: int | None = None) -> dict:
    """prev_end: use shot A only up to this frame (exclusive), e.g. to stop before a glitch."""
    os.makedirs(out_dir, exist_ok=True)
    tail, head = _valid_len(tail), _valid_len(head)
    n_prev, fps = frame_count_and_fps(prev)
    n_prev = min(n_prev, prev_end) if prev_end else n_prev
    a, _, _ = read_frames(prev, start=n_prev - tail, count=tail)
    b, _, _ = read_frames(nxt, start=head_start, count=head)
    total = _valid_len(int(round(seconds * fps)) + 1)
    end_idx = ((total - head) // 8) * 8          # latent-aligned slot for the end guide
    gs, ge = os.path.join(out_dir, "guide_start.mp4"), os.path.join(out_dir, "guide_end.mp4")
    _write([fit_crop(f, size) for f in a], gs, fps)
    _write([fit_crop(f, size) for f in b], ge, fps)
    # two helper files the LTX workflow needs as inputs (so it never depends on files
    # that only exist on one machine): a still of the first guide frame, and silence
    # for the audio branch (the bridge is rendered without sound)
    sf = os.path.join(out_dir, "start_frame.png")
    cv2.imwrite(sf, fit_crop(a[0], size))
    sil = os.path.join(out_dir, "silence.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=stereo", "-t", str(round(seconds + 2, 2)), sil], check=True)
    info = dict(previous_clip=prev, next_clip=nxt, fps=fps, size=list(size),
                start_frame=sf, silence=sil,
                prev_end=n_prev, guide_start=gs, guide_start_from_frame=n_prev - tail, tail=tail,
                guide_end=ge, guide_end_from_frame=head_start, head=head,
                bridge_frames=total, bridge_seconds=(total - 1) / fps, end_guide_index=end_idx)
    json.dump(info, open(os.path.join(out_dir, "bridge.json"), "w"), indent=2)
    return info


# ---------- 2. write them into a ComfyUI workflow ----------
# Node titles (right-click > Title in ComfyUI, then Export (API)):
#   NS guide start    LoadVideo for the shot-A tail
#   NS guide end      LoadVideo for the shot-B head
#   NS end guide      every LTXVAddGuide that places the end clip (frame_idx is set)
#   NS duration       seconds (float)      NS width / NS height   frame size
#   NS prompt / NS seed  as for `fill`

def _titled(wf):
    out = {}
    for nid, node in wf.items():
        t = (node.get("_meta") or {}).get("title", "").upper()
        if t.startswith("NS "):
            out.setdefault(t, []).append(nid)
    return out


def fill_bridge(template: dict, info: dict, prompt: str, guide_start_name: str,
                guide_end_name: str, seed: int | None = None, start_frame_name: str | None = None,
                audio_name: str | None = None) -> tuple[dict, list[str]]:
    wf = copy.deepcopy(template)
    t = _titled(wf)
    notes = []

    def put(title, key, value, required=False):
        ids = t.get(title, [])
        if not ids:
            (notes.append(f'no "{title.title()}" node') if not required else None)
            if required:
                raise ValueError(f'No node titled "{title.title()}" in the workflow')
        for i in ids:
            if key in wf[i]["inputs"]:
                wf[i]["inputs"][key] = value

    put("NS GUIDE START", "file", guide_start_name, required=True)
    put("NS GUIDE END", "file", guide_end_name, required=True)
    put("NS END GUIDE", "frame_idx", info["end_guide_index"], required=True)
    if start_frame_name:
        put("NS START FRAME", "image", start_frame_name)
    if audio_name:
        put("NS AUDIO", "audio", audio_name)
    put("NS DURATION", "value", float(info["bridge_seconds"]))
    put("NS WIDTH", "value", int(info["size"][0]))
    put("NS HEIGHT", "value", int(info["size"][1]))
    for i in t.get("NS PROMPT", []):
        for k in ("value", "text"):
            if k in wf[i]["inputs"]:
                wf[i]["inputs"][k] = prompt
    if seed is not None:
        for j, i in enumerate(t.get("NS SEED", [])):
            for k in ("noise_seed", "seed"):
                if k in wf[i]["inputs"]:
                    wf[i]["inputs"][k] = seed + j * 7919
    return wf, notes


# ---------- 3. find the invisible cuts ----------

def _grey(frames, size):
    return [cv2.resize(cv2.cvtColor(fit_crop(f, size), cv2.COLOR_BGR2GRAY), (320, 176)).astype(float)
            for f in frames]


def _d(a, b):
    return float(np.abs(a - b).mean())


def _seam_score(X, i, Y, j):
    """Score the cut  X[..i] | Y[j..]  (0 = looks like continuous footage).

    One-frame jump must match a normal step, AND the two-frame jumps must match a
    normal two-frame step: that is what tells a step forward from a step back
    (going back one frame looks like a normal-sized change, but X[i-1]->Y[j]
    would then be almost no change at all)."""
    s1 = (_d(X[i - 1], X[i]) + _d(Y[j], Y[j + 1])) / 2
    s2 = (_d(X[i - 2], X[i]) + _d(Y[j], Y[j + 2])) / 2
    s1, s2 = max(s1, 1e-6), max(s2, 1e-6)
    jump = _d(X[i], Y[j])
    e = (abs(jump - s1) / s1 + abs(_d(X[i - 1], Y[j]) - s2) / s2 + abs(_d(X[i], Y[j + 1]) - s2) / s2) / 3
    return e, jump, s1


def find_cuts(prev: str, bridge: str, nxt: str, size=DEFAULT_SIZE, tail: int = 25,
              search_next: int = 30, prev_end: int | None = None) -> dict:
    """Pick where A hands over to the bridge and where the bridge hands over to B.

    A cut is scored by how far the jump across it is from an ordinary frame step
    on either side (0% = indistinguishable from continuous footage)."""
    n_prev, _ = frame_count_and_fps(prev)
    n_prev = min(n_prev, prev_end) if prev_end else n_prev
    A, _, _ = read_frames(prev, start=n_prev - (tail + 2), count=tail + 2)
    Bf, _, _ = read_frames(bridge)
    C, _, _ = read_frames(nxt, start=0, count=search_next + 2)
    A, Bf, C = _grey(A, size), _grey(Bf, size), _grey(C, size)
    a0 = n_prev - (tail + 2)                 # index of A[0] in the original clip
    nb = len(Bf)

    # cut 1: keep A up to original frame a0+ai, then the bridge from frame c.
    # Both are searched freely: the model rarely puts guide frames exactly where asked.
    best1 = None
    for ai in range(2, len(A)):
        for c in range(1, min(tail + 8, nb - 3)):
            sc, jump, norm = _seam_score(A, ai, Bf, c)
            if best1 is None or sc < best1[0]:
                best1 = (sc, c, a0 + ai, jump, norm)

    # cut 2: keep the bridge up to frame e, then B from frame k
    best2 = None
    for e in range(max(best1[1] + 8, nb // 2), nb):
        for k in range(0, len(C) - 2):
            sc, jump, norm = _seam_score(Bf, e, C, k)
            if best2 is None or sc < best2[0]:
                best2 = (sc, e + 1, k, jump, norm)

    # slowdowns: the generated part slowing almost to a stop while both real shots keep
    # moving. Reported for REVIEW, not as an error: a pause can read as a glitch, or as a
    # natural beat (a climber gathering herself before pushing on). Only an eye can tell.
    keep = list(range(best1[1], best2[1]))
    steps = [_d(Bf[i], Bf[i + 1]) for i in keep[:-1]]
    ref = min(np.median([_d(A[i], A[i + 1]) for i in range(len(A) - 1)]),
              np.median([_d(C[i], C[i + 1]) for i in range(len(C) - 1)]))
    stalls, run = [], []
    for i, st in zip(keep, steps):
        if st < 0.4 * ref:
            run.append(i)
        else:
            if len(run) >= 4: stalls.append((run[0], run[-1]))
            run = []
    if len(run) >= 4: stalls.append((run[0], run[-1]))
    return dict(review_slowdowns=[dict(bridge_frames=f"{a}-{b}", seconds=round((b - a + 1) / 24, 2)) for a, b in stalls],
                prev_last_frame=best1[2], bridge_from=best1[1], bridge_to=best2[1] - 1,
                next_from=best2[2],
                cut1=dict(jump=round(best1[3], 2), normal_step=round(best1[4], 2), off_pct=round(best1[0] * 100, 1)),
                cut2=dict(jump=round(best2[3], 2), normal_step=round(best2[4], 2), off_pct=round(best2[0] * 100, 1)))


# ---------- 4. assemble ----------

def _dims(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def assemble(prev: str, bridge: str, nxt: str, cuts: dict, out: str, size=DEFAULT_SIZE,
             final_size=(1280, 720), fps: float | None = None) -> dict:
    fps = fps or frame_count_and_fps(prev)[1]
    pieces = []
    for p in (prev, bridge, nxt):
        w, h = _dims(p)
        pieces.append(_ffmpeg_crop(w, h, size))
    fw, fh = final_size
    fc = (f"[0:v]trim=end_frame={cuts['prev_last_frame'] + 1},setpts=PTS-STARTPTS,{pieces[0]},setsar=1,fps={fps},format=yuv420p[a];"
          f"[1:v]trim=start_frame={cuts['bridge_from']}:end_frame={cuts['bridge_to'] + 1},setpts=PTS-STARTPTS,{pieces[1]},setsar=1,fps={fps},format=yuv420p[b];"
          f"[2:v]trim=start_frame={cuts['next_from']},setpts=PTS-STARTPTS,{pieces[2]},setsar=1,fps={fps},format=yuv420p[c];"
          f"[a][b][c]concat=n=3:v=1:a=0,scale={fw}:{fh}:flags=lanczos,setsar=1[v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", prev, "-i", bridge, "-i", nxt,
                    "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    n, _ = frame_count_and_fps(out)
    return dict(out=out, frames=n, seconds=round(n / fps, 2), **cuts)
