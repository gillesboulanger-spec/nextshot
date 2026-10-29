"""Everything the next generation needs, taken from the end of the previous shot."""
from __future__ import annotations
import json, os, subprocess
import cv2
from .motion import analyze, describe, read_frames, frame_count_and_fps


def prepare(prev: str, out_dir: str, action: str, refs: list[str] | None = None,
            seconds: float = 1.0, guide_frames: int = 9) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    summ = analyze(prev, seconds=seconds, where="tail")
    desc = describe(summ)
    n, fps = frame_count_and_fps(prev)
    k = max(guide_frames, 2)
    tail, _, _ = read_frames(prev, start=-k, count=k)
    last = os.path.join(out_dir, "start_frame.png")
    cv2.imwrite(last, tail[-1])
    # the last few frames as a short clip: models that accept a video guide use these
    # to carry the camera's momentum, not just its position
    gdir = os.path.join(out_dir, "guide_frames"); os.makedirs(gdir, exist_ok=True)
    for i, f in enumerate(tail):
        cv2.imwrite(os.path.join(gdir, f"{i:03d}.png"), f)
    guide = os.path.join(out_dir, "guide_tail.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(gdir, "%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "14", guide], check=True)
    prompt = f"{action.strip()} {desc['camera_prompt']}".strip()
    info = dict(previous_clip=prev, start_frame=last, guide_clip=guide, guide_frame_count=k,
                references=refs or [], prompt=prompt, camera=desc, motion=summ.as_dict())
    with open(os.path.join(out_dir, "nextshot.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    with open(os.path.join(out_dir, "prompt.txt"), "w") as fh:
        fh.write(prompt + "\n")
    return info
