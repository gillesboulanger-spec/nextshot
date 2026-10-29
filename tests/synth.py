"""Synthetic clips with KNOWN camera moves, to prove the measurements are right."""
import cv2, numpy as np

W, H, FPS = 1280, 720, 30


def world(seed=3, size=(4000, 2600)):
    rng = np.random.default_rng(seed)
    img = (rng.random((size[1] // 8, size[0] // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    for _ in range(900):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, size[0])), int(rng.integers(0, size[1]))
        r = int(rng.integers(6, 40))
        if rng.random() < .5:
            cv2.circle(img, (x, y), r, c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + r // 2), c, -1)
    return img


def render(wimg, n, cx0, cy0, vx, vy, z0=1.0, dz=1.0, ax=0.0, tint=None):
    """Camera centre moves (vx, vy) px/frame in WORLD space (+vx = camera moves right),
    zoom multiplies by dz each frame (>1 = push in). ax = pan acceleration px/frame^2."""
    frames = []
    cx, cy, z, v = cx0, cy0, z0, vx
    for i in range(n):
        # map output pixel -> world: world = c + (p - centre)/z
        M = np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]], dtype=np.float32)
        f = cv2.warpAffine(wimg, M, (W, H), flags=cv2.INTER_LINEAR)
        if tint is not None:
            f = np.clip(f.astype(float) * np.array(tint), 0, 255).astype(np.uint8)
        frames.append(f)
        cx += v; cy += vy; z *= dz; v += ax
    return frames, (cx, cy, z, v)


def write(frames, path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
