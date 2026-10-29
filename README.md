# nextshot

Continue an AI video shot seamlessly, **even when the camera is moving**.

Start-frame continuation (feed the last frame of shot A into shot B, plus the
character references) works perfectly when the camera is still. When the camera
is moving, a single frame says *where* the camera is but not *how fast or which
way it is going*, so shot B starts with a jolt. nextshot measures the camera's
motion at the end of shot A and carries it into shot B, then checks the cut.


https://github.com/user-attachments/assets/5a56da33-226e-487e-ac61-d9048e1ae526


## Install

```
pip install .            # needs ffmpeg on your PATH
nextshot --help
```

## The loop

```
nextshot analyze  shotA.mp4                      # what is the camera doing at the end?
nextshot prepare  shotA.mp4 -o next/ -a "She reaches for the next hold."
nextshot fill     my_workflow_api.json next/ -o wf.json --ref face.png --ref backpack.png
nextshot run      wf.json --url https://<pod>-8188.proxy.runpod.net -o out/     # paid
nextshot check    shotA.mp4 out/shotB.mp4        # is the cut invisible? if not, why
nextshot stitch   shotA.mp4 out/shotB.mp4 -o AB.mp4
```

* **prepare** writes `start_frame.png`, `guide_tail.mp4` (the last ~9 frames, for
  models that accept a video guide and can carry momentum), and `prompt.txt`:
  your action line plus a measured camera instruction, e.g.
  *"…moderate steady pan right (about 9% of the frame width per second), slow push
  in (about 6% per second). Continue this exact motion without any pause, jolt,
  restart or change of direction in the first second."*
* **fill** writes those inputs into *your own* ComfyUI workflow (Kling, Seedance,
  LTX…). Title the nodes `NS start frame`, `NS prompt`, `NS ref 1…`, optionally
  `NS guide video` and `NS seed`, then Export (API).
* **check** compares camera direction and speed on both sides of the cut, the
  position jump, the colour shift, and finds how many leading frames of shot B to
  drop (models often repeat the start frame).
* **stitch** trims, eases shot B's colour into shot A's grade over 1.5 s, and joins.

## Bridging two shots that already exist

When shot A and shot B were made separately and the camera moves differently in
each (a drone flying in fast in A, drifting back slowly in B), nextshot generates
the camera move between them and hides both cuts:

```
nextshot bridge-prepare A.mp4 B.mp4 -o br/ --prev-end 294 --next-start 12 [--head 9 | --head 1]
nextshot bridge-fill    templates/ltx23_bridge_api.json br/ -o wf.json -p "The drone brakes smoothly..." --url <pod>
nextshot run            wf.json --url <pod> -o out/                     # paid
nextshot bridge-cut     A.mp4 out/nextshot_bridge_00001_.mp4 B.mp4 -o ABC.mp4 --prepared br/
```

Three things make it work:

1. **Motion, not stills.** The model gets the last 25 real frames of A as a video
   guide, so it sees the drone's actual 3D path, speed and braking, plus the
   foreground and texture it has to keep. Shot B's opening goes in as an end guide.
2. **One geometry.** Every piece is cropped the same way to a size the model renders
   natively (1280x704 for LTX) and scaled once at the end, so nothing shifts at a cut.
3. **Cuts found by measurement.** Models rarely put guide frames exactly where asked
   (LTX put ours about 8 frames early). `bridge-cut` tries every hand-over point on
   both sides and keeps the one where the jump across the cut matches an ordinary
   one-frame AND two-frame step of the footage. The two-frame test is what rules out
   a hidden step backwards, which a one-frame test cannot see.

### What it looks like in practice

Real test: a 15 s drone push-in on a free-solo climber (shot 1) joined to a separately
generated close shot drifting back (shot 2), bridged with LTX 2.3. "Change across the
cut" is the picture change across the cut compared with the frames around it; a few
percent is ordinary frame-to-frame variation.

| Bridge made from | Cut 1 | Cut 2 | Notes |
|---|---|---|---|
| one still frame at each end | +11%, foreground rock vanishes | +44% | camera speed and scenery jump |
| **nextshot** (25 real frames in, cuts measured) | +7%, rock stays | +3% | used in the final film |

The demo video (`nextshot_demo_v2.mp4`, attached to the release, about 1 minute) shows the
three ways this went wrong while making the film (camera rushes in then backs away,
head morph, foreground rock vanishing at the cut), what nextshot does about each, a
frame-by-frame chart of the measured camera motion with the two chosen cuts, and the
finished opening.

### Things the numbers can't judge (and what nextshot does about them)

* **Slow-downs.** If the generated move nearly stops while both real shots keep
  moving, `bridge-cut` lists it under `review_slowdowns`. It is a flag, not a failure:
  in our film a half-second ease-off reads as the climber gathering herself before
  pushing on, and it was kept. In another shot the same thing is a hiccup. Watch it.
* **End guide: clip or single frame.** A multi-frame end guide (`--head 9`) lands the
  move on shot B's motion most precisely, but LTX may linger on its first frame and
  ease off just before the hand-over. `--head 1` (a single end frame) removes that
  slow-down at the cost of a slightly larger second cut. Try both.
* **Hand over before an action, not during it.** If the character starts turning
  her head in shot A's last frames and your prompt says she faces the wall, the
  model will morph the head back. Use `--prev-end` to end shot A's guide just before
  the action starts, or describe the action in the prompt so the bridge completes it.

## How motion is measured
Background features are tracked frame to frame and one shift + zoom + rotation is
fitted with RANSAC, so the moving subject is ignored. Verified on synthetic clips
with known pans, tilts, push-ins and decelerations (`pytest tests`).

## Licence
MIT © 2026 Gilles Boulanger
