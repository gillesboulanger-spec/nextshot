import json, math, os, sys, tempfile
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import world, render, write, W, H, FPS
from nextshot.motion import analyze, describe
from nextshot.seam import check, stitch
from nextshot.prepare import prepare
from nextshot.comfy import fill

T = tempfile.mkdtemp()
WIMG = world()


def clip(name, **kw):
    fr, end = render(WIMG, kw.pop("n", 60), kw.pop("cx", 2000), kw.pop("cy", 1300), **kw)
    p = os.path.join(T, name); write(fr, p); return p, end


def test_static():
    p, _ = clip("static.mp4", vx=0, vy=0)
    d = describe(analyze(p))
    assert d["static"], d


def test_pan_right_speed():
    p, _ = clip("pan.mp4", vx=4, vy=0)          # 4 px/frame = 120 px/s
    m = analyze(p)
    assert abs(m.pan_px_s - 120) < 6, m.pan_px_s
    assert abs(m.tilt_px_s) < 6
    d = describe(m)
    assert "pan right" in d["camera_prompt"] and not d["static"], d


def test_pan_left_push_in():
    p, _ = clip("pushpan.mp4", vx=-3, vy=0, dz=1.003)   # zoom ~9.4%/s
    m = analyze(p)
    assert m.pan_px_s < -80, m.pan_px_s
    assert 1.07 < m.zoom_per_s < 1.12, m.zoom_per_s
    t = describe(m)["camera_prompt"]
    assert "pan left" in t and "push in" in t, t


def test_tilt_up():
    p, _ = clip("tilt.mp4", vx=0, vy=-4)       # camera moves up in world
    m = analyze(p)
    assert m.tilt_px_s > 100, m.tilt_px_s
    assert "tilt up" in describe(m)["camera_prompt"]


def test_slowing_pan():
    p, _ = clip("slow.mp4", vx=8, vy=0, ax=-0.12, n=40)
    assert "slowing" in describe(analyze(p))["camera_prompt"]


def test_check_seamless_continuation():
    prev, (cx, cy, z, v) = clip("prev.mp4", vx=4, vy=0)
    # the "model" restarts on the old last frame (duplicate it), then carries on
    nxt, _ = clip("good.mp4", vx=4, vy=0, cx=cx - 4, cy=cy)
    r = check(prev, nxt)
    assert r["ok"], r
    assert r["trim_frames"] == 1, r          # drop the repeated start frame


def test_check_catches_stop():
    prev, (cx, cy, z, v) = clip("prev2.mp4", vx=4, vy=0)
    nxt, _ = clip("stops.mp4", vx=0, vy=0, cx=cx - 4, cy=cy)
    r = check(prev, nxt)
    assert not r["ok"] and any("speed" in x for x in r["problems"]), r


def test_check_catches_reverse():
    prev, (cx, cy, z, v) = clip("prev3.mp4", vx=4, vy=0)
    nxt, _ = clip("reverse.mp4", vx=-4, vy=0, cx=cx - 4, cy=cy)
    r = check(prev, nxt)
    assert not r["ok"] and any("direction" in x for x in r["problems"]), r


def test_colour_shift_detected_and_graded_away():
    prev, (cx, cy, z, v) = clip("prev4.mp4", vx=4, vy=0)
    nxt, _ = clip("warm.mp4", vx=4, vy=0, cx=cx - 4, cy=cy, tint=(0.8, 0.95, 1.25))
    r = check(prev, nxt)
    assert any("colour" in x for x in r["problems"]), r
    out = os.path.join(T, "joined.mp4")
    stitch(prev, nxt, out)
    # after grading, the frames right after the cut match the old shot's colour
    from nextshot.motion import read_frames
    from nextshot.seam import _colour_gap
    fr, _, _ = read_frames(out)
    assert _colour_gap(fr[59], fr[60]) < 3, _colour_gap(fr[59], fr[60])


def test_prepare_and_fill():
    prev, _ = clip("prev5.mp4", vx=4, vy=0)
    od = os.path.join(T, "next")
    info = prepare(prev, od, "She reaches for the next hold.")
    assert os.path.exists(info["start_frame"]) and os.path.exists(info["guide_clip"])
    assert info["prompt"].startswith("She reaches") and "pan right" in info["prompt"]
    tpl = {"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}, "_meta": {"title": "NS start frame"}},
           "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["9", 0]}, "_meta": {"title": "NS prompt"}},
           "3": {"class_type": "LoadImage", "inputs": {"image": "y.png"}, "_meta": {"title": "NS ref 1"}},
           "4": {"class_type": "KSampler", "inputs": {"seed": 1}, "_meta": {"title": "NS seed"}}}
    wf, notes = fill(tpl, "start_frame.png", info["prompt"], ["face.png"], "guide_tail.mp4", seed=42)
    assert wf["1"]["inputs"]["image"] == "start_frame.png"
    assert wf["2"]["inputs"]["text"] == info["prompt"] and wf["2"]["inputs"]["clip"] == ["9", 0]
    assert wf["3"]["inputs"]["image"] == "face.png" and wf["4"]["inputs"]["seed"] == 42
    assert any("guide-video" in n for n in notes)
    assert tpl["1"]["inputs"]["image"] == "x.png"     # template untouched


def test_bridge_finds_continuous_cuts():
    """A, bridge and B are cut from ONE continuous decelerating drone move, but the
    'model' placed its frames 3 frames late at the start and 5 early at the end.
    find_cuts must still join them with no skipped or repeated moment."""
    from nextshot.bridge import prepare_bridge, find_cuts, assemble
    path, _ = render(WIMG, 260, 1500, 1100, vx=9, vy=-3, dz=1.004, ax=-0.035)
    A = path[:100]                       # shot A
    Bn = path[150:]                      # shot B
    # bridge as a model would render it: starts 3 frames AFTER A's guide claims,
    # slightly softer, and runs to path frame 160
    bridge = [cv2.GaussianBlur(f, (0, 0), 0.7) for f in path[100 - 25 + 3:161]]
    pa, pb, pbr = [os.path.join(T, n) for n in ("bA.mp4", "bB.mp4", "bridge.mp4")]
    write(A, pa); write(Bn, pb); write(bridge, pbr)
    info = prepare_bridge(pa, pb, os.path.join(T, "br"), tail=25, head=9, seconds=3)
    assert info["tail"] == 25 and info["head"] == 9 and (info["bridge_frames"] - 1) % 8 == 0
    assert os.path.exists(info["start_frame"]) and os.path.exists(info["silence"])
    cuts = find_cuts(pa, pbr, pb, size=(1280, 704))
    a_last = cuts["prev_last_frame"]
    b_first = 100 - 25 + 3 + cuts["bridge_from"]
    b_last = 100 - 25 + 3 + cuts["bridge_to"]
    n_first = 150 + cuts["next_from"]
    assert b_first == a_last + 1, cuts                 # no skipped / repeated frame at cut 1
    assert n_first == b_last + 1, cuts                 # nor at cut 2
    assert cuts["cut1"]["off_pct"] < 25 and cuts["cut2"]["off_pct"] < 25, cuts
    r = assemble(pa, pbr, pb, cuts, os.path.join(T, "abc.mp4"), size=(1280, 704))
    assert r["frames"] == (a_last + 1) + (cuts["bridge_to"] - cuts["bridge_from"] + 1) + (110 - cuts["next_from"]), r


def test_bridge_fill_template():
    from nextshot.bridge import fill_bridge
    tpl = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "ltx23_bridge_api.json")))
    info = dict(end_guide_index=88, bridge_seconds=4.0, size=[1280, 704])
    wf, notes = fill_bridge(tpl, info, "drone brakes", "nextshot/gs.mp4", "nextshot/ge.mp4", seed=5)
    assert wf["410"]["inputs"]["file"] == "nextshot/gs.mp4" and wf["420"]["inputs"]["file"] == "nextshot/ge.mp4"
    assert wf["403"]["inputs"]["frame_idx"] == 88 and wf["404"]["inputs"]["frame_idx"] == 88
    assert wf["340:319"]["inputs"]["value"] == "drone brakes" and wf["340:324"]["inputs"]["value"] == 704
    assert {wf["340:286"]["inputs"]["noise_seed"], wf["340:285"]["inputs"]["noise_seed"]} == {5, 5 + 7919}
    wf, _ = fill_bridge(tpl, info, "x", "gs.mp4", "ge.mp4", start_frame_name="sf.png", audio_name="sil.wav")
    assert wf["269"]["inputs"]["image"] == "sf.png" and wf["276"]["inputs"]["audio"] == "sil.wav"


def test_bridge_reports_stall():
    """A bridge that slows to a near stop for half a second before reaching shot B
    (the 'she takes a breath' glitch) must be reported."""
    from nextshot.bridge import find_cuts
    path, _ = render(WIMG, 220, 1500, 1100, vx=8, vy=-2)
    A, Bn = path[:100], path[140:]
    hold = [path[120]] * 12
    bridge = path[100 - 25:121] + hold + path[121:150]
    pa, pb, pbr = [os.path.join(T, n) for n in ("sA.mp4", "sB.mp4", "sbr.mp4")]
    write(A, pa); write(Bn, pb); write(bridge, pbr)
    cuts = find_cuts(pa, pbr, pb, size=(1280, 704))
    assert cuts["review_slowdowns"], cuts


def test_bridge_prepare_single_end_frame():
    from nextshot.bridge import prepare_bridge
    path, _ = render(WIMG, 160, 1500, 1100, vx=6, vy=0)
    pa, pb = os.path.join(T, "hA.mp4"), os.path.join(T, "hB.mp4")
    write(path[:100], pa); write(path[120:], pb)
    info = prepare_bridge(pa, pb, os.path.join(T, "h1"), head=1, seconds=4)
    assert info["head"] == 1 and info["end_guide_index"] == info["bridge_frames"] - 1
    from nextshot.motion import frame_count_and_fps
    assert frame_count_and_fps(info["guide_end"])[0] == 1
