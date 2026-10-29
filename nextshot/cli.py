"""nextshot — continue a shot seamlessly, even when the camera is moving.

  nextshot analyze  prev.mp4                       what is the camera doing at the end?
  nextshot prepare  prev.mp4 -o next/ -a "She reaches for the next hold."
                                                   start frame + guide clip + prompt
  nextshot fill     template_api.json next/ -o wf.json [--ref face.png ...]
                                                   write the inputs into your ComfyUI workflow
  nextshot run      wf.json --url https://<pod>-8188.proxy.runpod.net -o out/   (PAID: runs a generation)
  nextshot check    prev.mp4 new.mp4               is the cut invisible? what's wrong if not
  nextshot stitch   prev.mp4 new.mp4 -o joined.mp4 trim the overlap, match the grade, join

Bridging two shots that already exist (camera moving differently in each):
  nextshot bridge-prepare A.mp4 B.mp4 -o br/ [--prev-end 294] [--next-start 12]
                                                   motion guides from both shots (real frames, not stills)
  nextshot bridge-fill    templates/ltx23_bridge_api.json br/ -o wf.json -p "..." [--url ...]
  nextshot run            wf.json --url ... -o out/                  (PAID)
  nextshot bridge-cut     A.mp4 out/bridge.mp4 B.mp4 -o ABC.mp4 --prepared br/
                                                   find both invisible cuts by measurement, then join
"""
from __future__ import annotations
import argparse, json, os, sys
from .motion import analyze, describe
from .prepare import prepare
from .seam import check, stitch
from .comfy import fill, Comfy
from .bridge import prepare_bridge, fill_bridge, find_cuts, assemble


def main(argv=None):
    p = argparse.ArgumentParser(prog="nextshot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("analyze"); a.add_argument("clip"); a.add_argument("--seconds", type=float, default=1.0)
    r = sp.add_parser("prepare"); r.add_argument("clip"); r.add_argument("-o", "--out", required=True)
    r.add_argument("-a", "--action", default="", help="what happens next, in plain words")
    r.add_argument("--ref", action="append", default=[]); r.add_argument("--seconds", type=float, default=1.0)
    r.add_argument("--guide-frames", type=int, default=9)
    f = sp.add_parser("fill"); f.add_argument("template"); f.add_argument("prepared")
    f.add_argument("-o", "--out", required=True); f.add_argument("--ref", action="append", default=[])
    f.add_argument("--url", help="ComfyUI URL: also upload the start frame / guide / refs")
    f.add_argument("--seed", type=int)
    u = sp.add_parser("run"); u.add_argument("workflow"); u.add_argument("--url", required=True)
    u.add_argument("-o", "--out", required=True)
    c = sp.add_parser("check"); c.add_argument("prev"); c.add_argument("new"); c.add_argument("--seconds", type=float, default=1.0)
    s = sp.add_parser("stitch"); s.add_argument("prev"); s.add_argument("new"); s.add_argument("-o", "--out", required=True)
    s.add_argument("--trim", default="auto"); s.add_argument("--no-grade", action="store_true")
    bp = sp.add_parser("bridge-prepare"); bp.add_argument("prev"); bp.add_argument("next")
    bp.add_argument("-o", "--out", required=True); bp.add_argument("--prev-end", type=int)
    bp.add_argument("--next-start", type=int, default=0); bp.add_argument("--tail", type=int, default=25)
    bp.add_argument("--head", type=int, default=9); bp.add_argument("--seconds", type=float, default=4.0)
    bf = sp.add_parser("bridge-fill"); bf.add_argument("template"); bf.add_argument("prepared")
    bf.add_argument("-o", "--out", required=True); bf.add_argument("-p", "--prompt", required=True)
    bf.add_argument("--url"); bf.add_argument("--seed", type=int)
    bc = sp.add_parser("bridge-cut"); bc.add_argument("prev"); bc.add_argument("bridge"); bc.add_argument("next")
    bc.add_argument("-o", "--out", required=True); bc.add_argument("--prev-end", type=int)
    bc.add_argument("--tail", type=int, default=25)
    bc.add_argument("--prepared", help="the bridge-prepare folder: reuses its --prev-end and --tail")
    args = p.parse_args(argv)

    if args.cmd == "analyze":
        m = analyze(args.clip, args.seconds)
        print(json.dumps({"motion": m.as_dict(), **describe(m)}, indent=2))
    elif args.cmd == "prepare":
        info = prepare(args.clip, args.out, args.action, args.ref, args.seconds, args.guide_frames)
        print(json.dumps({k: info[k] for k in ("start_frame", "guide_clip", "prompt", "camera")}, indent=2))
    elif args.cmd == "fill":
        info = json.load(open(os.path.join(args.prepared, "nextshot.json")))
        tpl = json.load(open(args.template))
        start, guide, refs = info["start_frame"], info["guide_clip"], args.ref or info["references"]
        if args.url:
            cl = Comfy(args.url)
            start = cl.upload(start); guide = cl.upload(guide); refs = [cl.upload(x) for x in refs]
        else:
            start, guide, refs = os.path.basename(start), os.path.basename(guide), [os.path.basename(x) for x in refs]
        wf, notes = fill(tpl, start, info["prompt"], refs, guide, args.seed)
        json.dump(wf, open(args.out, "w"), indent=2)
        print(json.dumps({"written": args.out, "notes": notes}, indent=2))
    elif args.cmd == "run":
        cl = Comfy(args.url); pid = cl.queue(json.load(open(args.workflow)))
        print("queued", pid, file=sys.stderr)
        print(json.dumps(cl.download_outputs(cl.wait(pid), args.out), indent=2))
    elif args.cmd == "check":
        print(json.dumps(check(args.prev, args.new, args.seconds), indent=2))
    elif args.cmd == "bridge-prepare":
        print(json.dumps(prepare_bridge(args.prev, args.next, args.out, args.tail, args.head,
                                        args.next_start, args.seconds, prev_end=args.prev_end), indent=2))
    elif args.cmd == "bridge-fill":
        info = json.load(open(os.path.join(args.prepared, "bridge.json")))
        files = [info["guide_start"], info["guide_end"], info.get("start_frame"), info.get("silence")]
        if args.url:
            cl = Comfy(args.url); names = [cl.upload(f) if f else None for f in files]
        else:
            names = [os.path.basename(f) if f else None for f in files]
        wf, notes = fill_bridge(json.load(open(args.template)), info, args.prompt, names[0], names[1],
                                args.seed, start_frame_name=names[2], audio_name=names[3])
        json.dump(wf, open(args.out, "w"), indent=2)
        print(json.dumps({"written": args.out, "notes": notes}, indent=2))
    elif args.cmd == "bridge-cut":
        prev_end, tail = args.prev_end, args.tail
        if args.prepared:
            info = json.load(open(os.path.join(args.prepared, "bridge.json")))
            prev_end = prev_end or info.get("prev_end"); tail = info.get("tail", tail)
        cuts = find_cuts(args.prev, args.bridge, args.next, tail=tail, prev_end=prev_end)
        print(json.dumps(assemble(args.prev, args.bridge, args.next, cuts, args.out), indent=2))
    elif args.cmd == "stitch":
        trim = args.trim if args.trim == "auto" else int(args.trim)
        print(json.dumps(stitch(args.prev, args.new, args.out, trim, not args.no_grade), indent=2))


if __name__ == "__main__":
    main()
