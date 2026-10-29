"""Fill a ComfyUI workflow with the next-shot inputs, and (optionally) run it.

Works with ANY workflow you already trust (Kling, Seedance, LTX...). In ComfyUI:
  1. give the relevant nodes these titles (right-click a node > Title):
       NS start frame   - the Load Image node used as the first frame
       NS guide video   - (optional) a Load Video node for the last-frames guide
       NS prompt        - the positive prompt text node
       NS ref 1, NS ref 2, ...  - (optional) Load Image nodes for character/prop references
       NS seed          - (optional) the sampler node, to vary the seed
  2. Workflow menu > Export (API) and save the .json.
nextshot then writes those inputs into a copy of the file. Nothing else is touched.
"""
from __future__ import annotations
import copy, json, os, time, uuid
import urllib.request, urllib.parse

TEXT_KEYS = ("text", "prompt", "positive", "text_g", "value")
AUDIO_KEYS = ("audio",)
NUM_KEYS = ("value",)
IMAGE_KEYS = ("image",)
VIDEO_KEYS = ("video", "file", "filename")
SEED_KEYS = ("seed", "noise_seed")


def _by_title(wf: dict) -> dict:
    out = {}
    for nid, node in wf.items():
        t = (node.get("_meta") or {}).get("title", "")
        if t.upper().startswith("NS "):
            out[t.upper()] = nid
    return out


def _set(node: dict, keys, value) -> bool:
    for k in keys:
        if k in node.get("inputs", {}):
            node["inputs"][k] = value
            return True
    return False


def fill(template: dict, start_frame_name: str, prompt: str, refs: list[str] | None = None,
         guide_video_name: str | None = None, seed: int | None = None,
         audio_name: str | None = None, duration: float | None = None) -> tuple[dict, list[str]]:
    wf = copy.deepcopy(template)
    t = _by_title(wf)
    notes = []
    if "NS START FRAME" not in t:
        raise ValueError('No node titled "NS start frame" in the workflow')
    _set(wf[t["NS START FRAME"]], IMAGE_KEYS, start_frame_name)
    if "NS PROMPT" in t:
        if not _set(wf[t["NS PROMPT"]], TEXT_KEYS, prompt):
            notes.append("NS prompt node has no text input I recognise")
    else:
        notes.append('No "NS prompt" node: paste prompt.txt by hand')
    for i, r in enumerate(refs or [], 1):
        key = f"NS REF {i}"
        if key in t:
            _set(wf[t[key]], IMAGE_KEYS, r)
        else:
            notes.append(f"reference {i} has no matching node")
    if guide_video_name:
        if "NS GUIDE VIDEO" in t:
            _set(wf[t["NS GUIDE VIDEO"]], VIDEO_KEYS, guide_video_name)
        else:
            notes.append("workflow has no guide-video node: start frame only (still fine for static shots)")
    if seed is not None and "NS SEED" in t:
        _set(wf[t["NS SEED"]], SEED_KEYS, seed)
    if audio_name:
        if "NS AUDIO" in t:
            _set(wf[t["NS AUDIO"]], AUDIO_KEYS, audio_name)
        else:
            notes.append('No "NS audio" node: voice track not set')
    if duration is not None and "NS DURATION" in t:
        _set(wf[t["NS DURATION"]], NUM_KEYS, float(duration))
    return wf, notes


def free_memory(client: "Comfy") -> None:
    """Ask ComfyUI to unload models and free memory between long clips."""
    client._req("/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
                {"Content-Type": "application/json"})


class Comfy:
    """Minimal client for a ComfyUI server (e.g. your RunPod pod's port-8188 URL)."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _req(self, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def upload(self, path: str, subfolder: str = "nextshot") -> str:
        boundary = uuid.uuid4().hex
        name = os.path.basename(path)
        with open(path, "rb") as fh:
            payload = fh.read()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + payload + \
               (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n{subfolder}"
                f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
                f"\r\n--{boundary}--\r\n").encode()
        res = json.loads(self._req("/upload/image", body,
                                   {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return f"{res.get('subfolder')}/{res['name']}" if res.get("subfolder") else res["name"]

    def queue(self, wf: dict) -> str:
        body = json.dumps({"prompt": wf, "client_id": self.client_id}).encode()
        return json.loads(self._req("/prompt", body, {"Content-Type": "application/json"}))["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 5.0, timeout: float = 3600) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = json.loads(self._req(f"/history/{prompt_id}"))
            if prompt_id in h:
                return h[prompt_id]
            time.sleep(poll)
        raise TimeoutError(prompt_id)

    def download_outputs(self, result: dict, out_dir: str) -> list[str]:
        os.makedirs(out_dir, exist_ok=True); saved = []
        for node_out in result.get("outputs", {}).values():
            for kind in ("videos", "gifs", "images"):
                for item in node_out.get(kind, []):
                    q = urllib.parse.urlencode({"filename": item["filename"],
                                                "subfolder": item.get("subfolder", ""),
                                                "type": item.get("type", "output")})
                    data = self._req(f"/view?{q}")
                    p = os.path.join(out_dir, item["filename"])
                    with open(p, "wb") as fh:
                        fh.write(data)
                    saved.append(p)
        return saved
