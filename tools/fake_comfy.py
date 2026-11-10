# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""A pretend ComfyUI for tests: answers the few API calls LoRA Studio uses and paints a picture per prompt."""
# Test helper, not part of the program. Usage: fake_comfy.py <port> <LoRA folder> <output folder>
import json, sys, threading, time, uuid, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image, ImageDraw
PORT, LORAS, OUT = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
HIST = {}
LOG = []
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, d, code=200):
        b = json.dumps(d).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/system_stats": return self._json({"system": {"comfyui_version": "fake"}})
        if u.path == "/object_info/LoraLoader":
            names = sorted(str(p.relative_to(LORAS)) for p in LORAS.rglob("*.safetensors"))
            return self._json({"LoraLoader": {"input": {"required": {"lora_name": [names]}}}})
        if u.path.startswith("/history/"):
            pid = u.path.split("/")[-1]
            return self._json({pid: HIST[pid]} if pid in HIST else {})
        if u.path == "/view":
            q = urllib.parse.parse_qs(u.query); f = OUT / q["filename"][0]
            b = f.read_bytes(); self.send_response(200); self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if u.path == "/log": return self._json(LOG)
        self._json({}, 404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); raw = self.rfile.read(n)
        if self.path == "/upload/image":                  # multipart: keep the picture in OUT/input/
            b = self.headers["Content-Type"].split("boundary=")[1].encode()
            part = next(x for x in raw.split(b"--" + b) if b'name="image"' in x)
            head, body = part.split(b"\r\n\r\n", 1)
            name = head.split(b'filename="')[1].split(b'"')[0].decode()
            (OUT / "input").mkdir(exist_ok=True); (OUT / "input" / name).write_bytes(body[:-2])
            return self._json({"name": name, "subfolder": "lorastudio", "type": "input"})
        d = json.loads(raw or b"{}")
        if self.path == "/prompt" and any(v["class_type"] == "LoadImage" for v in d["prompt"].values()):
            g = d["prompt"]; pid = str(uuid.uuid4()); LOG.append(g)     # an upscale workflow: resize the picture
            src = OUT / "input" / next(v["inputs"]["image"] for v in g.values() if v["class_type"] == "LoadImage").split("/")[-1]
            up = next((v["inputs"] for v in g.values() if v["class_type"] == "SeedVR2VideoUpscaler"), {})
            im = Image.open(src); f = (up.get("resolution") or min(im.size)) / min(im.size)
            fn = f"{pid[:8]}.png"; im.resize((round(im.width * f), round(im.height * f))).save(OUT / fn)
            HIST[pid] = {"status": {"status_str": "success", "completed": True}, "outputs": {"15": {"images": [{"filename": fn, "subfolder": "", "type": "output"}]}}}
            return self._json({"prompt_id": pid, "number": 1, "node_errors": {}})
        if self.path == "/prompt":
            g = d["prompt"]; pid = str(uuid.uuid4()); LOG.append(g)
            lora = next((v["inputs"]["lora_name"] for v in g.values() if v["class_type"] == "LoraLoader"), "")
            if not (LORAS / lora).is_file():
                return self._json({"error": "lora missing", "node_errors": {"5": {"errors": [f"{lora} not found"]}}}, 400)
            ks = next(v["inputs"] for v in g.values() if v["class_type"] == "KSampler")
            lat = next(v["inputs"] for v in g.values() if v["class_type"] == "EmptyLatentImage")
            def work():
                time.sleep(0.3)
                im = Image.new("RGB", (lat["width"] // 4, lat["height"] // 4), (40 + ks["seed"] % 180, 90, 140))
                dr = ImageDraw.Draw(im); dr.text((8, 8), Path(lora).stem[-14:], fill=(255, 255, 255)); dr.text((8, 24), str(ks["seed"])[:8], fill=(255, 255, 0))
                fn = f"{pid[:8]}.png"; im.save(OUT / fn)
                HIST[pid] = {"status": {"status_str": "success", "completed": True}, "outputs": {"11": {"images": [{"filename": fn, "subfolder": "", "type": "output"}]}}}
            threading.Thread(target=work, daemon=True).start()
            return self._json({"prompt_id": pid, "number": 1, "node_errors": {}})
        self._json({}, 404)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
