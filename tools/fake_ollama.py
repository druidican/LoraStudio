# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
import json, re, sys
# Test helper, not part of the program. Usage: fake_ollama.py <port> <LoRA Studio folder>
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, sys.argv[2]); sys.path.insert(0, sys.argv[2] + "/program"); import lspaths; lspaths.bootstrap()
import consistency as C
LOADED = {}
MODELS = [{"name": "qwen2.5vl:7b", "size": 6000000000, "details": {"families": ["qwen25vl"]}},
          {"name": "llava:13b", "size": 8000000000, "details": {"families": ["llama", "clip"]}},
          {"name": "qwen3:8b", "size": 5200000000, "details": {"families": ["qwen3"]}}]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, d):
        b = json.dumps(d).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/api/ps"):
            return self._send({"models": [{"name": n, "size_vram": v} for n, v in LOADED.items()]})
        self._send({"models": MODELS})
    def do_DELETE(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MODELS[:] = [m for m in MODELS if m["name"] != d.get("model")]
        self._send({})
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/api/show"):
            n = d.get("model", "")
            return self._send({"capabilities": ["completion"] + (["vision"] if ("vl" in n or "llava" in n) else [])})
        if self.path.startswith("/api/pull"):
            MODELS.append({"name": d.get("model"), "size": 1e9, "details": {}})
            b = (json.dumps({"status": "pulling", "total": 10, "completed": 10}) + "\n" + json.dumps({"status": "success"}) + "\n").encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if self.path.startswith("/api/generate"):
            if d.get("keep_alive") == 0: LOADED.pop(d.get("model"), None)
            else: LOADED[d.get("model")] = 5_500_000_000
            return self._send({"done": True})
        if d.get("keep_alive") != 0 and d.get("model"): LOADED[d["model"]] = 5_500_000_000
        if d.get("keep_alive") == 0: LOADED.pop(d.get("model"), None)
        prompt = d["messages"][-1]["content"]
        if prompt.startswith("Look at this photo"):
            import hashlib
            h = int(hashlib.md5(d["messages"][-1]["images"][0][:4000].encode()).hexdigest(), 16)
            fr = ["close-up", "close-up", "close-up", "upper body", "full body"][h % 5]
            out = json.dumps({"people": 2 if h % 11 == 0 else 1, "main_subject_clear": True, "in_focus": True,
                              "face_visible": True, "face_problem": "", "text_or_watermark": False, "heavy_filter": False,
                              "collage_or_screenshot": False, "framing": fr,
                              "outfit": ["red dress", "red dress", "jeans and t-shirt", "armor"][h % 4],
                              "place": ["forest", "forest", "city street", "studio"][(h >> 3) % 4], "note": "", "teaches": [1, 3, 4, 5, 5][(h >> 5) % 5], "face_size": ["tiny", "small", "medium", "large"][(h >> 7) % 4], "features_visible": (h >> 9) % 6 != 0, "part_only": "feet" if h % 13 == 0 else "", "unique_detail": "a red scarf" if h % 7 == 0 else ""})
        elif prompt.startswith("This picture shows a character"):
            out = json.dumps({"details": ["long white hair", "blue eyes", "curved ram horns", "Pale skin"]})
        elif "should have these details" in prompt:
            import hashlib
            h = int(hashlib.md5(d["messages"][-1]["images"][0][:4000].encode()).hexdigest(), 16)
            n = len(re.findall(r"^\d+\. ", prompt, re.M))
            ans = [{"n": i + 1, "a": "yes"} for i in range(n)]
            if h % 6 == 0: ans[1]["a"] = "no"; ans[0]["a"] = "no"
            if h % 4 == 1: ans[-1]["a"] = "hidden"
            out = json.dumps({"answers": ans})
        elif "List ONLY traits" in prompt:
            cap = re.search(r'^"(.*)"$', prompt, re.M).group(1)
            tr = C.lexicon_traits(cap) + [{"trait": "dress colour", "value": "red", "phrase": "red dress"}]
            out = json.dumps({"traits": tr})
        elif "Group the words" in prompt:
            words = re.findall(r'"([^"]+)"', prompt.split("\n")[1])
            out = json.dumps({"groups": [[w] for w in words]})
        elif "Rewrite this picture caption" in prompt:
            cap = prompt.split("Caption: ", 1)[1]
            for r in re.findall(r'remove any mention of "([^"]+)"', prompt):
                cap = C.remove_phrase(cap, r) or cap.replace(r, "")
            out = cap
        elif "training captions" in " ".join(str(m.get("content", "")) for m in d["messages"]):
            import hashlib
            h = int(hashlib.md5(d["messages"][-1]["images"][0][:4000].encode()).hexdigest(), 16) if d["messages"][-1].get("images") else 0
            eyes = "brown eyes" if h % 9 == 0 else "blue eyes"
            if "booru" in " ".join(str(m.get("content", "")) for m in d["messages"]):
                out = f"mychar, 1girl, white hair, {eyes}, red dress, standing, forest, smiling"
            else:
                out = f"mychar stands in a forest in a red dress, with white hair and {eyes}, smiling at the viewer."
        else:
            out = "{}"
        self._send({"message": {"content": out}})
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
