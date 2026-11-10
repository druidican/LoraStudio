#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""LoRA Studio self-test: a quick check of the main parts, without training and without changing anything of yours.

    ./venv/bin/python selftest.py          (in the program folder; -v prints every check)
    ./venv/bin/python selftest.py --machine            (check THIS computer: GPU, trainers, model files, Ollama,
                                                        ComfyUI, then a 10-step training; see machinecheck.py)
    ./venv/bin/python selftest.py --machine --no-train (the same without the training)

It checks that all program files are of one version, that every part loads, that the page builds, and runs the main
logic on a small made-up dataset in a temporary folder: the launch script of every LoRA type (plan, form round trip,
command, version numbers), the dataset steps (overview, improvements, repeat folder rename), captions and the
ComfyUI test-picture graph. Your settings files are only read; the made-up dataset is deleted at the end.
Exit code 0 = all passed.
"""
from __future__ import annotations
__version__ = "1.101126b"  # LoRA Studio version, see version.py

import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "program" / "lspaths.py").is_file():          # an update from the old one-folder layout
    print("LoRA Studio self-test\n  ✕ the folder program/ is missing: this version has a new folder layout. "
          "Install it once with ./install.sh (your settings move to userdata/ by themselves).")
    sys.exit(1)
sys.path.insert(0, str(HERE / "program"))
import lspaths  # noqa: E402
lspaths.bootstrap()
sys.path.insert(len(lspaths.folders()), str(HERE))             # version.py (after the program folders)
os.chdir(HERE)
VERBOSE = "-v" in sys.argv
RESULTS = []

if "--machine" in sys.argv:
    import machinecheck
    machinecheck.run(train="--no-train" not in sys.argv, say=lambda s: print(s, flush=True))
    sys.exit(0)


def check(name):
    def wrap(fn):
        t = time.time()
        try:
            note = fn()
            RESULTS.append((True, name, note or ""))
            print(f"  ✓ {name}" + (f"  ({note})" if note and VERBOSE else "") + f"  [{time.time() - t:.1f} s]")
        except Exception as e:                           # noqa: BLE001
            RESULTS.append((False, name, repr(e)))
            print(f"  ✕ {name}: {e!r}")
            if VERBOSE:
                traceback.print_exc()
        return fn
    return wrap


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)


print("LoRA Studio self-test")

# ---------------------------------------------------------------- versions
import version  # noqa: E402

print(f"  version {version.VERSION}")


@check("all program files are the ones of this release")
def _versions():
    bad = version.problems()
    need(not bad, ", ".join(f"{p['name']} ({p['state']} {p['version']})" for p in bad))
    return f"{len(version.PARTS)} files"


# ---------------------------------------------------------------- every part loads
SKIP_IMPORT = {"pictools_worker.py", "lora_studio.py", "selftest.py"}   # runs in another venv / starts the program


@check("every part loads")
def _imports():
    import sdgui  # noqa: F401  (first: the parts need it)
    n = 0
    for name, _ in version.PARTS:
        if name.endswith(".py") and Path(name).name not in SKIP_IMPORT:
            importlib.import_module(Path(name).stem)
            n += 1
    return f"{n} modules"


import sdgui as G  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="lorastudio_selftest_"))
DS = "selftest_set"
_saved = {k: getattr(G, k) for k in ("IMAGES_DIR", "CONFIG_DIR", "SCAN_SCRIPT")}


def make_dataset():
    from PIL import Image, ImageDraw
    rep = TMP / "datasets" / DS / f"10_{DS}"
    rep.mkdir(parents=True)
    for i in range(8):
        im = Image.new("RGB", (768 + 64 * (i % 3), 1024), (40 + 20 * i, 90, 160 - 10 * i))
        d = ImageDraw.Draw(im)
        d.ellipse((200, 150 + 10 * i, 560, 800), fill=(220, 180, 140))
        d.rectangle((300, 820, 460, 1000), fill=(30, 30, 30))
        im.save(rep / f"pic_{i:02d}.png")
        (rep / f"pic_{i:02d}.txt").write_text(f"selftest_set, a person, red background, picture {i}\n")
    G.IMAGES_DIR = TMP / "datasets"
    G.CONFIG_DIR = TMP / "cfg"
    G.CONFIG_DIR.mkdir()
    if not G.SCAN_SCRIPT:                      # Lora_Datatypes.py next to the program (lora_studio.py sets this)
        G.SCAN_SCRIPT = G.find_scan_script(None)
    return rep


@check("the made-up dataset is found")
def _dataset():
    make_dataset()
    need(DS in G.list_lora_dirs(), f"{DS} not in {G.list_lora_dirs()}")
    st = G.dataset_stats(DS)
    rows = [r for r in st["rows"] if not r.get("virtual")]
    need(len(rows) == 1 and rows[0]["repeats"] == 10, f"repeat folder rows: {rows}")
    return f"{len(G.list_images(rows[0]['folder']))} pictures"


# ---------------------------------------------------------------- launch scripts
@check("a launch script for every LoRA type")
def _plans():
    sc, presets = G.merged_presets()
    need(sc is not None and presets, "Lora_Datatypes.py did not load")
    done = []
    for tkey in sorted(presets):
        p = G.make_plan(DS, tkey)
        t = p["script_text"]
        need(t.strip() and p["filename"].endswith(".sh"), f"{tkey}: empty script or name")
        vals = G.read_form(t)
        need(vals, f"{tkey}: the form could not read the script")
        again = G.apply_form(t, vals)
        again = again[0] if isinstance(again, tuple) else again
        need(G.read_form(again) == vals, f"{tkey}: form round trip changes the settings")
        need(G.training_command(t), f"{tkey}: no training command found")
        done.append(tkey)
    return ", ".join(done)


@check("character defaults: SDXL text encoders, FLUX/Chroma rank 32, Z-Image Base without the adapter")
def _chardefaults():
    sc, presets = G.merged_presets()
    G.set_choice("character", "balanced")
    ill = G.make_plan(DS, "illustrious")["script_text"]
    need("--text_encoder_lr 5e-5" in ill and "--network_train_unet_only" not in ill, "Illustrious: text encoders not trained")
    for t in ("flux", "chroma"):
        need("--network_dim 32 --network_alpha 32" in G.make_plan(DS, t)["script_text"], f"{t}: not rank 32/32")
    zib, zit = G.make_plan(DS, "zimagebase")["script_text"], G.make_plan(DS, "zimage")["script_text"]
    need("BASE_WEIGHTS" not in zib and "z_image_bf16" in zib, "Z-Image Base: adapter in the script or wrong model")
    need("BASE_WEIGHTS" in zit, "Z-Image Turbo lost its adapter")
    need(G.infer_type(zib, presets, "") == "zimagebase" and G.infer_type(zit, presets, "") == "zimage",
         f"type detection: {G.infer_type(zib, presets, '')} / {G.infer_type(zit, presets, '')}")


@check("each launch script trains on its own caption set; Anima/FLUX/Chroma can have both")
def _capsets():
    import dsflow as F
    import sd_script as SS
    old = F.meta(DS)
    try:
        G.set_choice("character", "balanced")
        F.meta_save(DS, {**old, "types": ["illustrious", "zimage"], "caption_choice": "auto", "train_style": {}})
        need([x["ext"] for x in F.caption_sets(DS)] == [".txt", ".sentences.txt"], F.caption_sets(DS))
        zit = G.make_plan(DS, "zimage")["script_text"]
        need('caption_extension = \\".sentences.txt\\"' in zit, "Z-Image does not train on the sentence set")
        need('--caption_extension ".txt"' in G.make_plan(DS, "illustrious")["script_text"], "Illustrious lost IMG.txt")
        F.meta_save(DS, {**F.meta(DS), "types": ["illustrious", "anima"], "caption_choice": "both"})
        need(len(F.train_sets(DS, "anima")) == 2 and len(F.train_sets(DS, "illustrious")) == 1, "anima should have both")
        need('--caption_extension ".txt"' in G.make_plan(DS, "anima")["script_text"], "Anima's usual set is the tags")
        F.set_train_style(DS, "anima", "sentences")
        t = G.make_plan(DS, "anima")["script_text"]
        need('--caption_extension ".sentences.txt"' in t, "the pick in step 9 is not used")
        need('--caption_extension ".txt"' in SS.set_caption_ext(t, ".txt"), "set_caption_ext")
        need('\\".tags.txt\\"' in SS.set_caption_ext(zit, ".tags.txt"), "set_caption_ext (musubi)")
    finally:
        F.meta_save(DS, old)


UPSCALE_WF = {"nodes": [
    {"id": 1, "type": "LoadImage", "mode": 0, "inputs": [{"name": "image", "type": "COMBO", "widget": {"name": "image"},
     "link": None}, {"name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None}],
     "outputs": [{"name": "IMAGE", "type": "IMAGE"}], "widgets_values": ["x.png", "image"]},
    {"id": 2, "type": "CompileSettings", "mode": 4, "inputs": [], "outputs": [{"name": "ARGS", "type": "ARGS"}],
     "widgets_values": []},
    {"id": 3, "type": "SeedVR2LoadDiTModel", "mode": 0, "inputs": [{"name": "compile", "type": "ARGS", "link": 2},
     {"name": "model", "type": "COMBO", "widget": {"name": "model"}, "link": None}],
     "outputs": [{"name": "DIT", "type": "SEEDVR2_DIT"}], "widgets_values": ["pip_upscaler-Q8_0.gguf"]},
    {"id": 4, "type": "SeedVR2VideoUpscaler", "mode": 0, "inputs": [
        {"name": "image", "type": "IMAGE", "link": 1}, {"name": "dit", "type": "SEEDVR2_DIT", "link": 3},
        {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
        {"name": "resolution", "type": "INT", "widget": {"name": "resolution"}, "link": None},
        {"name": "max_resolution", "type": "INT", "widget": {"name": "max_resolution"}, "link": None},
        {"name": "input_noise_scale", "type": "FLOAT", "widget": {"name": "input_noise_scale"}, "link": None},
        {"name": "latent_noise_scale", "type": "FLOAT", "widget": {"name": "latent_noise_scale"}, "link": None}],
     "outputs": [{"name": "IMAGE", "type": "IMAGE"}], "widgets_values": [42, "fixed", 2048, 2048, 0, 0]},
    {"id": 5, "type": "SaveImage", "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 4},
     {"name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None}],
     "outputs": [], "widgets_values": ["ComfyUI"]},
    {"id": 6, "type": "Note", "mode": 0, "inputs": [], "outputs": [], "widgets_values": ["a note"]}],
    "links": [[1, 1, 0, 4, 0, "IMAGE"], [2, 2, 0, 3, 0, "ARGS"], [3, 3, 0, 4, 1, "SEEDVR2_DIT"],
              [4, 4, 0, 5, 0, "IMAGE"]]}


def _pretend_comfy(folder: Path):
    """A tiny ComfyUI in this process: takes an uploaded picture and 'upscales' it to the requested short side."""
    from PIL import Image
    import threading
    import uuid
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    hist = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, b, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/system_stats":
                return self._send(b"{}")
            if u.path.startswith("/history/"):
                pid = u.path.rsplit("/", 1)[-1]
                return self._send(json.dumps({pid: hist[pid]} if pid in hist else {}).encode())
            if u.path == "/view":
                return self._send((folder / urllib.parse.parse_qs(u.query)["filename"][0]).read_bytes(), "image/png")
            self._send(b"{}")

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/upload/image":
                b = self.headers["Content-Type"].split("boundary=")[1].encode()
                part = next(x for x in raw.split(b"--" + b) if b'name="image"' in x)
                head, body = part.split(b"\r\n\r\n", 1)
                name = head.split(b'filename="')[1].split(b'"')[0].decode()
                (folder / name).write_bytes(body[:-2])
                return self._send(json.dumps({"name": name, "subfolder": "lorastudio"}).encode())
            g = json.loads(raw)["prompt"]
            src = next(v["inputs"]["image"] for v in g.values() if v["class_type"] == "LoadImage").split("/")[-1]
            up = next(v["inputs"] for v in g.values() if v["class_type"] == "SeedVR2VideoUpscaler")
            im = Image.open(folder / src)
            f = up["resolution"] / min(im.size)
            pid = uuid.uuid4().hex
            im.resize((round(im.width * f), round(im.height * f))).save(folder / f"{pid}.png")
            hist[pid] = {"status": {"status_str": "success", "completed": True},
                         "outputs": {"5": {"images": [{"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}}}
            self._send(json.dumps({"prompt_id": pid, "node_errors": {}}).encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@check("upscale workflow: your ComfyUI workflow gets the picture, the size and the noise")
def _upscalewf():
    from PIL import Image
    import comfytest as CT
    import comfyupscale as U
    api = U.to_api(UPSCALE_WF)
    need(set(api) == {"1", "3", "4", "5"}, f"bypassed / note nodes: {sorted(api)}")
    need(api["1"]["inputs"] == {"image": "x.png"} and "compile" not in api["3"]["inputs"], api)
    need(api["4"]["inputs"]["resolution"] == 2048 and api["4"]["inputs"]["seed"] == 42, api["4"])
    pl = U.plan(512, 768, 1024)
    need(pl["resolution"] == 1024 and pl["factor"] == 2.0 and pl["input_noise"] == 0.1, pl)
    need(U.plan(1024, 1024, 1024)["input_noise"] == 0.0 and U.plan(200, 200, 1024)["latent_noise"] > 0, "noise rule")
    g = U.graph(api, "lorastudio/a.png", pl, "p")
    need(g["4"]["inputs"]["resolution"] == 1024 and g["1"]["inputs"]["image"] == "lorastudio/a.png", g)
    tmp = Path(tempfile.mkdtemp())
    srv = _pretend_comfy(tmp)
    old = (U.WF, U.FILE, CT.settings, U._free_ollama)
    try:
        U.WF, U.FILE = tmp / "wf.json", tmp / "up.json"
        U.WF.write_text(json.dumps(UPSCALE_WF))
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        CT.settings = lambda: {"url": url}
        U._free_ollama = lambda: None
        src = tmp / "small.png"
        Image.new("RGB", (300, 400), (10, 120, 200)).save(src)
        r = U.upscale(src, tmp / "big.png", 1024)
        with Image.open(tmp / "big.png") as im:
            need(min(im.size) == 1024, f"upscaled to {im.size}")
        need(r["factor"] > 3, r)
    finally:
        U.WF, U.FILE, CT.settings, U._free_ollama = old
        srv.shutdown()


@check("detailed scan: gaps, and suggestions to add, upscale and crop (characters)")
def _detailscan():
    from PIL import Image
    import detailscan as DSC
    import dsflow as F
    old_meta, old_ask, old_sort = F.meta(DS), DSC.ask, F.sort_result(DS)
    rep = F.root(DS)
    pics = sorted(p for p in rep.rglob("*.png") if "_originals" not in p.parts)[:4]
    try:
        F.meta_save(DS, {**old_meta, "purpose": "character", "types": ["illustrious"], "detailed_scan": True, "pins": {}})
        need(DSC.enabled(DS), "not enabled")
        Image.new("RGB", (2048, 2048), (90, 90, 90)).save(pics[0])
        Image.new("RGB", (640, 640), (90, 90, 90)).save(pics[1])
        sort = {"pics": {str(p): {"cat": "good", "framing": "upper body", "outfit": "coat", "place": "park", "w": 1024,
                                  "h": 1024, "warn": [], "why": ""} for p in pics}}
        sort["pics"][str(pics[0])].update(w=2048, h=2048)
        sort["pics"][str(pics[1])].update(w=640, h=640)
        sort["pics"][str(pics[3])].update(cat="alike", why="one of 4 alike")
        F._save(DS, "sort.json", sort)
        answers = {pics[0].name: {"angle": "front", "pose": "standing", "face_box": [1000, 300, 1160, 460]},
                   pics[3].name: {"angle": "back", "pose": "standing"}}
        DSC.ask = lambda url, model, p: {"angle": "front", "pose": "standing", "expression": "neutral",
                                         "lighting": "daylight", "face_box": None, "detail_box": None, "detail": "",
                                         "W": 0, "H": 0, **answers.get(Path(p).name, {})}
        for line in DSC.run(DS, "http://x", "m"):
            pass
        r = DSC.result(DS)
        kinds = {(i["kind"], Path(i["path"]).name) for i in r["items"]}
        need(("add", pics[3].name) in kinds, f"the back view is not suggested: {kinds}")
        need(("upscale", pics[1].name) in kinds, f"the small picture is not suggested: {kinds}")
        need(("crop", pics[0].name) in kinds, f"the small face is not suggested: {kinds}")
        need(any("side" in g for g in r["gaps"]), r["gaps"])
        both = DSC._crops("p.png", "p.png", {"face_box": [535, 350, 725, 525], "detail_box": [490, 250, 795, 425],
                                             "detail": "a wreath"}, 1344, 1728, 896, 1024)
        need(len(both) == 1 and "face and wreath" in both[0]["label"], f"overlapping crops not merged: {both}")
        crop = next(i for i in r["items"] if i["kind"] == "crop")
        add = next(i for i in r["items"] if i["kind"] == "add")
        out = list(DSC.apply(DS, [crop["id"], add["id"]]))
        new = list(pics[0].parent.glob(pics[0].stem + "_face*.png"))
        need(new and F.pin_of(F.pins(DS), new[0]) == "keep" and F.pin_of(F.pins(DS), pics[3]) == "keep", out[-1])
        for f in new:
            f.unlink()
    finally:
        DSC.ask = old_ask
        F.meta_save(DS, old_meta)
        (F.root(DS) / ".lsflow" / "detail.json").unlink(missing_ok=True)
        if old_sort is None:
            (F.root(DS) / ".lsflow" / "sort.json").unlink(missing_ok=True)
        else:
            F._save(DS, "sort.json", old_sort)


@check("sentence captions of a character begin 'trigger, a class word, ...'")
def _classword():
    import captioner as C
    old = dict(C.PURPOSE)
    try:
        C.PURPOSE.update(value="character", **{"class": "robot"})
        need(C.clean("pip sits on a bench", "pip") == "pip, a robot, sits on a bench", C.clean("pip sits on a bench", "pip"))
        need(C.clean("A robot stands in the rain", "pip") == "pip, a robot, stands in the rain", "subject not replaced")
        need("pip, a robot," in C.build_prompt("sentences", "pip", "hair colour", 3, 3), "prompt rule missing")
        C.PURPOSE.update(value="style")
        need(C.clean("pip sits", "pip") == "pip sits", "style captions changed")
    finally:
        C.PURPOSE.clear(); C.PURPOSE.update(old)


@check("launch script names count up (V1, V2, ...)")
def _names():
    sc, presets = G.merged_presets()
    tkey = "sdxl" if "sdxl" in presets else sorted(presets)[0]
    p = G.make_plan(DS, tkey)
    lora = G.lora_path(DS)
    base, n = G.next_script_version(lora, p["script_text"])
    need(base, "no base name in the script")
    f = lora / p["filename"]
    f.write_text(G.set_script_version(p["script_text"], base, n))
    base2, n2 = G.next_script_version(lora, f.read_text())
    need(n2 == n + 1, f"after V{n} the next is V{n2}")
    f.unlink()
    return f"{base}V{n} → {base2}V{n2}"


# ---------------------------------------------------------------- dataset steps
@check("dataset steps: overview and improvements")
def _flow():
    import dsflow as F
    ov = F.overview([DS])
    need(ov and ov[0]["n"] == 8, f"overview: {ov}")
    imp = F.improvements(DS)
    need(isinstance(imp, list), "improvements is not a list")
    return f"next step {ov[0]['next']}, {len(imp)} improvement(s)"


@check("dataset steps: repeat folder rename (10 → 20 → 10)")
def _repeats():
    import dsflow as F
    msg = F.rename_repeats(DS, 20)
    need(msg and not msg.startswith("⚠"), f"rename: {msg!r}")
    need((G.lora_path(DS) / f"20_{DS}").is_dir(), "20_ folder missing")
    F.rename_repeats(DS, 10)
    need((G.lora_path(DS) / f"10_{DS}").is_dir(), "renaming back failed")


@check("automatic improvement settings")
def _batch():
    import dsflow as F
    b = F.batch_settings()
    need(set(F.BATCH_DEFAULTS) <= set(b), "settings miss keys")
    need(F.batch_summary(b), "no summary")


# ---------------------------------------------------------------- captions
@check("captions: purposes and cleaning")
def _captions():
    import captioner as CAP
    need(CAP.PURPOSE, "no purposes")
    out = CAP.clean("Caption: A photo of a person standing, red background.", "selftest_set")
    need(isinstance(out, str) and "selftest_set" in out and not out.lower().startswith("caption"), f"clean gave {out!r}")
    line, _ = CAP.clean_tags("Red Background, person, person, standing", "selftest_set", "", 10)
    need(line.startswith("selftest_set") and line.count("person") == 1, f"clean_tags gave {line!r}")


# ---------------------------------------------------------------- ComfyUI test pictures
TINY_WF = {
    "nodes": [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values_named": {"ckpt_name": "model.safetensors"}},
        {"id": 2, "type": "CLIPTextEncode", "inputs": [], "widgets_values_named": {"text": "masterpiece, best quality\n\na cat"}},
        {"id": 3, "type": "CLIPTextEncode", "inputs": [], "widgets_values_named": {"text": "blurry"}},
        {"id": 4, "type": "EmptyLatentImage", "widgets_values_named": {"width": 832, "height": 1216}},
        {"id": 5, "type": "KSampler", "inputs": [{"name": "positive", "link": 10}, {"name": "negative", "link": 11}],
         "widgets_values_named": {"steps": 28, "cfg": 5.5, "sampler_name": "euler_ancestral", "scheduler": "normal",
                                  "denoise": 1.0}},
    ],
    "links": [[10, 2, 0, 5, 1, "CONDITIONING"], [11, 3, 0, 5, 2, "CONDITIONING"]],
}


@check("ComfyUI: reading a workflow and the test graph")
def _comfy():
    import comfytest as CT
    wf = CT.read_workflow(TINY_WF)
    need(wf["ckpt"] == "model.safetensors" and wf["steps"] == 28 and wf["cfg"] == 5.5, f"read: {wf}")
    need((wf["width"], wf["height"]) == (832, 1216), f"size: {wf['width']}x{wf['height']}")
    need(wf["negative"] == "blurry", f"negative: {wf['negative']!r}")
    g = CT.graph(wf, "_lorastudio_test/x.safetensors", "a cat", 1, "t")
    kinds = {v["class_type"] for v in g.values()}
    need({"LoraLoader", "KSampler", "SaveImage"} <= kinds, f"graph nodes: {kinds}")
    eps = {k: Path(f"e{k}") for k in range(1, 11)}
    picked = CT.pick_epochs(eps, Path("final"), 6, 2)
    need([lab for lab, _ in picked] == ["epoch 4", "epoch 5", "epoch 6", "epoch 7", "epoch 8", "final"],
         f"picked {picked}")


# ---------------------------------------------------------------- best mark (in a temporary completed.json)
@check("best-version mark and Completed marks together")
def _best():
    import completed as CP
    old = CP.FILE
    CP.FILE = TMP / "completed.json"
    try:
        CP.set_done("selftest_set", "illustrious", True, lora="selftest_setILLV2")
        CP.set_best("selftest_set", "Illustrious", 1)
        need(CP.best_version("selftest_set", "illustrious") == 1, "best mark not read back")
        CP.set_done("selftest_set", "illustrious", True, lora="selftest_setILLV3")
        need(CP.best_version("selftest_set", "illustrious") == 1, "a Completed mark removed the best mark")
        need(CP.is_done("selftest_set", "illustrious", version=3), "Completed mark lost")
        CP.set_best("selftest_set", "illustrious", None)
        need(CP.best_version("selftest_set", "illustrious") is None, "best mark not removed")
    finally:
        CP.FILE = old


# ---------------------------------------------------------------- self-update: reading a LoraStudio.zip
@check("self-update reads a LoraStudio.zip")
def _selfupdate():
    import zipfile
    import selfupdate as SU
    z = TMP / "LoraStudio.zip"
    with zipfile.ZipFile(z, "w") as zz:
        for name, _ in version.PARTS:
            if (HERE / name).is_file():
                zz.write(HERE / name, f"LoraStudio/{name}")
    pl = SU.plan(z)
    need(pl["zip"]["version"] == version.VERSION and pl["same"], "the version in the zip was not read")
    changed = [r["name"] for r in pl["rows"] if r["state"] != "same"]
    need(not changed, f"the same files show as changed: {changed[:5]}")
    bad = TMP / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zz:
        zz.writestr("x/readme.txt", "no")
    try:
        SU.read_zip(bad)
        raise AssertionError("a zip without version.py was accepted")
    except ValueError:
        pass


# ---------------------------------------------------------------- dataset folder: only <N>_<dataset name> counts
@check("only the repeat folder <N>_<dataset name> is the dataset")
def _onlymain():
    import dsflow as F
    from PIL import Image
    d = G.lora_path(DS)
    (d / "5_other").mkdir()
    Image.new("RGB", (512, 512)).save(d / "5_other" / "x.png")
    (d / "_reserve").mkdir(exist_ok=True)
    Image.new("RGB", (512, 512)).save(d / "_reserve" / "y.png")
    try:
        st = G.dataset_stats(DS)
        need([r["folder"].name for r in st["rows"]] == [f"10_{DS}"], f"rows: {[r['folder'].name for r in st['rows']]}")
        need(len(F.images(DS)) == 8 and len(F.extra_pictures(DS)) == 2, "pictures outside the repeat folder counted")
        F.set_include_extra(DS, True)
        need(len(F.candidates(DS)) == 10, "the step-2 setting does not add the other pictures")
        note = F.script_saved(DS)
        need(not F.include_extra(DS) and (d / "_extra" / "5_other").is_dir(), f"after a launch script: {note}")
    finally:
        shutil.rmtree(d / "_extra", ignore_errors=True)
        shutil.rmtree(d / "5_other", ignore_errors=True)
        shutil.rmtree(d / "_reserve", ignore_errors=True)
        F.set_include_extra(DS, False)


# ---------------------------------------------------------------- what went wrong, the report, the manual
@check("failed runs are explained in a few words")
def _failcodes():
    import failcodes as FC
    cases = {"OOM": "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB\n--- exited with code 1 ---",
             "NAN": "steps: 10%| | 10/100 [avr_loss=nan]\n--- exited with code 1 ---",
             "IMPORT": "line 3: accelerate: command not found\n--- exited with code 127 ---",
             "ARGS": "train_network.py: error: unrecognized arguments: --foo",
             "EXIT": "--- exited with code 2 ---"}
    for code, log in cases.items():
        e = FC.explain(log)
        need(e and e["code"] == code, f"{code}: got {e and e['code']}")


@check("Report for Claude removes the token and the home folder")
def _report():
    import report as R
    home = str(Path.home())
    out = R.redact(f"HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123\npath {home}/Datasets/x\nkey sk-abcdefghijklmnopqrstuvwxyz")
    need("hf_abc" not in out and "sk-abc" not in out, "a token is left in the report")
    need(home == "/" or home not in out, "the home folder is left in the report")


@check("the manual has its chapters and pictures")
def _manual():
    import manual as M
    if not M.DIR.is_dir():
        return "not installed yet (an update from before 1.251026d: install the zip once more)"
    ch = M.chapters()
    need(len(ch) >= 5, f"only {len(ch)} chapters")
    missing = []
    import re as _re
    for c in ch:
        for img in _re.findall(r"\]\((img/[^)\s]+)\)", c["file"].read_text(encoding="utf-8")):
            if not (M.DIR / img).is_file():
                missing.append(img)
    need(not missing, f"pictures missing: {missing[:4]}")
    return f"{len(ch)} chapters"


@check("what is installed: missing parts are greyed out")
def _features():
    import features as FT
    saved = dict(FT._CACHE)
    try:
        FT._CACHE["data"] = {"ollama": "none", "sd_scripts": "existing", "musubi": "none"}
        need(not FT.have("ollama") and FT.have("sd_scripts") and not FT.have("musubi"), "have() reads features.json wrong")
        need(FT.label("Tags", "ollama") == "Tags (Ollama missing)", "label() does not add '(Ollama missing)'")
        need(FT.can_train("sdxl") and not FT.can_train("zimage"), "can_train() picks the wrong trainer")
        need(not FT.can_train_text('MUSUBI_DIR="x"') and FT.can_train_text('SD_SCRIPTS_DIR="x"'), "can_train_text() wrong")
        FT._CACHE["data"] = {}
        need(FT.have("ollama") and FT.have("musubi"), "without features.json everything must count as present")
    finally:
        FT._CACHE.clear(); FT._CACHE.update(saved)
    return ", ".join(f"{n} {'yes' if ok else 'missing'}" for n, _, ok in FT.summary())


@check("the training venv check finds a ROCm part that does not fit PyTorch")
def _venvcheck():
    import tempfile
    import venvcheck as VC
    with tempfile.TemporaryDirectory() as td:
        for name, ver in (("torch", "2.13.0+rocm10.0.0"), ("rocm-sdk-core", "10.0.0"), ("rocm-sdk-devel", "10.1.0"),
                          ("rocm-bootstrap", "0.3.0")):
            d = Path(td) / f"{name.replace('-', '_')}-{ver}.dist-info"
            d.mkdir()
            (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
        old = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = td
        try:
            r = VC.inspect(Path(sys.executable), only_path=True)
        finally:
            if old is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old
    need(r["kind"] == "rocm" and r["want"] == "10.0.0", f"read wrong: {r['kind']} {r['want']}")
    need([b[0] for b in r["bad"]] == ["rocm-sdk-devel"], f"found {r['bad']}")
    VC.inspect()                                     # the real one again, for the Health check
    return VC.LAST.get("text", "")


@check("WD tagger: tags cleaned like Ollama's (trigger first, traits out, 1girl kept)")
def _wdtags():
    import captioner as C
    line, dropped = C.clean_tags("1girl, solo, blue_eyes, blonde_hair, smile, 1. outdoors", "selftest",
                                 C.DEFAULT_LEAVE_OUT, 20)
    need(line.startswith("selftest, 1girl, solo"), f"got {line!r}")
    need("blue eyes" not in line and "blonde hair" not in line and "outdoors" in line, f"got {line!r}")
    import wdtagger as WD
    need(WD.REPO == "SmilingWolf/wd-eva02-large-tagger-v3", "another tagger model")
    return line


@check("the Hugging Face download list covers every built-in LoRA type and reads pasted links")
def _hfdl():
    import hfdownload as HD
    import Lora_Datatypes as L
    for t, cfg in L.PRESETS.items():
        k = HD.known(t)
        for flag, _ in cfg.get("model_args", []):
            need(flag.lstrip("-") in k, f"{t}: no download for {flag}")
    need(HD.parse_link("https://huggingface.co/a/b/blob/main/x/y.safetensors?download=true") == ("a/b", "main", "x/y.safetensors"),
         "a /blob/ link was not read")
    need(HD.parse_link("https://huggingface.co/a/b/resolve/v2/z.safetensors") == ("a/b", "v2", "z.safetensors"),
         "a /resolve/ link was not read")
    need(HD.parse_link("https://example.com/x.safetensors") is None, "a link to another site was taken")
    need(HD.part_paths("te/model-00001-of-00003.safetensors", 3)[2] == "te/model-00003-of-00003.safetensors",
         "the parts of a split file are wrong")
    return f"{len(L.PRESETS)} types"


@check("Personal sync merges .metadata.json both ways (missing and empty fields filled, newer file wins)")
def _metamerge():
    import lora_library as LL
    a = {"notes": "", "tags": ["a"], "civitai": {"trainedWords": ["trig"]}, "favorite": True}
    b = {"notes": "mine", "tags": ["b"], "civitai": {"trainedWords": [], "id": 5}, "extra": 1}
    m = LL.merge_meta(a, b, a_wins=False)
    need(m["notes"] == "mine" and m["civitai"] == {"trainedWords": ["trig"], "id": 5}, f"got {m}")
    need(m["tags"] == ["b"] and m["favorite"] is True and m["extra"] == 1, f"got {m}")
    need(LL.merge_meta(a, b, a_wins=True)["tags"] == ["a"], "the newer file did not win")


@check("training card: version and clip skip go to LoRA Manager, trigger words come back, old notes text goes")
def _tcard():
    import json as _j
    import lora_library as LL
    import trainingcard as TC
    ls = TC.apply({"notes": "Trained on 2026-10-20 from the dataset x. 10 epochs.\n\nMy note"},
                  {"version": "V1", "trigger": "x", "clip_skip": ""}, "sig")
    need(ls["notes"] == "My note", f"the old training text stayed in Notes: {ls['notes']!r}")
    need(ls["civitai"]["name"] == "V1" and _j.loads(ls["usage_tips"])["clip_skip"] == 1, "version / clip skip not set")
    lm = {"model_name": "mine", "civitai": {"trainedWords": ["a", "b"]}, "usage_tips": '{"strength": 0.8}',
          "modelDescription": ""}
    m = LL.sync_rules(lm, ls, LL.merge_meta(lm, ls, False))
    need(m["civitai"]["name"] == "V1", "LoRA Studio's version did not win")
    need(m["civitai"]["trainedWords"] == ["a", "b"], "LoRA Manager's trigger words were not taken")
    tips = _j.loads(m["usage_tips"])
    need(tips.get("strength") == 0.8 and tips.get("clip_skip") == 1, f"usage tips not merged: {tips}")
    need(m["model_name"] == "mine" and m["modelDescription"] == "", "LoRA Manager's own fields were changed")


@check("Insights: good datasets against the others, and the batch sizes the graphics card managed")
def _insights():
    import insights as IN
    mk = lambda v, n, c: {"type": "sdxl", "label": "SDXL", "verdict": v, "why": "", "stem": "x", "ds": "d",
                          "facts": {"pictures": n, "close": c}}
    s = IN.dataset_summary([mk("good", 40, 0.3), mk("good", 36, 0.35), mk("weak", 15, 0.8), mk("weak", 18, 0.7)])[0]
    need(s["n_good"] == 2 and s["weak_is_rated"], f"wrong counts: {s}")
    need(any("Pictures" in n and "more" in n for n in s["notes"]), f"no note about the number of pictures: {s['notes']}")
    runs = [{"type": "sdxl", "label": "SDXL", "batch": b, "res": "1024", "result": r, "sps": sp}
            for b, r, sp in ((2, "done", 1.5), (2, "done", 1.7), (4, "oom", None), (1, "done", 0.9))]
    g = IN.gpu_summary(runs)[0]
    need(g["safe"] == 2 and g["oom_at"] == 4, f"wrong batch advice: {g}")
    need(abs([c for c in g["batches"] if c["batch"] == 2][0]["sps"] - 1.6) < 1e-6, "wrong seconds per step")
    need(IN._speed("steps: 10/100 [00:01<00:10, 2.00it/s] steps: 20/100 [00:02, 2.00it/s]") == 0.5, "tqdm speed not read")
    need(IN._best_epoch("Try epoch 10 first, then the final (12)") == 10 and IN._best_epoch("Try the final (12) first") == 12,
         "the save point advice is not read")
    st = lambda v, lr, best: {"type": "sdxl", "label": "SDXL", "verdict": v, "facts": {}, "stem": "x",
                              "settings": {"lr": lr, "epochs": 10, "best_ratio": best / 10, "best_epoch": best}}
    rows = [st("good", 1e-4, 6), st("good", 1e-4, 5), st("weak", 3e-4, 7), st("weak", 3e-4, 6)]
    ss = IN.settings_summary(rows)[0]
    need(ss["good"]["lr"] == 1e-4 and any("Learning rate" in n for n in ss["notes"]), f"settings not compared: {ss}")
    es = IN.epochs_summary(rows)[0]
    need(es["n"] == 4 and es["ratio"] == 0.6 and "fewer epochs" in es["note"], f"early save points not seen: {es}")


@check("character description: suggested from the left-out tags, added next to the trigger word")
def _desc():
    import captioner as C
    import trainingcard as TC
    t = C.trait_tags("1girl, blonde_hair, blue eyes, dark skin, large breasts, freckles, smile", C.DEFAULT_LEAVE_OUT)
    need(t == ["blonde hair", "blue eyes", "freckles"], f"wrong traits suggested: {t}")
    sc = TC.apply({"civitai": {"trainedWords": ["pip"]}}, {"trigger": "pip", "description": "tall, blonde hair"}, "s")
    need(sc["civitai"]["trainedWords"] == ["pip", "tall", "blonde hair"], f"not next to the trigger: {sc['civitai']}")
    sc2 = TC.apply(sc, {"trigger": "pip", "description": "slim build"}, "s")
    need(sc2["civitai"]["trainedWords"] == ["pip", "slim build"], f"the old description stayed: {sc2['civitai']}")


@check("step 7: all tags written, ticked character tags move to the trigger words and back")
def _tagmove():
    import tempfile
    import dsflow as F
    d = Path(tempfile.mkdtemp())
    imgs = []
    for i, line in enumerate(["pip, blonde hair, blue eyes, smile, outdoors", "pip, blonde hair, blue eyes, indoors",
                              "pip, blonde hair, sitting"]):
        img = d / f"{i}.png"
        img.write_bytes(b"x")
        (d / f"{i}.txt").write_text(line + "\n", encoding="utf-8")
        imgs.append(img)
    store = {"m": {"purpose": "character", "types": []}}
    saved = {k: getattr(F, k) for k in ("images", "meta", "meta_save", "_flow", "set_file")}
    F.images, F.meta, F.meta_save = (lambda ds: imgs), (lambda ds: store["m"]), (lambda ds, m: store.update(m=m))
    F._flow, F.set_file = (lambda ds: d / ".lsflow"), (lambda img, ext: img.with_suffix(ext))
    try:
        rows, n = F.tag_counts("pip", ".txt", "pip")
        need(n == 3 and rows[0] == ("blonde hair", 3), f"wrong counts: {rows[:3]}")
        tick = F.default_ticks(rows, n)
        need(tick == ["blonde hair", "blue eyes"], f"wrong pre-ticks: {tick}")
        F.move_tags("pip", tick, ".txt", "pip")
        need((d / "0.txt").read_text().strip() == "pip, smile, outdoors", (d / "0.txt").read_text())
        need(F.description("pip") == "blonde hair, blue eyes", f"description: {F.description('pip')}")
        F.put_back("pip", ".txt")
        need((d / "0.txt").read_text().strip() == "pip, blonde hair, blue eyes, smile, outdoors", "not put back")
        need(F.description("pip") == "", f"description not undone: {F.description('pip')}")
    finally:
        for k, v in saved.items():
            setattr(F, k, v)


@check("worn items: kind and place words made the same, the item's look moved out, the kinds kept")
def _worn():
    import tempfile
    import dsflow as F
    d = Path(tempfile.mkdtemp())
    imgs = []
    for i, line in enumerate(["lsjc, 1girl, gold ring, rings, silver, on finger, brown hair, indoors",
                              "lsjc, 1girl, hairclip, silver, in hair, smile, outdoors",
                              "lsjc, 1girl, ring, silver, hairclip, on finger, in hair, red dress"]):
        img = d / f"{i}.png"
        img.write_bytes(b"x")
        (d / f"{i}.txt").write_text(line + "\n", encoding="utf-8")
        imgs.append(img)
    store = {"m": {"purpose": "worn", "worn_kind": "jewelry", "types": []}}
    saved = {k: getattr(F, k) for k in ("images", "meta", "meta_save", "_flow", "set_file", "sg")}
    F.images, F.meta, F.meta_save = (lambda ds: imgs), (lambda ds: store["m"]), (lambda ds, m: store.update(m=m))
    F._flow, F.set_file = (lambda ds: d / ".lsflow"), (lambda img, ext: img.with_suffix(ext))
    F.sg = lambda: type("S", (), {"record_change": staticmethod(lambda *a, **k: None)})()
    try:
        k, pl = F.suggest_worn("lsjc", ".txt", "lsjc")
        need("ring = rings" in k and "hair clip = hairclip" in k, f"kinds not suggested: {k!r}")
        F.set_worn_setup("lsjc", "ring = rings\nhair clip = hairclip", "on finger\nin hair")
        msg = F.same_words("lsjc", ".txt", "lsjc")
        need((d / "0.txt").read_text().startswith("lsjc, ring, on finger, 1girl, gold ring"), (d / "0.txt").read_text())
        need("**ring** 2" in msg and "**hair clip** 2" in msg, f"balance: {msg}")
        rows, n = F.tag_counts("lsjc", ".txt", "lsjc")
        tick = F.default_ticks(rows, n, "worn", "jewelry", ["ring", "hair clip"], ["on finger", "in hair"])
        need("silver" in tick and "ring" not in tick and "1girl" not in tick, f"wrong pre-ticks: {tick}")
        F.move_tags("lsjc", ["silver", "gold ring"], ".txt", "lsjc")
        t0 = (d / "0.txt").read_text()
        need("silver" not in t0 and "gold ring" not in t0 and t0.startswith("lsjc, ring,"), t0)
        need(F.description("lsjc") == "silver, gold ring", F.description("lsjc"))
    finally:
        for k_, v in saved.items():
            setattr(F, k_, v)


@check("AMD GPU values are read from amd-smi (ROCm 10+) as from rocm-smi")
def _amdsmi():
    import gpusmi as GS
    a = [{"gpu": 0, "usage": {"gfx_activity": {"value": 37, "unit": "%"}},
          "power": {"socket_power": {"value": 212, "unit": "W"}, "power_limit": {"value": 304, "unit": "W"}},
          "temperature": {"edge": {"value": 61, "unit": "C"}, "hotspot": {"value": 78, "unit": "C"}},
          "mem_usage": {"total_vram": {"value": 16304, "unit": "MB"}, "used_vram": {"value": 9120, "unit": "MB"}}}]
    c = GS.parse_amd_smi(a)[0]
    need((c["load"], c["temp"], c["junction"], c["power"]) == (37, 61, 78, 212), f"amd-smi metric: {c}")
    need(abs(c["vram_total"] / 2 ** 20 - 16304) < 1, f"VRAM not in bytes: {c}")
    m = GS.parse_amd_smi({"gpu_data": [{"gpu": 0, "gfx": "55 %", "vram_used": "8000 MB", "vram_total": "16304 MB",
                                        "hotspot_temperature": "80 °C", "power_usage": "180 W"}]})[0]
    need((m["load"], m["junction"], m["power"]) == (55, 80, 180) and m["vram_used"] == 8000 * 2 ** 20, f"monitor: {m}")


@check("Pony gets score_9 … in front of the captions, Illustrious masterpiece, best quality behind")
def _quality():
    import sd_script as SS
    pony = SS.add_quality("pony", {"label": "Pony (SDXL)", "flags": ["--shuffle_caption --keep_tokens 1"]})
    need('--caption_prefix "score_9, score_8_up, score_7_up, source_anime,"' in pony["flags"], pony["flags"])
    need("--shuffle_caption --keep_tokens 5" in pony["flags"], f"keep_tokens not moved: {pony['flags']}")
    ill = SS.add_quality("illustrious", {"label": "Illustrious (SDXL)", "flags": ["--keep_tokens 1"]})
    need('--caption_suffix ", masterpiece, best quality"' in ill["flags"] and "--keep_tokens 1" in ill["flags"], ill)
    need(not SS.add_quality("sdxl", {"label": "SDXL", "flags": []}).get("quality_words"), "plain SDXL got quality words")


@check("a launch script never samples without prompts, and SDXL scripts carry no --clip_skip")
def _fixflags():
    import sdgui as G
    import sd_script as SS
    a = 'accelerate launch sdxl_train_network.py \\\n  --seed 42 --sample_every_n_epochs 1 --clip_skip 2\n'
    b, n = G.fix_flags(a)
    need("--sample_every_n_epochs" not in b and "--clip_skip" not in b and len(n) == 2, b)
    c = 'accelerate launch sdxl_train_network.py --sample_every_n_epochs 1 --sample_prompts "p.txt"\n'
    need(G.fix_flags(c)[0] == c, "sampling with prompts was changed")
    vals = ["1" if f == "sample_every_n_epochs" else "2" if f == "clip_skip" else "" for f, _ in SS.FORM_FIELDS]
    t, _, _ = SS.apply_form('accelerate launch sdxl_train_network.py --seed 42\n', vals)
    need("--sample_every_n_epochs" not in t and "--clip_skip" not in t, t)


@check("a chosen LoRA goes into the folder of its LoRA type, never a LoRA folder's root")
def _copytarget():
    import tempfile
    import housekeeping as HK
    r = Path(tempfile.mkdtemp())
    (r / "illustrious").mkdir()
    (r / "oldILLV1.safetensors").write_bytes(b"x")
    t = HK.copy_target("oldILLV2", [str(r)], "Illustrious (SDXL)")
    need(t["dir"] == r / "illustrious", f"went to {t['dir']}")
    need(HK.copy_target("newSDXLV1", [str(r)], "SDXL")["dir"] == r / "SDXL", "no new type folder")


@check("the queue stops what a run leaves behind")
def _sweep():
    import subprocess
    import time as _t
    pr = subprocess.Popen(["bash", "-c", "sleep 60 & exit 0"], start_new_session=True)
    pr.wait()
    _t.sleep(0.3)
    need(G.sweep_run(pr) == 1, "a left-over process was not found")
    need(G.sweep_run(pr) == 0, "the process group is still there after the sweep")


@check("step 5 pins win over the rules; the sort leaves out pictures that teach nothing about the subject")
def _pins():
    import datacheck as DC
    import dsflow as F
    need(DC.teaches({"teaches": "1"}) == 1 and DC.teaches({}) is None, "the usefulness rating is not read")
    bad = [t for s, t in DC.content_issues({"people": 1, "teaches": 1}, "character") if s == DC.BAD]
    need(any("teaches almost nothing" in t for t in bad), f"a useless picture is not left out: {bad}")
    need(not any("teaches" in t for s, t in DC.content_issues({"people": 1, "teaches": 1}, "style")),
         "the usefulness rating should not judge a style LoRA")
    dress = {"people": 1, "teaches": 3, "face_size": "tiny", "features_visible": True, "framing": "full body"}
    need(any(s == DC.BAD and "face too small" in t for s, t in DC.content_issues(dress, "character")),
         "a tiny figure with a tiny face is not left out")
    pose = {"people": 1, "teaches": 3, "face_size": "small", "features_visible": False}
    need(any(s == DC.BAD and "hides nearly everything" in t for s, t in DC.content_issues(pose, "character")),
         "a pose that hides the character is not left out")
    need(DC.plain_part("both feet") and DC.plain_part("hands") and not DC.plain_part("tail") and
         not DC.plain_part("horns"), "plain parts (hands, feet) are not told from features (tail, horns)")
    old = F.pins
    try:
        F.pins = lambda ds: {"a": "keep", "b": "never"}
        rows = [{"path": "/x/10_d/a.png", "cat": "out"}, {"path": "/x/10_d/b.png", "cat": "good"},
                {"path": "/x/10_d/c.png", "cat": "good"}]
        first, cands = F._apply_pins("d", rows, [r for r in rows if r["cat"] == "good"])
        need([r["path"] for r in first] == ["/x/10_d/a.png"], "an 'always keep' pin is not taken first")
        need([r["path"] for r in cands] == ["/x/10_d/c.png"], "a 'never' pin is still a candidate")
        F._pin_reasons(rows)
        need(rows[0]["keep"] is True and rows[1]["keep"] is False, "the pins do not decide keep")
    finally:
        F.pins = old


@check("sorting stays valid when pictures move or are removed (only new pictures ask for a new sort)")
def _sortkeep():
    import dsflow as F
    pics = {"/a/10_x/p1.png": {"cat": "good"}, "/a/10_x/p2.png": {"cat": "out"}}
    find = F.sort_lookup(pics)
    need(find("/a/20_x/p1.png") is not None, "a renamed repeat folder lost the sort")
    need(find("/a/10_x/p2.webp") is not None, "another format lost the sort")
    need(find("/a/10_x/p9.png") is None, "a new picture was taken as sorted")


# ---------------------------------------------------------------- the page
@check("the training page builds")
def _page():
    demo = G.build_ui("")
    n = len(demo.blocks) if hasattr(demo, "blocks") else 0
    need(n > 50, f"only {n} parts on the page")
    return f"{n} components"


# ---------------------------------------------------------------- done
for k, v in _saved.items():
    setattr(G, k, v)
shutil.rmtree(TMP, ignore_errors=True)
bad = [r for r in RESULTS if not r[0]]
print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed" + ("" if not bad else ":"))
for _, name, why in bad:
    print(f"  ✕ {name}: {why}")
if "--json" in sys.argv:
    print(json.dumps([{"ok": a, "name": b, "note": c} for a, b, c in RESULTS]))
sys.exit(1 if bad else 0)
