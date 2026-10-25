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
__version__ = "1.251026m"  # LoRA Studio version, see version.py

import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
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
        if name.endswith(".py") and name not in SKIP_IMPORT:
            importlib.import_module(name[:-3])
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
