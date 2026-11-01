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
__version__ = "1.011126"  # LoRA Studio version, see version.py

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
