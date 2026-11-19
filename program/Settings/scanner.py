# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Find on this computer: looks through the folders you list (only those, and their subfolders) for what LoRA Studio
needs and suggests what to use. Nothing is changed until you tick and press Use.

  program folders   sd-scripts (sdxl_train_network.py), musubi-tuner (src/musubi_tuner), ComfyUI (main.py + comfy/),
                    its output folder (pictures), LoRA Manager's example pictures folder (its settings.json)
  LoRA folders      folders with LoRA files (a ComfyUI models/loras first), not the model/ folders of datasets
  dataset folders   the folders that hold datasets (a dataset = a folder with an <N>_name repeat folder of pictures)
  model folder      ComfyUI's models folder, else the folder that holds most model files
  model files       every .safetensors is identified by its header (model_guide.identify: SDXL checkpoint, VAE,
                    text encoder, DiT …, read once and remembered); per LoRA type and slot the fitting files, best first
The search skips hidden folders, venvs, caches and system folders, and goes at most 8 folders deep.
"""
from __future__ import annotations
__version__ = "1.191126c"  # LoRA Studio version, see version.py

import json
import os
import re
from collections import Counter
from pathlib import Path

import lspaths as _LP  # noqa: E402

ROOTS_FILE = _LP.USER / "scan_roots.txt"
SKIP = {"proc", "sys", "dev", "run", "tmp", "boot", "lost+found", "snap", "node_modules", "__pycache__", "site-packages",
        "venv", ".venv", "env", ".git", "cache", ".cache", "trash", ".trash", "temp", "input", "_reserve",
        "_originals", "_set_aside", "logs", "log"}
IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REPEAT = re.compile(r"^\d+_\S")
DEPTH = 8
MAX_FILES = 6000


def roots() -> list[str]:
    try:
        return [x.strip() for x in ROOTS_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def save_roots(text: str) -> list[str]:
    rs = [x.strip() for x in (text or "").splitlines() if x.strip()]
    ROOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROOTS_FILE.write_text("\n".join(rs) + "\n", encoding="utf-8")
    return rs


def _has_pictures(d: Path) -> bool:
    try:
        return any(f.suffix.lower() in IMG for f in d.iterdir())
    except OSError:
        return False


def _lm_examples(comfy: Path) -> str:
    for name in ("ComfyUI-Lora-Manager", "comfyui-lora-manager"):
        f = comfy / "custom_nodes" / name / "settings.json"
        try:
            p = (json.loads(f.read_text(encoding="utf-8")).get("example_images_path") or "").strip()
        except (OSError, ValueError):
            continue
        if p and Path(p).is_dir():
            return p
    return ""


def scan(start, say=None, stop=None) -> dict:
    """Look through the start folders. Returns {"folders": {setting: [(path, why)]}, "models": [(path, info)],
    "seen": folders looked at, "roots": [...]}."""
    say = say or (lambda t: None)
    found = {k: Counter() for k in ("sd", "mu", "comfy", "pictures", "lm_examples", "loras", "datasets", "models")}
    files_in = Counter()                                 # how many LoRA / model files a folder holds (shown)
    why = {}
    safes, ds_dirs = [], []
    seen = 0
    for r in [Path(os.path.expanduser(x)) for x in start]:
        if not r.is_dir():
            continue
        base = len(r.parts)
        for dp, dns, fns in os.walk(r, followlinks=False):
            if stop is not None and stop.is_set():
                break
            d = Path(dp)
            seen += 1
            if seen % 400 == 0:
                say(f"Looking … {seen} folders, {len(safes)} model files so far ({d})")
            names = set(fns)
            dns[:] = [x for x in dns if not x.startswith(".") and x.lower() not in SKIP]
            if len(d.parts) - base >= DEPTH:
                dns[:] = []
            if "sdxl_train_network.py" in names and "train_network.py" in names:
                found["sd"][str(d)] += 1
                why[("sd", str(d))] = "has sdxl_train_network.py"
                dns[:] = []
                continue
            if (d / "src" / "musubi_tuner").is_dir():
                found["mu"][str(d)] += 1
                why[("mu", str(d))] = "has src/musubi_tuner"
                dns[:] = []
                continue
            if "main.py" in names and (d / "comfy").is_dir() and (d / "custom_nodes").is_dir():
                found["comfy"][str(d)] += 1
                why[("comfy", str(d))] = "ComfyUI (main.py, comfy/)"
                if (d / "output").is_dir():
                    found["pictures"][str(d / "output")] += 5
                    why[("pictures", str(d / "output"))] = "ComfyUI's output folder"
                if (d / "models").is_dir():
                    found["models"][str(d / "models")] += 50
                    why[("models", str(d / "models"))] = "ComfyUI's models folder"
                if (d / "models" / "loras").is_dir():
                    found["loras"][str(d / "models" / "loras")] += 50
                    why[("loras", str(d / "models" / "loras"))] = "ComfyUI's LoRA folder"
                ex = _lm_examples(d)
                if ex:
                    found["lm_examples"][ex] += 1
                    why[("lm_examples", ex)] = "from LoRA Manager's settings"
                dns[:] = [x for x in dns if x in ("models",)]
            reps = [x for x in dns if REPEAT.match(x) and _has_pictures(d / x)]
            if reps:                                     # a dataset: its parent holds datasets
                ds_dirs.append(d)
                found["datasets"][str(d.parent)] += 1
                dns[:] = []
                continue
            for f in fns:
                if f.lower().endswith((".safetensors", ".sft")) and len(safes) < MAX_FILES:
                    safes.append(d / f)
    say(f"Reading the headers of {len(safes)} model files (remembered, so a next search is quick) …")
    import model_guide as G
    models = []
    in_ds = tuple(str(x) + os.sep for x in ds_dirs)
    for i, f in enumerate(safes):
        if stop is not None and stop.is_set():
            break
        if i and i % 200 == 0:
            say(f"Reading the headers: {i} of {len(safes)}")
        if str(f).startswith(in_ds):
            continue                                     # a dataset's own LoRAs (model/) are not your LoRA folder
        info = G.identify(f)
        if info.get("kind") in ("lora", "zimage_lora"):
            top = next((a for a in [f.parent] + list(f.parents) if a.name.lower() in ("loras", "lora")), f.parent)
            found["loras"][str(top)] += 1
            files_in[str(top)] += 1
        elif info.get("kind") not in ("unknown", "missing"):
            models.append((f, info))
            top = next((a for a in f.parents if a.name.lower() in ("models", "model")), None)
            if top:
                found["models"][str(top)] += 1
                files_in[str(top)] += 1
    G.save_cache()
    for k, c in found.items():
        for p in c:
            why.setdefault((k, p), {"loras": "LoRA files", "models": "model files", "datasets": "datasets"}.get(k, ""))
    def label(k, p, n):
        w = why.get((k, p), "")
        if k in ("loras", "models"):
            return f"{w}, {files_in[p]} file{'s' if files_in[p] != 1 else ''}"
        return f"{w} ({n})" if k == "datasets" else w
    folders = {k: [(p, label(k, p, n)) for p, n in c.most_common(6)] for k, c in found.items()}
    return {"folders": folders, "models": models, "seen": seen, "roots": list(start)}


NAME_HINTS = {"sdxl": ["sdxl", "sdxlbase"], "illustrious": ["illustrious"], "noobai": ["noobai", "noob"],
              "noobai_vpred": ["vpred"], "pony": ["pony"]}
AVOID = {"sdxl": ["illustrious", "noob", "pony"], "illustrious": ["noob", "pony"], "noobai": ["vpred"],
         "pony": ["illustrious", "noob"]}


def model_picks(models) -> dict:
    """{tkey: {slot: [(path, description, sure)]}}: per LoRA type and slot the fitting files, best first (at most 4);
    sure = False when the best base model is only a file of the same architecture (another SDXL family's model)."""
    import model_guide as G
    import sd_settings as SS
    import sdgui as S
    _, presets = S.merged_presets()
    out = {}
    for tkey in presets:
        req = G.requirements_for(tkey, SS._type_base(tkey))
        if not req:
            continue
        slots = {}
        for flag, _ in SS.type_slots(tkey):
            kinds = req.get(flag)
            if not kinds:
                continue
            ok = [(p, i) for p, i in models if not G.problems(i, flag, kinds)]
            words = NAME_HINTS.get(tkey) or ([w for w in re.split(r"[^a-z0-9]+", tkey.lower()) if len(w) > 2]
                                             if tkey not in presets or presets[tkey].get("base") else [])
            avoid = AVOID.get(tkey, [])

            def fit(p):                                  # an SDXL slot fits SDXL, Illustrious and Pony files alike:
                n = p.name.lower().replace("-", "").replace("_", "")      # the type's name in the file name first
                return (0 if any(w in n for w in words) else 1) + (2 if any(w in n for w in avoid) else 0)
            if flag in G.BASE_FLAGS and tkey in NAME_HINTS and tkey != "sdxl":
                ok = [(p, i) for p, i in ok if fit(p) == 0]     # its header looks like any SDXL: go by the name
            ok.sort(key=lambda pi: (fit(pi[0]) if flag in G.BASE_FLAGS else 0, kinds.index(pi[1]["kind"]),
                                    *G._rank(pi[1], flag)))
            if ok:                                       # sure: the name says it is this type's model (or no need)
                slots[flag] = [(str(p), G.describe(i), flag not in G.BASE_FLAGS or fit(p) == 0) for p, i in ok[:4]]
        if slots:
            out[tkey] = slots
    return out


def current() -> dict:
    """What is set now: {"sd", "mu", "datasets", "loras", "models", "pictures", "lm_examples", "files": {tkey: {slot}}}."""
    import lora_library as LL
    import sd_settings as SS
    import sdgui as S
    st = LL.load_settings()
    files = {}
    for tkey, sec in SS.read_settings().items():
        files[tkey] = {k: v for k, v in sec.items() if not k.endswith(".dir")}
    return {"sd": str(S._general_path(S.SD_KEY) or ""), "mu": str(S._general_path(S.MUSUBI_KEY) or ""),
            "datasets": "\n".join(st.get("dataset_dirs") or []), "loras": "\n".join(st.get("lora_dirs") or []),
            "models": st.get("models_dir") or "", "pictures": st.get("pictures_dir") or "",
            "lm_examples": str(LL.LIB.data.get("lm_examples_dir") or ""), "files": files}


def apply(folders: dict, files: dict) -> str:
    """Save the ticked suggestions. folders: {setting: path}; files: {tkey: {slot: path}}."""
    import firstrun as FR
    import lora_library as LL
    out = []
    lib = {}
    if folders.get("datasets"):
        lib["dataset_dirs"] = [folders["datasets"]]
    if folders.get("loras"):
        lib["lora_dirs"] = [folders["loras"]]
    if folders.get("models"):
        lib["models_dir"] = folders["models"]
    if folders.get("pictures"):
        lib["pictures_dir"] = folders["pictures"]
    if lib:
        vals = LL.default_settings_values()
        if FR.LIB_SETTINGS.exists():
            cur = LL.load_settings()
            vals.update({k: cur[k] for k in cur if k in vals})
        for k, v in lib.items():                         # a found folder joins the ones you have (first in the list)
            if isinstance(v, list):
                vals[k] = list(dict.fromkeys(v + [x for x in vals.get(k) or [] if x]))
            else:
                vals[k] = v
        FR.LIB_SETTINGS.write_text(LL.render_settings(vals), encoding="utf-8")
        out.append("folders: " + ", ".join(lib))
    gen = {}
    if folders.get("sd"):
        gen["SDGUI_SD_SCRIPTS"] = FR.tilde(Path(folders["sd"]))
    if folders.get("mu"):
        gen["SDGUI_MUSUBI"] = FR.tilde(Path(folders["mu"]))
    if gen:
        FR.set_keys(FR.TRAIN_SETTINGS, "general", gen)
        out.append("trainers")
    if folders.get("lm_examples"):
        import maint
        maint.set_example_folder(folders["lm_examples"])
        out.append("LoRA Manager's example folder")
    n = 0
    if any(files.values() if files else []):
        import sd_settings as SS
        data = SS.read_settings()
        for tkey, slots in files.items():
            sec = data.setdefault(tkey, {})
            for slot, path in slots.items():
                sec[slot] = path
                sec[slot + ".dir"] = str(Path(path).parent)
                n += 1
        SS.write_settings(data)
    if n:
        out.append(f"{n} model file(s)")
    if not out:
        return "Nothing ticked."
    try:
        LL.LIB.scan()
    except Exception:                                    # noqa: BLE001
        pass
    return ("✔ Saved: " + "; ".join(out) + ". New launch scripts use the model files; a changed trainer folder is used "
            "from the next run on.")
