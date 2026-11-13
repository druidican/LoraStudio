# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings of the trainer side: the Hugging Face token, the library scan, repeat folders, model files, LoRA subtypes
and the GPU / GFX values.

Part of sdgui.py (split by topic): the names of sdgui are reached as S.<name>; sdgui imports
the names of this file back, so callers use sdgui.<name> as before."""
from __future__ import annotations
__version__ = "1.131126c"  # LoRA Studio version, see version.py

import glob
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import notify
import sdgui as S


# --------------------------------------------------------------------------- Hugging Face token
HF_FILE = "huggingtoken.txt"


def hf_path() -> Path:
    """huggingtoken.txt: in the config folder (LoRA Studio's own folder), else in <sd-scripts>."""
    return (S.CONFIG_DIR or S.ROOT) / HF_FILE


def read_hf_token() -> str:
    """First non-empty, non-comment line of huggingtoken.txt (see hf_path(); empty string if missing)."""
    try:
        for line in (hf_path()).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line.strip("\"'")
    except OSError:
        pass
    return ""


def mask_token(tok: str) -> str:
    if not tok:
        return ""
    return f"{tok[:4]}…{tok[-4:]}" if len(tok) > 12 else "•" * len(tok)


def hf_status_text() -> str:
    tok = read_hf_token()
    if tok:
        return f"**HF token:** `{mask_token(tok)}` (from {HF_FILE})"
    return f"**HF token:** not set. Put it in `{hf_path()}`"


def get_export(text: str, name: str):
    m = re.search(rf"^export\s+{name}=(?:\"([^\"]*)\"|'([^']*)'|(\S*))", text, re.M)
    if not m:
        return None
    return next((g for g in m.groups() if g is not None), "")


def set_export(text: str, name: str, value: str) -> str:
    line = f'export {name}="{value}"'
    pat = re.compile(rf"^export\s+{name}=.*$", re.M)
    if pat.search(text):
        return pat.sub(lambda m: line, text, count=1)
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("cd "):                       # keep it above the cd into sd-scripts
            lines.insert(i, line)
            return "\n".join(lines)
    lines.insert(1 if lines and lines[0].startswith("#!") else 0, line)
    return "\n".join(lines)


def fill_hf(text: str, force: bool):
    """Write the token from huggingtoken.txt into `export HF_TOKEN=...`.
    force=False only fills an empty value. Returns (text, changed)."""
    tok = read_hf_token()
    if not tok or not re.fullmatch(r"[\w.\-]+", tok):
        return text, False
    cur = get_export(text, "HF_TOKEN")
    if cur == tok or (cur and not force):
        return text, False
    return set_export(text, "HF_TOKEN", tok), True


def cs_fill_hf(text, plan):
    tok = read_hf_token()
    if not tok:
        notify.warning(f"No token found in {hf_path()}.")
        return S._cs_result(status=f"No token found in `{hf_path()}`.", plan=plan)
    if not re.fullmatch(r"[\w.\-]+", tok):
        notify.warning("The token contains unusual characters; not written into the script.")
        return S._cs_result(status="The token contains unusual characters, so it was not written into the script.", plan=plan)
    new, changed = fill_hf(text or "", force=True)
    return S._cs_result(text=new, plan=plan,
                      status="`HF_TOKEN` set in the script text (not saved yet)." if changed
                      else "`HF_TOKEN` already matches huggingtoken.txt.")


# --------------------------------------------------------------------------- repeat folders (rename)
def base_name(name: str) -> str:
    return S.REPEAT_RE.sub("", name, count=1)


def folder_rows(ds):
    return [(f, r, len(S.list_images(f))) for f, r in S.repeat_folders(S.lora_path(ds))]


def rp_choices(ds):
    lora, out = S.lora_path(ds), []
    for f, r, n in folder_rows(ds):
        if f == lora:
            label = f"(images directly in {lora.name}/)  {n} images, no repeat folder"
        else:
            label = f"{f.relative_to(lora)}  ({n} images, " + (f"{r} repeats)" if r else "no repeat count)")
        out.append((label, str(f)))
    return out


def rp_new_name(ds, folder, repeats):
    if not ds or not folder:
        return ""
    lora, f = S.lora_path(ds), Path(folder)
    base = lora.name.lower() if f == lora else base_name(f.name)
    try:
        r = int(repeats)
    except (TypeError, ValueError):
        r = 0
    return f"{r}_{base}" if r > 0 else ""


def rp_refresh(ds, tkey, steps_over):
    if not ds or not S.lora_path(ds).is_dir():
        return {"choices": [], "folder": None, "repeats": 0, "newname": "", "info": ""}
    rows = folder_rows(ds)
    if not rows:
        return {"choices": [], "folder": None, "repeats": 0, "newname": "", "info": "No folders with images found in this dataset."}
    n_total = sum(n for _, _, n in rows)
    rec, label = None, tkey
    try:
        sc, presets = S.merged_presets()
        cfg = dict(presets[tkey])
        so = S._num(steps_over)
        if so:
            cfg["steps"] = so
        label = cfg["label"]
        if sc:
            bs, ga, _ = sc.suggest_settings(n_total, cfg)
            rec = sc.suggest_repeats(n_total, bs, ga, cfg["epochs"], cfg["steps"])
    except Exception:
        pass
    pick = next((f for f, r, n in rows if r is None), None)
    if pick is None and rec:
        pick = next((f for f, r, n in rows if r != rec), None)
    if pick is None:
        pick = rows[0][0]
    repeats = rec or next((r for f, r, n in rows if f == pick and r), 0)
    lora = S.lora_path(ds)
    lines = [f"Recommended repeats for {label} with {n_total} images: **{rec}**." if rec else f"{n_total} images."]
    for f, r, n in rows:
        nm = "(LoRA folder itself)" if f == lora else f"`{f.relative_to(lora)}`"
        lines.append(f"- {nm}: {n} images, " + (f"{r} repeats" + (" (matches)" if rec and r == rec else "")
                                                if r else "**no repeat count: sd-scripts ignores these images**"))
    return {"choices": rp_choices(ds), "folder": str(pick), "repeats": repeats,
            "newname": rp_new_name(ds, str(pick), repeats), "info": "\n".join(lines)}


def rp_rename(ds, folder, newname, move_ok):
    """Rename a repeat folder to <N>_name, or move loose images into a new repeat folder. Returns (message, done)."""
    keep = False
    if not ds or not folder:
        notify.warning("Pick a folder first.")
        return "Pick a folder first.", keep
    lora, f, name = S.lora_path(ds), Path(folder), (newname or "").strip()
    if not re.fullmatch(r"\d+_[^/\\\x00]+", name):
        notify.warning("The name must look like 4_name (a repeat count, an underscore, then a name).")
        return "The name must look like `4_name`.", keep
    if f == lora:                                   # loose images directly in the LoRA folder
        if not move_ok:
            notify.warning("Tick the box to allow moving the images into a new folder.")
            return "Tick the box above to allow moving these images into a new folder.", keep
        dest = lora / name
        if dest.exists():
            notify.warning(f"{name} already exists.")
            return f"`{name}` already exists.", keep
        dest.mkdir()
        moved = 0
        for im in S.list_images(lora):
            for src in [im] + [im.with_name(im.stem + e) for e in S.CAPTION_EXTS]:
                if src.is_file():
                    shutil.move(str(src), str(dest / src.name))
                    moved += 1
        notify.info(f"Moved {moved} files into {name}/")
        return f"Moved {moved} files (images and captions) into `{name}/`.", True
    dest = f.with_name(name)
    if dest == f:
        return "The folder already has that name.", keep
    if dest.exists():
        notify.warning(f"{name} already exists.")
        return f"`{name}` already exists next to it.", keep
    f.rename(dest)
    notify.info(f"Renamed {f.name} -> {name}")
    return f"Renamed `{f.name}` to `{name}`. Epochs are re-planned for the folder's actual repeat count.", True


def rp_make(ds, tkey, text, plan):
    if not ds:
        notify.warning("Pick a LoRA folder first.")
        return "Pick a LoRA folder first."
    notes = S.ensure_folders(ds, text or "", plan, tkey)
    notify.info("Folders checked.")
    return ("Done: " + "; ".join(notes) + ".") if notes else "Nothing to create: the repeat folder, `model` and `log` already exist."


def cs_set_epochs(ds, tkey, script, text, plan, steps_over):
    """Set epochs so the run lands on the target step count with the script's batch/accum and the folder's repeats."""
    try:
        _, presets = S.merged_presets()
        cfg = dict(presets[tkey])
        steps = S._num(steps_over) or cfg["steps"]
        stats = S.dataset_stats(ds)
        f = S.form_dict(text or "")
        bs = max(1, int(float(f.get("train_batch_size") or 1)))
        ga = max(1, int(float(f.get("gradient_accumulation_steps") or 1)))
        reps = (plan or {}).get("repeats") or [r["repeats"] for r in stats["rows"]]
        samples = sum(r["n"] * rep for r, rep in zip(stats["rows"], reps))
        spe = max(1, math.ceil(samples / (bs * ga)))
        ep = max(1, round(steps / spe))
    except Exception as e:
        notify.warning(f"Could not calculate epochs: {e}")
        return S._cs_result(status=f"Could not calculate epochs: {e}", plan=plan)
    vals = [""] * len(S.FORM_FIELDS)
    vals[[n for n, _ in S.FORM_FIELDS].index("max_train_epochs")] = str(ep)
    new, _, _ = S.apply_form(text or "", vals)
    return S._cs_result(text=new, fields=S.read_form(new), explain=S.explain_for(ds, tkey, script, new, plan),
                      status=f"Epochs set to {ep}: {spe} steps/epoch × {ep} = ~{spe * ep} steps (target {steps}). Not saved yet.",
                      plan=plan)


MODEL_EXTS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin"}


SLOT_LABELS = {
    "pretrained_model_name_or_path": "Model (checkpoint / diffusion model)", "vae": "VAE", "ae": "AE (VAE)",
    "qwen3": "Qwen3 text encoder", "clip_l": "CLIP-L text encoder", "clip_g": "CLIP-G text encoder",
    "t5xxl": "T5-XXL text encoder", "gemma2": "Gemma2 text encoder", "text_encoder": "Text encoder",
    "byt5": "ByT5 text encoder", "dit": "DiT (Z-Image Turbo diffusion model)",
    "base_weights": "Training adapter (--base_weights)",
}


OPTIONAL_VAE_TYPES = {"sdxl", "illustrious", "sd15"}


def vae_optional(tkey, presets) -> bool:
    return tkey in OPTIONAL_VAE_TYPES or (presets.get(tkey) or {}).get("base") in OPTIONAL_VAE_TYPES


def settings_path() -> Path:
    """Training settings: in the config folder (LoRA Studio's own folder), else <sd-scripts>/settings.txt."""
    return (S.CONFIG_DIR or S.ROOT) / S.SETTINGS_FILE


def slot_label(key: str) -> str:
    return SLOT_LABELS.get(key, key.replace("_", " "))


SECTION_NAMES: dict = {}       # lower-case section name -> the spelling used in the file, so [Pony] is written back as [Pony]


def read_settings() -> dict:
    """settings.txt -> {lora_type: {flag: path, "<flag>.dir": folder}}. Section names are case-insensitive."""
    out, sec = {}, None
    try:
        lines = settings_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for ln in lines:
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        if t.startswith("[") and t.endswith("]"):
            name = t[1:-1].strip()
            sec = name.lower()
            SECTION_NAMES[sec] = name
            out.setdefault(sec, {})
        elif sec and "=" in t:
            k, _, v = t.partition("=")
            out[sec][k.strip()] = v.strip()
    return out


def write_settings(data: dict):
    lines = ["# sdgui settings: model, VAE and text-encoder files for each LoRA type.",
             "# Written by sdgui.py (Settings tab); safe to edit by hand.",
             "# One [section] per LoRA type, `flag = path` lines, and `<flag>.dir` remembers the folder shown in the file list.", ""]
    for sec, kv in data.items():
        kv = {k: v for k, v in kv.items() if v}
        if kv:
            lines.append(f"[{SECTION_NAMES.get(sec, sec)}]")
            lines += [f"{k} = {v}" for k, v in kv.items()]
            lines.append("")
    path = settings_path()
    if path.is_file():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    path.write_text("\n".join(lines), encoding="utf-8")


def settings_models(tkey) -> dict:
    """{flag_without_dashes: path} saved for a LoRA type (used for new launch scripts)."""
    return {k: v for k, v in read_settings().get(tkey, {}).items() if v and not k.endswith(".dir") and k not in S.META_KEYS}


def type_slots(tkey):
    _, presets = S.merged_presets()
    cfg = presets.get(tkey) or {}
    slots = [(f.lstrip("-"), path) for f, path in cfg.get("model_args", [])]
    if vae_optional(tkey, presets) and not any(k in ("vae", "ae") for k, _ in slots):
        slots.append(("vae", ""))                    # optional external VAE (--vae) for SD1.5 / SDXL family
    return slots


def list_model_files(folder, recursive=True, limit=1000):
    """[(label, full_path)] of model files in a folder (and up to two levels of subfolders)."""
    base = Path(os.path.expanduser((folder or "").strip()))
    if not (folder or "").strip() or not base.is_dir():
        return []
    found = []
    try:
        if recursive:
            for dirpath, dirnames, filenames in os.walk(base):
                depth = len(Path(dirpath).relative_to(base).parts)
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")) if depth < 2 else []
                found += [Path(dirpath) / f for f in filenames if Path(f).suffix.lower() in MODEL_EXTS]
                if len(found) >= limit:
                    break
        else:
            found = [f for f in base.iterdir() if f.is_file() and f.suffix.lower() in MODEL_EXTS]
    except OSError:
        pass
    out = []
    for f in sorted(found, key=lambda x: str(x).lower())[:limit]:
        try:
            size = f" ({f.stat().st_size / 1024 ** 3:.1f} GB)"
        except OSError:
            size = ""
        out.append((f"{f.relative_to(base)}{size}", str(f)))
    return out


def _slot_dir(path, saved_dir):
    """Folder shown for a model-file slot: the saved one, else the folder of the preset file if it exists
    on this computer, else DEFAULT_MODEL_DIR (the model folder chosen in LoRA Studio's setup)."""
    if saved_dir:
        return saved_dir
    if path and "/PATH/TO" not in path:
        parent = Path(os.path.expanduser(path)).parent
        if parent.is_dir() or not S.DEFAULT_MODEL_DIR:
            return str(parent)
    return S.DEFAULT_MODEL_DIR or ""


def _type_base(tkey):
    _, presets = S.merged_presets()
    return (presets.get(tkey) or {}).get("base", "")


def st_suggest(tkey, flags, *args):
    """Suggestions for the other model files of this LoRA type, from the files found in the slot folders."""
    if S.GUIDE is None:
        return "model_guide.py is missing next to sdgui.py, so no suggestions.", {}
    if not tkey:
        return "", {}
    dirs, files = list(args[:S.ST_N]), list(args[S.ST_N:2 * S.ST_N])
    current = {f: (v or "") for f, v in zip(flags, files) if f}
    roots = [d for d in dirs if d] + ([S.DEFAULT_MODEL_DIR] if S.DEFAULT_MODEL_DIR else [])
    try:
        r = S.GUIDE.suggest(tkey, _type_base(tkey), [f for f in flags if f], current, roots)
    except Exception as e:                          # noqa: BLE001
        return f"Could not build suggestions: {e}", {}
    return r["md"], r["picks"]


def st_online(tkey, base_file, progress=None):
    if S.GUIDE is None:
        return "model_guide.py is missing next to sdgui.py."
    base = (base_file or "").strip()
    if not base or "/PATH/TO" in base:
        return "Choose the base model file first."
    try:
        return S.GUIDE.lookup_online(tkey, os.path.expanduser(base), read_hf_token() or "")
    except Exception as e:                          # noqa: BLE001
        return f"Online lookup failed: {e}"


# ---- Hugging Face downloads (hfdownload.py), on a click
def _hd():
    import hfdownload
    return hfdownload


def st_download_stop():
    _hd().STOP.set()
    notify.info("Stopping the download: it goes on from there next time.")


def st_raw() -> str:
    try:
        return settings_path().read_text(encoding="utf-8")
    except OSError:
        return "(settings.txt does not exist yet)"


def st_save(tkey, flags, *args):
    dirs, files = args[:S.ST_N], args[S.ST_N:2 * S.ST_N]
    if not tkey:
        notify.warning("Pick a LoRA type first.")
        return "Pick a LoRA type first.", st_raw()
    data = read_settings()
    sec = {}
    rows = ["| Slot | File | |", "|---|---|---|"]
    for key, d, f in zip(flags, dirs, files):
        if not key:
            continue
        f = (f or "").strip()
        d = (d or "").strip()
        if "\n" in f or "\n" in d:
            notify.warning(f"{key}: paths cannot contain line breaks.")
            return f"{key}: paths cannot contain line breaks.", st_raw()
        if d:
            sec[key + ".dir"] = d
        if f:
            sec[key] = f
        ok = bool(f) and Path(os.path.expanduser(f)).is_file()
        rows.append(f"| {slot_label(key)} | `{f or '(not set: the preset default is used)'}` | "
                    + ("found" if ok else ("**file not found**" if f else "")) + " |")
    meta = S.custom_subtypes().get(tkey)
    if meta:                                          # keep (and make explicit) what defines a subtype
        sec = {"base": meta["base"], "label": meta["label"], "tag": meta["tag"], **sec}
    data[tkey] = sec
    try:
        write_settings(data)
    except OSError as e:
        notify.warning(f"Could not write settings.txt: {e}")
        return f"Could not write `{settings_path()}`: {e}", st_raw()
    notify.info(f"Saved {Path(settings_path()).name}")
    return (f"**Saved to `{settings_path()}`** for this LoRA type (other types are untouched; the previous file is kept as "
            "`settings.txt.bak`). New launch scripts of this type now use these files. For an existing script press "
            "**Use model files from Settings** on a dataset's Train.\n\n" + "\n".join(rows)), st_raw()


# ---- LoRA subtypes (e.g. Pony or another SDXL / Anima variant), stored as their own section in settings.txt
def sub_list_md() -> str:
    subs = S.custom_subtypes()
    if not subs:
        return ("No subtypes yet. **Illustrious** is already built in. Add others (for example Pony) below, "
                "or write a section like `[pony]` with `base = sdxl` in settings.txt.")
    rows = ["| Name | Based on | File tag | Section |", "|---|---|---|---|"]
    for k, c in subs.items():
        rows.append(f"| {c['label']} | {S.SUBTYPE_BASES[c['base']]} | `{c['tag']}` | `[{SECTION_NAMES.get(k, k)}]`"
                    + ("" if c["explicit"] else " (found in settings.txt, base guessed from its keys)") + " |")
    return "\n".join(rows)


def sub_choices():
    return [(f"{c['label']} ({S.SUBTYPE_BASES[c['base']]})", k) for k, c in S.custom_subtypes().items()]


def sub_add(name, base, tag):
    name = (name or "").strip()
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    if not key:
        notify.warning("Give the subtype a name, for example Pony.")
        return "Give the subtype a name, for example **Pony**."
    _, builtin = S.builtin_presets()
    if key in S.RESERVED_TYPES or key in builtin:
        notify.warning(f"'{name}' is already a built-in type.")
        return f"`{key}` is already a built-in type, so it cannot be added again. Pick it in the LoRA type list below."
    if key in S.custom_subtypes():
        notify.warning(f"The subtype '{name}' already exists.")
        return f"The subtype `{key}` already exists. Pick it in the LoRA type list below."
    if base not in S.SUBTYPE_BASES:
        notify.warning("Pick SDXL or Anima as the base.")
        return "Pick SDXL or Anima as the base."
    tag = re.sub(r"[^A-Za-z0-9]", "", tag or "") or re.sub(r"[^A-Za-z0-9]", "", name).upper()
    data = read_settings()
    data[key] = {"base": base, "label": name, "tag": tag}
    SECTION_NAMES[key] = key
    try:
        write_settings(data)
    except OSError as e:
        return f"Could not write `{settings_path()}`: {e}"
    notify.info(f"Subtype {name} added")
    return (f"Added **{name}** (based on {S.SUBTYPE_BASES[base]}, file tag `{tag}`). It is now in the LoRA type "
            f"lists. Choose its model files below and press **Save to settings.txt**.")


def sub_remove(key, confirm):
    if not key:
        notify.warning("Pick a subtype to remove.")
        return "Pick a subtype to remove."
    if not confirm:
        notify.warning("Tick the confirmation box first.")
        return "Tick the confirmation box first."
    data = read_settings()
    if key not in S.custom_subtypes() or key not in data:
        return "That subtype does not exist any more."
    del data[key]
    try:
        write_settings(data)
    except OSError as e:
        return f"Could not write `{settings_path()}`: {e}"
    notify.info("Subtype removed")
    return (f"Removed `{key}` from `{settings_path()}` (a `.bak` copy of the previous file was kept). Launch scripts "
            "that were made with it are not changed.")


# ---- GPU / GFX (HSA_OVERRIDE_GFX_VERSION), stored in settings.txt under [general]
GFX_VAR = "HSA_OVERRIDE_GFX_VERSION"
# (label, value). The value is what the ROCm override expects. Cards with a native ROCm target need no override
# in theory, but your scripts set one, so the list gives the matching number. Unsupported chips are mapped to
# the nearest supported target (RDNA2 -> 10.3.0, 780M -> 11.0.0). Check ROCm's compatibility page for your version.
KNOWN_GFX = [
    ("RX 9070 XT / 9070 / 9070 GRE, RDNA4 (gfx1201)", "12.0.1"),
    ("RX 9060 XT / 9060, RDNA4 (gfx1200)", "12.0.0"),
    ("RX 7900 XTX / XT / GRE, RDNA3 (gfx1100)", "11.0.0"),
    ("RX 7800 XT / 7700 XT, RDNA3 (gfx1101)", "11.0.1"),
    ("RX 7600 / 7600 XT, RDNA3 (gfx1102)", "11.0.2"),
    ("Radeon 780M / 760M iGPU (gfx1103, use gfx1100 override)", "11.0.0"),
    ("Strix Halo / Radeon 8060S (gfx1151)", "11.5.1"),
    ("Strix Point / Radeon 890M (gfx1150)", "11.5.0"),
    ("RX 6950 XT / 6900 XT / 6800 XT / 6800, RDNA2 (gfx1030)", "10.3.0"),
    ("RX 6750 XT / 6700 XT, RDNA2 (gfx1031, use gfx1030 override)", "10.3.0"),
    ("RX 6650 XT / 6600 XT / 6600, RDNA2 (gfx1032, use gfx1030 override)", "10.3.0"),
    ("RX 5700 XT / 5700, RDNA1 (gfx1010)", "10.1.0"),
    ("Radeon VII / MI50 (gfx906)", "9.0.6"),
    ("Vega 56 / 64 (gfx900)", "9.0.0"),
    ("Instinct MI100 (gfx908)", "9.0.8"),
    ("Instinct MI200 series (gfx90a)", "9.0.10"),
    ("Instinct MI300 series (gfx942)", "9.4.2"),
]


def gfx_name_to_version(name: str) -> str:
    """'gfx1201' -> '12.0.1', 'gfx90a' -> '9.0.10', 'gfx942' -> '9.4.2'."""
    m = re.fullmatch(r"gfx([0-9]+)([0-9a-f])([0-9a-f])", (name or "").strip().lower())
    if not m:
        return ""
    major, minor, step = m.groups()
    return f"{int(major)}.{int(minor, 16)}.{int(step, 16)}"


# other environment variables written into every launch script: (name, label, preset default hint)
ENV_EXTRA_AMD = [
    ("OMP_NUM_THREADS", "CPU threads (OMP_NUM_THREADS)", "12"),
    ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "AOTriton experimental (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL)", "1"),
    ("PYTORCH_ROCM_ALLOC_CONF", "ROCm allocator (PYTORCH_ROCM_ALLOC_CONF)", "max_split_size_mb:512,garbage_collection_threshold:0.8"),
]
ENV_EXTRA_NVIDIA = [
    ("CUDA_VISIBLE_DEVICES", "GPU to train on (CUDA_VISIBLE_DEVICES: 0 = the first; a laptop's NVIDIA card is usually 0)", "all"),
    ("OMP_NUM_THREADS", "CPU threads (OMP_NUM_THREADS)", "12"),
    ("PYTORCH_CUDA_ALLOC_CONF", "CUDA allocator (PYTORCH_CUDA_ALLOC_CONF)", "max_split_size_mb:512,garbage_collection_threshold:0.8"),
]
try:
    import features as _FT
    NVIDIA = _FT.gpu_kind() == "nvidia"
except Exception:                                        # noqa: BLE001
    NVIDIA = False
ENV_EXTRA = ENV_EXTRA_NVIDIA if NVIDIA else ENV_EXTRA_AMD
ENV_NAMES = ([] if NVIDIA else [GFX_VAR]) + [n for n, _, _ in ENV_EXTRA]
AMD_ONLY = ("HSA_OVERRIDE_GFX_VERSION", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")


def vendor_env(env: dict) -> dict:
    """The environment of a launch script for this computer's GPU. The presets are written for AMD (ROCm); on
    NVIDIA the ROCm-only values are left out and the ROCm allocator setting becomes the CUDA one."""
    if not NVIDIA:
        return dict(env)
    out = {}
    for k, v in env.items():
        if k in AMD_ONLY:
            continue
        out["PYTORCH_CUDA_ALLOC_CONF" if k == "PYTORCH_ROCM_ALLOC_CONF" else k] = v
    return out


def get_gfx() -> str:
    return (read_settings().get("general", {}).get(GFX_VAR) or "").strip()


def env_saved() -> dict:
    """{name: value} for every environment variable saved in settings.txt [general]."""
    g = read_settings().get("general", {})
    return {n: (g.get(n) or "").strip() for n in ENV_NAMES if (g.get(n) or "").strip()}


def apply_gfx_env():
    """Make this process (and everything it starts) use the saved value."""
    os.environ.update(env_saved())


def detect_gfx():
    """(version, message) from rocminfo, falling back to amd-smi or rocm-smi."""
    names = []
    exe = shutil.which("rocminfo") or next(iter(sorted(glob.glob("/opt/rocm*/bin/rocminfo"))), None)
    if exe:
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=15).stdout
            names = re.findall(r"\bgfx[0-9a-f]{3,4}\b", out)
        except Exception:
            pass
    if not names:
        try:
            import gpusmi
            names = gpusmi.gfx_names()                   # amd-smi (ROCm 10+) or rocm-smi
        except Exception:                                # noqa: BLE001
            pass
    names = [n for n in names if n != "gfx000"]
    if not names:
        return "", "Could not detect the GPU (rocminfo / amd-smi / rocm-smi gave no gfx name). Pick it from the list instead."
    n = names[0]
    v = gfx_name_to_version(n)
    return v, f"Detected **{n}**, which is `{v}`. If the card is not officially supported by your ROCm, pick the nearest supported entry from the list instead."


def gfx_load():
    if NVIDIA:
        saved = env_saved()
        return ["", None, "NVIDIA graphics card: no GFX value is needed (that is for AMD ROCm)."] + \
            [saved.get(n, "") for n, _, _ in ENV_EXTRA]
    v = get_gfx() or os.environ.get(GFX_VAR, "")
    label = next((lab for lab, val in KNOWN_GFX if val == v), None)
    saved = env_saved()
    note = (f"`{GFX_VAR}` is `{v}` ({'from settings.txt' if get_gfx() else 'from the environment, not saved in settings.txt'})."
            if v else f"No `{GFX_VAR}` saved yet.")
    return [v, (v if label else None), note] + [saved.get(n, "") for n, _, _ in ENV_EXTRA]


def env_save(gfx, *extra):
    vals = {} if NVIDIA else {GFX_VAR: (gfx or "").strip()}
    vals.update({n: (x or "").strip() for (n, _, _), x in zip(ENV_EXTRA, extra)})
    if vals.get(GFX_VAR) and not re.fullmatch(r"\d+\.\d+\.\d+", vals[GFX_VAR]):
        notify.warning("The GFX version must look like 12.0.1")
        return "The GFX version must look like `12.0.1` (major.minor.step).", st_raw()
    for n, v in vals.items():
        if "\n" in v or '"' in v:
            notify.warning(f"{n}: no line breaks or double quotes please.")
            return f"{n}: no line breaks or double quotes please.", st_raw()
    data = read_settings()
    sec = data.setdefault("general", {})
    for n, v in vals.items():
        if v:
            sec[n] = v
        else:
            sec.pop(n, None)
    try:
        write_settings(data)
    except OSError as e:
        return f"Could not write `{settings_path()}`: {e}", st_raw()
    for n, v in vals.items():
        if v:
            os.environ[n] = v
        else:
            os.environ.pop(n, None)
    notify.info(f"Saved {Path(settings_path()).name}")
    shown = "\n".join(f"- `{n} = {v}`" for n, v in vals.items() if v) or "- (all empty: the preset defaults are used)"
    return (f"Saved to `{settings_path()}` (`[general]`):\n\n{shown}\n\nThe GUI uses them now and every new launch script gets them "
            "(an empty field means the preset's own value is kept). Use **Update all launch scripts** for existing ones."), st_raw()


def env_update_scripts(confirm):
    saved = env_saved()
    if not saved:
        return "Save at least one value first."
    if not confirm:
        notify.warning("Tick the confirmation box first.")
        return "Tick the confirmation box first."
    changed = []
    for _, path in S.list_scripts():
        f = Path(path)
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        new = text
        for n, v in saved.items():
            if get_export(new, n) != v:
                new = set_export(new, n, v)
        if new != text:
            try:
                shutil.copy2(f, f.with_name(f.name + ".bak"))
                f.write_text(new, encoding="utf-8")
                changed.append(str(f))
            except OSError:
                pass
    return (f"Wrote {len(saved)} environment value(s) into {len(changed)} launch script(s) (a `.bak` copy of each was kept)."
            + ("\n\n" + "\n".join(f"- `{c}`" for c in changed[:20]) if changed else " All scripts already match."))


def set_arg(text: str, name: str, value: str):
    """Set --name to value in the script text. Returns (text, changed, ok)."""
    m = S._arg_re(name).search(text)
    if m:
        old = m.group(2)
        q = old[0] if old[0] in "\"'" else ""
        if (old[1:-1] if q else old) == value:
            return text, False, True
        tok = f"{q}{value}{q}" if q else (f'"{value}"' if re.search(r"\s", value) else value)
        return text[:m.start(2)] + tok + text[m.end(2):], True, True
    tok = f'"{value}"' if re.search(r"\s", value) else value
    new, ok = S.append_args(text, f"--{name} {tok}")
    return new, ok, ok


def cs_use_settings(ds, tkey, script, text, plan):
    """Write the model files saved in settings.txt for this LoRA type into the script text."""
    ov = settings_models(tkey)
    if not ov and not env_saved():
        notify.warning("No saved model files for this LoRA type: set them in Settings › Model files first.")
        return S._cs_result(status="No saved model files for this LoRA type: set them in the **Settings** tab first.", plan=plan)
    new, changed, failed = text or "", [], []
    for key, val in ov.items():
        new, ch, ok = set_arg(new, key, val)
        if ch:
            changed.append(key)
        if not ok:
            failed.append(key)
    for n, v in env_saved().items():
        if get_export(new, n) != v:
            new = set_export(new, n, v)
            changed.append(n)
    msg = (f"Settings from settings.txt applied to the script text (not saved yet): {', '.join(k if k.isupper() else '--' + k for k in changed)}."
           if changed else "The script already uses the model files from settings.txt.")
    if failed:
        msg += f" No training command found to add: {', '.join(failed)}."
    return S._cs_result(text=new, fields=S.read_form(new), explain=S.explain_for(ds, tkey, script, new, plan), status=msg, plan=plan)
