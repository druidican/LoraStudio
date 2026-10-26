# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings of the trainer side: the Hugging Face token, the library scan, repeat folders, model files, LoRA subtypes
and the GPU / GFX values.

Part of sdgui.py (split by topic): the names of sdgui are reached as S.<name>; sdgui imports
the names of this file back, so callers use sdgui.<name> as before."""
from __future__ import annotations
__version__ = "1.261026v"  # LoRA Studio version, see version.py

import glob
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
import gradio as gr

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


def header_status():
    return S.gpu_status_text(), hf_status_text()


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
        gr.Warning(f"No token found in {hf_path()}.")
        return S._cs_result(status=f"No token found in `{hf_path()}`.", plan=plan)
    if not re.fullmatch(r"[\w.\-]+", tok):
        gr.Warning("The token contains unusual characters; not written into the script.")
        return S._cs_result(status="The token contains unusual characters, so it was not written into the script.", plan=plan)
    new, changed = fill_hf(text or "", force=True)
    return S._cs_result(text=new, plan=plan,
                      status="`HF_TOKEN` set in the script text (not saved yet)." if changed
                      else "`HF_TOKEN` already matches huggingtoken.txt.")


# --------------------------------------------------------------------------- library scan
KNOWN_LIBS = S.CACHE_DIR / "known_libraries.json"


def scan_library(cur_cs, cur_rv):
    img = S.images_dir()
    if not img.is_dir():
        return gr.update(), gr.update(), f"`{img}` does not exist.", gr.update()
    try:
        known = set(json.loads(KNOWN_LIBS.read_text()))
    except Exception:
        known = None
    names = S.list_lora_dirs()
    rows, new = [], []
    for name in names:
        fr = S.repeat_folders(S.lora_path(name))
        imgs = capt = 0
        for f, _ in fr:
            for im in S.list_images(f):
                imgs += 1
                capt += bool(S.read_caption(im))
        reps = sorted({r for _, r in fr if r})
        loose = [f for f, r in fr if r is None]
        nscr = len(S.launch_scripts(name))
        notes = []
        if known is not None and name not in known:
            new.append(name)
            notes.append("**NEW**")
        if not imgs:
            notes.append("no images")
        elif not reps:
            notes.append("no `N_name` repeat folder: sd-scripts would ignore the images")
        elif loose:
            notes.append("some images are outside repeat folders")
        if imgs and capt < imgs:
            notes.append(f"{imgs - capt} without caption")
        if imgs and not nscr:
            notes.append("no launch script yet")
        rows.append(f"| {name} | {imgs} | {capt} | {', '.join(str(r) for r in reps) or '-'} | {nscr} | {'; '.join(notes) or 'ready'} |")
    gone = sorted(known - set(names)) if known else []
    try:
        S.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        KNOWN_LIBS.write_text(json.dumps(names))
    except OSError:
        pass
    if known is None:
        head = f"**First scan:** {len(names)} image libraries in `{img}`. They are remembered, so the next scan can point out new ones."
    elif new:
        head = f"**{len(new)} new image librar{'y' if len(new) == 1 else 'ies'} since the last scan:** {', '.join(new)}. ({len(names)} in total.)"
    else:
        head = f"No new image libraries since the last scan. ({len(names)} in total.)"
    if gone:
        head += f" No longer present: {', '.join(gone)}."
    table = ["", "| Library | Images | Captions | Repeats | Launch scripts | Notes |", "|---|---|---|---|---|---|"] + rows
    ds_cs = gr.update(choices=names, value=cur_cs if cur_cs in names else None)
    ds_rv = gr.update(choices=names, value=cur_rv if cur_rv in names else None)
    return ds_cs, ds_rv, head + "\n".join(table), gr.update(open=True)


def scan_short(cur):
    """Scan library: re-read every dataset folder (and Personal Data's scan), refresh the list, one status line."""
    _, _, md, _ = scan_library(cur, None)              # also remembers the names, to point out new ones
    names = S.list_lora_dirs()
    head = md.split("\n", 1)[0]
    LL = S._library()
    extra = ""
    if LL is not None:
        try:
            LL.LIB.scan()
            extra = f" Collection rescanned ({len(LL.LIB.loras)} LoRAs)."
        except Exception as e:              # noqa: BLE001
            extra = f" (Collection rescan failed: {e})"
    no_script = [n for n in names if not S.launch_scripts(n)]
    tail = f" {len(no_script)} without a launch script yet." if no_script else ""
    return gr.update(choices=names, value=cur if cur in names else None), head + tail + extra


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
        return [gr.update(choices=[], value=None), 0, "", ""]
    rows = folder_rows(ds)
    if not rows:
        return [gr.update(choices=[], value=None), 0, "", "No folders with images found in this dataset."]
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
    return [gr.update(choices=rp_choices(ds), value=str(pick)), repeats, rp_new_name(ds, str(pick), repeats), "\n".join(lines)]


def rp_name_update(ds, folder, repeats):
    return rp_new_name(ds, folder, repeats)


def rp_rename(ds, folder, newname, move_ok):
    """Rename a repeat folder to <N>_name, or move loose images into a new repeat folder. Returns (message, userep)."""
    keep = gr.update()
    if not ds or not folder:
        gr.Warning("Pick a folder first.")
        return "Pick a folder first.", keep
    lora, f, name = S.lora_path(ds), Path(folder), (newname or "").strip()
    if not re.fullmatch(r"\d+_[^/\\\x00]+", name):
        gr.Warning("The name must look like 4_name (a repeat count, an underscore, then a name).")
        return "The name must look like `4_name`.", keep
    if f == lora:                                   # loose images directly in the LoRA folder
        if not move_ok:
            gr.Warning("Tick the box to allow moving the images into a new folder.")
            return "Tick the box above to allow moving these images into a new folder.", keep
        dest = lora / name
        if dest.exists():
            gr.Warning(f"{name} already exists.")
            return f"`{name}` already exists.", keep
        dest.mkdir()
        moved = 0
        for im in S.list_images(lora):
            for src in [im] + [im.with_name(im.stem + e) for e in S.CAPTION_EXTS]:
                if src.is_file():
                    shutil.move(str(src), str(dest / src.name))
                    moved += 1
        gr.Info(f"Moved {moved} files into {name}/")
        return f"Moved {moved} files (images and captions) into `{name}/`.", gr.update(value=False)
    dest = f.with_name(name)
    if dest == f:
        return "The folder already has that name.", keep
    if dest.exists():
        gr.Warning(f"{name} already exists.")
        return f"`{name}` already exists next to it.", keep
    f.rename(dest)
    gr.Info(f"Renamed {f.name} -> {name}")
    return f"Renamed `{f.name}` to `{name}`. Epochs are re-planned for the folder's actual repeat count.", gr.update(value=False)


def rp_make(ds, tkey, text, plan):
    if not ds:
        gr.Warning("Pick a LoRA folder first.")
        return "Pick a LoRA folder first."
    notes = S.ensure_folders(ds, text or "", plan, tkey)
    gr.Info("Folders checked.")
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
        gr.Warning(f"Could not calculate epochs: {e}")
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


def st_load(tkey, recursive):
    if not tkey:
        return [gr.update()] * (3 * S.ST_N) + [[""] * S.ST_N, "Pick a LoRA type."]
    slots, saved = type_slots(tkey), read_settings().get(tkey, {})
    out, flags = [], []
    for i in range(S.ST_N):
        if i < len(slots):
            key, default = slots[i]
            path = saved.get(key) or default
            d = _slot_dir(path, saved.get(key + ".dir"))
            lab = slot_label(key)
            flags.append(key)
            out += [gr.update(visible=True), gr.update(label=f"{lab}: folder", value=d),
                    gr.update(label=f"{lab}: file (--{key})", choices=list_model_files(d, recursive), value=path or None)]
        else:
            flags.append("")
            out += [gr.update(visible=False), gr.update(value=""), gr.update(choices=[], value=None)]
    have = settings_path().is_file()
    info = (f"Settings for this type are read from `{settings_path()}`." if saved else
            (f"No saved settings for this type yet: showing the defaults from your preset. **Save** writes them to `{settings_path()}`."
             if have else f"`{settings_path()}` does not exist yet: showing the defaults from your preset. **Save** creates it."))
    return out + [flags, info]


def st_refresh(folder, recursive, current):
    return gr.update(choices=list_model_files(folder, recursive), value=current)


def st_refresh_all(recursive, *args):
    dirs, cur = args[:S.ST_N], args[S.ST_N:2 * S.ST_N]
    return [st_refresh(d, recursive, c) for d, c in zip(dirs, cur)]


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


def st_apply_suggestions(tkey, flags, picks, replace_fitting, recursive, *args):
    """Put the suggested files into the slots: empty, missing or unsuitable ones (all of them when ticked).
    The base model is only replaced by another file of the same model family when the chosen one is unusable."""
    files = list(args[S.ST_N:2 * S.ST_N])
    req = S.GUIDE.requirements_for(tkey, _type_base(tkey)) if S.GUIDE else {}
    out_d, out_f, changed = [], [], []
    for i in range(S.ST_N):
        flag = flags[i] if i < len(flags) else ""
        cur = os.path.expanduser((files[i] or "").strip())
        new = (picks or {}).get(flag)
        fits = bool(cur) and "/PATH/TO" not in cur and Path(cur).is_file() and S.GUIDE is not None \
            and not S.GUIDE.problems(S.GUIDE.identify(cur), flag, req.get(flag, []))
        if new and (replace_fitting or not fits) and new != cur:
            d = str(Path(new).parent)
            out_d.append(gr.update(value=d))
            out_f.append(gr.update(choices=list_model_files(d, recursive), value=new))
            changed.append(f"{slot_label(flag)}: {Path(new).name}")
        else:
            out_d.append(gr.update())
            out_f.append(gr.update())
    if changed:
        gr.Info("Filled " + "; ".join(changed) + ". Press Save to keep them.")
    else:
        gr.Info("Nothing to change: the slots already hold fitting files (tick 'Also replace files that already fit' "
                "to use the ★ ones).")
    return out_d + out_f


def st_online(tkey, base_file, progress=gr.Progress()):
    if S.GUIDE is None:
        return "model_guide.py is missing next to sdgui.py."
    base = (base_file or "").strip()
    if not base or "/PATH/TO" in base:
        return "Choose the base model file first."
    progress(0.2, desc="Asking Hugging Face and Civitai ...")
    try:
        return S.GUIDE.lookup_online(tkey, os.path.expanduser(base), read_hf_token() or "")
    except Exception as e:                          # noqa: BLE001
        return f"Online lookup failed: {e}"


# ---- Hugging Face downloads (hfdownload.py), on a click
def _hd():
    import hfdownload
    return hfdownload


def st_dl_visible(tkey, flags, *files):
    """A download button on every slot whose file is missing and that has a built-in file; 'Download all' when
    there are two or more; the slot list for the paste-a-link field."""
    HD = _hd()
    kn = HD.known(tkey) if tkey else {}
    btns, n = [], 0
    for i in range(S.ST_N):
        flag = flags[i] if i < len(flags) else ""
        f = files[i] if i < len(files) else ""
        vis = bool(flag) and flag in kn and HD.is_missing(f or "")
        n += vis
        btns.append(gr.update(visible=vis))
    slots = [(slot_label(f), f) for f in flags if f]
    return btns + [gr.update(visible=n >= 2), gr.update(choices=slots, value=slots[0][1] if slots else None)]


def _slot_after(path, recursive=True):
    """(folder update, file update) for a slot after a download."""
    if path is None:
        return gr.update(), gr.update()
    d = str(path.parent.parent if _hd().GENERIC.match(path.name) else path.parent)
    return gr.update(value=d), gr.update(choices=list_model_files(d, recursive), value=str(path))


def st_download(tkey, flags, i, d, f):
    flag = flags[i] if i < len(flags) else ""
    if not tkey or not flag:
        yield "Pick a LoRA type first.", gr.update(), gr.update()
        return
    gen, res = _hd().fetch(tkey, flag, slot_dir=d or ""), None
    try:
        while True:
            yield next(gen), gr.update(), gr.update()
    except StopIteration as e:
        res = e.value
    yield (gr.update(), *_slot_after(res))


def _all_slots(flags, results, recursive, n):
    out_d, out_f = [], []
    for i in range(n):
        flag = flags[i] if i < len(flags) else ""
        a, b = _slot_after(results.get(flag), recursive) if flag in results else (gr.update(), gr.update())
        out_d.append(a)
        out_f.append(b)
    return out_d + out_f


def st_download_all(tkey, flags, recursive, *args):
    dirs = list(args[:S.ST_N])
    HD = _hd()
    todo = HD.missing(tkey)
    keep = [gr.update()] * (2 * S.ST_N)
    if not todo:
        yield ["✔ Every model file of this LoRA type is on this computer."] + keep
        return
    results, log = {}, []
    for flag in todo:
        i = flags.index(flag) if flag in flags else -1
        gen, last = HD.fetch(tkey, flag, slot_dir=dirs[i] if i >= 0 else ""), ""
        try:
            while True:
                last = next(gen)
                yield ["\n\n".join(log + [last])] + keep
        except StopIteration as e:
            if e.value is not None:
                results[flag] = e.value
        log.append(last)
        if HD.STOP.is_set():
            break
    yield ["\n\n".join(log + [f"**{len(results)} of {len(todo)} files downloaded.**"])] + _all_slots(
        flags, results, recursive, S.ST_N)


def st_download_link(tkey, flags, slot, link, recursive, *args):
    dirs = list(args[:S.ST_N])
    keep = [gr.update()] * (2 * S.ST_N)
    if not tkey or not slot:
        yield ["Pick a LoRA type and the slot first."] + keep
        return
    if not (link or "").strip():
        yield ["Paste the link of the file on huggingface.co first."] + keep
        return
    i = flags.index(slot) if slot in flags else -1
    gen, res = _hd().fetch(tkey, slot, slot_dir=dirs[i] if i >= 0 else "", link=link), None
    try:
        while True:
            yield [next(gen)] + keep
    except StopIteration as e:
        res = e.value
    yield [gr.update()] + _all_slots(flags, {slot: res} if res else {}, recursive, S.ST_N)


def st_download_stop():
    _hd().STOP.set()
    gr.Info("Stopping the download: it goes on from there next time.")


def st_raw() -> str:
    try:
        return settings_path().read_text(encoding="utf-8")
    except OSError:
        return "(settings.txt does not exist yet)"


def st_save(tkey, flags, *args):
    dirs, files = args[:S.ST_N], args[S.ST_N:2 * S.ST_N]
    if not tkey:
        gr.Warning("Pick a LoRA type first.")
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
            gr.Warning(f"{key}: paths cannot contain line breaks.")
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
        gr.Warning(f"Could not write settings.txt: {e}")
        return f"Could not write `{settings_path()}`: {e}", st_raw()
    gr.Info(f"Saved {Path(settings_path()).name}")
    return (f"**Saved to `{settings_path()}`** for this LoRA type (other types are untouched; the previous file is kept as "
            "`settings.txt.bak`). New launch scripts of this type now use these files. For an existing script press "
            "**Use model files from Settings** on Training › Launch script.\n\n" + "\n".join(rows)), st_raw()


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


def _sub_outs(msg, st_val=None, cs_val=None):
    ch = S.type_choices()
    return (msg, gr.update(choices=ch, value=st_val) if st_val else gr.update(choices=ch),
            gr.update(choices=ch, value=cs_val) if cs_val else gr.update(choices=ch),
            gr.update(choices=sub_choices(), value=None), sub_list_md(), st_raw())


def sub_add(name, base, tag):
    name = (name or "").strip()
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    if not key:
        gr.Warning("Give the subtype a name, for example Pony.")
        return _sub_outs("Give the subtype a name, for example **Pony**.")
    _, builtin = S.builtin_presets()
    if key in S.RESERVED_TYPES or key in builtin:
        gr.Warning(f"'{name}' is already a built-in type.")
        return _sub_outs(f"`{key}` is already a built-in type, so it cannot be added again. Pick it in the LoRA type list below.")
    if key in S.custom_subtypes():
        gr.Warning(f"The subtype '{name}' already exists.")
        return _sub_outs(f"The subtype `{key}` already exists. Pick it in the LoRA type list below.", st_val=key)
    if base not in S.SUBTYPE_BASES:
        gr.Warning("Pick SDXL or Anima as the base.")
        return _sub_outs("Pick SDXL or Anima as the base.")
    tag = re.sub(r"[^A-Za-z0-9]", "", tag or "") or re.sub(r"[^A-Za-z0-9]", "", name).upper()
    data = read_settings()
    data[key] = {"base": base, "label": name, "tag": tag}
    SECTION_NAMES[key] = key
    try:
        write_settings(data)
    except OSError as e:
        return _sub_outs(f"Could not write `{settings_path()}`: {e}")
    gr.Info(f"Subtype {name} added")
    return _sub_outs(f"Added **{name}** (based on {S.SUBTYPE_BASES[base]}, file tag `{tag}`). It is now in the LoRA type lists. "
                     f"Choose its model files below and press **Save to settings.txt**.", st_val=key)


def sub_remove(key, confirm, st_cur, cs_cur):
    if not key:
        gr.Warning("Pick a subtype to remove.")
        return _sub_outs("Pick a subtype to remove.")
    if not confirm:
        gr.Warning("Tick the confirmation box first.")
        return _sub_outs("Tick the confirmation box first.")
    data = read_settings()
    if key not in S.custom_subtypes() or key not in data:
        return _sub_outs("That subtype does not exist any more.")
    del data[key]
    try:
        write_settings(data)
    except OSError as e:
        return _sub_outs(f"Could not write `{settings_path()}`: {e}")
    gr.Info("Subtype removed")
    return _sub_outs(f"Removed `{key}` from `{settings_path()}` (a `.bak` copy of the previous file was kept). Launch scripts that were "
                     "made with it are not changed.",
                     st_val="sdxl" if st_cur == key else None, cs_val="sdxl" if cs_cur == key else None)


def sub_reload():
    return _sub_outs("Subtypes re-read from settings.txt.")


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
    """(version, message) from rocminfo, falling back to rocm-smi."""
    names = []
    exe = shutil.which("rocminfo") or next(iter(sorted(glob.glob("/opt/rocm*/bin/rocminfo"))), None)
    if exe:
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=15).stdout
            names = re.findall(r"\bgfx[0-9a-f]{3,4}\b", out)
        except Exception:
            pass
    if not names:
        exe = shutil.which("rocm-smi") or next(iter(sorted(glob.glob("/opt/rocm*/bin/rocm-smi"))), None)
        if exe:
            try:
                out = subprocess.run([exe, "--showproductname"], capture_output=True, text=True, timeout=10).stdout
                names = re.findall(r"\bgfx[0-9a-f]{3,4}\b", out)
            except Exception:
                pass
    names = [n for n in names if n != "gfx000"]
    if not names:
        return gr.update(), "Could not detect the GPU (rocminfo / rocm-smi gave no gfx name). Pick it from the list instead."
    n = names[0]
    v = gfx_name_to_version(n)
    return v, f"Detected **{n}**, which is `{v}`. If the card is not officially supported by your ROCm, pick the nearest supported entry from the list instead."


def gfx_pick(choice):
    return choice or gr.update()


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
        gr.Warning("The GFX version must look like 12.0.1")
        return "The GFX version must look like `12.0.1` (major.minor.step).", st_raw()
    for n, v in vals.items():
        if "\n" in v or '"' in v:
            gr.Warning(f"{n}: no line breaks or double quotes please.")
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
    gr.Info(f"Saved {Path(settings_path()).name}")
    shown = "\n".join(f"- `{n} = {v}`" for n, v in vals.items() if v) or "- (all empty: the preset defaults are used)"
    return (f"Saved to `{settings_path()}` (`[general]`):\n\n{shown}\n\nThe GUI uses them now and every new launch script gets them "
            "(an empty field means the preset's own value is kept). Use **Update all launch scripts** for existing ones."), st_raw()


def env_update_scripts(confirm):
    saved = env_saved()
    if not saved:
        return "Save at least one value first."
    if not confirm:
        gr.Warning("Tick the confirmation box first.")
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
        gr.Warning("No saved model files for this LoRA type: set them in Settings › Model files first.")
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
