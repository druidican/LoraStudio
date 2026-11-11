# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Launch scripts: LoRA types, the recommended plan, the explanation, the report and the handlers of the launch script
page (Datasets › 9).

Part of sdgui.py (split by topic): the names of sdgui are reached as S.<name>; sdgui imports
the names of this file back, so callers use sdgui.<name> as before."""
from __future__ import annotations
__version__ = "1.111126e"  # LoRA Studio version, see version.py

import copy
import functools
import difflib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import gradio as gr

import sdgui as S


# --------------------------------------------------------------------------- LoRA types (architectures)
VIEWS_OK = (20, 100)       # rough "views per image" window that works for character LoRAs
_BIG_ENV = {"PYTORCH_ROCM_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8"}
_PH = "/PATH/TO/"          # placeholder for model files this tool cannot know


# All LoRA types (SDXL, Illustrious, Anima, SD 1.5, SD3, FLUX, Lumina, HunyuanImage, ...) live in Lora_Datatypes.py.
EXTRA_PRESETS: dict = {}   # kept empty for compatibility; add new types to Lora_Datatypes.py instead
PRESETS_ERROR = {"msg": ""}  # why Lora_Datatypes.py could not be loaded (shown on the Create Data Launch Script tab)


def builtin_presets():
    """(module or None, {key: preset}): every LoRA type defined in Lora_Datatypes.py."""
    try:
        sc = S.load_scan_module()
        PRESETS_ERROR["msg"] = ""
    except Exception as e:
        PRESETS_ERROR["msg"] = f"{type(e).__name__}: {e}"
        return None, dict(EXTRA_PRESETS)
    merged = {k: dict(v) for k, v in sc.PRESETS.items()}   # a copy of each preset: callers set keys on theirs
    for k, v in EXTRA_PRESETS.items():
        merged.setdefault(k, v)
    return sc, merged


SUBTYPE_BASES = {"sdxl": "SDXL", "anima": "Anima"}
RESERVED_TYPES = {"general", "sdxl", "anima", "illustrious"}   # never treated as a user subtype, even if Lora_Datatypes.py is missing
META_KEYS = ("base", "label", "tag")        # keys of a subtype section in settings.txt that are not model files


def custom_subtypes() -> dict:
    """{key: {base, label, tag, explicit}} for every settings.txt section that is not a built-in type.
    A subtype trains like its base type (SDXL or Anima) but has its own model files and file tag."""
    _, builtin = builtin_presets()
    out = {}
    for sec, kv in S.read_settings().items():
        if sec in RESERVED_TYPES or sec in builtin:
            continue
        base = (kv.get("base") or "").strip().lower()
        explicit = base in SUBTYPE_BASES
        if not explicit:
            base = "anima" if kv.get("qwen3") else "sdxl"          # hand-written section without `base =`
        label = (kv.get("label") or S.SECTION_NAMES.get(sec, sec)).strip()
        tag = re.sub(r"[^A-Za-z0-9]", "", kv.get("tag") or label) or sec.upper()
        if not kv.get("tag"):
            tag = tag.upper()
        out[sec] = {"base": base, "label": label, "tag": tag, "explicit": explicit}
    return out


# Quality words of a model family: put in front of (Pony) or behind (Illustrious) every caption while training (with
# sd-scripts' --caption_prefix / --caption_suffix, so the caption files stay as they are) and every sample prompt.
QUALITY_PREFIX = {"pony": "score_9, score_8_up, score_7_up, source_anime"}
QUALITY_SUFFIX = {"illustrious": "masterpiece, best quality"}


def quality_family(key: str, preset: dict) -> str:
    """'pony', 'illustrious' or '' for a LoRA type (a subtype counts by its name: Pony, PonyXL, …)."""
    words = f"{key} {preset.get('label', '')} {preset.get('suffix', '')}".lower()
    if "pony" in words:
        return "pony"
    if key == "illustrious" or "illustrious" in words:
        return "illustrious"
    return ""


def add_quality(key: str, p: dict) -> dict:
    fam = quality_family(key, p)
    pre, suf = QUALITY_PREFIX.get(fam, ""), QUALITY_SUFFIX.get(fam, "")
    if not (pre or suf) or p.get("quality_words"):
        return p
    flags = list(p.get("flags") or [])
    if pre:
        n = len([w for w in pre.split(",") if w.strip()])
        # the quality words come first, so --keep_tokens must also keep them before the trigger word
        flags = [re.sub(r"--keep_tokens\s+(\d+)", lambda m: f"--keep_tokens {int(m.group(1)) + n}", f) for f in flags]
        flags.append(f'--caption_prefix "{pre},"')
    if suf:
        flags.append(f'--caption_suffix ", {suf}"')
    p.update(flags=flags, quality_words=(pre, suf))
    return p


def merged_presets():
    """(scan module or None, {key: preset}): built-in types plus the subtypes defined in settings.txt."""
    sc, merged = builtin_presets()
    for key, c in custom_subtypes().items():
        base = merged.get(c["base"])
        if not base:
            continue
        p = copy.deepcopy(base)
        p.update(label=f"{c['label']} ({SUBTYPE_BASES[c['base']]})", suffix=c["tag"], base=c["base"], custom=True)
        p.pop("template", None)
        merged[key] = p
    for key in list(merged):
        merged[key] = add_quality(key, copy.deepcopy(merged[key]))
    return sc, merged


def type_choices():
    _, presets = merged_presets()
    return [(v["label"] + ("  [template]" if v.get("template") else ""), k) for k, v in presets.items()]


# --------------------------------------------------------------------------- create launch script tab
NEW = "__new__"


FORM_GROUPS = [
    ("Training", [
        ("max_train_epochs", "Epochs"), ("train_batch_size", "Batch size"),
        ("gradient_accumulation_steps", "Grad accum"), ("learning_rate", "Learning rate"),
        ("optimizer_type", "Optimizer"), ("lr_scheduler", "LR scheduler"),
        ("lr_warmup_steps", "Warmup steps"), ("network_dim", "Network dim"),
        ("network_alpha", "Network alpha"), ("resolution", "Resolution"),
        ("seed", "Seed"), ("mixed_precision", "Mixed precision")]),
    ("Saving and samples", [
        ("output_name", "Output name"), ("save_every_n_epochs", "Save every N epochs"),
        ("sample_every_n_epochs", "Sample every N epochs"), ("max_data_loader_n_workers", "Data loader workers"),
        ("logging_dir", "Logging dir (TensorBoard)")]),
    ("Advanced", [
        ("network_module", "Network module"), ("attn_mode", "Attn mode (Anima)"),
        ("noise_offset", "Noise offset"), ("min_snr_gamma", "Min SNR gamma"),
        ("text_encoder_lr", "Text encoder LR"), ("max_grad_norm", "Max grad norm"), ("keep_tokens", "Keep tokens"), ("weighting_scheme", "Weighting scheme"),
        ("model_version", "Model version (FLUX.2 / Qwen)"),
        ("clip_skip", "Clip skip"), ("timestep_sampling", "Timestep sampling"),
        ("discrete_flow_shift", "Discrete flow shift"), ("blocks_to_swap", "Blocks to swap")]),
    ("Model files", [
        ("pretrained_model_name_or_path", "Base model"), ("vae", "VAE"), ("clip_l", "CLIP-L"),
        ("clip_g", "CLIP-G"), ("t5xxl", "T5-XXL"), ("ae", "AE (Flux/Lumina)"), ("gemma2", "Gemma2"),
        ("qwen3", "Qwen3 (Anima)"), ("text_encoder", "Text encoder"), ("byt5", "ByT5"),
        ("dit", "DiT (Z-Image, musubi)"), ("base_weights", "Base weights / adapter (Z-Image)")]),
]
FORM_FIELDS = [f for _, fs in FORM_GROUPS for f in fs]
# shown in the settings table of Datasets › 9 (LoRA Studio), so not again under More settings
STUDIO_TABLE = {"max_train_epochs", "train_batch_size", "learning_rate", "network_dim", "network_alpha", "resolution",
                "save_every_n_epochs"}
MODEL_FIELDS = {name for name, _ in FORM_GROUPS[3][1]}


@functools.lru_cache(maxsize=None)
def _arg_re(name: str):
    return re.compile(rf"((?<![\w-])--{name}(?:=|[ \t]+))(\"[^\"]*\"|'[^']*'|[^\s\\]+)")


# A musubi-tuner script (Z-Image) keeps these in shell variables at the top, because musubi has no such command-line
# options (batch size and resolution go into dataset.toml) or because several stages share them (model files, output name).
MUSUBI_VARS = {"dit": "DIT", "vae": "VAE", "text_encoder": "TEXT_ENCODER", "base_weights": "BASE_WEIGHTS",
               "train_batch_size": "BATCH_SIZE", "resolution": "RESOLUTION", "output_name": "OUTPUT_NAME",
               "model_version": "MODEL_VERSION"}


@functools.lru_cache(maxsize=None)
def _var_re(var: str):
    return re.compile(rf"^({var}=)(\"[^\"]*\"|'[^']*'|\S*)", re.M)


def read_form(text: str):
    vals = []
    musubi = S.is_musubi_text(text)
    for name, _ in FORM_FIELDS:
        m = (_var_re(MUSUBI_VARS[name]).search(text) if musubi and name in MUSUBI_VARS else None) or _arg_re(name).search(text)
        v = m.group(2) if m else ""
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        vals.append(v)
    return vals


def form_value(text: str, name: str) -> str:
    """One field of read_form (much cheaper than reading all of them)."""
    musubi = S.is_musubi_text(text)
    m = (_var_re(MUSUBI_VARS[name]).search(text) if musubi and name in MUSUBI_VARS else None) or _arg_re(name).search(text)
    v = m.group(2) if m else ""
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" else v


def form_dict(text: str) -> dict:
    return {name: v for (name, _), v in zip(FORM_FIELDS, read_form(text))}


def apply_form(text: str, vals):
    """Write non-empty form values into the script text. Existing args are changed in place; missing ones
    are appended to the training command. An empty field leaves the script alone."""
    changes, failed = [], []
    for (name, label), v in zip(FORM_FIELDS, vals):
        v = (v or "").strip()
        if not v:
            continue
        if S.is_musubi_text(text) and name in MUSUBI_VARS:
            mv = _var_re(MUSUBI_VARS[name]).search(text)
            if mv:
                old = mv.group(2)
                oldv = old[1:-1] if old[:1] in "\"'" and len(old) >= 2 else old
                if oldv != v:
                    text = text[:mv.start(2)] + '"' + v.replace('"', "") + '"' + text[mv.end(2):]
                    changes.append(f"{label}: {oldv} -> {v}")
                continue
        m = _arg_re(name).search(text)
        if m:
            old = m.group(2)
            q = old[0] if old[0] in "\"'" else ""
            oldv = old[1:-1] if q else old
            if oldv == v:
                continue
            tok = f"{q}{v}{q}" if q else (f'"{v}"' if re.search(r"\s", v) else v)
            text = text[:m.start(2)] + tok + text[m.end(2):]
            changes.append(f"{label}: {oldv} -> {v}")
        else:
            tok = f'"{v}"' if re.search(r"\s", v) else v
            text, ok = S.append_args(text, f"--{name} {tok}")
            if ok:
                changes.append(f"{label}: added {v}")
            else:
                failed.append(label)
    fixed, notes = S.fix_flags(text)
    if notes:
        text = fixed
        changes += [n[:1].upper() + n[1:] for n in notes]
    return text, changes, failed


def _num(v):
    try:
        return int(v) if v and int(v) > 0 else None
    except (TypeError, ValueError):
        return None


def training_command(text: str):
    """(trainer script name, text after it) for the first training command, or (None, "")."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") or not S.TRAIN_RE.search(line):
            continue
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
        cmd = "\n".join(lines[i:j + 1])
        m = S.TRAIN_RE.search(cmd)
        return m.group(0), cmd[m.end():]
    return None, ""


def infer_type(text, presets, fallback):
    script, _ = training_command(text)
    if not script:
        return fallback
    cands = [k for k, v in presets.items() if v.get("script") == script]
    if not cands:
        return fallback
    if len(cands) == 1:
        return cands[0]
    # a type can name a pattern that only its scripts contain (Chroma: --model_type chroma, same trainer as FLUX.1)
    detected = [k for k in cands if presets[k].get("detect") and re.search(presets[k]["detect"], text)]
    if detected:
        return detected[0]
    cands = [k for k in cands if not presets[k].get("detect")] or cands
    # the output name carries the type's suffix (e.g. mycharPONYV1), so look there first, then in the script head
    out_name = form_value(text, "output_name")
    for hay in (out_name, out_name + " " + text[:600]):
        for k in sorted(cands, key=lambda k: -len(presets[k].get("suffix", ""))):
            suf = presets[k].get("suffix", "")
            if suf and suf in hay:
                return k
    return cands[0]


def prompts_path_from_text(text: str, lora: Path):
    m = re.search(r"^PROMPTS_FILE=(?:\"([^\"]*)\"|'([^']*)'|(\S+))", text, re.M)
    if not m:
        return None
    raw = next(g for g in m.groups() if g is not None)
    raw = raw.replace("${SCRIPT_DIR}", str(lora)).replace("$SCRIPT_DIR", str(lora)).replace("${HOME}", str(S.HOME))
    return Path(os.path.expanduser(raw))


OLD_RUNS = "_old_runs"         # older launch scripts of a dataset, moved here when a newer version is saved
SCRIPT_DIR_LINE = 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
SCRIPT_DIR_OLD = 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # moved to _old_runs/: the dataset is one up'


def script_home(p) -> Path:
    """The dataset folder of a launch script (one up for a script in _old_runs/)."""
    p = Path(p)
    return p.parent.parent if p.parent.name == OLD_RUNS else p.parent


def find_script(path) -> str:
    """A launch script by its old path, also when it has been moved to _old_runs/ since."""
    p = Path(path or "")
    if not path or p.is_file():
        return str(path or "")
    moved = p.parent / OLD_RUNS / p.name
    if moved.is_file():
        return str(moved)
    try:
        ren = json.loads((p.parent / ".lsflow" / "script_renames.json").read_text(encoding="utf-8"))
        new = p.parent / ren.get(p.name, "")
        if ren.get(p.name) and new.is_file():
            return str(new)
        if ren.get(p.name):
            return find_script(new)
    except Exception:                       # noqa: BLE001
        pass
    return str(path)


def name_by_version(p: Path) -> Path:
    """Rename a launch script to its output name (bobSDXLV2.sh), unless it is queued or running. Old names are
    remembered in .lsflow/script_renames.json so the run history still finds it."""
    p = Path(p)
    try:
        _, out_name = S.script_args(p.read_text(encoding="utf-8", errors="ignore"), script_home(p))
    except OSError:
        return p
    if not out_name or not _ver_of(out_name)[1] or p.stem == out_name or str(p.resolve()) in _busy_scripts():
        return p
    dest = p.with_name(f"{out_name}.sh")
    if dest.exists():
        return p
    p.rename(dest)
    f = p.parent / ".lsflow" / "script_renames.json"
    try:
        ren = json.loads(f.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001
        ren = {}
    ren[p.name] = dest.name
    try:
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(ren, indent=1), encoding="utf-8")
    except OSError:
        pass
    return dest


def _ver_of(name: str):
    m = re.search(r"^(.*?)V(\d{1,3})$", name or "")
    return (m.group(1), int(m.group(2))) if m else (name or "", 0)


def next_script_version(lora: Path, text: str, script_dir=None) -> tuple[str, int]:
    """(base, n): the next free version for the output name in `text`, counting the model folder and every launch
    script of the dataset (also the ones in _old_runs/)."""
    out_dir, out_name = S.script_args(text, script_dir or lora)
    base, _ = _ver_of(out_name or "")
    if not base:
        return "", 0
    top = 0
    md = Path(out_dir) if out_dir else lora / "model"
    if md.is_dir():
        pat = re.compile(r"^" + re.escape(base) + r"V(\d+)")
        for f in md.iterdir():
            m = pat.match(f.name)
            if m:
                top = max(top, int(m.group(1)))
    for f in list(lora.glob("*.sh")) + list((lora / OLD_RUNS).glob("*.sh")):
        try:
            b, n = _ver_of(S.script_args(f.read_text(encoding="utf-8", errors="ignore"), lora)[1] or "")
        except OSError:
            continue
        if b == base:
            top = max(top, n)
    return base, top + 1


def set_script_version(text: str, base: str, n: int) -> str:
    """Make the output name in a launch script end in V<n> (and the '# Output' comment line)."""
    def fix(m):
        val = m.group(2)
        val = re.sub(r"V\d{1,3}$", "", val) + f"V{n}"
        return m.group(1) + val + m.group(3)
    text = re.sub(r'(--output_name[ =]+")([^"]*)(")', fix, text, count=1)
    text = re.sub(r"(--output_name[ =]+')([^']*)(')", fix, text, count=1)
    text = re.sub(r"^(# Output )" + re.escape(base) + r"V\d+", lambda m: f"{m.group(1)}{base}V{n}", text, count=1,
                  flags=re.M)
    return text


def _busy_scripts() -> set:
    busy = set()
    try:
        busy |= {str(Path(i["path"]).resolve()) for i in S.QUEUE["items"] if i.get("state") in ("queued", "running")}
    except Exception:                       # noqa: BLE001
        pass
    try:
        if S.RUN.running() and S.RUN.script:
            busy.add(str(Path(S.RUN.script).resolve()))
    except Exception:                       # noqa: BLE001
        pass
    return busy


def archive_old_scripts(lora: Path, keep: Path | None = None) -> list[str]:
    """Keep one launch script per LoRA type in the dataset folder: the newest version (or `keep`). The others move to
    _old_runs/ (still runnable: their SCRIPT_DIR points one folder up). Scripts in the queue or running stay."""
    lora = Path(lora)
    groups: dict = {}
    for p in lora.glob("*.sh"):
        if p.name.startswith(".gui_run_"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        groups.setdefault(S.script_type(str(p), text), []).append((p, text))
    busy = _busy_scripts()
    moved = []
    for tk, items in groups.items():
        if not tk or len(items) < 2:                   # an unknown type is never moved
            continue
        if keep is not None and any(p.resolve() == Path(keep).resolve() for p, _ in items):
            best = Path(keep).resolve()
        else:
            best = max(items, key=lambda it: (_ver_of(S.script_args(it[1], lora)[1] or "")[1], it[0].stat().st_mtime))[0].resolve()
        for p, text in items:
            if p.resolve() == best or str(p.resolve()) in busy:
                continue
            d = lora / OLD_RUNS
            d.mkdir(exist_ok=True)
            dest = d / p.name
            n = 2
            while dest.exists():
                dest = d / f"{p.stem}_{n}.sh"
                n += 1
            new = text.replace(SCRIPT_DIR_LINE, SCRIPT_DIR_OLD, 1)
            dest.write_text(new, encoding="utf-8")
            dest.chmod(p.stat().st_mode)
            shutil.copystat(p, dest)
            p.unlink()
            moved.append(p.name)
    return moved


def archive_all_datasets() -> int:
    """Once at start: tidy every dataset folder that has launch scripts (one per LoRA type)."""
    n = 0
    try:
        folders = {Path(p).parent for _, p in S.list_scripts()}
        for d in folders:
            if d == S.ROOT:
                continue
            if len(list(d.glob("*.sh"))) > 1:
                n += len(archive_old_scripts(d))
            for p in d.glob("*.sh"):
                if not p.name.startswith(".gui_run_"):
                    name_by_version(p)
    except Exception:                       # noqa: BLE001
        pass
    return n


def launch_scripts(ds):
    d = S.lora_path(ds) if ds else None
    if not d or not d.is_dir():
        return []
    return sorted((p for p in d.glob("*.sh") if not p.name.startswith(".gui_run_")), key=lambda p: p.name.lower())


def cs_script_choices(ds):
    return [("+ New launch script", NEW)] + [(p.name, str(p)) for p in launch_scripts(ds)]


def unique_script_name(lora: Path, name: str) -> str:
    if not (lora / name).exists():
        return name
    stem, n = name[:-3], 2
    while (lora / f"{stem}_{n}.sh").exists():
        n += 1
    return f"{stem}_{n}.sh"


# ---- dataset numbers
EXTRA_DIR = "_extra"          # other repeat folders are moved here when a launch script is saved (not trained)


def is_main_folder(folder: Path, lora: Path) -> bool:
    """The dataset's own repeat folder: <N>_<dataset name> directly in the dataset folder."""
    m = S.REPEAT_RE.match(folder.name)
    return bool(m) and folder.parent == lora and folder.name[m.end():].lower() == lora.name.lower()


def main_repeat_folders(lora: Path, found=None) -> tuple[list, list]:
    """([(folder, repeats)] that are the dataset, [(folder, repeats)] other repeat folders). Only <N>_<dataset name>
    counts; a dataset without such a folder (an older or renamed one) uses its repeat folders directly in the
    dataset folder, as sd-scripts does."""
    reps = [(f, r) for f, r in (found if found is not None else S.repeat_folders(lora)) if r is not None]
    main = [(f, r) for f, r in reps if is_main_folder(f, lora)]
    if not main:
        main = [(f, r) for f, r in reps if f.parent == lora]
    return main, [x for x in reps if x not in main]


def dataset_stats(ds):
    """The training pictures of a dataset: its own repeat folder <N>_<dataset name> only. Pictures anywhere else
    (_reserve/, other repeat folders, other folders, the dataset folder itself) are not part of it: they are listed
    in 'loose' / 'others' and only used when the usable scan is told to look at them (Datasets › 2)."""
    lora = S.lora_path(ds)
    rows, loose = [], []
    found = S.repeat_folders(lora)
    main, others = main_repeat_folders(lora, found)
    for f, r in main:
        imgs = S.list_images(f)
        rows.append(dict(folder=f, repeats=r, n=len(imgs), captioned=sum(1 for im in imgs if S.read_caption(im))))
    for f, r in found:
        if r is None:
            loose.append((f, len(S.list_images(f))))
    return dict(lora=lora, rows=rows, loose=loose, others=[(f, len(S.list_images(f))) for f, _ in others],
                total=sum(r["n"] for r in rows), captioned=sum(r["captioned"] for r in rows))


def with_virtual_row(stats):
    """A dataset whose images sit directly in the LoRA folder has no <N>_name folder yet. Plan it as if
    the folder existed: one virtual row, which Save creates (and moves the images into)."""
    root_imgs = S.list_images(stats["lora"])
    if stats["rows"] or not root_imgs:
        return stats
    caps = sum(1 for im in root_imgs if S.read_caption(im))
    row = dict(folder=stats["lora"], repeats=1, n=len(root_imgs), captioned=caps, virtual=True)
    return {**stats, "rows": [row], "loose": [(f, n) for f, n in stats["loose"] if f != stats["lora"]],
            "total": len(root_imgs), "captioned": caps}


def recommended_repeats(ds, tkey, steps_over=0):
    stats = with_virtual_row(dataset_stats(ds))
    n = stats["total"]
    try:
        sc, presets = merged_presets()
        if not n or sc is None or tkey not in presets:
            return 10
        cfg = dict(presets[tkey])
        if _num(steps_over):
            cfg["steps"] = _num(steps_over)
        bs, ga, _ = sc.suggest_settings(n, cfg)
        return sc.suggest_repeats(n, bs, ga, cfg["epochs"], cfg["steps"])
    except Exception:
        return 10


def build_prompts(sc, stats, cfg):
    """(path, text, is_new): the type's sample prompts file for this dataset (existing file, else generated)."""
    lora = stats["lora"]
    path = lora / cfg["prompts_name"]
    if path.is_file():
        return path, path.read_text(encoding="utf-8", errors="ignore"), False
    caps = sc.read_captions([(r["folder"], r["repeats"]) for r in stats["rows"]]) if stats["rows"] else []
    return path, sc.build_prompts_text(caps, lora.name.lower(), lora.name, cfg)[0], True


def _speed(tkey, cfg) -> dict:
    try:
        import speedjobs
        return speedjobs.apply(tkey, cfg)
    except Exception:                                  # noqa: BLE001
        return cfg


def caption_ext_for(ds, tkey) -> str:
    """The caption set a new launch script of this type trains on (Datasets: step 3 and 9), else IMG.txt."""
    try:
        import dsflow
        return dsflow.caption_ext(ds, tkey) if ds else ".txt"
    except Exception:                                  # noqa: BLE001
        return ".txt"


_TOML_EXT = re.compile(r'(caption_extension = \\")([^"\\]*)(\\")')


_CAP_VAR = re.compile(r'^(CAPTION_EXT=")([^"]*)(")', re.M)


def set_caption_ext(text: str, ext: str) -> str:
    """Point a launch script at another caption set: --caption_extension, or the CAPTION_EXT of a musubi script (older
    musubi scripts: the dataset.toml line)."""
    if _CAP_VAR.search(text or ""):
        return _CAP_VAR.sub(lambda m: m.group(1) + ext + m.group(3), text)
    if _TOML_EXT.search(text or ""):
        return _TOML_EXT.sub(lambda m: m.group(1) + ext + m.group(3), text)
    m = _arg_re("caption_extension").search(text or "")
    if m:
        return text[:m.start(2)] + f'"{ext}"' + text[m.end(2):]
    return S.append_args(text or "", f'--caption_extension "{ext}"')[0]


def has_samples(text) -> bool:
    return re.search(r"(?<![\w-])--sample_prompts(?![\w-])", text or "") is not None


def add_logging(text: str) -> str:
    """New scripts write TensorBoard logs to <train dir>/log (sd-scripts turns on TensorBoard by itself
    when --logging_dir is given)."""
    if _has(text, "--logging_dir"):
        return text
    if not re.search(r"^LOG_DIR=", text, re.M):
        m = re.search(r"^OUTPUT_DIR=.*$", text, re.M)
        if not m:
            return text
        text = text[:m.end()] + '\nLOG_DIR="${TRAIN_DIR}/log"' + text[m.end():]
    return S.append_args(text, '--logging_dir "${LOG_DIR}"')[0]


def _inside(p: Path, lora: Path) -> bool:
    try:
        p.resolve().relative_to(lora.resolve())
        return True
    except ValueError:
        return False


def _rel(p: Path, lora: Path) -> str:
    try:
        return str(p.resolve().relative_to(lora.resolve())) + "/"
    except ValueError:
        return str(p)


def ensure_folders(ds, text, plan, tkey):
    """Create what a training run needs inside the LoRA folder: the <N>_name repeat folder (if there is none),
    and the model and log folders the script writes to. Images lying directly in the LoRA folder are moved
    into a newly created repeat folder. Returns a list of what was done."""
    lora = S.lora_path(ds)
    notes = []
    v = S.script_vars(text or "", lora)

    def resolve(name, default):
        raw = v.get(name)
        val = S._expand(raw, v) if raw else ""
        return Path(val) if val and "$" not in val and "`" not in val else default

    train_dir = resolve("TRAIN_DIR", lora)
    v["TRAIN_DIR"] = str(train_dir)
    for label, d in (("model", resolve("OUTPUT_DIR", train_dir / "model")), ("log", resolve("LOG_DIR", train_dir / "log"))):
        if not _inside(d, lora):
            notes.append(f"`{d}` is outside the LoRA folder, so it was not created")
        elif not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            notes.append(f"created `{_rel(d, lora)}`")
    stats = dataset_stats(ds)
    if not stats["rows"]:
        name = (plan or {}).get("create") or f"{recommended_repeats(ds, tkey)}_{lora.name.lower()}"
        dest = train_dir / name
        if not _inside(dest, lora):
            notes.append(f"`{dest}` is outside the LoRA folder, so it was not created")
        elif not dest.exists():
            dest.mkdir(parents=True)
            notes.append(f"created repeat folder `{_rel(dest, lora)}`")
            if train_dir.resolve() == lora.resolve():
                moved = 0
                for im in S.list_images(lora):
                    for src in [im] + [im.with_name(im.stem + e) for e in S.CAPTION_EXTS]:
                        if src.is_file():
                            shutil.move(str(src), str(dest / src.name))
                            moved += 1
                if moved:
                    notes.append(f"moved {moved} files (images and captions) from the LoRA folder into `{name}/`")
            if not S.list_images(dest):
                notes.append(f"`{name}/` is empty: copy your images and .txt captions into it")
        left = [f for f, n in dataset_stats(ds)["loose"] if n]          # re-read: images may just have been moved
        if left:
            notes.append("images in " + ", ".join(f"`{_rel(f, lora)}`" for f in left)
                         + " have no repeat count: use the Repeat folders panel to rename that folder")
    return notes


def recommend_numbers(sc, cfg, stats, use_sug_rep=True):
    """Batch / accumulation / repeats / epochs recommended for this type and dataset (same maths as Lora_Datatypes.py)."""
    total, rows = stats["total"], stats["rows"]
    bs, ga, _ = sc.suggest_settings(total, cfg)
    eff = bs * ga
    repeats = [r["repeats"] for r in rows]
    rename, want, create = None, None, None
    if len(rows) == 1:
        want = sc.suggest_repeats(total, bs, ga, cfg["epochs"], cfg["steps"])
        if rows[0].get("virtual"):
            repeats = [want]
            create = f"{want}_{rows[0]['folder'].name.lower()}"
        elif want != repeats[0] and use_sug_rep:
            old = rows[0]["folder"]
            rename = f'mv "{old}" "{old.parent / S.REPEAT_RE.sub(f"{want}_", old.name, count=1)}"'
            repeats = [want]
    samples = sum(r["n"] * rep for r, rep in zip(rows, repeats))
    spe = max(1, math.ceil(samples / eff))
    targets = cfg["targets"]
    main_target = targets[len(targets) // 2]
    plan = {t: max(1, round(t / spe)) for t in targets}
    epochs = plan[main_target]
    return dict(bs=bs, ga=ga, eff=eff, repeats=repeats, want_repeats=want, rename=rename, create=create, samples=samples,
                spe=spe, plan=plan, main_target=main_target, epochs=epochs, total_steps=epochs * spe)


RECS_FILE = "Training_Recommendations.py"
CHOICE = {"purpose": "character", "priority": "balanced"}      # set by the Purpose / Priority dropdowns
RECS_ERROR = {"msg": ""}


def load_recs():
    """Training_Recommendations.py, read fresh every time (so a renewed file applies without a restart)."""
    import importlib.util
    for d in (Path(__file__).resolve().parent, S.CONFIG_DIR, S.ROOT):
        if not d:
            continue
        f = Path(d) / RECS_FILE
        if f.is_file():
            try:
                spec = importlib.util.spec_from_file_location("training_recommendations", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                RECS_ERROR["msg"] = ""
                return mod
            except Exception as e:                      # noqa: BLE001
                RECS_ERROR["msg"] = f"{f}: {e}"
                return None
    RECS_ERROR["msg"] = f"{RECS_FILE} not found"
    return None


def tuned_cfg(tkey, cfg, stats):
    """The preset with the recommendations for the chosen purpose and priority applied (rec_notes says what changed)."""
    recs = load_recs()
    if recs is None:
        cfg = dict(cfg)
        cfg["rec_notes"] = [f"Recommendations not applied: {RECS_ERROR['msg']}"]
        return cfg
    files = {f: p for f, p in cfg.get("model_args", [])}
    files.update({"--" + k: v for k, v in S.settings_models(tkey).items()})
    try:
        bp = "concept" if CHOICE["purpose"] in ("object", "worn", "action", "concept") else CHOICE["purpose"]
        out, _ = recs.apply(tkey, cfg, bp, CHOICE["priority"], int(stats.get("total") or 0), files,
                            S.GUIDE.identify if S.GUIDE else None)
        return out
    except Exception as e:                              # noqa: BLE001
        cfg = dict(cfg)
        cfg["rec_notes"] = [f"Recommendations not applied ({RECS_FILE}: {e})."]
        return cfg


def set_choice(purpose, priority):
    CHOICE["purpose"], CHOICE["priority"] = purpose or "character", priority or "balanced"
    if S.CAP is not None:
        S.CAP.PURPOSE["value"] = CHOICE["purpose"]          # the caption writers follow the purpose too


def recs_status() -> str:
    recs = load_recs()
    if recs is None:
        return f"Recommendations: **not loaded** ({RECS_ERROR['msg']})."
    return f"Recommendations: `{RECS_FILE}` v{recs.VERSION} ({recs.UPDATED})."


def make_plan(ds, tkey, steps_over=0, use_sug_rep=True, hf=True, samples=True):
    sc, presets = merged_presets()
    if sc is None:
        raise FileNotFoundError("Lora_Datatypes.py could not be loaded: " + (PRESETS_ERROR["msg"] or "file not found"))
    if tkey not in presets:
        raise ValueError(f"Unknown LoRA type: {tkey}")
    cfg = dict(presets[tkey])
    ov = S.settings_models(tkey)                       # model files chosen in the Settings tab win over the preset
    if ov:
        have = {f.lstrip("-") for f, _ in cfg["model_args"]}
        cfg["model_args"] = [(f, ov.get(f.lstrip("-"), pth)) for f, pth in cfg["model_args"]]
        if S.vae_optional(tkey, presets) and "vae" not in have and ov.get("vae"):
            cfg["model_args"].append(("--vae", ov["vae"]))
    musubi = S.is_musubi_cfg(cfg)
    if musubi:
        samples = False                              # musubi types train without sample images
    stats = with_virtual_row(dataset_stats(ds))
    cfg = tuned_cfg(tkey, cfg, stats)
    cfg = _speed(tkey, cfg)                           # Settings › Speed test: "use the faster settings" for this type
    cfg["caption_ext"] = caption_ext_for(ds, tkey)
    so = _num(steps_over)
    if so:
        cfg["steps"], cfg["targets"] = so, (round(so * 0.7), so, round(so * 1.3))
    virtual = any(r.get("virtual") for r in stats["rows"])
    if not stats["rows"]:
        if stats["loose"]:
            raise ValueError("The images are not in a `<N>_name` repeat folder, so sd-scripts would ignore them. "
                             "Rename their folder in the **Repeat folders** panel below (for example to `4_name`).")
        raise ValueError("No images found in this dataset yet. Copy your images (and .txt captions) into the folder, "
                         "or press **Create missing folders now** in the Repeat folders panel to set up the folders first.")
    if not stats["total"]:
        raise ValueError("Repeat folders found, but no images inside them.")
    rec = recommend_numbers(sc, cfg, stats, use_sug_rep)
    lora = stats["lora"]
    train_dir = lora if virtual else sorted({r["folder"].parent for r in stats["rows"]})[0]
    if samples:
        prompts_path, prompts_text, prompts_new = build_prompts(sc, stats, cfg)
    else:
        prompts_path, prompts_text, prompts_new = lora / cfg["prompts_name"], "", False
    base_name = lora.name + cfg["suffix"]
    version = sc.next_version(train_dir / "model", base_name)
    out_name = f"{base_name}V{version}"
    run_cfg = {**cfg, "suffix": f"{cfg['suffix']}V{version}"}
    cmd = sc.build_command(run_cfg, "${CHARACTER_NAME}", "${TRAIN_DIR}", "${OUTPUT_DIR}", rec["bs"], rec["ga"],
                           rec["epochs"], cfg.get("save_every", 1), cfg["lr"], "${PROMPTS_FILE}" if samples else None)
    if not cmd:
        raise ValueError("This LoRA type has no launch template (no 'script' in its preset).")
    plan_line = (f"Output {out_name}. Plan: {stats['total']} images x {'/'.join(str(x) for x in rec['repeats'])} "
                 f"repeats, batch {rec['bs']} x accum {rec['ga']}, ~{rec['spe']} steps/epoch, {rec['epochs']} epochs "
                 f"= ~{rec['total_steps']} steps (target {rec['main_target']})")
    extra = {"cfg": run_cfg, "bs": rec["bs"]} if musubi else {}
    text = sc.render_training_script(sc.under(S.trainer_root(cfg), S.HOME, "${HOME}"), sc.under(train_dir, lora, "${SCRIPT_DIR}"),
                                     sc.under(prompts_path, lora, "${SCRIPT_DIR}") if samples else None, plan_line, cmd,
                                     S.vendor_env({**sc.ROCM_ENV, **cfg.get("env", {}), **S.env_saved()}), **extra)
    if not musubi:
        text = add_logging(text)
    if hf:
        text, _ = S.fill_hf(text, force=True)
    return dict(cfg=cfg, stats=stats, rec=rec, script_text=text, prompts_path=prompts_path,
                prompts_text=prompts_text, prompts_new=prompts_new, out_name=out_name, musubi=musubi,
                filename=unique_script_name(lora, f"{out_name}.sh"))


# ---- explanation
OPT_WHY = {
    "adamw8bit": "AdamW8bit keeps optimizer state in 8 bit (about a quarter of AdamW's memory) with practically the same quality. That is why it is the usual choice for LoRA on 16 GB.",
    "adamw": "AdamW is the reference optimizer. It needs more VRAM than the 8-bit version for the same result.",
    "adafactor": "Adafactor needs very little optimizer memory. With scale_parameter=False relative_step=False it behaves like a fixed-LR optimizer.",
    "prodigy": "Prodigy adapts the learning rate by itself, so the LR should stay at 1.0. It removes LR guesswork at the cost of some memory.",
    "lion": "Lion is memory-light but wants a 3-10x lower LR than AdamW.",
}
SCHED_WHY = {
    "cosine": "cosine decays the LR smoothly towards 0, so the last epochs refine what was learned instead of overwriting it.",
    "cosine_with_restarts": "cosine_with_restarts decays like cosine but restarts a few times, which can shake the run out of a weak solution. The last cycle still ends near 0.",
    "constant": "constant keeps the LR fixed: predictable, but later epochs can overshoot.",
    "constant_with_warmup": "constant_with_warmup holds the LR fixed after the warmup: predictable, but later epochs can overshoot.",
    "linear": "linear decays the LR in a straight line to 0.",
    "polynomial": "polynomial decays the LR along a curve to 0.",
}
FLAG_WHY = [
    ("--gradient_checkpointing", "Gradient checkpointing recomputes activations instead of storing them: roughly 20-30% slower, but it is what keeps this model inside 16 GB."),
    ("--cache_latents", "Images are encoded by the VAE once and cached (to disk with --cache_latents_to_disk), so the VAE does not run every step. Faster and lighter; random crop/flip augmentation is off."),
    ("--cache_text_encoder_outputs", "Text-encoder outputs are cached, so the text encoder is not held in VRAM while training. Caption shuffling is therefore off and tag order stays as written."),
    ("--network_train_unet_only", "Only the image model is trained (text encoder frozen): less VRAM, and less overfitting to caption wording."),
    ("--fp8_base", "The base model is held in 8-bit float and upcast layer by layer, roughly halving its VRAM so large models fit 16 GB. The quality cost is small for LoRA training."),
    ("--blocks_to_swap", "Transformer blocks are swapped between GPU and system RAM to save VRAM, at a speed cost. Raise the number if you still run out of memory."),
    ("--sdpa", "PyTorch scaled-dot-product attention is the fast attention path on ROCm (xformers is not available there)."),
    ("--shuffle_caption", "Tags are shuffled every step so the model does not memorize their order. --keep_tokens pins the first tag (the trigger word)."),
    ("--min_snr_gamma", "Min-SNR weighting (gamma 5) balances the loss across noise levels, which speeds up convergence and reduces blotchy artifacts."),
    ("--noise_offset", "A small noise offset lets the model learn very dark and very bright images. Keep it small (0.05 or less) or colors drift."),
    ("--base_weights", "The training adapter is merged into the Turbo model before training. Turbo is a distilled model "
                       "that loses its few-step ability within a few hundred steps of normal training; the adapter "
                       "keeps it intact. It is NOT used when you generate: the LoRA goes on the normal Turbo model."),
    ("--fp8_scaled", "The bf16 DiT is quantised to scaled fp8 while loading (about half the memory). Needed for the 20B "
                     "Qwen-Image on 16 GB VRAM / 32 GB RAM; if the loss turns NaN, fp8 is the first suspect."),
    ("--model_version", "Which variant of the model family musubi builds (FLUX.2: klein-base-4b / klein-base-9b, Qwen-Image: "
                        "original). It must match the DiT and text-encoder files."),
    ("--weighting_scheme", "How the loss is weighted over noise levels; `none` is the flow-matching default for musubi."),
    ("--save_state", "Training state is saved with the checkpoints, so a crashed run can be resumed from Training › Run."),
    ("--model_type", "Tells flux_train_network.py which model family the base file is. `chroma` switches to Chroma: "
                     "T5-XXL only (no CLIP-L), Chroma's own modulation layers and its timestep handling."),
    ("--apply_t5_attn_mask", "T5 padding tokens are masked, the way Chroma was trained. sd-scripts refuses to train "
                             "Chroma without it, and leaving it out would teach the LoRA to rely on padding."),
    ("--guidance_scale", "The embedded (distilled) guidance value used during training. Chroma has no distilled "
                         "guidance, so it must be 0.0; you use real CFG with a negative prompt when generating. "
                         "FLUX.1-dev uses 1.0 here."),
]


def _has(text, flag):
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text) is not None


def safetensors_keys(path: Path, limit: int = 4000) -> list:
    """Tensor names from a .safetensors header (only the header is read, not the weights)."""
    import json as _json
    import struct as _struct
    try:
        with open(path, "rb") as fh:
            (n,) = _struct.unpack("<Q", fh.read(8))
            if n > 200_000_000:
                return []
            head = _json.loads(fh.read(n))
        return [k for k in head if k != "__metadata__"][:limit]
    except Exception:
        return []


# Separate component files (text encoders, VAE/AE) must contain only that component. A full checkpoint that
# bundles them (e.g. sd3_medium_incl_clips_t5xxlfp16) has prefixed names, so sd-scripts finds none of the weights
# it expects and fails later with "Cannot copy out of meta tensor".
_BUNDLE_PREFIXES = ("model.diffusion_model.", "first_stage_model.", "text_encoders.", "conditioner.", "cond_stage_model.")
_COMPONENT = {"t5xxl": "T5-XXL text encoder", "clip_l": "CLIP-L text encoder", "clip_g": "CLIP-G text encoder",
              "ae": "AE (VAE)", "vae": "VAE", "qwen3": "Qwen3 text encoder", "gemma2": "Gemma2 text encoder",
              "text_encoder": "text encoder", "byt5": "ByT5 text encoder"}


def model_file_problem(flag: str, path: Path) -> str:
    """'' when the file looks right for this slot, else a warning for the analysis."""
    if path.suffix.lower() != ".safetensors" or (flag not in _COMPONENT and flag not in ("pretrained_model_name_or_path", "dit")):
        return ""
    keys = safetensors_keys(path, limit=200000)
    if not keys:
        return ""
    # ComfyUI "fp8_scaled" files store fp8 weights plus a scale per layer (".scale_weight" / "scaled_fp8").
    # sd-scripts ignores the scales, so it would train on wrong weights without any error.
    if any(k.endswith(".scale_weight") or k == "scaled_fp8" or k.endswith(".scale_input") for k in keys):
        return (f"`--{flag}` points to `{path.name}`, a ComfyUI **fp8_scaled** file. sd-scripts cannot use its scale "
                "values and would train on wrong weights (no error, just a broken LoRA). Use the normal bf16/fp16 file "
                "(sd-scripts converts it to fp8 itself with --fp8_base)"
                + (", for T5-XXL `t5xxl_fp16.safetensors` or the unscaled `t5xxl_fp8_e4m3fn.safetensors`." if flag == "t5xxl" else
                   ", for Chroma `chroma_v10HD.safetensors` / `Chroma1-HD.safetensors`." if flag == "pretrained_model_name_or_path" else
                   ", for Z-Image Turbo the plain `z_image_turbo_bf16.safetensors`." if flag == "dit" else "."))
    if flag not in _COMPONENT:
        return ""
    hits = sorted({p for p in _BUNDLE_PREFIXES for k in keys if k.startswith(p)})
    if hits:
        what = ", ".join(f"`{h}...`" for h in hits)
        return (f"`--{flag}` points to `{path.name}`, which is a **full model checkpoint** (its weights are named {what}), "
                f"not a standalone {_COMPONENT[flag]} file. sd-scripts would find none of the weights it needs and stop "
                "with *Cannot copy out of meta tensor*. Use the standalone file instead"
                + (" (for T5-XXL: `t5xxl_fp16.safetensors` or `t5xxl_fp8_e4m3fn.safetensors`, the same files ComfyUI uses for FLUX)." if flag == "t5xxl" else "."))
    if flag == "t5xxl" and not any(k.startswith(("encoder.", "shared.")) for k in keys):
        return (f"`--t5xxl` points to `{path.name}`, which does not look like a T5-XXL encoder (no `encoder.block...` "
                "weights). Use `t5xxl_fp16.safetensors` or `t5xxl_fp8_e4m3fn.safetensors`.")
    return ""


def type_dataset_checks(cfg, stats) -> list:
    """Checks a LoRA type asks for in its preset: caption style ("captions": "natural") and image size ("min_side":
    images with fewer pixels than min_side x min_side get upscaled by bucketing)."""
    out = []
    folders = [r["folder"] for r in stats["rows"] if not r.get("virtual")] or [stats["lora"]]
    imgs = [p for d in folders if d.is_dir() for p in sorted(d.iterdir()) if p.suffix.lower() in S.IMG_EXT]
    if cfg.get("captions") == "natural":
        tagged = natural = 0
        for p in imgs[:400]:
            c = p.with_suffix(".txt")
            if not c.is_file():
                continue
            text = c.read_text(encoding="utf-8", errors="ignore").strip()
            parts = [x.strip() for x in text.split(",") if x.strip()]
            if not parts:
                continue
            words = sum(len(x.split()) for x in parts) / len(parts)
            if words < 2.5 and len(parts) >= 4:
                tagged += 1
            else:
                natural += 1
        if tagged and tagged >= natural:
            out.append(f"{tagged} of {tagged + natural} captions are short tag lists. {cfg['label'].split(' (')[0]} reads "
                       "captions through a language model (T5 / Qwen / Gemma), which understands sentences: it trains better on "
                       "short natural sentences that start with the trigger word (tags work, but teach less about how things "
                       "relate). **Recaption with Ollama** (further down this tab) rewrites them with a local vision model.")
    side = cfg.get("min_side")
    if side and imgs:
        small = []
        try:
            from PIL import Image
            for p in imgs[:1500]:
                try:
                    with Image.open(p) as im:
                        if (im.size[0] * im.size[1]) ** 0.5 < side:     # total pixels, so 1216x832 counts as 1024-class
                            small.append(p)
                except Exception:
                    pass
        except ImportError:
            pass
        if small:
            ex = ", ".join(p.name for p in small[:4]) + (" ..." if len(small) > 4 else "")
            m = re.search(r"--resolution[ =](\d+)", " ".join(cfg.get("flags", [])))
            out.append(f"{len(small)} of {len(imgs)} images are smaller than about {side}x{side} pixels ({ex}). This type trains "
                       f"at {m.group(1) if m else 1024} px; bucketing upscales small images, which teaches softness. "
                       "Use larger sources or upscale them first.")
    return out




def explain_md(ds, tkey, cfg, stats, text, rec, existing, planned_repeats=None, rename=None, create=None):
    f = form_dict(text)

    def I(name, default=None):
        try:
            return int(float(str(f.get(name, "")).strip()))
        except ValueError:
            return default

    rows, imgs = stats["rows"], stats["total"]
    reps = planned_repeats or [r["repeats"] for r in rows]
    samples = sum(r["n"] * rep for r, rep in zip(rows, reps))
    bs, ga = max(1, I("train_batch_size", 1) or 1), max(1, I("gradient_accumulation_steps", 1) or 1)
    ep = I("max_train_epochs")
    eff = bs * ga
    spe = max(1, math.ceil(samples / eff))
    total = spe * ep if ep else None
    views = total * eff / imgs if total else None
    lo, hi = cfg["targets"][0], cfg["targets"][-1]
    rep_txt = " / ".join(str(x) for x in sorted(set(reps)))

    L = []
    L.append(f"### {'Your launch script compared with the recommendation' if existing else 'Recommended plan'}: {cfg['label']}")
    miss = imgs - stats["captioned"]
    L.append(f"**Dataset `{ds}`:** {imgs} images in {len(rows)} repeat folder(s), {stats['captioned']} with captions"
             + (f", **{miss} without**" if miss else "") + ".")

    # why these settings
    W = []
    if ga > 1:
        W.append(f"**Batch {bs} × accumulation {ga} = {eff}.** A larger effective batch gives smoother gradients. "
                 f"The batch size alone is capped at {cfg['max_batch']} for this architecture on 16 GB, so gradient "
                 "accumulation supplies the rest without extra VRAM.")
    else:
        W.append(f"**Batch {bs}, no accumulation.** At this dataset size one batch of {bs} already gives about the "
                 f"{cfg['views']} views per image the plan aims for, so no accumulation is needed.")
    save_every = I("save_every_n_epochs", 1) or 1
    if ep:
        W.append(f"**{rep_txt} repeats × {ep} epochs.** Low repeats with more epochs give a checkpoint every {save_every} "
                 f"epoch(s) ({max(1, ep // save_every)} in total) to compare. The best one is usually around epoch "
                 f"{max(1, round(ep * 0.6))}-{max(1, round(ep * 0.8))} of {ep}, not the last.")
    opt, lr = (f.get("optimizer_type") or "").lower(), f.get("learning_rate") or ""
    if opt or lr:
        W.append(f"**{f.get('optimizer_type') or 'optimizer'}, LR {lr or '?'}.** " + OPT_WHY.get(opt, "")
                 + " If the loss turns to NaN, retry at half the LR. If likeness is weak at the end, raise it or add epochs.")
    sched = (f.get("lr_scheduler") or "").lower()
    warm = I("lr_warmup_steps")
    if sched:
        pct = f" The {warm} warmup steps ({100 * warm / total:.0f}% of the run) ramp the LR up gradually so the first noisy steps cannot damage the weights." if warm and total else ""
        W.append(f"**Scheduler {sched}.** " + SCHED_WHY.get(sched, "") + pct)
    dim, alpha = I("network_dim"), I("network_alpha")
    if dim and alpha:
        W.append(f"**Dim {dim}, alpha {alpha} (scale {alpha / dim:.2f}).** Dim is the LoRA's capacity; 16-32 is plenty for "
                 "one character, and more mostly adds file size and overfitting. A scale below 1 acts like a lower "
                 "learning rate and keeps training stable.")
    mp = (f.get("mixed_precision") or "").lower()
    if mp:
        W.append("**Precision " + mp + ".** " + ("bf16 has fp32's number range, which avoids the NaN/overflow problems of fp16, and your RDNA4 GPU supports it natively." if mp == "bf16" else "fp16 is fast but can overflow; switch to bf16 if you see NaN."))
    if f.get("resolution"):
        W.append(f"**Resolution {f['resolution']}** with aspect-ratio bucketing, so non-square images are trained without cropping.")
    if (f.get("timestep_sampling") or "").lower() == "sigmoid":
        W.append("**Timestep sampling sigmoid** (with raw model prediction). Training timesteps are drawn mostly from the "
                 "middle of the noise range, where a LoRA learns shapes and likeness best. sd-scripts recommends it for "
                 "Chroma; no flow shift is used with it.")
    elif f.get("timestep_sampling") or f.get("discrete_flow_shift"):
        W.append(f"**Flow-matching settings** (timestep sampling {f.get('timestep_sampling') or 'default'}, shift "
                 f"{f.get('discrete_flow_shift') or 'default'}). The shift moves training towards noisier timesteps, which suits large images; "
                 "it is a common starting value for LoRA training of this model family.")
    for flag, why in FLAG_WHY:
        if _has(text, flag):
            W.append(f"**`{flag}`.** {why}")
    if f.get("sample_every_n_epochs"):
        W.append(f"**Sample images every {f['sample_every_n_epochs']} epoch(s)** from the prompts file with a fixed seed, so you can compare epochs visually.")
    L += ["", "**Why these settings**", ""] + [f"- {w}" for w in W]

    # warnings
    A = []
    if create:
        A.append(f"There is no `<N>_name` repeat folder yet. **`{create}/` will be created when you save** (with the `model` and `log` "
                 "folders), and images lying directly in the LoRA folder are moved into it, if the checkbox next to Save is ticked.")
    if rename:
        A.append("The repeat folder gets the recommended repeat count when you press **Fill with recommended settings** "
                 "or save this new script (its copies in `_originals/` and `_reserve/` too).")
    elif not existing and rec and rec.get("want_repeats") and rec["want_repeats"] != rec["repeats"][0]:
        A.append(f"The folder uses {rec['repeats'][0]} repeats; about {rec['want_repeats']} would hit the step target best.")
    if total is not None:
        if total < lo:
            A.append(f"Total steps ({total}) is below the recommended {lo}-{hi}: likely undertrained. About {max(1, round(cfg['steps'] / spe))} epochs would give ~{cfg['steps']} steps.")
        elif total > hi:
            A.append(f"Total steps ({total}) is above the recommended {lo}-{hi}: overfit risk. About {max(1, round(cfg['steps'] / spe))} epochs would give ~{cfg['steps']} steps.")
    if views is not None and views < VIEWS_OK[0]:
        A.append(f"Only ~{views:.0f} views per image: weak likeness is likely.")
    if views is not None and views > VIEWS_OK[1]:
        A.append(f"~{views:.0f} views per image: overfitting and memorized poses are likely.")
    if bs > cfg["max_batch"]:
        A.append(f"Batch size {bs} is above the {cfg['max_batch']} that fits in 16 GB for this type: out-of-memory risk.")
    if miss:
        A.append(f"{miss} image(s) have no caption file (check the dataset's repeat folder).")
    if stats["loose"]:
        A.append("Images outside `<N>_name` repeat folders are ignored by sd-scripts: " + ", ".join(f"{p.name} ({n})" for p, n in stats["loose"]))
    top_others = [(f, n) for f, n in stats.get("others") or [] if f.parent == stats["lora"]]
    if top_others:
        A.append("Not part of this dataset (only `<N>_" + stats["lora"].name + "` is): " + ", ".join(
            f"{p.name} ({n})" for p, n in top_others) + ". sd-scripts would train them too, so they are moved to `"
            + EXTRA_DIR + "/` when you save the launch script (nothing is deleted).")
    for d in S.renamed_archives(stats["lora"]):
        A.append(f"🔴 `{d.name}` is the Personal sync folder (LoRA previews and example images) renamed into a repeat folder: "
                 f"sd-scripts would TRAIN on those pictures and the Library no longer finds its data. Rename it back: "
                 f"`mv \"{d}\" \"{d.with_name('LoraManager')}\"` (it is left out of the plan above).")
    if imgs < 15:
        A.append(f"Only {imgs} images: expect weak or overfit results. 20-100 good, varied images is the sweet spot.")
    A += type_dataset_checks(cfg, stats)
    if S.get_export(text, "HF_TOKEN") == "" and S.read_hf_token():
        A.append("`HF_TOKEN` is empty in this script but huggingtoken.txt has a token. Press **Fill HF_TOKEN** if you use gated Hugging Face models.")
    if not _has(text, "--logging_dir") and not S.is_musubi_text(text):
        A.append("No `--logging_dir`: nothing is written to the `log` folder, so TensorBoard has no loss curves. "
                 "Put `${LOG_DIR}` (or a folder path) in the form field 'Logging dir'.")
    if not _has(text, "--save_state"):
        A.append("No `--save_state`: a crashed run cannot be resumed (resuming in Training › Run needs it).")
    for name in MODEL_FIELDS:
        v = os.path.expandvars(os.path.expanduser(f.get(name, "") or ""))
        if v and "$" not in v and not Path(v).exists():
            A.append(f"Model file not found for `--{name}`: `{v}`" + (" (placeholder: set the real path)" if "/PATH/TO/" in v else ""))
        elif v and "$" not in v and Path(v).is_file():
            bad = model_file_problem(name, Path(v))
            if bad:
                A.append(bad)
            elif S.GUIDE is not None:                       # right kind of file for this slot? (e.g. an SD3 VAE as FLUX AE)
                try:
                    kinds = S.GUIDE.requirements_for(tkey, S._type_base(tkey)).get(name, [])
                    info = S.GUIDE.identify(v)
                    why = S.GUIDE.problems(info, name, kinds) if kinds and info.get("kind") not in ("unknown", "missing") else ""
                except Exception:
                    why = ""
                if why:
                    A.append(f"`--{name}` `{Path(v).name}` {why}. Settings › Model files › Suggestions lists the files that fit.")
    script = cfg.get("script", "")
    mu = S.is_musubi_cfg(cfg) or S.is_musubi_text(text)
    troot = S.musubi_root() if mu else S.sd_root()
    if script and not S.trainer_script_path(script, mu).is_file():
        A.append(f"`{script}` is not in `{troot}`: " + ("check the musubi-tuner folder in Settings › Trainers + GPU, or update musubi-tuner (git pull)."
                                                       if mu else "this architecture is not supported by your sd-scripts version."))
    if not (troot / "venv").is_dir():
        A.append(f"No `venv/` in `{troot}`" + (" (the script activates `<musubi-tuner>/venv`)." if mu else "."))
    if cfg.get("template"):
        A.append("This type's settings are a template written from the sd-scripts documentation, not tested on your machine. Use **Check flags** before training.")
    if A:
        L += ["", "**Check before you train**", ""] + [f"- {a}" for a in A]
    if cfg.get("rec_notes"):
        L += ["", "**Recommendations used**", ""] + [f"- {n}" for n in cfg["rec_notes"]]
    if cfg.get("notes"):
        L += ["", f"**Notes for {cfg['label']}**", ""] + [f"- {n}" for n in cfg["notes"]]
    return "\n".join(L)


def explain_for(ds, tkey, script, text, plan):
    """Explanation markdown for whatever is in the editor right now."""
    try:
        sc, presets = merged_presets()
        if tkey not in presets:
            return "Pick a LoRA type."
        cfg = dict(presets[tkey])
        stats = with_virtual_row(dataset_stats(ds))
        if not stats["rows"] or not stats["total"]:
            return "**No analysis:** this dataset has no images in `<N>_name` repeat folders yet."
        cfg = tuned_cfg(tkey, cfg, stats)
        rec = recommend_numbers(sc, cfg, stats, True) if sc else None
        return explain_md(ds, tkey, cfg, stats, text, rec, existing=(script != NEW),
                          planned_repeats=(plan or {}).get("repeats") or (rec or {}).get("repeats"),
                          rename=(plan or {}).get("rename"), create=(plan or {}).get("create") or (rec or {}).get("create"))
    except Exception as e:
        return f"**Could not build the explanation:** {e}"


# ---- report for renewing the recommendations
def _cmd(args, timeout=20) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:                              # noqa: BLE001
        return f"(not available: {e})"


def _mask(text: str) -> str:
    text = re.sub(r"(HF_TOKEN=)\"?[^\s\"]+\"?", r'\1"***"', text)
    return re.sub(r"hf_[A-Za-z0-9]{10,}", "hf_***", text)


def _log_summary(f: Path) -> list:
    try:
        lines = re.split(r"[\r\n]+", f.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return [f"- `{f.name}`: unreadable"]
    exit_line = next((l for l in reversed(lines) if l.startswith("--- exited with code")), "--- (no exit line) ---")
    steps = [l for l in lines if "steps:" in l and "it" in l]
    speed = steps[-1].strip()[-120:] if steps else "(no step line)"
    cmd = next((l for l in lines if "_train_network.py" in l or "train_network.py" in l), "")
    errs = [l.strip() for l in lines if re.search(r"(Error|Exception|Traceback|out of memory|NaN)", l)][-4:]
    m = re.search(r"code (-?\d+)(?:.*?ran ([\d:]+))?", exit_line)
    ended = (("finished OK" if m.group(1) == "0" else f"exit code {m.group(1)}") + (f" after {m.group(2)}" if m.group(2) else "")) \
        if m else "no exit line (still running, or from an older version)"
    out = [f"- **{f.name}**: {ended}", f"  - last progress: `{speed}`"]
    if cmd:
        out.append(f"  - command: `{_mask(cmd.strip())[:600]}`")
    out += [f"  - {e[:240]}" for e in errs]
    return out


def export_report():
    """One Markdown file with everything needed to renew Training_Recommendations.py: versions, hardware,
    software, model settings, recent runs (speed, loss, errors) and the LoRAs' status and your notes."""
    now = datetime.now()
    recs = load_recs()
    _, presets = merged_presets()
    L = [f"# LoRA Studio report, {now:%Y-%m-%d %H:%M}", "",
         "Send this file together with `Training_Recommendations.py` to renew the recommendations.", "",
         "## Versions",
         "- Training_Recommendations.py: " + (f"v{recs.VERSION} ({recs.UPDATED})" if recs else RECS_ERROR["msg"]),
         f"- LoRA types: {', '.join(sorted(presets))}",
         f"- sd-scripts: `{S.sd_root()}` " + _cmd(["git", "-C", str(S.sd_root()), "log", "-1", "--format=%h %cd", "--date=short"]),
         f"- musubi-tuner: `{S.musubi_root()}` " + _cmd(["git", "-C", str(S.musubi_root()), "log", "-1", "--format=%h %cd", "--date=short"]),
         "", "## Hardware and software",
         f"- GPU: {_cmd(['bash', '-c', 'lspci | grep -iE \"vga|3d\" | head -2'])}",
         f"- VRAM / ROCm: {_cmd(['bash', '-c', '(amd-smi static --asic --vram 2>/dev/null | grep -iE \"market_name|size\" | head -3) || (rocm-smi --showproductname --showmeminfo vram 2>/dev/null | grep -iE \"card series|vram total|Total Memory\" | head -3)'])}",
         f"- RAM: {_cmd(['bash', '-c', 'free -h | head -2 | tail -1'])}",
         f"- Kernel: {_cmd(['uname', '-r'])}"]
    wd, venv, _ = S.runner_settings()
    py = Path(os.path.expanduser(venv or str(S.sd_root() / "venv"))) / "bin" / "python"
    if py.is_file():
        L.append("- sd-scripts venv: " + _cmd([str(py), "-c", "import sys, torch; print('Python', sys.version.split()[0], "
                                                "'| torch', torch.__version__, '| HIP', torch.version.hip, '| GPU', "
                                                "torch.cuda.is_available())"], 60))
    L += ["", "## Model files per LoRA type (Settings › Model files)"]
    for tkey, kv in S.read_settings().items():
        files = {k: v for k, v in kv.items() if not k.endswith(".dir") and k not in META_KEYS}
        if not files:
            continue
        L.append(f"- **{tkey}**")
        for k, v in files.items():
            desc = S.GUIDE.describe(S.GUIDE.identify(v)) if S.GUIDE and os.path.isfile(os.path.expanduser(v)) else "not found"
            L.append(f"  - {k}: `{Path(v).name}` ({desc})")
    L += ["", "## Recent runs (newest first)"]
    logs = sorted([f for f in S.LOG_DIR.glob("*.log") if not f.name.startswith("caption_")],
                  key=lambda f: f.stat().st_mtime, reverse=True)[:12] if S.LOG_DIR.is_dir() else []
    for f in logs:
        L += _log_summary(f)
    if not logs:
        L.append("- (no run logs yet)")
    try:
        import lora_library as LL
        lib = LL.LIB
        lib.ensure_loaded()
        L += ["", "## Your LoRAs (newest version per dataset + type)"]
        for gid, members in sorted(lib.groups.items(), key=lambda kv: (kv[1][-1].character.lower(), kv[1][-1].base)):
            l = members[-1]
            if l.dataset is None:
                continue
            m = l.meta
            ov = lib.overrides(l.key)
            notes = (l.sidecar or {}).get("notes", "")
            loss = f"{l.loss[-1][1]:.4f}" if l.loss else "-"
            L.append(f"- **{l.character} · {l.base}** `{l.stem}` ({len(members)} version(s)): status {l.status}; "
                     f"dim {m.get('ss_network_dim', '?')}/{m.get('ss_network_alpha', '?')}, lr {m.get('ss_learning_rate', '?')}, "
                     f"epochs {m.get('ss_epoch', '?')}/{m.get('ss_num_epochs', '?')}, steps {m.get('ss_steps', '?')}, "
                     f"images {m.get('ss_num_train_images', '?')}, last loss {loss}"
                     + (f"; issues: {'; '.join(t for _, t in l.issues)}" if l.issues else "")
                     + (f"; your flag: {ov.get('flag')} {ov.get('flag_reason', '')}" if ov.get("flag") else "")
                     + (f"; your notes: {notes}" if notes else ""))
    except Exception as e:                              # noqa: BLE001
        L.append(f"\n(LoRA list not available: {e})")
    L += ["", "## What I would like improved", "", "(write here: what worked, what did not, what you want next)", ""]
    out = (S.CONFIG_DIR or S.ROOT) / f"LoRA_Studio_report_{now:%Y%m%d_%H%M}.md"
    out.write_text(_mask("\n".join(L)), encoding="utf-8")
    return (gr.update(value=str(out), visible=True),
            f"**Report written:** `{out}`. Add a few words under *What I would like improved*, then send it with "
            f"`{RECS_FILE}`. Hugging Face tokens are masked.")


# ---- handlers
KEEP = object()


def _cs_result(script=KEEP, type_=KEEP, text=KEEP, fields=None, explain=KEEP, prompts_path=KEEP,
               prompts_text=KEEP, newname=KEEP, save_over=KEEP, status=KEEP, plan=None, samples=KEEP):
    def k(v):
        return gr.update() if v is KEEP else v
    flds = fields if fields is not None else [gr.update()] * len(FORM_FIELDS)
    return ([k(script), k(type_), k(text)] + list(flds)
            + [k(explain), k(prompts_path), k(prompts_text), k(newname), k(save_over), k(status), plan or {}, k(samples)])


def _plan_state(plan):
    return {"repeats": plan["rec"]["repeats"], "rename": plan["rec"]["rename"], "create": plan["rec"]["create"]}


def _new_status(plan):
    n = " (a launch script for this folder already exists; the new one gets a different name)" if launch_scripts(plan["stats"]["lora"].name) else ""
    return (f"**New launch script:** recommended settings for {plan['cfg']['label']}{n}. Nothing is written until you press Save as new.")


def script_for_type(ds, tkey) -> str:
    """The launch script of this dataset for a LoRA type (the newest version), or NEW when it has none yet."""
    best, best_v = None, -1
    for p in launch_scripts(ds):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if S.script_type(p, text) != tkey:
            continue
        _, out_name = S.script_args(text, script_home(p))
        v = _ver_of(out_name or "")[1]
        if v > best_v or (v == best_v and best is not None and p.stat().st_mtime > best.stat().st_mtime):
            best, best_v = p, v
    return str(best) if best else NEW


def cs_load(ds, script, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    if not ds:
        return _cs_result(status="Pick a LoRA folder.")
    files = launch_scripts(ds)
    if script != NEW and script not in {str(p) for p in files}:
        if tkey:                                    # the script of the chosen LoRA type, never another type's
            script = script_for_type(ds, tkey)
        else:
            script = str(files[0]) if files else NEW
    upd = gr.update(choices=cs_script_choices(ds), value=script)
    n = len(FORM_FIELDS)
    if script == NEW:
        try:
            plan = make_plan(ds, tkey, steps_over, use_rep, hf_auto, samples)
        except Exception as e:
            return _cs_result(script=upd, text="", fields=[""] * n, explain="", prompts_path="", prompts_text="",
                              newname="", save_over=gr.update(visible=False),
                              status=f"**Cannot build a recommendation:** {e}")
        text, ps = plan["script_text"], _plan_state(plan)
        return _cs_result(script=upd, text=text, fields=read_form(text),
                          explain=explain_md(ds, tkey, plan["cfg"], plan["stats"], text, plan["rec"], False,
                                             ps["repeats"], ps["rename"], ps["create"]),
                          prompts_path=str(plan["prompts_path"]) if samples else "",
                          prompts_text=plan["prompts_text"] if samples else "",
                          newname=plan["filename"], save_over=gr.update(visible=False),
                          status=_new_status(plan), plan=ps)
    text = Path(script).read_text(encoding="utf-8", errors="ignore")
    filled = False
    if hf_auto:
        text, filled = S.fill_hf(text, force=False)
    _, presets = merged_presets()
    tk = infer_type(text, presets, tkey)
    pp = prompts_path_from_text(text, S.lora_path(ds))
    ptext = pp.read_text(encoding="utf-8", errors="ignore") if pp and pp.is_file() else ""
    label = presets.get(tk, {}).get("label", tk)
    return _cs_result(script=upd, type_=tk, text=text, fields=read_form(text),
                      explain=explain_for(ds, tk, script, text, {}),
                      samples=has_samples(text),
                      prompts_path=str(pp) if pp else "", prompts_text=ptext,
                      newname=_next_name(ds, text, script),
                      save_over=gr.update(visible=True),
                      status=f"**Loaded `{Path(script).name}`** (detected type: {label}). Edit the form or the text, then Save. "
                             "Use Save as new to keep the original untouched."
                             + (" `HF_TOKEN` was empty and has been filled from huggingtoken.txt (not saved yet)." if filled else ""))


def _next_name(ds, text, script) -> str:
    lora = S.lora_path(ds)
    base, n = next_script_version(lora, text, script_home(script))
    return unique_script_name(lora, f"{base}V{n}.sh" if base else Path(script).stem + "_new.sh")


def cs_ds_change(ds, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    return cs_load(ds, "", tkey, steps_over, use_rep, hf_auto, samples)


def cs_script_change(ds, script, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    return cs_load(ds, script, tkey, steps_over, use_rep, hf_auto, samples)


def cs_suggest(ds, script, tkey, steps_over, use_rep, hf_auto=True, samples=True):
    """Replace the editor with the recommended script for the chosen type (does not save anything)."""
    if not ds:
        return _cs_result(status="Pick a LoRA folder.")
    try:
        plan = make_plan(ds, tkey, steps_over, use_rep, hf_auto, samples)
    except Exception as e:
        return _cs_result(status=f"**Cannot build a recommendation:** {e}")
    renamed = ""
    if use_rep and plan["rec"].get("rename") and plan["rec"].get("want_repeats"):
        renamed = _auto_rename(ds, plan["rec"]["want_repeats"])
        if renamed and not renamed.startswith("⚠"):
            plan = make_plan(ds, tkey, steps_over, use_rep, hf_auto, samples)   # the folder has the count now
    text, ps = plan["script_text"], _plan_state(plan)
    gr.Info("The editor now shows the recommended script. Nothing is saved until you press a Save button.")
    return _cs_result(text=text, fields=read_form(text),
                      explain=explain_md(ds, tkey, plan["cfg"], plan["stats"], text, plan["rec"], False,
                                         ps["repeats"], ps["rename"], ps["create"]),
                      prompts_path=(str(plan["prompts_path"]) if samples else "") if script == NEW else KEEP,
                      prompts_text=(plan["prompts_text"] if samples else "") if script == NEW else KEEP,
                      status=f"Recommended settings for {plan['cfg']['label']} loaded into the editor (not saved)."
                             + (f" {renamed}." if renamed else ""),
                      plan=ps if script == NEW else {})


def _auto_rename(ds, want) -> str:
    try:
        import dsflow
        r = dsflow.rename_repeats(ds, int(want))
    except Exception as e:                  # noqa: BLE001
        return f"⚠ The repeat folder could not be renamed: {e}"
    return ("✔ " + r[0].upper() + r[1:]) if r and not r.startswith("⚠") else r


def cs_apply_quiet(ds, tkey, script, text, plan, *vals):
    """LoRA Studio: a field of More settings was changed: write it into the script text, without a message."""
    new, _, _ = apply_form(text or "", list(vals))
    if new == (text or ""):
        return [gr.update()] * len(_cs_result())
    return _cs_result(text=new, explain=explain_for(ds, tkey, script, new, plan), plan=plan)


def cs_retype(ds, script, tkey, text, steps_over, use_rep, hf_auto=True, samples=True):
    if script == NEW:
        return cs_load(ds, NEW, tkey, steps_over, use_rep, hf_auto, samples)
    if ds and tkey:
        # another LoRA type means its own script, or a new one with that type's recommended settings;
        # never the settings of the type that was open before
        try:
            cur = S.script_type(script, Path(script).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            cur = ""
        if cur != tkey:
            return cs_load(ds, script_for_type(ds, tkey), tkey, steps_over, use_rep, hf_auto, samples)
    return _cs_result(explain=explain_for(ds, tkey, script, text, {}),
                      status="Comparison updated for the chosen type. Your script text is unchanged.")


def cs_apply(ds, tkey, script, text, plan, *vals):
    new, changes, failed = apply_form(text or "", list(vals))
    if changes:
        gr.Info("Applied to the script text (not saved yet): " + "; ".join(changes))
    if failed:
        gr.Warning("No training command found to add these to: " + ", ".join(failed))
    if not changes and not failed:
        gr.Info("Nothing to change: the form matches the text, or its fields are empty.")
    return _cs_result(text=new, explain=explain_for(ds, tkey, script, new, plan), plan=plan)


def cs_read(ds, tkey, script, text, plan):
    return _cs_result(fields=read_form(text or ""), explain=explain_for(ds, tkey, script, text or "", plan), plan=plan)


def cs_check(text):
    script, tail = training_command(text or "")
    if not script:
        return "No training command (a line with `*train*.py`) found in the script."
    mu = S.is_musubi_text(text or "")
    troot = S.musubi_root() if mu else S.sd_root()
    py, sp = troot / "venv" / "bin" / "python", S.trainer_script_path(script, mu)
    if not py.is_file():
        return f"venv python not found: `{py}`"
    if not sp.is_file():
        return (f"`{script}` is not in `{sp.parent}`: update musubi-tuner (git pull) or check its folder in Settings › Trainers + GPU."
                if mu else f"`{script}` is not in `{troot}`: your sd-scripts version does not support this architecture.")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    try:
        r = subprocess.run([str(py), str(sp), "--help"], cwd=str(troot), capture_output=True, text=True,
                           timeout=240, env=env)
    except subprocess.TimeoutExpired:
        return "`--help` timed out after 4 minutes."
    helptxt = (r.stdout or "") + (r.stderr or "")
    known = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", helptxt))
    if len(known) < 20:
        return "Could not read the option list from `--help`:\n\n```\n" + helptxt[-800:] + "\n```"
    used = []
    for fl in re.findall(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9_-]*)", tail):
        if fl not in used:
            used.append(fl)
    bad = [x for x in used if x not in known]
    if not bad:
        return f"All {len(used)} options in the training command are accepted by `{script}` in your {'musubi-tuner' if mu else 'sd-scripts'}."
    out = [f"{len(bad)} of {len(used)} options are **not** known to `{script}` in your {'musubi-tuner' if mu else 'sd-scripts'}:"]
    for b in bad:
        close = difflib.get_close_matches(b, known, n=2, cutoff=0.7)
        out.append(f"- `{b}`" + (f"  (did you mean {', '.join('`' + c + '`' for c in close)}?)" if close else ""))
    return "\n".join(out)


def _folder_note(notes):
    return (" Folders: " + "; ".join(notes) + ".") if notes else ""


def cs_save_over(ds, script, text, tkey, plan, makedirs):
    if script in (None, "", NEW) or not Path(script).is_file():
        gr.Warning("No existing launch script is selected: use Save as new.")
        return "No existing launch script is selected."
    p = Path(script)
    text = S.fix_flags(text or "")[0]
    shutil.copy2(p, p.with_name(p.name + ".bak"))
    p.write_text((text or "").rstrip("\n") + "\n", encoding="utf-8")
    msg = f"**Saved `{p.name}`** (previous version: `{p.name}.bak`). The queue always runs the saved file."
    if has_samples(text):                     # create the prompts file if the script points at a missing one
        pp = prompts_path_from_text(text, S.lora_path(ds))
        if pp and not pp.exists() and _inside(pp, S.lora_path(ds)):
            try:
                _, presets = merged_presets()
                pp.write_text(build_prompts(merged_presets()[0], with_virtual_row(dataset_stats(ds)), presets[tkey])[1], encoding="utf-8")
                msg += f" Created the prompts file `{pp.name}`."
            except Exception:
                pass
    notes = ensure_folders(ds, text, plan, tkey) if makedirs else []
    gr.Info(f"Saved {p.name}")
    return msg + _folder_note(notes)


def _bump_if_taken(lora: Path, text: str) -> tuple[str, str]:
    """A new script must not train under a version that already exists: move its output name to the next free one."""
    out_dir, out_name = S.script_args(text, lora)
    base, n = _ver_of(out_name or "")
    if not base or not n:
        return text, ""
    _, nxt = next_script_version(lora, text, lora)
    if n >= nxt:
        return text, ""
    return set_script_version(text, base, nxt), f" (output name moved up to **{base}V{nxt}**: V{n} already exists)"


def cs_save_new(ds, newname, text, prompts_path, prompts_text, tkey, script, plan, makedirs):
    if not ds:
        return _cs_result(status="Pick a LoRA folder.", plan=plan) + [gr.update()]
    name = (newname or "").strip()
    if name and not name.endswith(".sh"):
        name += ".sh"
    if not name or not re.fullmatch(r"[\w][\w .\-]*\.sh", name):
        gr.Warning("Enter a file name like MyCharacterFLUXtraining.sh (letters, digits, space, _ - . only).")
        return _cs_result(status="Invalid file name.", plan=plan) + [gr.update()]
    dest = S.lora_path(ds) / name
    if dest.exists():
        gr.Warning(f"{name} already exists. Pick another name, or select it and use Save (overwrite).")
        return _cs_result(status=f"`{name}` already exists. Choose another name.", plan=plan) + [gr.update()]
    text, vnote = _bump_if_taken(S.lora_path(ds), S.fix_flags(text or "")[0])
    if (plan or {}).get("rename") and (plan or {}).get("repeats"):
        r = _auto_rename(ds, plan["repeats"][0])
        if r:
            vnote += f" {r}."
    dest.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    dest.chmod(dest.stat().st_mode | 0o755)
    msg = f"**Saved new launch script `{name}`**" + vnote
    pp = Path(os.path.expanduser(prompts_path)) if (prompts_path or "").strip() else None
    if pp and has_samples(text) and (prompts_text or "").strip() and not pp.exists():
        pp.write_text(prompts_text.rstrip("\n") + "\n", encoding="utf-8")
        msg += f" and the prompts file `{pp.name}`"
    elif not has_samples(text):
        msg += " (no sample images, so no prompts file)"
    msg += "."
    notes = ensure_folders(ds, text, plan, tkey) if makedirs else []
    msg += _folder_note(notes)
    moved = archive_old_scripts(S.lora_path(ds), keep=dest)
    if moved:
        msg += (f" The older script{'s' if len(moved) > 1 else ''} of this type ({', '.join(f'`{m}`' for m in moved)}) "
                f"moved to `{OLD_RUNS}/`.")
    tidy = S.script_saved_hook(ds)
    if tidy:
        msg += f" {tidy[:1].upper()}{tidy[1:]}."
    msg += " Add it to the queue in Training › Run."
    gr.Info(f"Saved {name}")
    res = _cs_result(script=gr.update(choices=cs_script_choices(ds), value=str(dest)), save_over=gr.update(visible=True),
                     status=msg, plan={})
    return res + [gr.update(choices=S.list_scripts(), value=str(dest))]
