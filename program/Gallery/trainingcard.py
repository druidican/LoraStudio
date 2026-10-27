# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The training card of a LoRA: what it was trained with, in separate fields (Collection › LoRA page).

Read from what is there, the most exact source first:
  1. the LoRA file's own training metadata (sd-scripts writes ss_* keys: what really ran),
  2. its launch script (the one whose --output_name is this LoRA, also in _old_runs/),
  3. the repeat folders of the dataset (pictures and repeats when the file does not say),
  4. the run log in userdata/gui_logs/ (training time, final loss, GPU) and the TensorBoard log (save point to try
     first, via savepoint.py).

The card is stored in the dataset copy of the LoRA Manager file (<dataset>/LoraManager/<lora>.metadata.json) under
"lorastudio": {"training": {...}, "scanned": {...}}. "scanned" holds the LoRA file's size and time: a version that was
scanned is not read again until its file changes (a new version is a new file, so it is scanned). Scan all LoRAs
(Collection › Maintenance) fills every card; a LoRA page shows a card that is missing at once.

LoRA Manager's own fields are filled from it too: the version name (civitai.name, LoRA Studio knows it best), the base
model when empty and the clip skip (usage_tips; 1 when nothing says otherwise). The trigger words are left to the
sync: LoRA Manager's when LoRA Studio has none, the dataset's trigger word when neither has any.
"""
from __future__ import annotations
__version__ = "1.261026z"  # LoRA Studio version, see version.py

import json
import math
import re
import sys
import time
from pathlib import Path

CARD_VERSION = 2                       # raise when the fields change: every card is filled again once

GROUPS = ("Dataset", "Schedule", "Network and optimizer", "Result")
FIELDS = [   # (group, key, label)
    ("Dataset", "dataset", "Dataset"), ("Dataset", "lora_type", "LoRA type"), ("Dataset", "version", "Version"),
    ("Dataset", "trigger", "Trigger words"), ("Dataset", "description", "Character description"),
    ("Dataset", "pictures", "Pictures"),
    ("Dataset", "repeats", "Repeat folders"), ("Dataset", "captions", "Captions"),
    ("Schedule", "epochs", "Epochs"), ("Schedule", "steps", "Steps"), ("Schedule", "steps_calc", "How the steps add up"),
    ("Schedule", "batch", "Batch size"), ("Schedule", "grad_acc", "Gradient accumulation"),
    ("Schedule", "resolution", "Resolution"),
    ("Network and optimizer", "base_model", "Base model"), ("Network and optimizer", "trainer", "Trainer"),
    ("Network and optimizer", "network", "Network (dim / alpha)"), ("Network and optimizer", "lr", "Learning rate"),
    ("Network and optimizer", "optimizer", "Optimizer"), ("Network and optimizer", "scheduler", "Scheduler"),
    ("Network and optimizer", "clip_skip", "Clip skip"), ("Network and optimizer", "noise_offset", "Noise offset"),
    ("Network and optimizer", "min_snr", "Min SNR gamma"), ("Network and optimizer", "precision", "Precision"),
    ("Network and optimizer", "seed", "Seed"),
    ("Result", "trained_on", "Trained on"), ("Result", "duration", "Training time"),
    ("Result", "final_loss", "Final loss"), ("Result", "best_epoch", "Save point to try first"), ("Result", "gpu", "GPU"),
]
LABEL = {k: lab for _, k, lab in FIELDS}
OLD_RUNS = "_old_runs"
REPEAT_RE = re.compile(r"^(\d+)_(.+)$")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# the long text older versions wrote into Notes (now in the fields): "Trained on ... . 10 epochs, ... ."
AUTO_NOTE = re.compile(r"^Trained( on \d{4}-\d{2}-\d{2})?( from the dataset [^\n.]+)?( \(\d+ pictures with repeats\))?\..*$",
                       re.S)


def _sg():
    return sys.modules.get("sdgui")


def _ok(v) -> bool:
    return v not in (None, "", "None", "none", [], {})


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def fmt_lr(v) -> str:
    f = _num(v)
    if f is None:
        return str(v or "")
    return f"{f:.0e}".replace("e-0", "e-") if 0 < f < 0.01 else f"{f:g}"


def hms(sec) -> str:
    try:
        s = int(float(sec))
    except (TypeError, ValueError):
        return ""
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


# --------------------------------------------------------------------------- the launch script of a LoRA
def _scripts(ds_root: Path) -> list[Path]:
    out = sorted(ds_root.glob("*.sh"))
    if (ds_root / OLD_RUNS).is_dir():
        out += sorted((ds_root / OLD_RUNS).glob("*.sh"))
    return [p for p in out if not p.name.startswith(".gui_run_")]


def find_script(lora) -> tuple[Path | None, str]:
    """(script, text): the one whose --output_name is this LoRA; else a script named like it; else the matched one."""
    sg = _sg()
    if lora.dataset is None:
        return None, ""
    want = {lora.stem.lower(), (lora.output_name or "").lower()}
    named = None
    for p in _scripts(lora.dataset.root):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        home = lora.dataset.root
        out = sg.script_arg(text, "output_name", home) if sg is not None else None
        if not out:
            m = re.search(r'^OUTPUT_NAME="?([^"\n]+)"?', text, re.M)
            out = m.group(1).replace("${CHARACTER_NAME}", home.name) if m else ""
        if out and out.lower() in want:
            return p, text
        if p.stem.lower() in want and named is None:
            named = (p, text)
    if named:
        return named
    if lora.scripts:
        p = Path(lora.scripts[0][0])
        try:
            return p, p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    return None, ""


def _desc(lora) -> str:
    """The character description of the dataset (step 7), '' when none."""
    if lora.dataset is None:
        return ""
    try:
        import dsflow
        return dsflow.description(lora.dataset.name)
    except Exception:                                  # noqa: BLE001
        return ""


def _run_log(script: Path | None, stem: str) -> Path | None:
    """The newest run log of this LoRA in userdata/gui_logs/ (named <time>_<script>.log)."""
    sg = _sg()
    d = getattr(sg, "LOG_DIR", None) if sg is not None else None
    if not d or not Path(d).is_dir():
        return None
    names = {stem.lower()} | ({script.stem.lower()} if script else set())
    logs = [f for f in Path(d).glob("*.log") if re.match(r"^\d{8}_\d{6}_(.+)\.log$", f.name)
            and re.match(r"^\d{8}_\d{6}_(.+)\.log$", f.name).group(1).lower() in names]
    return max(logs, key=lambda f: f.name) if logs else None


def _log_facts(log: Path | None) -> dict:
    out = {}
    if log is None:
        return out
    try:
        size = log.stat().st_size
        with open(log, "rb") as fh:
            head = fh.read(4000).decode("utf-8", "replace")
            fh.seek(max(0, size - 400_000))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return out
    m = re.search(r"^##LS_GPU (.+)$", head, re.M)
    if m:
        out["gpu"] = m.group(1).strip()
    ex = re.findall(r"--- exited with code (-?\d+) ---(?: \(ran ([\d:]+)\))?", tail)
    if ex and ex[-1][1]:
        parts = [int(x) for x in ex[-1][1].split(":")]
        sec = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
        out["duration"] = hms(sec)
    loss = re.findall(r"avr_loss[=:]\s*([0-9.]+(?:e-?\d+)?)", tail)
    if loss:
        out["final_loss"] = loss[-1]
    st = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", log.name)
    if st:
        out["started"] = f"{st.group(1)}-{st.group(2)}-{st.group(3)} {st.group(4)}:{st.group(5)}"
    return out


# --------------------------------------------------------------------------- the card
def _json(meta: dict, key: str, default):
    v = meta.get(key)
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return default
    return v if v is not None else default


def _folders_from_meta(meta: dict) -> list[tuple[str, int, int]]:
    """[(folder, repeats, pictures)] from sd-scripts' ss_dataset_dirs / ss_datasets."""
    out = []
    dd = _json(meta, "ss_dataset_dirs", {})
    if isinstance(dd, dict):
        for name, info in dd.items():
            if isinstance(info, dict):
                out.append((Path(name).name, int(_num(info.get("n_repeats")) or 1), int(_num(info.get("img_count")) or 0)))
    if not out:
        for ds in _json(meta, "ss_datasets", []) or []:
            for sub in (ds.get("subsets") or []) if isinstance(ds, dict) else []:
                if isinstance(sub, dict):
                    out.append((Path(str(sub.get("image_dir") or "")).name or "?", int(_num(sub.get("num_repeats")) or 1),
                                int(_num(sub.get("img_count")) or 0)))
    return [f for f in out if f[2]]


def _folders_now(ds_root: Path) -> list[tuple[str, int, int]]:
    out = []
    for d in sorted(ds_root.iterdir()) if ds_root.is_dir() else []:
        m = REPEAT_RE.match(d.name)
        if d.is_dir() and m:
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
            if n:
                out.append((d.name, int(m.group(1)), n))
    return out


def build(lib, lora) -> dict:
    """{field: text} for every field that is known (unknown ones are left out)."""
    sg = _sg()
    m = lora.meta or {}
    script, text = find_script(lora)
    home = lora.dataset.root if lora.dataset else (script.parent if script else None)

    def arg(*names):
        if not text or sg is None:
            return None
        for n in names:
            v = sg.script_arg(text, n, home)
            if _ok(v):
                return v
        return None

    def pick(meta_key, *script_keys):
        v = m.get(meta_key) if meta_key else None
        return v if _ok(v) else arg(*script_keys)

    c = {}
    if lora.dataset:
        c["dataset"] = lora.dataset.name
    t = sg.script_type(str(script), text) if (sg is not None and script) else ""
    if t:
        c["_tkey"] = t
        try:
            lab = {k: v for v, k in sg.type_choices()}.get(t, t)
            c["lora_type"] = re.sub(r"\s*\[template\]", "", lab)
        except Exception:                              # noqa: BLE001
            c["lora_type"] = t
    elif lora.base and lora.base != "Unknown":
        c["lora_type"] = lora.base
    if lora.version is not None:
        c["version"] = f"V{lora.version}"
    trig = [w for w in (lora.trigger_words or []) if str(w).strip()]
    if not trig and lora.dataset:
        try:
            import dsflow
            tw = (dsflow.meta(lora.dataset.name).get("trigger") or "").strip()
            trig = [tw] if tw else []
        except Exception:                              # noqa: BLE001
            pass
    if trig:
        c["trigger"] = ", ".join(trig)
    desc = _desc(lora)
    if desc:
        c["description"] = desc
    # pictures and repeats: what the file says was trained, else the repeat folders now
    folders = _folders_from_meta(m)
    plan = re.search(r"Plan: (\d+) images x (\d+) repeats, batch (\d+) x accum (\d+)", text or "")
    if not folders and plan and lora.dataset:            # what the script was made for (the folder may be renamed since)
        now = _folders_now(lora.dataset.root)
        name = now[0][0].split("_", 1)[-1] if len(now) == 1 else lora.dataset.name
        folders = [(f"{plan.group(2)}_{name}", int(plan.group(2)), int(plan.group(1)))] if len(now) <= 1 else now
    if not folders and lora.dataset:
        folders = _folders_now(lora.dataset.root)
    total = sum(r * n for _, r, n in folders)
    if folders:
        c["pictures"] = str(sum(n for _, _, n in folders))
        c["repeats"] = " · ".join(f"{f} ({n} pictures × {r})" for f, r, n in folders)
    if lora.dataset:
        try:
            import dsflow
            dm = dsflow.meta(lora.dataset.name)
            purpose = dsflow.PURPOSE_LABEL.get(dm.get("purpose"), dm.get("purpose") or "")
        except Exception:                              # noqa: BLE001
            purpose = ""
        ext = arg("caption_extension") or ".txt"
        c["captions"] = (f"{purpose} · " if purpose else "") + f"{ext} files" + \
                        (" · shuffled, trigger word kept first" if text and "--shuffle_caption" in text else "")
    # schedule
    ep_done, ep_plan = _num(m.get("ss_epoch")), _num(pick("ss_num_epochs", "max_train_epochs"))
    if ep_plan:
        c["epochs"] = f"{ep_done} of {ep_plan}" if ep_done and ep_done != ep_plan else str(ep_plan)
    bs = _num(m.get("ss_total_batch_size") or m.get("ss_batch_size_per_device") or arg("train_batch_size", "batch_size")
              or (plan.group(3) if plan else None)) or 1
    ga = _num(m.get("ss_gradient_accumulation_steps") or arg("gradient_accumulation_steps")
              or (plan.group(4) if plan else None)) or 1
    c["batch"] = str(bs)
    c["grad_acc"] = str(ga)
    st_done, st_plan = _num(m.get("ss_steps")), _num(pick("ss_max_train_steps", "max_train_steps"))
    if total and ep_plan:
        per = math.ceil(total / bs / ga)
        calc = f"{total} pictures with repeats ÷ batch {bs}" + (f" ÷ accumulation {ga}" if ga != 1 else "") + \
               f" = {per} steps per epoch × {ep_plan} epochs = {per * ep_plan} steps"
        c["steps_calc"] = calc
        st_plan = st_plan or per * ep_plan
    if st_plan:
        c["steps"] = f"{st_done} of {st_plan}" if st_done and st_done != st_plan else str(st_plan)
    res = pick("ss_resolution", "resolution")
    if res:
        c["resolution"] = str(res).strip("()").replace(", ", "×").replace(",", "×")
    # network and optimizer
    base = pick("ss_sd_model_name", "pretrained_model_name_or_path", "dit")
    if base:
        c["base_model"] = Path(str(base)).name
    trainer = re.search(r"\b(\w+_train_network\.py|\w+_train\.py)", text or "")
    if trainer:
        c["trainer"] = ("musubi-tuner" if "musubi" in (text or "").lower() else "sd-scripts") + f" ({trainer.group(1)})"
    dim, alpha = pick("ss_network_dim", "network_dim"), pick("ss_network_alpha", "network_alpha")
    if dim:
        c["network"] = f"{_num(dim)} / {_num(alpha) if alpha else '?'}"
    lr = m.get("ss_unet_lr") if _ok(m.get("ss_unet_lr")) else pick("ss_learning_rate", "learning_rate", "unet_lr")
    if lr:
        c["lr"] = fmt_lr(lr)
    opt = pick("ss_optimizer", "optimizer_type")
    if opt:
        c["optimizer"] = str(opt).split(".")[-1].split("(")[0]
    sch = pick("ss_lr_scheduler", "lr_scheduler")
    if sch:
        warm = pick("ss_lr_warmup_steps", "lr_warmup_steps")
        c["scheduler"] = str(sch) + (f" · warmup {warm} steps" if _num(warm) else "")
    cs = pick("ss_clip_skip", "clip_skip")
    if _num(cs):
        c["clip_skip"] = str(_num(cs))
    for key, mk, sk in (("noise_offset", "ss_noise_offset", "noise_offset"), ("min_snr", "ss_min_snr_gamma", "min_snr_gamma"),
                        ("precision", "ss_mixed_precision", "mixed_precision"), ("seed", "ss_seed", "seed")):
        v = pick(mk, sk)
        if not _ok(v) or (key != "seed" and str(v) in ("0", "0.0")):
            continue
        c[key] = str(_num(v) if _num(v) is not None else v)
    # result
    lf = _log_facts(_run_log(script, lora.stem))
    try:
        c["trained_on"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(m["ss_training_started_at"])))
    except (KeyError, TypeError, ValueError):
        if lf.get("started"):
            c["trained_on"] = lf["started"]
    try:
        c["duration"] = hms(float(m["ss_training_finished_at"]) - float(m["ss_training_started_at"]))
    except (KeyError, TypeError, ValueError):
        if lf.get("duration"):
            c["duration"] = lf["duration"]
    good = [v for _, v in (lora.loss or []) if isinstance(v, (int, float)) and math.isfinite(v)]
    if good:
        c["final_loss"] = f"{good[-1]:.4f}"
    elif lf.get("final_loss"):
        c["final_loss"] = lf["final_loss"]
    if lf.get("gpu"):
        c["gpu"] = lf["gpu"]
    if script is not None and sg is not None:
        try:
            import savepoint
            md, ld = sg.script_paths(str(script))
            adv = savepoint.advice(md, ld, lora.output_name or lora.stem)
            if adv:
                c["best_epoch"] = adv["short"][:1].upper() + adv["short"][1:]
        except Exception:                              # noqa: BLE001
            pass
    if script is not None:
        c["_script"] = str(script)
    return {k: v for k, v in c.items() if _ok(v)}


# --------------------------------------------------------------------------- stored in the LoRA Manager file
def file_sig(lora) -> str:
    try:
        st = Path(lora.path).stat()
        return f"{st.st_size}|{int(st.st_mtime)}|{CARD_VERSION}"
    except (OSError, TypeError):
        return ""


def stored(sidecar: dict | None) -> dict:
    return dict(((sidecar or {}).get("lorastudio") or {}).get("training") or {})


def is_scanned(lora) -> bool:
    """Scanned, and the character description has not changed since."""
    if not lora.sidecar:
        return False
    ls = lora.sidecar.get("lorastudio") or {}
    return (ls.get("scanned") or {}).get("sig") == file_sig(lora) and (ls.get("desc_added") or "") == _desc(lora)


def strip_auto_notes(notes: str) -> str:
    """Notes without the long training text older versions wrote there (your own notes stay)."""
    parts = re.split(r"\n\s*\n", notes or "")
    keep = [p for p in parts if not AUTO_NOTE.match(p.strip())]
    return "\n\n".join(keep).strip()


def apply(sidecar: dict, card: dict, sig: str) -> dict:
    """The card into a LoRA Manager sidecar: lorastudio.training, the scanned mark, the version name and the clip
    skip (when none is set), and the old training text taken out of Notes."""
    out = dict(sidecar)
    ls = dict(out.get("lorastudio") or {})
    ls["training"] = {k: v for k, v in card.items()}
    ls["scanned"] = {"sig": sig, "at": time.strftime("%Y-%m-%d %H:%M")}
    out["lorastudio"] = ls
    civ = dict(out.get("civitai") or {})
    if card.get("version"):
        civ["name"] = card["version"]                   # LoRA Studio knows the version: it wins (LoRA Manager shows N/A)
    desc = card.get("description") or ""
    if desc != (ls.get("desc_added") or ""):            # a new or changed description: its words join the trigger words
        words = list(civ.get("trainedWords") or [])
        if not words and card.get("trigger"):
            words = [w.strip() for w in card["trigger"].split(",") if w.strip()]
        old = {w.strip().lower() for w in (ls.get("desc_added") or "").split(",") if w.strip()}
        words = [w for w in words if w.strip().lower() not in old]          # the old description's words go
        for w in [x.strip() for x in desc.split(",") if x.strip()]:
            if w.lower() not in {x.lower() for x in words}:
                words.append(w)
        civ["trainedWords"] = words
        ls["desc_added"] = desc
        out["lorastudio"] = ls
    out["civitai"] = civ
    tips = usage_tips(out)
    if not _num(tips.get("clip_skip")):
        tips["clip_skip"] = int(_num(card.get("clip_skip")) or 1)   # none known anywhere: the default, 1
    out["usage_tips"] = json.dumps(tips)
    out["notes"] = strip_auto_notes(out.get("notes") or "")
    return out


def usage_tips(sc: dict) -> dict:
    raw = (sc or {}).get("usage_tips") or "{}"
    try:
        t = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return t if isinstance(t, dict) else {}
    except (ValueError, TypeError):
        return {}


def scan_one(lib, lora, force=False) -> str:
    """'' when the card was already up to date, else 'scanned' (the dataset copy of the LoRA Manager file is written)."""
    import lora_library as LL
    if lora.dataset is None:
        return "no dataset"
    if not force and is_scanned(lora):
        return ""
    card = build(lib, lora)
    # no file yet: only what the card brings, so a later sync takes everything else from LoRA Manager's file
    base = LL.copy.deepcopy(lora.sidecar) if lora.sidecar else {"file_name": lora.stem}
    if lora.base and lora.base != "Unknown" and not base.get("base_model"):
        base["base_model"] = lora.base
    new = apply(base, card, file_sig(lora))
    path = lora.sidecar_path or (lora.dataset.archive_dir / (lora.stem + ".metadata.json"))
    LL.write_sidecar(path, new)
    lora.sidecar, lora.sidecar_path = new, path
    return "scanned"


def scan_all(lib, force=False, progress=None) -> str:
    lib.ensure_loaded()
    mine = [l for l in lib.loras.values() if l.dataset is not None and l.final]
    done, same, errs = 0, 0, []
    for i, l in enumerate(mine, 1):
        if progress is not None:
            progress(i / max(1, len(mine)), desc=f"Reading {l.stem}")
        try:
            r = scan_one(lib, l, force)
        except Exception as e:                         # noqa: BLE001
            errs.append(f"{l.stem}: {e}")
            continue
        done += r == "scanned"
        same += r == ""
    lines = [f"Read **{done}** LoRA{'s' if done != 1 else ''} of your datasets into their training cards; "
             f"{same} were already scanned (a version is read again only when its file changes)."]
    if errs:
        lines.append("Could not read: " + "; ".join(errs[:8]))
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- the LoRA page
def card_html(card: dict, esc) -> str:
    if not card:
        return "<p class='ll-muted'>No training data found for this LoRA.</p>"
    blocks = []
    for g in GROUPS:
        rows = [(lab, card[k]) for gg, k, lab in FIELDS if gg == g and card.get(k)]
        if rows:
            dl = "".join(f"<dt>{esc(a)}</dt><dd>{esc(str(b))}</dd>" for a, b in rows)
            blocks.append(f"<div class='ll-tcard'><div class='rs-label'>{esc(g.upper())}</div><dl class='ll-ddl'>{dl}</dl></div>")
    src = card.get("_script")
    foot = f"<p class='ll-muted ll-mono' style='margin-top:6px'>from {esc(Path(src).name)}</p>" if src else ""
    return "<div class='ll-tgrid'>" + "".join(blocks) + "</div>" + foot
