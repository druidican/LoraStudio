# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Which save point of a finished LoRA to try first, from its TensorBoard log (avr_loss per epoch).

Per epoch the loss is the trainer's own epoch average ("loss/epoch") when it logged one, else the mean of the
per-step loss over the second half of that epoch. Among the epochs that have a saved file, the advice is the
EARLIEST one within 1 % of the lowest: a longer run that only flattens does not win by a hair. It also says when the
loss rose at the end (too long?), went NaN, or was still falling (the final one, maybe train longer).

Loss is only a rough guide for image LoRAs: the advice is "try this one first", followed by a few test pictures.
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import math
import re
from pathlib import Path

FLAT = 0.01          # within 1 % of the lowest = as good
ROSE = 0.03          # final more than 3 % above the lowest = the loss rose at the end
FALLING = 0.02       # last epoch still 2 % lower than the one before = still falling
EPOCH_FILE = re.compile(r"^(?P<name>.+)-(?P<ep>\d{6})$")


def _tb_scalars(log_dir: Path, name: str) -> dict:
    """Scalars of the newest TensorBoard run in log_dir that belongs to `name` (else the newest run)."""
    import lora_library as LL
    if not log_dir or not Path(log_dir).is_dir():
        return {}
    events = sorted(Path(log_dir).rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not events:
        return {}
    low = (name or "").lower()
    mine = [e for e in events if low and any(part.lower().startswith(low.rstrip("_-")) for part in e.relative_to(log_dir).parts[:-1])]
    pick = mine or events
    run_dir = pick[-1].parent
    out: dict = {}
    for ev in run_dir.glob("events.out.tfevents.*"):
        for tag, pts in LL.read_tb_scalars(ev).items():
            out.setdefault(tag, []).extend(pts)
    for v in out.values():
        v.sort()
    return out


def _saved_epochs(model_dir: Path, name: str) -> tuple[dict, int | None]:
    """{epoch: file} of the epoch files, and the final file's epoch (from its metadata, else None)."""
    files, final_ep = {}, None
    if not model_dir or not Path(model_dir).is_dir():
        return files, final_ep
    for f in Path(model_dir).glob("*.safetensors"):
        m = EPOCH_FILE.match(f.stem)
        if m and m.group("name").lower() == name.lower():
            files[int(m.group("ep"))] = f
        elif f.stem.lower() == name.lower():
            try:
                import lora_library as LL
                meta = LL.read_safetensors_meta(f)
                final_ep = int(meta.get("ss_epoch") or meta.get("ss_num_epochs") or 0) or None
            except Exception:
                final_ep = None
            files["final"] = f
    return files, final_ep


def per_epoch(scalars: dict, epochs: int | None) -> dict:
    """{epoch: loss}."""
    if scalars.get("loss/epoch"):
        return {int(s): v for s, v in scalars["loss/epoch"] if s > 0}
    tag = next((t for t in ("loss/average", "loss/current") if scalars.get(t)), None)
    if not tag or not epochs:
        return {}
    pts = scalars[tag]
    last = max(s for s, _ in pts) or 1
    per = last / epochs
    out = {}
    for e in range(1, epochs + 1):
        lo, hi = (e - 0.5) * per, e * per
        vals = [v for s, v in pts if lo < s <= hi]
        if vals:
            out[e] = sum(vals) / len(vals)
    return out


def advice(model_dir, log_dir, name: str) -> dict | None:
    """{"best", "final", "short", "text", "points": [(epoch, loss)]} or None when there is nothing to go on."""
    if not name:
        return None
    files, final_ep = _saved_epochs(Path(model_dir) if model_dir else None, name)
    if not files:
        return None
    eps = sorted(e for e in files if e != "final")
    total = final_ep or ((max(eps) + (eps[1] - eps[0] if len(eps) > 1 else 1)) if eps else None)
    loss = per_epoch(_tb_scalars(Path(log_dir) if log_dir else None, name), total)
    if not loss:
        return None
    saved = set(eps) | ({total} if "final" in files and total else set())
    pts = sorted(loss.items())
    nan_at = next((e for e, v in pts if not math.isfinite(v)), None)
    cands = [(e, v) for e, v in pts if e in saved and math.isfinite(v) and (nan_at is None or e < nan_at)]
    if not cands:
        return None
    low = min(v for _, v in cands)
    best = next(e for e, v in cands if v <= low * (1 + FLAT))
    final = total if total in saved else max(saved)
    final_v = loss.get(final)
    lab = lambda e: f"the final ({e})" if e == final else f"epoch {e}"
    if nan_at is not None:
        why = f"the loss went NaN at epoch {nan_at}: use a save point before it"
    elif final_v is not None and math.isfinite(final_v) and best != final and final_v > low * (1 + ROSE):
        why = f"the loss rose after epoch {best} (trained too long?)"
    elif best == final:
        prev = [v for e, v in pts if e < final and math.isfinite(v)]
        why = ("still falling at the end: maybe train longer" if prev and (prev[-1] - low) / prev[-1] > FALLING
               else "the lowest loss is at the end")
    else:
        why = f"flat after epoch {best}: within 1 % of the lowest"
    then = f", then {lab(final)}" if best != final else ""
    return {"best": best, "final": final, "points": pts,
            "short": f"try {lab(best)} first{then}",
            "text": f"Try **{lab(best)}** first{then}: {why}.",
            "file": str(files.get(best) or files.get("final") or "")}


def for_script(script: str) -> dict | None:
    """Advice for a launch script's newest run (its model and log folders, its output name)."""
    import sys
    sg = sys.modules.get("sdgui")
    if sg is None or not script or not Path(script).is_file():
        return None
    model, log = sg.script_paths(script)
    text = Path(script).read_text(encoding="utf-8", errors="ignore")
    name = sg.script_arg(text, "output_name", sg.script_home(script))   # the dataset folder, also for a script in _old_runs/
    if not name:
        m = re.search(r'^OUTPUT_NAME="?([^"\n]+)"?', text, re.M)
        name = m.group(1) if m else ""
    return advice(model, log, name)
