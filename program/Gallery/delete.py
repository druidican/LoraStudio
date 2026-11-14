# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Collection › Delete: what belongs to a LoRA (every version), and moving the parts you tick to the desktop Trash.

A LoRA here is one card of the Collection (a dataset + LoRA type, all its versions). Per version:
  lora     the file in your LoRA folder (what ComfyUI uses) and its LoRA Manager files (.metadata.json, preview,
           .civitai.info, example pictures)
  model    the dataset's copy in model/, its save points (epoch files), the resume states and the dataset's
           LoraManager/ files
  runs     the run logs, the test pictures (model/compare/<name>), the sample pictures and the TensorBoard logs
  script   the launch script
and, for the LoRA as a whole, the dataset folder (pictures, captions, everything in it). Nothing is deleted for
good: everything goes to the Trash, so it can be put back from the file manager.
"""
from __future__ import annotations
__version__ = "1.141126b"  # LoRA Studio version, see version.py

import re
from pathlib import Path

import lora_library as LL

LIB = LL.LIB
PARTS = [("lora", "LoRA folder: the file ComfyUI uses and its LoRA Manager files"),
         ("model", "Dataset's model/ folder: the final file, save points and resume states"),
         ("runs", "Runs: logs, test pictures, sample pictures, TensorBoard logs"),
         ("script", "Launch script")]


def _expand(gids) -> list[str]:
    """Card ids to version groups (a Pictures tile 'ds:<name>' stands for every LoRA type of that dataset)."""
    out = []
    for g in gids or []:
        if g.startswith("ds:"):
            name = g[3:].lower()
            out += [k for k, m in LIB.groups.items() if m and m[-1].dataset is not None and m[-1].dataset.name.lower() == name]
        elif g in LIB.groups:
            out.append(g)
    return list(dict.fromkeys(out))


def _uniq(paths) -> list[Path]:
    seen, out = set(), []
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        k = str(p)
        if k not in seen and (p.exists() or p.is_symlink()):
            seen.add(k)
            out.append(p)
    return out


def _runs_by_name() -> dict:
    """{output name: [run dicts]} of every run in gui_logs/."""
    import training_ui as T
    by = {}
    for r in T.runs(5000):
        info = r.get("info") or {}
        if info.get("out_name"):
            by.setdefault(info["out_name"], []).append(r)
    return by


def _log_dirs(script: Path, stem: str) -> list[Path]:
    """TensorBoard folders of a run: <logging_dir>/<output name>_<time>."""
    try:
        text = script.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    m = re.search(r'--logging_dir[= ]+"?([^"\s]+)"?', text)
    if not m:
        return []
    d = Path(m.group(1)).expanduser()
    return [p for p in d.glob(stem + "_*") if p.is_dir()] if d.is_dir() else []


def version_parts(lora, runs_by: dict) -> dict:
    """{part: [paths]} of one version (only what exists)."""
    import sdgui as sg
    stem = lora.stem
    lora_side = []
    if lora.lora_path is not None:
        lora_side.append(lora.lora_path)
        lora_side += [f for f in lora.lora_path.parent.glob(stem + ".*") if f != lora.lora_path]
    lora_side += list(lora.examples or [])
    model = [lora.dataset_path] + [p for _, p in (lora.epoch_files or [])]
    if lora.dataset is not None:
        model += list(lora.dataset.archive_dir.glob(stem + ".*"))
    runs, scripts = [], [p for p, _ in (lora.scripts or [])]
    for r in runs_by.get(stem, []):
        runs.append(sg.LOG_DIR / r["log"])
        info = r["info"]
        out_dir = info["out_dir"]
        runs.append(out_dir / "compare" / stem)
        if out_dir.is_dir():
            model += [st for st in sg.scan_states(out_dir) if st.name.startswith(stem + "-") or st.name == stem + "-state"]
        if r.get("script"):
            scripts.append(Path(r["script"]))
    runs += list(lora.samples or [])
    for s in scripts:
        runs += _log_dirs(Path(s), stem)
    return {"lora": _uniq(lora_side), "model": _uniq(model), "runs": _uniq(runs), "script": _uniq(scripts)}


def plan(gids) -> list[dict]:
    """What the dialog shows: one entry per LoRA (card) with its versions and parts, sizes in bytes."""
    HK = LL.HK
    runs_by = _runs_by_name()
    picked = _expand(gids)
    out = []
    for gid in picked:
        members = LIB.groups.get(gid) or []
        if not members:
            continue
        ds = next((m.dataset for m in members if m.dataset is not None), None)
        vers = []
        for m in members:
            parts = version_parts(m, runs_by)
            vers.append({"key": m.key, "stem": m.stem, "label": m.vlabel, "parts": {k: [str(p) for p in v] for k, v in parts.items()},
                         "sizes": {k: sum(HK.size_of(p) for p in v) for k, v in parts.items()}})
        others = []
        if ds is not None:
            others = sorted({LL.group_title(m) for g, m in LIB.groups.items()
                             if g not in picked and m and m[-1].dataset is not None and m[-1].dataset.root == ds.root})
        out.append({"gid": gid, "title": LL.group_title(members), "versions": vers,
                    "dataset": str(ds.root) if ds is not None else "", "dataset_name": ds.name if ds is not None else "",
                    "dataset_size": HK.size_of(ds.root) if ds is not None else 0, "others": others})
    return out


def run(plan_: list[dict], picks: dict) -> str:
    """Move the ticked parts to the Trash. picks = {gid: {"versions": [keys], "parts": [part names], "dataset": bool}}."""
    import sdgui as sg
    HK = LL.HK
    busy = LL.busy_datasets()
    queued = {str(Path(it.get("path", ""))) for it in sg.QUEUE["items"] if it.get("state") == "queued"}
    todo, skipped, gone_ds, done_gids = [], [], [], []
    ds_out = {e["dataset"] for e in plan_ if e["dataset"] and (picks.get(e["gid"]) or {}).get("dataset")}
    for e in plan_:
        p = picks.get(e["gid"]) or {}
        ds_root = Path(e["dataset"]) if e["dataset"] else None
        if ds_root is not None and any(ds_root == b or ds_root in b.parents or b in ds_root.parents for b in busy):
            skipped.append(f"{e['title']}: a training run is using its dataset")
            continue
        if e["dataset"] in ds_out and e["dataset_name"] not in gone_ds:
            todo.append(ds_root)
            gone_ds.append(e["dataset_name"])
        for v in e["versions"]:
            if v["key"] not in (p.get("versions") or []):
                continue
            for part in p.get("parts") or []:
                for f in v["parts"].get(part, []):
                    if part == "script" and f in queued:
                        skipped.append(f"{Path(f).name}: waiting in the queue")
                        continue
                    if ds_root is not None and e["dataset"] in ds_out and Path(f).is_relative_to(ds_root):
                        continue                         # goes with the dataset folder
                    todo.append(Path(f))
        if e["dataset"] in ds_out or set(p.get("versions") or []) >= {v["key"] for v in e["versions"]}:
            done_gids.append(e["gid"])
    todo = _uniq(todo)
    if not todo:
        return "Nothing ticked." + ("\n\nSkipped: " + "; ".join(skipped) if skipped else "")
    size = sum(HK.size_of(p) for p in todo)
    moved, err = HK.trash(todo)
    _forget(done_gids, gone_ds)
    msg = f"✔ Moved {moved} file(s) / folder(s), {HK.human(size)}, to the Trash. Put them back from the file manager's Trash if needed."
    if skipped:
        msg += "\n\nSkipped: " + "; ".join(skipped)
    if err:
        msg += "\n\nCould not move: " + "; ".join(err[:5])
    return msg


def _forget(gids: list[str], datasets: list[str]):
    """Data LoRA Studio keeps about what is gone: Completed marks and the recent datasets on Home."""
    try:
        if gids:
            LL.mark_completed(gids, False)
    except Exception:                                    # noqa: BLE001
        pass
    if datasets:
        try:
            import home
            home.forget(datasets)
        except Exception:                                # noqa: BLE001
            pass
