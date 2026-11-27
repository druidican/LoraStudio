# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Review and export (a dataset's page › Review and export): pick the save point of a run by its test pictures, check
what goes to LoRA Manager, and export it.

Export: the chosen save point becomes the run's final file (<name>.safetensors: the name never has the epoch in it;
the old final file becomes its own save point <name>-0000NN, or goes to the Trash with Clean up), is copied into your
LoRA folder, gets LoRA Manager's data next to it (<name>.metadata.json: trigger words, base model, the training and
your note in the notes, usage tips with the strength, your rating), its preview (<name>.preview.png) and the whole row
of test pictures of that save point as LoRA Manager example pictures (<example_images_path>/<sha256>/custom_*.png,
listed in civitai.customImages). The dataset's own LoRA card gets the same data; the version is marked Completed.
Every step is checked at the end. The page is program/UI/ds_review.py.
"""
from __future__ import annotations
__version__ = "1.271126b"  # LoRA Studio version, see version.py

import json
import re
import shutil
import time
from pathlib import Path


def _T():
    import training_ui
    return training_ui


def _CT():
    import comfytest
    return comfytest


def _LL():
    import lora_library
    return lora_library


# --------------------------------------------------------------------------- what the page shows
def runs_of(ds: str) -> list[dict]:
    """The finished runs of a dataset that have test pictures (newest first): [{"log", "name", "started"}]."""
    T, CT = _T(), _CT()
    out = []
    for r in T.runs():
        info = r.get("info")
        if not info or info["dataset"] != ds or r.get("result") != "done":
            continue
        if Path(info["path"]).parent.name == "_compare":
            continue
        if not CT.results(info):
            continue
        out.append({"log": r["log"], "name": T.run_name(info), "started": r["started"]})
    return out


def view(log: str) -> dict | None:
    """Everything the page shows for one run, or None when its launch script is gone."""
    T, CT = _T(), _CT()
    facts = CT._run_facts(log)
    if not facts:
        return None
    info, final = facts["info"], facts["final"]
    grid = CT.results(info) or {}
    a = T._advice(facts["script"])
    pts = dict(a["points"]) if a else {}
    best = facts["best"]
    files = grid.get("files") or {}
    rows = []
    for lab in grid.get("labels") or []:
        f = files.get(lab) or ""
        m = re.match(r"epoch (\d+)", lab)
        ep = int(m.group(1)) if m else (a["final"] if a and isinstance(a.get("final"), int) else None)
        loss = pts.get(ep) if ep is not None else None
        rows.append({"label": lab, "epoch": ep, "loss": loss, "file": f, "ok": bool(f) and Path(f).is_file(),
                     "best": best is not None and lab == f"epoch {best}", "final": lab == "final",
                     "pics": [p for p in (grid.get("pics") or {}).get(lab) or []]})
    kept = grid.get("kept")
    pick = kept if kept in files else next((r["label"] for r in rows if r["best"] and r["ok"]), None) \
        or next((r["label"] for r in rows if r["final"] and r["ok"]), None) or next((r["label"] for r in rows if r["ok"]), "")
    f = T._training_facts(facts["script"])
    dest = ""
    try:
        import sdgui as sg
        dest = str(sg.HK.copy_target(info["out_name"], sg.lora_dirs_setting(), info["type"])["dir"] or "")
    except Exception:                                    # noqa: BLE001
        pass
    lo = T._lora_of(info)
    sc = (lo.sidecar or {}) if lo is not None else {}
    trig = ", ".join((sc.get("civitai") or {}).get("trainedWords") or []) or f["trigger"]
    try:
        tips = json.loads(sc.get("usage_tips") or "{}")
    except ValueError:
        tips = {}
    strength = tips.get("strength") or CT.lora_strength(info["tkey"])
    started = next((r["started"] for r in T.runs() if r["log"] == log), "")
    chosen = Path(files.get(pick) or "")
    others = [p for p in list(facts["eps"].values()) + ([final] if final else [])
              if p.is_file() and (not chosen.name or p.resolve() != chosen.resolve())]
    size = sum(p.stat().st_size for p in others) + _states_size(info)
    return {"log": log, "name": T.run_name(info), "out": info["out_name"], "dataset": info["dataset"],
            "type": info["type"], "tkey": info["tkey"], "started": started, "at": grid.get("at", ""), "rows": rows,
            "pick": pick, "kept": kept, "dest": dest, "trigger": trig, "strength": float(strength),
            "base": _base_of(final, info), "training": _training_text(f, info),
            "rating": T.rating_of(info), "note": "", "cleanup_size": size,
            "completed": _completed(lo), "lm_root": lm_root()}


def _states_size(info) -> int:
    import sdgui as sg
    return sum(sg._size(d) for d in sg.scan_states(info["out_dir"]) if d.name.startswith(info["out_name"] + "-"))


def _base_of(final, info) -> str:
    LL = _LL()
    try:
        return LL.detect_base(LL.read_safetensors_meta(Path(final)), None, Path(final), LL.LIB.settings["bases"])
    except Exception:                                    # noqa: BLE001
        return info["type"]


def _training_text(f: dict, info) -> str:
    bits = [f"{f['epochs']} epochs" if f["epochs"] else "", f"lr {f['lr']}" if f["lr"] else "",
            f"rank {f['dim']}/{f['alpha']}" if f["dim"] else "",
            f"{f['pics']} pictures × {f['repeats']}" if f["pics"] else "", f"resolution {f['res']}" if f["res"] else "",
            f["base_file"]]
    return " · ".join(b for b in bits if b)


def _completed(lo) -> bool:
    LL = _LL()
    if lo is None or LL.COMPLETED is None:
        return False
    ds, t = LL.done_key(lo)
    return bool(ds) and bool(LL.COMPLETED.info(ds, t, version=LL.COMPLETED.version_of(lo.stem)).get("done"))


def lm_root() -> str:
    """LoRA Manager's example picture folder (its setting example_images_path, or the one set in Collection ›
    Maintenance), '' when unknown."""
    LL = _LL()
    own = str((LL.LIB.data.get("lm_examples_dir") or "")).strip()
    if own and Path(own).expanduser().is_dir():
        return str(Path(own).expanduser())
    LL._LM_ROOTS.clear()
    roots = LL.lm_example_roots(LL.LIB)
    return roots[0] if roots else ""


# --------------------------------------------------------------------------- export
def export(log: str, label: str, form: dict, say=None) -> list[tuple[str, str, str]]:
    """Export one save point of a run to LoRA Manager. form: dest, trigger, strength, preview, note, rating, done,
    cleanup. Returns the steps [(state ✔ / ! / ❌, title, detail)]; the last one is the check."""
    import libsync
    say = say or (lambda t: None)
    T, CT, LL = _T(), _CT(), _LL()
    import sdgui as sg
    steps: list = []
    facts = CT._run_facts(log)
    if not facts:
        return [("❌", "The launch script of this run is gone", "")]
    if sg.RUN.running() and sg.RUN.logfile and Path(sg.RUN.logfile).name == log:
        return [("❌", "This run is still training", "")]
    info = facts["info"]
    grid = CT.results(info) or {}
    src = Path((grid.get("files") or {}).get(label) or "")
    if not src.is_file():
        return [("❌", "That save point's file is gone", "Kept or cleaned up already? Make the test pictures again.")]
    dest_dir = Path(str(form.get("dest") or "").strip()).expanduser()
    if not str(form.get("dest") or "").strip():
        return [("❌", "No LoRA folder", "Enter the folder to copy to (Settings › Folders sets the LoRA folders).")]
    out = info["out_name"]
    final = info["out_dir"] / f"{out}.safetensors"
    pics = [p for p in (grid.get("pics") or {}).get(label) or [] if p and Path(p).is_file()]
    prompts = grid.get("prompts") or []

    # 1. the chosen save point becomes the run's final file
    say("the chosen save point becomes the final file")
    if src.resolve() != final.resolve():
        old_lab = next((k for k, v in (grid.get("files") or {}).items() if v and Path(v).resolve() == final.resolve()), "")
        if final.exists():
            epochs = T._training_facts(facts["script"])["epochs"]
            keep_as = info["out_dir"] / f"{out}-{int(epochs):06d}.safetensors" if str(epochs).isdigit() else None
            if form.get("cleanup") or keep_as is None or keep_as.exists():
                n, failed = sg.HK.trash([final])
                if final.exists():
                    return [("❌", "The old final file could not be moved to the Trash", "; ".join(failed[:2]))]
                if old_lab:
                    grid["files"][old_lab] = ""
            else:
                final.rename(keep_as)
                if old_lab:
                    grid["files"][old_lab] = str(keep_as)
        src.rename(final)
        grid["files"][label] = str(final)
        steps.append(("✔", f"{label} is the run's final file", f"{final.name} (no epoch in the name)"))
    else:
        steps.append(("✔", f"{label} is the run's final file", final.name))
    grid["kept"] = label
    if form.get("cleanup"):
        _, eps = T._final_and_epochs(info)
        trash = [p for p in eps.values() if p.exists() and p.resolve() != final.resolve()]
        trash += [d for d in sg.scan_states(info["out_dir"]) if d.name.startswith(out + "-")]
        size = sum(sg._size(p) for p in trash)
        n, failed = sg.HK.trash(trash)
        for k, v in list((grid.get("files") or {}).items()):
            if k != label and v and not Path(v).exists():
                grid["files"][k] = ""
        steps.append(("✔" if not failed else "!", f"{n} other save point(s) and states cleaned up",
                      f"{T._gb(size)} moved to the Trash" + (f"; could not move: {'; '.join(failed[:2])}" if failed else "")))
    CT._save_grid(CT.compare_dir(info), grid)
    T._LEFT.clear()

    # 2. copied into the LoRA folder
    say("copying into your LoRA folder")
    ok, msg = sg.HK.copy_lora(final, dest_dir, True)
    if not ok:
        steps.append(("❌", "Not copied into your LoRA folder", msg))
        return steps + [check_step(steps)]
    copy = dest_dir / final.name
    steps.append(("✔", "LoRA copied", f"{copy} · {T._gb(copy.stat().st_size)}"))

    # 3. LoRA Manager data, preview, example pictures (the dataset's card first, then the copy's)
    say("writing LoRA Manager's data")
    import lorapage
    preview = Path(form.get("preview") or "") if form.get("preview") else (Path(pics[0]) if pics else None)
    if preview is not None and not preview.is_file():
        preview = Path(pics[0]) if pics else None
    note = (form.get("note") or "").strip()
    lorapage.write_card(final, preview, f"exported {label} of run {log}" + (f": {note}" if note else ""))
    lo = lorapage._lora_of(final)
    data = json.loads(json.dumps(lo.sidecar)) if lo is not None and lo.sidecar else T._sidecar(final, facts["script"], None)
    sha = LL.file_sha256(final, LL.LIB.hash_cache)
    LL.LIB.save_hash_cache()
    words = [w.strip() for w in re.split(r"[,\n]", form.get("trigger") or "") if w.strip()]
    civ = data.setdefault("civitai", {})
    civ["trainedWords"] = words
    try:
        tips = json.loads(data.get("usage_tips") or "{}")
    except ValueError:
        tips = {}
    try:
        tips["strength"] = round(float(form.get("strength") or 1.0), 2)
    except (TypeError, ValueError):
        tips["strength"] = 1.0
    data["usage_tips"] = json.dumps(tips)
    data["sha256"] = sha
    m_ep = re.match(r"epoch (\d+)", label)            # the training line names the save point that was exported
    auto = T._sidecar(final, facts["script"], int(m_ep.group(1)) if m_ep else None)["notes"]
    kept_lines = [x for x in (data.get("notes") or "").split("\n") if x.strip() and not x.startswith("Trained with LoRA Studio")]
    data["notes"] = "\n".join([auto] + kept_lines)
    # example pictures: the whole row of the chosen save point
    root = lm_root()
    ex_state, ex_text, entries = "!", "", []
    if not pics:
        ex_text = "This save point has no test pictures."
    elif not root:
        ex_text = ("LoRA Manager's example picture folder is not known: set it in Collection › Maintenance (or in "
                   "LoRA Manager's settings), then export again. The LoRA and its data are exported anyway.")
    else:
        folder = Path(root) / sha.lower()
        folder.mkdir(parents=True, exist_ok=True)
        mine = "ls" + sha[:6].lower()
        for old in folder.glob(f"custom_{mine}*"):
            old.unlink(missing_ok=True)
        civ["customImages"] = [e for e in civ.get("customImages") or [] if not str(e.get("id", "")).startswith(mine)]
        from PIL import Image
        for i, p in enumerate(pics):
            sid = f"{mine}{i:02d}"
            tgt = folder / f"custom_{sid}{Path(p).suffix.lower()}"
            shutil.copy2(p, tgt)
            try:
                with Image.open(tgt) as im:
                    w, h = im.size
            except Exception:                            # noqa: BLE001
                w = h = 0
            pr = prompts[i] if i < len(prompts) else {}
            meta = {"prompt": pr.get("text", ""), "seed": pr.get("seed")} if pr else None
            entries.append({"url": "", "id": sid, "nsfwLevel": 0, "width": w, "height": h, "type": "image",
                            "meta": meta, "hasMeta": bool(meta), "hasPositivePrompt": bool(meta and meta["prompt"])})
        civ["customImages"] = list(civ.get("customImages") or []) + entries
        ex_state, ex_text = "✔", f"LoRA Manager's example picture folder › {sha[:4]}…{sha[-4:]}/ (the LoRA's hash)"
    if lo is not None and lo.sidecar_path is not None:      # the dataset's card (Collection)
        libsync.write_sidecar(lo.sidecar_path, data)
        lo.sidecar = data
    c_data = dict(data, file_name=copy.stem, file_path=str(copy), size=copy.stat().st_size,
                  modified=copy.stat().st_mtime)
    if preview is not None:
        prev = copy.with_name(copy.stem + ".preview" + preview.suffix.lower())
        shutil.copy2(preview, prev)
        c_data["preview_url"] = str(prev)
        c_data["preview_nsfw_level"] = 0
    libsync.write_sidecar(copy.with_name(copy.stem + ".metadata.json"), c_data)
    steps.append(("✔", "LoRA Manager data written",
                  f"{copy.stem}.metadata.json: trigger {', '.join(words) or '(none)'}, base model "
                  f"{data.get('base_model', '?')}, the training in the notes, strength {tips['strength']}"))
    steps.append(("✔", "Preview picture", f"{copy.stem}.preview{preview.suffix.lower()}") if preview is not None
                 else ("!", "No preview picture", "This save point has no test pictures."))
    steps.append((ex_state, f"{len(entries)} example pictures" if entries else "No example pictures", ex_text))

    # 4. rating and Completed
    T.set_rating({"script": facts["script"]}, int(form.get("rating") or 0))
    if lo is not None and form.get("done"):
        msg = lorapage.set_done(lo.key, True)
        steps.append(("✔" if msg else "!", f"{info['type']} {info.get('version', '')} marked Completed".replace("  ", " "),
                      "its launch script is hidden in the Queue" if msg else "this LoRA has no dataset"))
    try:
        LL.LIB.scan()
    except Exception:                                    # noqa: BLE001
        pass
    steps.append(("!", "LoRA Manager reads new files at its next scan",
                  "In ComfyUI: LoRA Manager › Refresh, or restart ComfyUI."))
    return steps + [check_step(steps, copy, sha, len(entries), root)]


def check_step(steps, copy: Path | None = None, sha: str = "", n_ex: int = 0, root: str = "") -> tuple[str, str, str]:
    """The check at the end: the files are where they should be and LoRA Manager's data reads back."""
    if copy is None:
        return ("❌", "Check: the export did not finish", "")
    bad = []
    if not copy.is_file():
        bad.append("the LoRA copy is missing")
    else:
        h = __import__("hashlib").sha256()
        with open(copy, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if sha and h.hexdigest() != sha:
            bad.append("the copy differs from the final file")
    md = copy.with_name(copy.stem + ".metadata.json")
    try:
        d = json.loads(md.read_text(encoding="utf-8"))
        if d.get("sha256") != sha:
            bad.append("the metadata has another hash")
        pv = d.get("preview_url")
        if pv and not Path(pv).is_file():
            bad.append("the preview is missing")
    except (OSError, ValueError):
        bad.append("the metadata does not read back")
    if n_ex and root:
        have = len(list((Path(root) / sha.lower()).glob(f"custom_ls{sha[:6].lower()}*")))
        if have != n_ex:
            bad.append(f"{have} of {n_ex} example pictures are there")
    if bad:
        return ("❌", "Check: " + "; ".join(bad), "")
    return ("✔", "Checked: the copy, its data, the preview and the example pictures are all there", time.strftime("%H:%M"))
