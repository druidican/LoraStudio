# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""What the LoRA page shows and does, without any page code: one LoRA version's data (details), saving the Edit
fields, Completed, accepting warnings, copying to the LoRA folder, reading the training again, and the Pictures part
(picture folder, scans, old names). The page itself is program/UI/lora_page.py; the parts it draws as HTML
(head, training card, versions, picture grid) come from lora_library.
"""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import copy
import json
import math
import os
import re
from pathlib import Path

import lora_library as LL

LIB = LL.LIB


def resolve(key: str) -> str:
    """A LoRA key from 'gid:<group>' (its newest version) or a key; '' when unknown."""
    key = (key or "").split("|")[0]
    if key.startswith("gid:"):
        n = LIB.newest(key[4:])
        key = n.key if n else ""
    return key if key in LIB.loras else ""


# --------------------------------------------------------------------------- one version
def details(key: str) -> dict | None:
    """Everything the LoRA page shows for one version (empty Edit fields filled from the training, not saved)."""
    lora = LIB.loras.get(key)
    if not lora:
        return None
    import trainingcard as TC
    if lora.dataset is not None and lora.final and not TC.is_scanned(lora):
        try:                                     # a version not read yet: its training card is made now
            TC.scan_one(LIB, lora)
        except Exception as e:                   # noqa: BLE001
            print(f"[LoRA page] training card of {lora.stem}: {e!r}")
    head, settings = LL.details_html(LIB, lora)
    tips = LL.usage_tips(lora)
    level, _ = lora.nsfw_level()
    ov = LIB.overrides(lora.key)
    sc = lora.sidecar or {}
    trig = "\n".join(lora.trigger_words)
    strength, clip_s, char = tips.get("strength"), tips.get("clip_strength"), lora.character
    filled = []
    try:
        sug = LL.suggest_edit(lora)
    except Exception:                            # noqa: BLE001
        sug = {}
    if not trig.strip() and sug.get("trigger"):
        trig = sug["trigger"]
        filled.append("trigger word")
    use = None
    try:
        import insights
        use = insights.usage_of(lora)
    except Exception:                            # noqa: BLE001
        pass
    if strength is None and use and use.get("strength") is not None and use["pictures"] >= 3:
        strength = round(float(use["strength"]), 2)
        filled.append(f"strength {strength}: the median of the {use['pictures']} pictures you made with it")
    elif strength is None and sug.get("strength") is not None:
        strength = sug["strength"]
        filled.append("strength")
    if clip_s is None and sug.get("clip_strength") is not None:
        clip_s = sug["clip_strength"]
        filled.append("CLIP strength")
    if not (char or "").strip() and sug.get("character"):
        char = sug["character"]
        filled.append("character")
    cskip = tips.get("clip_skip")
    if cskip in (None, ""):
        cskip = int(TC.stored(lora.sidecar).get("clip_skip") or 1) if lora.sidecar else None
    pts = [(s, v) for s, v in lora.loss if not (math.isnan(v) or math.isinf(v))]
    if len(pts) > 2000:
        pts = pts[:: len(pts) // 2000 + 1]
    ds, t = LL.done_key(lora)
    done = {"can": bool(ds), "value": False, "label": "Completed (only for LoRAs with a dataset in your dataset folders)"}
    if ds:
        inf = LL.COMPLETED.info(ds, t, version=LL.COMPLETED.version_of(lora.stem))
        how = (f" (set {'automatically' if inf.get('how') == 'auto' else 'by you'} {inf.get('at', '')})"
               if inf.get("done") else "")
        done = {"can": True, "value": bool(inf.get("done")), "label": f"Completed: {ds} · {lora.base} {lora.vlabel}{how}"}
    return {
        "lora": lora, "key": lora.key, "gid": lora.gid, "preview": lora.preview, "blur": bool(LIB.blur(lora)),
        "examples": list(lora.examples[:40]), "head": head, "settings": settings, "loss": pts,
        "versions": [(label, k) for label, k in LL.version_choices(lora)], "versions_html": LL.versions_html(lora),
        "types": siblings(lora), "title": lora.character or lora.display_name, "subtitle": lora.display_name,
        "done": done,
        "edit": {"name": lora.display_name, "character": char, "trigger": trig,
                 "notes": TC.strip_auto_notes(sc.get("notes") or ""), "strength": strength, "clip": clip_s,
                 "nsfw": level if level in (0, 1, 2, 4, 8, 16) else 0, "favorite": bool(sc.get("favorite")),
                 "flag": ov.get("flag", "none"), "reason": ov.get("flag_reason", ""),
                 "version": (sc.get("civitai") or {}).get("name") or (lora.vlabel if lora.version is not None else ""),
                 "clip_skip": cskip, "description": sc.get("modelDescription", ""), "rating": int(ov.get("rating") or 0)},
        "filled": ("Filled in from the training (" + ", ".join(filled) + "). Not saved yet: check them and press "
                   "Save to keep them.") if filled else "",
    }


def siblings(lora) -> list[tuple[str, str]]:
    """The other LoRA types of the same dataset (or character): [(base, gid)]."""
    ds = lora.dataset
    sib = [g for g, m in LIB.groups.items() if m and (
        (ds is not None and m[-1].dataset is not None and m[-1].dataset.name == ds.name)
        or (ds is None and m[-1].character.lower() == lora.character.lower()))]
    out = []
    for g in sib:
        n = LIB.newest(g)
        out.append((n.base if n else g, g))
    return sorted(out)


def save_edits(key: str, e: dict) -> str:
    lora = LIB.loras.get(key)
    if not lora:
        return "Open a LoRA first."
    data = copy.deepcopy(lora.sidecar) if lora.sidecar else LL.new_sidecar(LIB, lora)
    data["model_name"] = (e.get("name") or lora.stem).strip()
    civ = data.get("civitai") or {}
    data["civitai"] = civ
    civ["trainedWords"] = [t.strip() for t in (e.get("trigger") or "").splitlines() if t.strip()]
    data["notes"] = e.get("notes") or ""
    if (e.get("version") or "").strip():
        civ["name"] = e["version"].strip()
    data["modelDescription"] = e.get("description") or ""
    tips = LL.usage_tips(lora)
    for k, v in (("clip_skip", e.get("clip_skip")), ("strength", e.get("strength")), ("clip_strength", e.get("clip"))):
        if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
            tips.pop(k, None)
        else:
            tips[k] = int(v) if k == "clip_skip" else float(v)
    data["usage_tips"] = json.dumps(tips)
    data["favorite"] = bool(e.get("favorite"))
    data["preview_nsfw_level"] = int(e.get("nsfw") or 0)
    target = lora.sidecar_path        # <dataset>/LoraManager/<model file>.metadata.json (the LoRA folder is untouched)
    if target is None:
        return "No place to save: this LoRA has no dataset."
    LL.write_sidecar(target, data)
    ov = LIB.overrides(key)
    char = (e.get("character") or "").strip()
    if char and char != lora.character:
        ov["character"] = char
    if e.get("rating"):
        ov["rating"] = int(e["rating"])
    else:
        ov.pop("rating", None)
    try:
        import insights
        insights.forget()
    except Exception:                            # noqa: BLE001
        pass
    if e.get("flag") and e["flag"] != "none":
        ov["flag"], ov["flag_reason"] = e["flag"], e.get("reason") or ""
    else:
        ov.pop("flag", None)
        ov.pop("flag_reason", None)
    LIB.save_data()
    lora.sidecar, lora.sidecar_path, lora.sidecar_from_archive = data, target, False
    if char and LIB.dataset_by_name(char) and lora.dataset is None:
        lora.dataset = LIB.dataset_by_name(char)
    if ov.get("character"):
        lora.character = ov["character"]
    LIB._evaluate(lora)
    LIB.save_cache()
    return f"Saved to {target}."


def set_done(key: str, value: bool) -> str:
    lora = LIB.loras.get(key)
    if not lora:
        return ""
    ds, t = LL.done_key(lora)
    if not ds:
        return ""
    LL.COMPLETED.set_done(ds, t, bool(value), how="manual", lora=lora.stem)
    return (f"{ds} · {lora.base} {lora.vlabel} marked Completed: its launch script is hidden in the Queue. A new "
            "version starts unmarked." if value else
            f"{ds} · {lora.base} {lora.vlabel} is no longer Completed: its launch script shows in the Queue again.")


def accept_warnings(key: str) -> str:
    lora = LIB.loras.get(key)
    if not lora:
        return "Open a LoRA first."
    ov = LIB.overrides(key)
    warn = [t for lvl, t in lora.issues if lvl == "warning"]
    if warn:
        ov["accepted"] = sorted(set(ov.get("accepted") or []) | set(warn))
        msg = f"Accepted: {'; '.join(warn)}. The orange mark is gone."
    elif ov.get("accepted"):
        ov.pop("accepted", None)
        msg = "The accepted warnings show again."
    else:
        return "This version has no warnings to accept" + (
            " (a red problem, such as a NaN loss, cannot be accepted: train it again)." if lora.issues else ".")
    LIB.save_data()
    LIB._evaluate(lora)
    LIB.save_cache()
    return msg


def read_training_again(key: str):
    import trainingcard as TC
    lora = LIB.loras.get(key)
    if lora is not None and lora.dataset is not None:
        TC.scan_one(LIB, lora, force=True)


def copy_info(key: str) -> tuple[str, str]:
    """(suggested LoRA folder, note) for Copy to LoRA folder."""
    lora = LIB.loras.get(key)
    if LL.HK is None or not lora or not lora.dataset_path:
        return "", "This LoRA has no file in a dataset's model folder."
    HK = LL.HK
    t = HK.copy_target(lora.stem, LIB.settings.get("lora_dirs") or [], lora.base)
    same = [f for f in HK.lora_files(LIB.settings.get("lora_dirs") or []) if f.name == lora.dataset_path.name]
    note = f"{lora.dataset_path.name} → {t['reason']}."
    if same:
        note += f" Already in {same[0].parent}" + (" (same size)." if same[0].stat().st_size ==
                                                    lora.dataset_path.stat().st_size else " (a DIFFERENT file).")
    return str(t["dir"] or ""), note


def copy_to(key: str, dest: str, replace: bool) -> str:
    lora = LIB.loras.get(key)
    if LL.HK is None or not lora or not lora.dataset_path:
        return "Open a LoRA first."
    if not (dest or "").strip():
        return "Enter the LoRA folder to copy to."
    ok, msg = LL.HK.copy_lora(lora.dataset_path, Path(os.path.expanduser(dest.strip())), bool(replace))
    return ("✔ " if ok else "❌ ") + msg


# --------------------------------------------------------------------------- Pictures made with it
def picture_folder_note(gid: str) -> tuple[str, str]:
    n = LIB.newest(gid)
    base = n.base if n else ""
    if not base:
        return "", ""
    folder, how = LL.picture_folder(base)
    if folder:
        done = LL.PG.index()["folders"].get(str(Path(folder).resolve()))
        when = (f" Last scanned {done['scanned']} ({done['pictures']} pictures)." if done
                else " Not scanned yet: press Scan for new pictures.")
        return folder, f"{base} pictures: {how}.{when}"
    return "", (f"{how} Enter where ComfyUI saves your {base} pictures and press Scan for new pictures; the folder "
                "is remembered for this LoRA type.")


def scan_pictures(gid: str, folder: str) -> tuple[str, str]:
    """Scan one picture folder (only new and changed pictures are read). Returns (message, folder)."""
    n = LIB.newest(gid)
    base = n.base if n else ""
    if not base:
        return "Open a LoRA first.", folder
    folder = os.path.expanduser((folder or "").strip())
    if not folder:
        return f"Where does ComfyUI save your {base} pictures? Enter the folder, then press the button again.", folder
    if not Path(folder).is_dir():
        return f"❌ The folder {folder} does not exist.", folder
    auto, _ = LL.picture_folder(base)
    saved = ""
    if not auto or Path(auto).resolve() != Path(folder).resolve():
        LL.set_picture_dir(base, folder)
        LIB.settings["picture_dirs"] = LL.load_settings().get("picture_dirs") or {}
        saved = f" The folder is remembered for {base}."
    st = LL.PG.scan(folder, None)
    if st["stopped"]:
        return (f"⏸ Stopped after {st['seconds']} s: {st['read']} pictures read and kept. Scan again to continue.{saved}",
                folder)
    return (f"✔ Scanned {folder} in {st['seconds']} s: {st['pictures']} pictures, {st['read']} new or changed, "
            f"{st['removed']} gone, {st['with_lora']} with LoRAs, {st['no_meta']} without metadata (skipped).{saved}",
            folder)


def scan_all_pictures() -> str:
    s = LIB.settings
    roots = [d for d in [s.get("pictures_dir")] + list((s.get("picture_dirs") or {}).values()) if d and Path(d).is_dir()]
    keep = []
    for r in sorted(set(str(Path(d).resolve()) for d in roots), key=len):
        if not any(r.startswith(k + os.sep) for k in keep):
            keep.append(r)
    if not keep:
        return ("No picture folders yet: set the ComfyUI picture folder in Settings › Folders, or enter a folder and "
                "press Scan for new pictures.")
    lines, total = [], 0
    for r in keep:
        st = LL.PG.scan(r, None)
        if st["stopped"]:
            lines.append(f"{r}: stopped, {st['read']} read so far (press again to continue)")
            break
        total += st["with_lora"]
        lines.append(f"{r}: {st['pictures']} pictures, {st['read']} new or changed, {st['removed']} gone")
    return f"✔ Scanned {len(keep)} folder(s), {total} pictures with LoRAs.\n" + "\n".join(lines)


def old_names(gid: str) -> tuple[list[tuple[str, str]], list[str]]:
    """(suggestions [(label, name)], linked names)."""
    ch = LL.alias_suggestions(gid) if gid in LIB.groups else []
    linked = set(LL.picture_aliases(gid))
    return ch, [n for _, n in ch if n in linked]


def save_old_names(gid: str, picked: list, typed: str) -> str:
    if gid not in LIB.groups:
        return "Open a LoRA first."
    names = list(picked or []) + [t for t in re.split(r"[,\n]", typed or "") if t.strip()]
    LL.set_picture_aliases(gid, names)
    n = len(LL.picture_aliases(gid))
    return f"✔ {n} old name{'s' if n != 1 else ''} linked to this LoRA." if n else "No old names linked."
