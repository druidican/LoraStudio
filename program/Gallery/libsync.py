# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""libsync.py - LoRA Manager's files of your LoRAs (<name>.metadata.json, preview, example pictures): made new
(new_sidecar, write_sidecar), merged both ways between the copy in the LoRA folder and the dataset's LoraManager/
folder (merge_meta, sync_rules, sync_lora, quick_sync, personal_sync), and the housekeeping of the datasets: training
leftovers of finished LoRAs and LoRA Manager files whose LoRA is gone (leftovers_md, orphans_list).
"""
from __future__ import annotations
__version__ = "1.271126b"  # LoRA Studio version, see version.py

import html
import json
import shutil
import sys
import time
from pathlib import Path

import lora_library as LL


def new_sidecar(lib: LL.Library, lora: LL.Lora) -> dict:
    st = lora.path.stat()
    return {
        "file_name": lora.stem,
        "model_name": lora.stem,
        "file_path": str(lora.path),
        "size": st.st_size,
        "modified": st.st_mtime,
        "sha256": LL.file_sha256(lora.path, lib.hash_cache),
        "base_model": lora.base,
        "preview_url": str(lora.preview) if lora.preview else "",
        "preview_nsfw_level": 0,
        "notes": "",
        "from_civitai": False,
        "civitai": {"trainedWords": []},
        "tags": [],
        "modelDescription": "",
        "favorite": False,
        "exclude": False,
        "db_checked": False,
        "skip_metadata_refresh": False,
        "last_checked_at": 0,
        "hash_status": "completed",
        "usage_tips": "{}",
    }


def write_sidecar(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    LL._save_json(path, data)


def _sync_one(src: Path, dst: Path, stats: dict, same_if_equal: bool = False) -> None:
    """Copy when missing, or when the LoRA folder's file is newer (an edit made in the dataset copy is kept).
    same_if_equal (the preview): copied whenever it is another picture, also an older one you picked in LoRA Manager."""
    try:
        s = src.stat()
        if dst.exists():
            d = dst.stat()
            if same_if_equal:
                if s.st_size == d.st_size and abs(s.st_mtime - d.st_mtime) <= 1:
                    stats["same"] += 1
                    return
            elif s.st_mtime <= d.st_mtime + 1:
                stats["same"] += 1
                return
            if dst.suffix == ".json":
                shutil.copy2(dst, dst.with_name(dst.name + ".bak"))
            stats["updated"] += 1
        else:
            stats["new"] += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except OSError as e:
        stats["errors"].append(f"{src.name}: {e}")


# fields that describe where a file lies: each side keeps its own
_META_OWN = {"file_name", "file_path", "folder", "preview_url", "modified"}


def _empty(v) -> bool:
    """Missing for the merge: None, "", "{}", [], {}, 0 (booleans are values: False is kept)."""
    if isinstance(v, bool):
        return False
    if v is None or v == 0 or v == "" or v == [] or v == {}:
        return True
    return isinstance(v, str) and v.strip() in ("", "{}", "[]")


def merge_meta(a: dict, b: dict, a_wins: bool) -> dict:
    """Both .metadata.json files together: a field that is missing or empty on one side is taken from the other;
    when both have a different value, the newer file (a_wins) decides. Nested parts (civitai …) merge the same way."""
    out = {}
    for k in list(a) + [k for k in b if k not in a]:
        if k not in b:
            out[k] = a[k]
        elif k not in a:
            out[k] = b[k]
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out[k] = merge_meta(a[k], b[k], a_wins)
        elif _empty(a[k]):
            out[k] = b[k]
        elif _empty(b[k]):
            out[k] = a[k]
        else:
            out[k] = a[k] if a_wins else b[k]
    return out


def _tips(sc: dict) -> dict:
    raw = (sc or {}).get("usage_tips") or "{}"
    try:
        t = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return t if isinstance(t, dict) else {}
    except (ValueError, TypeError):
        return {}


def sync_rules(lm: dict, ls: dict, m: dict, lm_newer: bool = False) -> dict:
    """What LoRA Studio knows best wins over the merge: the version name (from the training card or your Edit), and
    the trigger words of the dataset when neither side has any. Usage tips (strength, clip skip, …) are merged field
    by field; no clip skip anywhere means the default, 1."""
    m = dict(m)
    civ = dict(m.get("civitai") or {})
    ls_civ = (ls.get("civitai") or {}) if isinstance(ls.get("civitai"), dict) else {}
    card = ((ls.get("lorastudio") or {}).get("training") or {}) if isinstance(ls.get("lorastudio"), dict) else {}
    ver = ls_civ.get("name") or card.get("version")
    if ver:
        civ["name"] = ver
    if not civ.get("trainedWords") and card.get("trigger"):
        civ["trainedWords"] = [w.strip() for w in str(card["trigger"]).split(",") if w.strip()]
    if civ or "civitai" in m:
        m["civitai"] = civ
    t_lm, t_ls = _tips(lm), _tips(ls)
    tips = merge_meta(t_lm, t_ls, lm_newer)            # tips: fill what one side lacks; the newer file on a clash
    try:
        cs = int(float(tips.get("clip_skip")))
    except (TypeError, ValueError):
        cs = 0
    tips["clip_skip"] = cs if cs >= 1 else int(float(card.get("clip_skip") or 1))
    m["usage_tips"] = json.dumps(tips)
    return m


def _sync_meta(src: Path, dst: Path, stats: dict) -> None:
    """Sync both ways: data only one .metadata.json has is copied into the other (the LoRA folder's file and the
    dataset copy); for fields both have, the newer file wins. Location fields (file path, folder, preview) stay as
    they are on each side. A .bak is made before a file is changed; files that already agree are not touched."""
    if not dst.exists():
        _sync_one(src, dst, stats)
        return
    a, b = LL._load_json(src, None), LL._load_json(dst, None)
    if not isinstance(a, dict) or not isinstance(b, dict):
        _sync_one(src, dst, stats)                       # one of them is not readable JSON: the old rule
        return
    try:
        src_newer = src.stat().st_mtime > dst.stat().st_mtime + 1
    except OSError as e:
        stats["errors"].append(f"{src.name}: {e}")
        return
    m = sync_rules(a, b, merge_meta(a, b, src_newer), src_newer)
    changed = False
    for path, cur in ((src, a), (dst, b)):
        new = dict(m)
        for k in _META_OWN:                              # this side's own location fields
            if k in cur:
                new[k] = cur[k]
            else:
                new.pop(k, None)
        if path == src and "lorastudio" not in cur:      # the training card stays in LoRA Studio's copy
            new.pop("lorastudio", None)
        if new != cur:
            try:
                write_sidecar(path, new)
                changed = True
                stats["merged"] = stats.get("merged", 0) + 1
                if path == src:
                    stats["back"] = stats.get("back", 0) + 1
            except OSError as e:
                stats["errors"].append(f"{path.name}: {e}")
    if changed:
        stats["updated"] += 1
    else:
        stats["same"] += 1


# ----------------------------------------------------------------------------------- housekeeping (Clean up tab)
def _trainer():
    """sdgui when it runs in the same program (LoRA Studio): for the active run and the musubi-tuner folder."""
    return sys.modules.get("sdgui")


def busy_datasets() -> list[Path]:
    import sd_script
    sg = _trainer()
    try:
        if sg is not None and sg.RUN.running() and sg.RUN.script:
            return [sd_script.script_home(sg.RUN.script)]
    except Exception:                  # noqa: BLE001
        pass
    return []


def musubi_cache_dir() -> Path | None:
    sg = _trainer()
    try:
        return (sg.musubi_root() / "cache") if sg is not None else None
    except Exception:                  # noqa: BLE001
        return None


def dataset_roots(lib: LL.Library) -> list[Path]:
    lib.ensure_loaded()
    return sorted({d.root for d in lib.datasets.values()})


def leftovers_md(lib: LL.Library):
    """(markdown, [paths]) of the training leftovers of finished LoRAs in every dataset. Nothing is touched."""
    if LL.HK is None:
        return "housekeeping.py is missing next to lora_library.py.", []
    plan = LL.HK.leftovers(dataset_roots(lib), musubi_cache_dir(), busy_datasets())
    items = plan["items"]
    L = []
    if items:
        total = sum(i[3] for i in items)
        L.append(f"**{len(items)} items, {LL.HK.human(total)}** can go to the Trash:\n")
        L += ["| Dataset | Epoch files | States | .npz caches | Other | Size |", "|---|---|---|---|---|---|"]
        per = {}
        for p, kind, ds, size in items:
            r = per.setdefault(ds, {"epoch checkpoint": 0, "saved state": 0, "latent cache (.npz)": 0, "other": 0, "size": 0})
            r[kind if kind in r else "other"] += 1
            r["size"] += size
        for ds, r in sorted(per.items(), key=lambda kv: -kv[1]["size"]):
            L.append(f"| {html.escape(ds)} | {r['epoch checkpoint']} | {r['saved state']} | {r['latent cache (.npz)']} | "
                     f"{r['other']} | {LL.HK.human(r['size'])} |")
        L.append("\nKept: pictures, captions, launch scripts, logs, `LoraManager/` and every **final** LoRA file. "
                 "Epoch files and states are only taken for LoRAs that have a final file; a later run re-creates the caches.")
    else:
        L.append("Nothing to clean up: no leftovers of finished LoRAs found.")
    if plan["skipped"]:
        L.append("\n**Skipped:** " + "; ".join(f"{html.escape(d)} ({why})" for d, why in plan["skipped"]))
    return "\n".join(L), [str(i[0]) for i in items]


def orphans_list(lib: LL.Library):
    """(markdown, checkbox choices, all values) for LoRA Manager meta files whose LoRA no longer exists."""
    if LL.HK is None:
        return "housekeeping.py is missing next to lora_library.py.", [], []
    found = LL.HK.orphans(lib.settings.get("lora_dirs") or [], dataset_roots(lib))
    if not found:
        return "No orphaned meta files: every `.metadata.json` / `.civitai.info` has its LoRA file.", [], []
    choices = []
    for o in found:
        label = (f"[{o['where']}] {o['folder']} / {o['stem']} — {len(o['files'])} file(s), {LL.HK.human(o['size'])}")
        choices.append((label, "\x1f".join(str(f) for f in o["files"])))
    md = (f"**{len(found)} meta file group(s)** whose LoRA file is gone ({LL.HK.human(sum(o['size'] for o in found))}). "
          "Untick what you want to keep, tick the confirmation and press **Move ticked to Trash**. In a dataset's "
          "`LoraManager/`, a group only counts as orphaned when the LoRA is in neither the dataset's `model/` folder "
          "nor a LoRA folder (your Details edits live there).")
    return md, choices, [c[1] for c in choices]


def _sync_pair(lib: LL.Library, f: Path, lora, stats: dict) -> bool:
    """Copy one LoRA's LoRA Manager files (f: its copy in the LoRA folder) into its dataset's LoraManager/ folder.
    True when something was copied or changed."""
    src_dir, old, new = f.parent, f.stem, lora.stem
    dst_dir = lora.dataset.archive_dir
    before = stats["new"] + stats["updated"]
    sc_path = f.with_name(old + ".metadata.json")
    sidecar = LL._load_json(sc_path, None) if sc_path.exists() else None
    ls_path = dst_dir / (new + ".metadata.json")
    if sc_path.exists():
        _sync_meta(sc_path, ls_path, stats)
        try:                        # the training card into LoRA Studio's copy, then what it adds to LoRA Manager
            import trainingcard as TC
            sc_new = LL._load_json(ls_path, None)
            lora.sidecar, lora.sidecar_path = (sc_new if isinstance(sc_new, dict) else None), ls_path
            if TC.scan_one(lib, lora) == "scanned":
                _sync_meta(sc_path, ls_path, {"new": 0, "updated": 0, "same": 0, "errors": stats["errors"]})
        except Exception as e:      # noqa: BLE001
            stats["errors"].append(f"{new} training card: {e}")
    preview = LL.find_preview(src_dir, old, sidecar if isinstance(sidecar, dict) else None)
    if preview:                     # the preview LoRA Manager's card shows: the LoRA page shows the same one
        target = dst_dir / f"{new}.preview{preview.suffix.lower()}"
        for ext in LL.PREVIEW_EXT:
            other = dst_dir / f"{new}.preview{ext}"
            if other != target and other.exists() and preview.stat().st_mtime > other.stat().st_mtime:
                try:
                    other.unlink()
                except OSError:
                    pass
        _sync_one(preview, target, stats, same_if_equal=True)
    sha = (((sidecar or {}) if isinstance(sidecar, dict) else {}).get("sha256") or "").lower()
    exs = [str(x) for x in LL.find_examples(src_dir, old, sha, preview, list(lib.settings["lora_dirs"]) + LL.lm_example_roots(lib))]
    if exs or ls_path.exists():                     # LoRA Manager's example pictures: linked, not copied
        cur = LL._load_json(ls_path, None) if ls_path.exists() else None
        cur = cur if isinstance(cur, dict) else {"file_name": new}
        lsd = dict(cur.get("lorastudio") or {})
        if (lsd.get("examples") or []) != exs:
            lsd["examples"] = exs
            cur["lorastudio"] = lsd
            try:
                write_sidecar(ls_path, cur)
                stats["linked"] = stats.get("linked", 0) + len(exs)
                stats["updated"] += 1
            except OSError as e:
                stats["errors"].append(f"{ls_path.name}: {e}")
    return stats["new"] + stats["updated"] > before


def sync_lora(lib: LL.Library, lora) -> bool:
    """The Personal sync for one LoRA, when its page opens: its copy in the LoRA folder is found by name and size
    (no hashing), and the LoRA Manager files are copied when they are newer. True when something changed."""
    p = lora.dataset_path
    if lora.dataset is None or p is None:
        return False
    try:
        size = p.stat().st_size
    except OSError:
        return False
    want = p.name.lower()
    ds_roots = {str(ds.root.resolve()) for ds in lib.datasets.values()}
    for root in lib.settings["lora_dirs"]:
        for d, dns, fns in LL.walk(root):
            if str(d.resolve()) in ds_roots:
                dns[:] = []
                continue
            for fn in fns:
                if fn.lower() == want:
                    f = d / fn
                    try:
                        if f.stat().st_size != size:
                            continue
                    except OSError:
                        continue
                    LL._LISTINGS.clear()
                    stats = {"new": 0, "updated": 0, "same": 0, "errors": []}
                    changed = _sync_pair(lib, f, lora, stats)
                    LL._LISTINGS.clear()
                    return changed
    return False


_QUICK = {"t": 0.0}


def quick_sync(lib: LL.Library, every: float = 60.0) -> int:
    """What LoRA Manager changed since the last look (a preview you set, notes, example pictures), for all your
    LoRAs, when the Collection opens: the LoRA folder is listed once and copies are matched by name and size only (the
    full Personal sync also finds renamed copies by their hash). At most once a minute. Returns how many changed."""
    if time.time() - _QUICK["t"] < every:
        return 0
    _QUICK["t"] = time.time()
    lib.ensure_loaded()
    want = {}
    for l in lib.loras.values():
        if l.dataset is not None and l.dataset_path is not None:
            want.setdefault(l.dataset_path.name.lower(), []).append(l)
    if not want:
        return 0
    ds_roots = {str(ds.root.resolve()) for ds in lib.datasets.values()}
    stats = {"new": 0, "updated": 0, "same": 0, "errors": []}
    changed = 0
    LL._LISTINGS.clear()
    for root in lib.settings["lora_dirs"]:
        for d, dns, fns in LL.walk(root):
            if str(d.resolve()) in ds_roots:
                dns[:] = []
                continue
            for fn in fns:
                for lora in want.get(fn.lower(), []):
                    f = d / fn
                    try:
                        if f.stat().st_size != lora.dataset_path.stat().st_size:
                            continue
                    except OSError:
                        continue
                    if _sync_pair(lib, f, lora, stats):
                        changed += 1
    LL._LISTINGS.clear()
    if changed:
        lib.scan()
    return changed


def personal_sync(lib: LL.Library, progress=None) -> str:
    """Read the LoRA folder(s) once and copy the LoRA Manager files of your own LoRAs (.metadata.json, preview
    and example images) into each dataset's LoraManager/ folder, named after the dataset's model file.
    A LoRA in the LoRA folder belongs to a dataset when its name and size match a model file in the dataset,
    or (for renamed copies) its sha256 does. Everything else in the LoRA folder (downloads) is left alone."""
    lib.ensure_loaded()
    by_name, by_size = {}, {}
    for l in lib.loras.values():
        p = l.dataset_path
        if l.dataset is None or p is None:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        by_name[(p.stem.lower(), size)] = l
        by_size.setdefault(size, []).append(l)

    LL._LISTINGS.clear()
    ds_roots = {str(ds.root.resolve()) for ds in lib.datasets.values()}
    candidates = []
    for root in lib.settings["lora_dirs"]:
        for d, dns, fns in LL.walk(root):
            if str(d.resolve()) in ds_roots:
                dns[:] = []
                continue
            candidates += [d / f for f in fns if f.endswith(".safetensors")]

    stats = {"new": 0, "updated": 0, "same": 0, "errors": []}
    matched, done_loras = 0, []
    total = max(1, len(candidates))
    for n, f in enumerate(candidates):
        if progress is not None:
            progress((n + 1) / total, desc=f"Checking {f.name}")
        try:
            size = f.stat().st_size
        except OSError:
            continue
        lora = by_name.get((f.stem.lower(), size))
        if lora is None:
            for cand in by_size.get(size, []):
                if LL.file_sha256(cand.dataset_path, lib.hash_cache) == LL.file_sha256(f, lib.hash_cache):
                    lora = cand
                    break
        if lora is None:
            continue
        matched += 1
        if _sync_pair(lib, f, lora, stats):
            done_loras.append(f"{lora.dataset.name} · {lora.stem}")
    LL._LISTINGS.clear()
    lib.save_hash_cache()
    lib.scan()          # pick up the copied files

    lines = [f"Checked **{len(candidates)}** LoRA files in the LoRA folder; **{matched}** belong to your datasets.",
             f"Copied **{stats['new']}** new and **{stats['updated']}** updated files; "
             f"{stats['same']} were already up to date."]
    if done_loras:
        shown = ", ".join(done_loras[:30]) + (f" and {len(done_loras) - 30} more" if len(done_loras) > 30 else "")
        lines.append(f"Updated: {shown}.")
    if stats.get("linked"):
        lines.append(f"Linked **{stats['linked']}** example pictures from LoRA Manager (shown on the LoRA page; not "
                     "copied).")
    if stats.get("back"):
        lines.append(f"**{stats['back']}** `.metadata.json` file(s) in the LoRA folder got data that only the dataset "
                     "copy had (a `.bak` was made next to each).")
    if stats["errors"]:
        lines.append("Could not copy: " + "; ".join(stats["errors"][:10]))
    return "\n\n".join(lines)
