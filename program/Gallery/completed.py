# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
completed.py - shared "Completed" marks for LoRA Studio.

A mark says: this VERSION of the LoRA of this LoRA type for this dataset is finished (for example
"mychar · Anima V3"). A new version (V4) starts unmarked, so its launch script shows on the Train tab.
  * Training LoRA sets it automatically when a training run ends without error and wrote its final
    .safetensors, and hides that dataset's launch scripts of that type from the Train tab's list.
  * Personal Data shows it as a ✓ badge and lets you tick or untick it in Details.

Stored in completed.json next to this file:
  {"mychar|anima": {"done": true, "versions": [1, 2, 3], "how": "auto", "lora": "mycharAnimaV3", "at": "2026-10-13 22:41",
                    "best": 2, "best_at": "2026-10-24 16:10"}}
"best" is the version marked as the best one in Training › Results (the Collection shows it first).
Marks from before versions were tracked have no "versions": they count for every version up to the one in "lora"
(mycharAnimaV3 -> V1-V3), so the versions that existed then stay Completed and V4 does not.
"""

from __future__ import annotations
__version__ = "1.261026v"  # LoRA Studio version, see version.py

import json
import re
import time

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
FILE = _LP.USER / "completed.json"


def type_key(name: str) -> str:
    """'Illustrious' / 'illustrious' / 'Z-Image' -> 'illustrious' / 'zimage' (same key on both sides)."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def key(dataset: str, tkey: str) -> str:
    return f"{str(dataset or '').strip().lower()}|{type_key(tkey)}"


_CACHE = {"sig": None, "data": {}}


def load() -> dict:
    """The marks (read again only when completed.json changed; the Collection asks for every card)."""
    try:
        st = FILE.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    if _CACHE["sig"] == sig:
        return json.loads(json.dumps(_CACHE["data"]))
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
    _CACHE.update(sig=sig, data=data)
    return json.loads(json.dumps(data))


def _save(data: dict):
    tmp = FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(FILE)


def best_version(dataset: str, tkey: str, data: dict | None = None):
    """The version marked as the best of a dataset + LoRA type (Training › Results), or None."""
    if not dataset or not tkey:
        return None
    return ((data if data is not None else load()).get(key(dataset, tkey)) or {}).get("best")


def set_best(dataset: str, tkey: str, version) -> None:
    """Mark one version as the best (the Collection shows it first); version None removes the mark."""
    if not dataset or not tkey:
        return
    data = load()
    k = key(dataset, tkey)
    e = dict(data.get(k) or {})
    if version is None:
        e.pop("best", None)
        e.pop("best_at", None)
    else:
        e["best"] = int(version)
        e["best_at"] = time.strftime("%Y-%m-%d %H:%M")
    if e:
        data[k] = e
    else:
        data.pop(k, None)
    _save(data)


ANY = object()                       # is_done(..., version=ANY): any version of this dataset + type is marked


def version_of(name) -> int | None:
    """'mycharAnimaV3' / 'mycharAnimaV3-000004' -> 3; None when the name has no version."""
    n = re.sub(r"-(?:step)?\d{3,}$", "", str(name or ""))
    m = re.search(r"[vV](\d{1,3})$", n)
    return int(m.group(1)) if m else None


def _versions(entry: dict) -> set:
    """Completed versions of an entry; old entries count up to the version they were set for."""
    if not entry or not entry.get("done"):
        return set()
    if isinstance(entry.get("versions"), list):
        return {v for v in entry["versions"]}
    upto = version_of(entry.get("lora"))
    return set(range(1, upto + 1)) if upto else {None}


def is_done(dataset: str, tkey: str, data: dict | None = None, version=ANY) -> bool:
    if not dataset or not tkey:
        return False
    vs = _versions((data if data is not None else load()).get(key(dataset, tkey)) or {})
    return bool(vs) if version is ANY else version in vs


def info(dataset: str, tkey: str, version=ANY) -> dict:
    """The stored entry, with 'done' for that version (or for any version)."""
    e = dict(load().get(key(dataset, tkey)) or {})
    if e:
        e["done"] = is_done(dataset, tkey, {key(dataset, tkey): e}, version)
    return e


def set_done(dataset: str, tkey: str, done: bool, how: str = "manual", lora: str = "", version=ANY) -> None:
    """Tick (done=True) or untick ONE version of a dataset + LoRA type (the version of `lora`, or `version`)."""
    if not dataset or not tkey:
        return
    data = load()
    k = key(dataset, tkey)
    v = version_of(lora) if version is ANY else version
    vs = _versions(data.get(k) or {})
    if done:
        vs.add(v)
    else:
        vs.discard(v)
    old = data.get(k) or {}
    if vs:
        data[k] = {**{x: old[x] for x in ("best", "best_at") if x in old},
                   "done": True, "versions": sorted(vs, key=lambda x: (x is None, x or 0)),
                   "how": how if done else old.get("how", how), "lora": lora if done else old.get("lora", lora),
                   "at": time.strftime("%Y-%m-%d %H:%M") if done else old.get("at", "")}
    elif "best" in old:
        data[k] = {x: old[x] for x in ("best", "best_at") if x in old}
    else:
        data.pop(k, None)
    _save(data)
