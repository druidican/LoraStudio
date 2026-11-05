# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""AMD GPU status: amd-smi (ROCm 10 and newer, where rocm-smi is gone) or rocm-smi (older ROCm).

Which one is used: the one install.sh found (userdata/features.json "gpu_tool"), else ROCm 10+ → amd-smi, older
ROCm → rocm-smi, and whichever exists when only one does. Every caller gets the same card dicts:
    {card, load (%), vram_total (bytes), vram_used (bytes), temp (°C), junction (°C), power (W)}
amd-smi's JSON differs between versions (a list, or {"gpu_data": [...]}, values as {"value", "unit"} or "38 C"),
so its keys are matched loosely, as for rocm-smi.
"""
from __future__ import annotations
__version__ = "1.051126b"  # LoRA Studio version, see version.py

import glob
import json
import re
import shutil
import subprocess
from pathlib import Path

import lspaths as _LP

_TOOL: dict = {}


def _find(name: str) -> str | None:
    cands = [shutil.which(name)] + sorted(glob.glob(f"/opt/rocm*/bin/{name}"), reverse=True)
    cands += [str(_LP.ROOT / "venv" / "bin" / name)]
    cands += sorted(glob.glob(str(_LP.ROOT / "venv" / "lib" / "python*" / "site-packages" / "_rocm_sdk*" / "bin" / name)))
    return next((c for c in cands if c and Path(c).is_file()), None)


def rocm_major() -> int | None:
    """The ROCm version's first number: the system's /opt/rocm, else the one PyTorch is built for."""
    for f in sorted(glob.glob("/opt/rocm*/.info/version"), reverse=True):
        try:
            return int(Path(f).read_text().strip().split(".")[0])
        except (OSError, ValueError):
            pass
    cons = _LP.ROOT / "constraints.txt"
    try:
        m = re.search(r"^torch==[^\n]*\+rocm(\d+)", cons.read_text(encoding="utf-8"), re.M)
        if m:
            return int(m.group(1))
    except OSError:
        pass
    return None


def tool() -> tuple[str, str] | None:
    """('amd-smi' | 'rocm-smi', path) or None. Cached for the run."""
    if "v" in _TOOL:
        return _TOOL["v"]
    pick = None
    try:
        f = json.loads((_LP.USER / "features.json").read_text(encoding="utf-8")).get("gpu_tool") or ""
        if f and Path(f).is_file():
            pick = ("amd-smi" if "amd-smi" in Path(f).name else "rocm-smi", f)
    except (OSError, ValueError):
        pass
    if pick is None:
        amd, old = _find("amd-smi"), _find("rocm-smi")
        new = (rocm_major() or 0) >= 10
        if amd and (new or not old):
            pick = ("amd-smi", amd)
        elif old:
            pick = ("rocm-smi", old)
        elif amd:
            pick = ("amd-smi", amd)
    _TOOL["v"] = pick
    return pick


def label() -> str:
    t = tool()
    return t[0] if t else "amd-smi / rocm-smi"


# ---------------------------------------------------------------- reading the values
def _num(v):
    if isinstance(v, dict):
        v = v.get("value")
    m = re.search(r"-?\d+(?:\.\d+)?", str(v)) if v is not None else None
    return float(m.group()) if m else None


def _unit(v) -> str:
    if isinstance(v, dict):
        return str(v.get("unit") or "")
    m = re.search(r"[A-Za-z]+", str(v or ""))
    return m.group() if m else ""


def _bytes(v):
    n = _num(v)
    if n is None:
        return None
    u = _unit(v).upper()
    return n * (1024 ** 3 if u.startswith("G") else 1024 ** 2 if u.startswith("M") else 1024 if u.startswith("K")
                else 1 if u in ("B", "BYTES") else 1024 ** 2)            # amd-smi reports MB when no unit is given


def _flat(d, path=""):
    """{'usage.gfx_activity': {...} or 'value'} for nested amd-smi dicts (a {value, unit} dict stays one value)."""
    out = {}
    if isinstance(d, dict) and not ({"value", "unit"} >= set(d) and "value" in d):
        for k, v in d.items():
            out.update(_flat(v, f"{path}.{k}" if path else str(k)))
    else:
        out[path.lower()] = d
    return out


def _get(f: dict, *pats, exclude=()):
    for k, v in f.items():
        if any(re.search(p, k) for p in pats) and not any(re.search(x, k) for x in exclude):
            if _num(v) is not None:
                return v
    return None


def parse_amd_smi(data) -> list[dict]:
    """amd-smi metric/monitor --json -> card dicts."""
    if isinstance(data, dict):
        data = data.get("gpu_data") or data.get("gpus") or [data]
    cards = []
    for i, g in enumerate(data if isinstance(data, list) else []):
        if not isinstance(g, dict):
            continue
        f = _flat(g)
        cards.append(dict(
            card=f"card{int(_num(g.get('gpu')) or i) if g.get('gpu') is not None else i}",
            load=_num(_get(f, r"gfx_activity", r"gfx_util", r"(^|\.)gfx$", r"gpu_use", r"gfx_busy")),
            vram_total=_bytes(_get(f, r"total_vram", r"vram_total", r"vram\.total")),
            vram_used=_bytes(_get(f, r"used_vram", r"vram_used", r"vram\.used")),
            temp=_num(_get(f, r"temperature\.edge", r"edge_temp", r"(^|\.)gpu_temp", r"temperature_edge")),
            junction=_num(_get(f, r"hotspot", r"junction")),
            power=_num(_get(f, r"socket_power", r"power_usage", r"(^|\.)power$", exclude=(r"limit", r"cap", r"max"))),
        ))
    return cards


def _run(args, timeout=8) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout


def _json(out: str):
    a = [i for i in (out.find("["), out.find("{")) if i >= 0]
    start = min(a)
    end = max(out.rfind("]"), out.rfind("}")) + 1
    return json.loads(out[start:end])


def cards() -> list[dict]:
    """Every AMD card with its values (empty when no tool or no answer)."""
    t = tool()
    if not t:
        return []
    kind, exe = t
    if kind == "amd-smi":
        for args in ([exe, "metric", "--json"], [exe, "monitor", "--json"]):
            try:
                c = parse_amd_smi(_json(_run(args)))
                if c and any(x["vram_total"] or x["load"] is not None for x in c):
                    return c
            except Exception:                            # noqa: BLE001
                continue
        return []
    import sdgui
    out = _run([exe, "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"])
    res = sdgui.parse_rocm_smi(_json(out))
    for c in res:                                        # rocm-smi: bytes, or MB in old ROCm versions
        tot = c.get("vram_total") or 0
        if tot and tot < 1e6:
            c["vram_total"], c["vram_used"] = tot * 1024 ** 2, (c.get("vram_used") or 0) * 1024 ** 2
    return res


def _static() -> list:
    t = tool()
    if not t or t[0] != "amd-smi":
        return []
    try:
        d = _json(_run([t[1], "static", "--asic", "--json"], timeout=10))
        if isinstance(d, dict):
            d = d.get("gpu_data") or [d]
        return d if isinstance(d, list) else []
    except Exception:                                    # noqa: BLE001
        return []


def name() -> str:
    """The first card's name."""
    t = tool()
    if not t:
        return ""
    if t[0] == "amd-smi":
        for g in _static():
            f = _flat(g)
            v = next((f[k] for k in f if re.search(r"market_name|product_name|asic\.name", k)), "")
            if v:
                return str(v.get("value") if isinstance(v, dict) else v).strip()
        return ""
    out = _run([t[1], "--showproductname"])
    m = re.search(r"Card Series:\s*(.+)", out) or re.search(r"Card Model:\s*(.+)", out)
    return m.group(1).strip() if m else ""


def gfx_names() -> list[str]:
    """gfx names of the cards (gfx1201 …), from amd-smi static or rocm-smi."""
    t = tool()
    if not t:
        return []
    out = json.dumps(_static()) if t[0] == "amd-smi" else _run([t[1], "--showproductname"], timeout=10)
    return [n for n in re.findall(r"\bgfx[0-9a-f]{3,4}\b", out) if n != "gfx000"]
