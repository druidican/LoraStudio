# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The training venv: do PyTorch and its GPU parts belong together, and do they match constraints.txt?

PyTorch decides. It is built for one ROCm version (torch 2.13.0+rocm10.0.0 -> every rocm / rocm-sdk-* part must be
10.0.0) or one set of CUDA libraries (torch's own requirements name the nvidia-* versions). A part with another
version, even a newer one, is wrong: rocm-sdk-devel 10.1 next to 10.0 made SDXL runs crash with "Memory access fault
by GPU". When PyTorch itself is newer than constraints.txt (upgraded on purpose), the locks are written again for it
and the requirements are installed again within the new locks.

inspect() is quick (it reads the installed versions, PyTorch is not started) and runs at every start; fix() is the
"Fix the training venv" button in Settings › Updates. Only the venv LoRA Studio installed (install.sh) is changed;
your own venv only gets the commands to run.
"""
from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
HERE = _LP.ROOT                      # venv/, constraints.txt, requirements.txt
CONSTRAINTS = HERE / "constraints.txt"
ROCM_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"
LOCK_RE = re.compile(r"(torch|torchvision|torchaudio|triton|pytorch-triton-rocm|rocm[a-z0-9_-]*|nvidia-[a-z0-9_-]+)", re.I)
LAST: dict = {}                     # the newest inspect() result (Health check, Settings › Updates)

_PROBE = r"""
import json, re
from importlib.metadata import distributions, distribution
out = {"pk": {}, "req": []}
for d in distributions():
    n = (d.metadata["Name"] or "").strip().lower()
    if n and n not in out["pk"]:
        out["pk"][n] = d.version
try:
    out["req"] = [r for r in (distribution("torch").requires or [])]
except Exception:
    pass
print(json.dumps(out))
"""


def _norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", (n or "").lower())


def _vkey(v: str):
    """'2.13.0+rocm10.0.0' -> ((2, 13, 0), (10, 0, 0)) for comparing PyTorch builds."""
    base, _, local = (v or "").partition("+")
    nums = tuple(int(x) for x in re.findall(r"\d+", base)[:3])
    return nums, tuple(int(x) for x in re.findall(r"\d+", local))


def training_python() -> tuple[Path | None, bool]:
    """(python of the venv the trainings run in, True when LoRA Studio installed it)."""
    shared = (HERE / "venv" / ".lorastudio-shared").exists()
    sg = sys.modules.get("sdgui")
    venv = ""
    if sg is not None:
        try:
            venv = sg.runner_settings()[1]
        except Exception:                                # noqa: BLE001
            venv = ""
    cands = []
    if venv:
        v = Path(os.path.expanduser(venv))
        v = v.parent.parent if v.name == "activate" else v
        cands.append(v / "bin" / "python")
    if sg is not None:
        try:
            cands.append(Path(sg.sd_root()) / "venv" / "bin" / "python")
        except Exception:                                # noqa: BLE001
            pass
    cands.append(HERE / "venv" / "bin" / "python")
    for c in cands:
        if c.exists():
            try:
                managed = shared and c.resolve().parent.parent == (HERE / "venv").resolve()
            except OSError:
                managed = False
            return c, managed
    return None, False


def locks() -> dict:
    """name -> version of the PyTorch / ROCm / CUDA lines in constraints.txt."""
    out = {}
    if CONSTRAINTS.is_file():
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.lstrip().startswith("#"):
                n, _, v = line.strip().partition("==")
                if LOCK_RE.fullmatch(n.strip()):
                    out[_norm(n)] = v.strip()
    return out


def inspect(py: Path | None = None, only_path: bool = False) -> dict:
    """{ok, kind, torch, want, bad: [(name, have, want)], lock_torch, newer, managed, py, text}"""
    managed = False
    if py is None:
        py, managed = training_python()
    res = {"ok": True, "kind": "", "torch": "", "want": "", "bad": [], "lock_bad": [], "lock_torch": "",
           "newer": False, "managed": managed, "py": str(py or ""), "text": ""}
    if not py:
        res.update(ok=True, text="no training venv found")
        LAST.clear(); LAST.update(res)
        return res
    try:
        # only_path (the self-test): -S leaves out the site-packages, so only the made-up packages count
        r = subprocess.run([str(py), *(["-S"] if only_path else []), "-c", _PROBE], capture_output=True, text=True,
                           timeout=60)
        data = json.loads(r.stdout or "{}")
    except Exception as e:                               # noqa: BLE001
        res.update(ok=True, text=f"could not read the venv: {e!r}")
        LAST.clear(); LAST.update(res)
        return res
    pk = {_norm(k): v for k, v in data.get("pk", {}).items()}
    tv = pk.get("torch", "")
    res["torch"] = tv
    if not tv:
        res.update(text="PyTorch is not installed in the training venv")
        LAST.clear(); LAST.update(res)
        return res
    bad = []
    if "+rocm" in tv:
        res["kind"] = "rocm"
        want = tv.partition("+rocm")[2]
        res["want"] = want
        for n, v in sorted(pk.items()):
            if n.startswith("rocm") and n != "rocm-bootstrap" and v != want:
                bad.append((n, v, want))
    elif any(n.startswith("nvidia-") for n in pk):
        res["kind"] = "cuda"
        for req in data.get("req", []):
            m = re.match(r"^\s*(nvidia-[A-Za-z0-9_.-]+)\s*==\s*([0-9][^;\s,]*)", req)
            if m and ";" in req and "extra ==" in req:
                continue
            if m:
                n, want = _norm(m.group(1)), m.group(2)
                if n in pk and pk[n] != want:
                    bad.append((n, pk[n], want))
        res["want"] = tv.partition("+")[2] or "CUDA"
    else:
        res["kind"] = "cpu"
    res["bad"] = bad
    lk = locks() if managed else {}      # constraints.txt belongs to the venv install.sh made, not to your own
    res["lock_torch"] = lk.get("torch", "")
    if res["lock_torch"] and res["lock_torch"] != tv:
        res["newer"] = _vkey(tv) > _vkey(res["lock_torch"])
    if res["kind"] == "rocm":       # a lock of a ROCm part that does not fit PyTorch (rocm-sdk-devel==10.1.0)
        res["lock_bad"] = [(n, v) for n, v in sorted(lk.items()) if n.startswith("rocm") and n != "rocm-bootstrap"
                           and v != res["want"]]
    res["ok"] = not bad and not res["newer"] and not res["lock_bad"] and (not res["lock_torch"] or
                                                                          res["lock_torch"] == tv)
    res["text"] = summary(res)
    LAST.clear(); LAST.update(res)
    return res


def summary(r: dict) -> str:
    if not r.get("torch"):
        return r.get("text") or "no PyTorch"
    gpu = {"rocm": f"ROCm {r['want']}", "cuda": "CUDA", "cpu": "no GPU build"}.get(r["kind"], "")
    if r["ok"]:
        return f"torch {r['torch']} · every GPU part fits ({gpu})"
    parts = []
    if r["bad"]:
        parts.append("parts that do not fit PyTorch: " + ", ".join(f"{n} {h} (needs {w})" for n, h, w in r["bad"]))
    if r["lock_bad"]:
        parts.append("constraints.txt locks " + ", ".join(f"{n} {v}" for n, v in r["lock_bad"]))
    if r["newer"]:
        parts.append(f"PyTorch {r['torch']} is newer than the lock ({r['lock_torch']}): the locks and requirements "
                     "need to be renewed")
    elif r["lock_torch"] and r["lock_torch"] != r["torch"]:
        parts.append(f"PyTorch {r['torch']} differs from the lock ({r['lock_torch']})")
    return f"torch {r['torch']} · " + "; ".join(parts)


def commands(r: dict) -> list[str]:
    """The pip commands fix() runs, as text (shown for a venv LoRA Studio does not manage)."""
    py = r.get("py") or "python"
    out = []
    if r.get("bad"):
        idx = f" --index-url {ROCM_INDEX}" if r["kind"] == "rocm" else ""
        out.append(f"{py} -m pip install --no-deps --force-reinstall{idx} "
                   + " ".join(f"{n}=={w}" for n, _, w in r["bad"]))
    if r.get("newer"):
        out.append(f"{py} -m pip install -r {HERE / 'requirements.txt'} -c {CONSTRAINTS}")
    return out


def _write_locks(py: Path, log: list):
    """PyTorch / ROCm / CUDA lines of constraints.txt from what is installed now; the other lines are kept."""
    r = subprocess.run([str(py), "-c", _PROBE], capture_output=True, text=True, timeout=60)
    pk = json.loads(r.stdout or "{}").get("pk", {})
    keep = []
    if CONSTRAINTS.is_file():
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
            n = line.partition("==")[0].strip()
            if line.lstrip().startswith("#") or not LOCK_RE.fullmatch(n):
                keep.append(line)
    new = [f"{n}=={v}" for n, v in sorted(pk.items()) if LOCK_RE.fullmatch(n) and n != "rocm-bootstrap"]
    head = [l for l in keep if l.lstrip().startswith("#")] or [
        "# versions that later 'pip install' runs must keep (pip install ... -c constraints.txt)"]
    rest = [l for l in keep if l.strip() and not l.lstrip().startswith("#")]
    if CONSTRAINTS.is_file():
        CONSTRAINTS.with_name("constraints.txt.bak").write_text(CONSTRAINTS.read_text(encoding="utf-8"), encoding="utf-8")
    CONSTRAINTS.write_text("\n".join(head + new + rest) + "\n", encoding="utf-8")
    log.append(f"   constraints.txt: {len(new)} PyTorch / GPU lines written from the venv (the old file is "
               "constraints.txt.bak)")


def fix():
    """Generator for the button: yields the log text. Only for the venv LoRA Studio installed."""
    import updater as U
    log: list[str] = []
    r = inspect()
    if r["ok"]:
        yield "Nothing to fix: " + r["text"]
        return
    if not r["managed"]:
        yield ("This is your own venv (" + r["py"] + "): LoRA Studio does not change it. Run:\n\n"
               + "\n".join(commands(r)))
        return
    if U.training_running():
        yield "A training is running: fix the venv after it."
        return
    if not U.LOCK.acquire(blocking=False):
        yield "An update of the trainers is running: try again when it has finished."
        return
    try:
        py = Path(r["py"])
        log.append("== The training venv: " + r["text"])
        if r["bad"]:
            log.append(f"\n== Putting {len(r['bad'])} part(s) back to the version PyTorch is built for")
            cmd = [py, "-m", "pip", "install", "--no-deps", "--force-reinstall"]
            if r["kind"] == "rocm":
                cmd += ["--index-url", ROCM_INDEX]
            cmd += [f"{n}=={w}" for n, _, w in r["bad"]]
            rc = yield from _stream(U.run_stream(cmd, None, log), log)
            if rc:
                log.append("❌ pip could not install them (see above). Nothing else was changed.")
                yield "\n".join(log)
                return
        if r["bad"] or r["lock_bad"] or r["newer"] or (r["lock_torch"] and r["lock_torch"] != r["torch"]):
            log.append("\n== Writing the locks again")
            _write_locks(py, log)
        if r["newer"]:
            log.append("\n== Installing the requirements again within the new locks")
            req = HERE / "requirements.txt"
            if req.is_file():
                rc = yield from _stream(U.run_stream([py, "-m", "pip", "install", "-r", req, "-c", CONSTRAINTS],
                                                     None, log), log)
                if rc:
                    log.append("❌ LoRA Studio's requirements could not be installed (see above).")
            roots = dict(U.repos())
            for cmd, cwd in U._install_cmds(py, roots, CONSTRAINTS, dry=False):
                rc = yield from _stream(U.run_stream(cmd, cwd, log), log)
                if rc:
                    log.append("❌ a trainer's requirements could not be installed (see above).")
        after = inspect()
        log.append("\n== " + ("✔ " if after["ok"] else "! ") + after["text"])
        yield "\n".join(log)
    finally:
        U.LOCK.release()


def _stream(gen, log):
    """Pass a run_stream() generator on as log texts; returns its exit code."""
    while True:
        try:
            next(gen)
        except StopIteration as s:
            return s.value
        yield "\n".join(log)


def start_check(say=print):
    """At start, in the background: read the venv and say so in the terminal when something does not fit."""
    import threading

    def work():
        try:
            r = inspect()
            if not r["ok"]:
                say(f"[Training venv] {r['text']} · Settings › Updates › Fix the training venv")
        except Exception:                                # noqa: BLE001
            pass
    threading.Thread(target=work, daemon=True, name="venvcheck").start()


# --------------------------------------------------------------------------- a newer PyTorch / ROCm
NEWER: dict = {}                    # the newest look() result


def _gfx(py: Path) -> str:
    """gfx1201 from the installed rocm-sdk-device-gfx1201 package ('' when there is none)."""
    r = subprocess.run([str(py), "-c", _PROBE], capture_output=True, text=True, timeout=60)
    for n in json.loads(r.stdout or "{}").get("pk", {}):
        m = re.match(r"rocm-sdk-device-(gfx[0-9a-z]+)$", _norm(n))
        if m:
            return m.group(1)
    return ""


def _index_versions(py: Path, name: str, index: str | None) -> list[str]:
    """The stable versions of a package on an index (pip index versions; pre-releases left out)."""
    cmd = [str(py), "-m", "pip", "index", "versions", name, "--disable-pip-version-check"]
    if index:
        cmd += ["--index-url", index]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    m = re.search(r"Available versions:\s*(.+)", r.stdout)
    if not m:
        raise RuntimeError((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip()
                           else "pip found no versions")
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


def look() -> dict:
    """{ok, have, newest, newer, kind, text} · asks AMD's index (ROCm) or PyPI (CUDA) which PyTorch builds exist."""
    r = inspect()
    res = {"ok": False, "have": r.get("torch", ""), "newest": "", "newer": False, "kind": r.get("kind", ""),
           "managed": r.get("managed", False), "text": ""}
    if not r.get("torch") or r["kind"] not in ("rocm", "cuda"):
        res["text"] = "no GPU build of PyTorch in the training venv"
    else:
        try:
            vers = _index_versions(Path(r["py"]), "torch", ROCM_INDEX if r["kind"] == "rocm" else None)
            if r["kind"] == "rocm":
                vers = [v for v in vers if "+rocm" in v]
            else:
                vers = [v for v in vers if "+rocm" not in v]
            vers = [v for v in vers if not re.search(r"(a|b|rc|dev)\d", v.partition("+")[0])]   # stable only
            newest = max(vers, key=_vkey) if vers else ""
            res.update(ok=True, newest=newest)
            have = r["torch"] if r["kind"] == "rocm" else r["torch"].partition("+")[0]
            res["newer"] = bool(newest) and _vkey(newest) > _vkey(have)
            res["text"] = (f"newer: torch {newest} (installed {r['torch']})" if res["newer"] else
                           f"torch {r['torch']} is the newest stable build" + (" on AMD's server" if r["kind"] == "rocm"
                                                                             else " on PyPI"))
        except Exception as e:                       # noqa: BLE001
            res["text"] = f"could not ask the server: {e}"
    NEWER.clear(); NEWER.update(res)
    return res


GPU_OK = ("import torch, json; ok = torch.cuda.is_available(); x = torch.randn(256, 256, device='cuda') if ok else None; "
          "print('##OK' if ok and bool(torch.isfinite(x @ x).all()) else '##NO', torch.__version__)")


def _gpu_ok(py: Path, log: list) -> bool:
    r = subprocess.run([str(py), "-c", GPU_OK], capture_output=True, text=True, timeout=300)
    log.append("   " + ((r.stdout.strip() or r.stderr.strip().splitlines()[-1]) if (r.stdout + r.stderr).strip() else "?"))
    return "##OK" in r.stdout


def upgrade(target: str):
    """Generator: install torch `target` (with the torchvision / torchaudio pip picks for it), put the ROCm parts to
    its version, test the GPU, then the locks and the requirements; the old versions go back when the GPU test fails."""
    import time as _t
    import updater as U
    log: list[str] = []
    r = inspect()
    if not r.get("managed"):
        yield ("This is your own venv: LoRA Studio does not change it. To upgrade it yourself:\n\n"
               + _upgrade_cmd(r, target))
        return
    if U.training_running():
        yield "A training is running: upgrade after it."
        return
    if not U.LOCK.acquire(blocking=False):
        yield "An update of the trainers is running: try again when it has finished."
        return
    try:
        py = Path(r["py"])
        snap = json.loads(subprocess.run([str(py), "-c", _PROBE], capture_output=True, text=True,
                                         timeout=60).stdout or "{}").get("pk", {})
        old = {n: v for n, v in snap.items() if LOCK_RE.fullmatch(n) and n != "rocm-bootstrap"}
        bdir = _LP.USER / "update_backups"
        bdir.mkdir(exist_ok=True)
        bfile = bdir / f"venv_before_{_t.strftime('%Y%m%d_%H%M%S')}.txt"
        bfile.write_text("\n".join(f"{n}=={v}" for n, v in sorted(old.items())) + "\n", encoding="utf-8")
        log.append(f"== Upgrading PyTorch {r['torch']} → {target}")
        log.append(f"   the versions before are kept in update_backups/{bfile.name}")
        cmd = _upgrade_list(r, target, py)
        rc = yield from _stream(U.run_stream(cmd, None, log), log)
        ok = rc == 0
        if ok:
            r2 = inspect()
            if r2["bad"]:
                log.append("\n== Putting the ROCm parts to the version of the new PyTorch")
                fixc = [py, "-m", "pip", "install", "--no-deps", "--force-reinstall"]
                if r2["kind"] == "rocm":
                    fixc += ["--index-url", ROCM_INDEX]
                fixc += [f"{n}=={w}" for n, _, w in r2["bad"]]
                rc = yield from _stream(U.run_stream(fixc, None, log), log)
                ok = rc == 0
        if ok:
            log.append("\n== Testing the GPU with the new PyTorch")
            yield "\n".join(log)
            ok = _gpu_ok(py, log)
        if not ok:
            log.append("\n❌ The new PyTorch did not work: the old versions go back")
            back = [py, "-m", "pip", "install", "--no-deps", "--force-reinstall"]
            if r["kind"] == "rocm":
                back += ["--index-url", ROCM_INDEX, "--extra-index-url", "https://pypi.org/simple"]
            back += [f"{n}=={v}" for n, v in sorted(old.items())]
            yield from _stream(U.run_stream(back, None, log), log)
            log.append("   " + ("the GPU works again" if _gpu_ok(py, log) else
                                 "the GPU test still fails: run ./install.sh --reinstall-torch"))
            yield "\n".join(log)
            return
        log.append("\n== Writing the locks again")
        _write_locks(py, log)
        log.append("\n== Installing the requirements again within the new locks")
        req = HERE / "requirements.txt"
        if req.is_file():
            yield from _stream(U.run_stream([py, "-m", "pip", "install", "-r", req, "-c", CONSTRAINTS], None, log), log)
        for c, cwd in U._install_cmds(py, dict(U.repos()), CONSTRAINTS, dry=False):
            yield from _stream(U.run_stream(c, cwd, log), log)
        after = inspect()
        log.append("\n== " + ("✔ " if after["ok"] else "! ") + after["text"])
        log.append("   New trainings use the new PyTorch. Restart LoRA Studio if the picture tools were open.")
        NEWER.clear()
        yield "\n".join(log)
    finally:
        U.LOCK.release()


def _upgrade_list(r: dict, target: str, py: Path) -> list:
    if r["kind"] == "rocm":
        gfx = _gfx(py)
        ext = f"[device-{gfx}]" if gfx else ""
        # --pre: AMD pairs a stable torch with an alpha torchvision (0.29.0a0 for 2.14.0); torch itself is pinned
        rv = target.partition("+rocm")[2]
        rocm = (f"rocm[libraries,devel,device-{gfx}]=={rv}" if gfx else f"rocm[libraries,devel]=={rv}")
        # ROCm itself in the same pip run, at exactly the version the new torch is built for
        return [py, "-m", "pip", "install", "--pre", "--index-url", ROCM_INDEX, rocm,
                f"torch{ext}=={target}", f"torchvision{ext}", "torchaudio"]
    return [py, "-m", "pip", "install", f"torch=={target}", "torchvision", "torchaudio"]


def _upgrade_cmd(r: dict, target: str) -> str:
    return " ".join(f'"{c}"' if "[" in str(c) else str(c) for c in _upgrade_list(r, target, Path(r.get("py") or "python")))
