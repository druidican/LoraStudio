#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Speed test for a LoRA Studio launch script: which settings make training faster on THIS computer?

    python3 ~/AI/LoraStudio/program/Training/speedtest.py /path/to/dataset/pipZITV1.sh
    python3 .../speedtest.py pipFLUXV1.sh --steps 20 --quick      # fewer variants
    python3 .../speedtest.py pipZITV1.sh --only baseline,pinned   # only some variants

What it does (nothing in your scripts or datasets is changed):
  1. runs the launch script up to the training command WITHOUT training: every stage before it runs as usual
     (musubi-tuner: checks, dataset, latent and text-encoder caches), then the exact training command, its folder
     and its environment are recorded;
  2. asks the trainer which options it has (--help), so it only tries what your sd-scripts / musubi-tuner support;
  3. runs the training command for a few steps per variant, each changing ONE thing (fewer swapped blocks, pinned
     memory for swapping, no gradient checkpointing, another attention, fp8, torch.compile, ROCm settings), and
     measures seconds per step (its own clock, after the first warm-up steps) and the peak video memory;
  4. tries the winners together, and writes a report (Markdown) next to the launch script.

Test LoRAs go to a temporary folder and are deleted. A variant that runs out of memory or gets a NaN loss is
reported as such. Close ComfyUI and stop Ollama first, so the video memory is free. Runs take a while: the model
is loaded again for every variant (about 1-3 minutes each, plus the steps).
"""
from __future__ import annotations
__version__ = "1.111126b"  # LoRA Studio version, see version.py

import argparse
import glob
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROCM_ENV = {
    "tunableop": ("PyTorch TunableOp (picks the fastest matrix kernels; the first steps are slow while it tunes)",
                  {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1"}),
    "miopen_fast": ("MIOpen fast kernel search", {"MIOPEN_FIND_MODE": "FAST"}),
    "expandable": ("expandable memory segments (less fragmentation, can allow fewer swapped blocks)",
                   {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
}
ATTN = ["sdpa", "flash_attn", "sage_attn", "xformers", "flash3"]
DROP_ALWAYS = ["save_every_n_epochs", "save_every_n_steps", "save_last_n_epochs", "save_last_n_epochs_state",
               "save_state", "save_state_on_train_end", "sample_every_n_epochs", "sample_every_n_steps",
               "sample_prompts", "sample_at_first", "logging_dir", "log_with", "wandb_api_key", "max_train_epochs",
               "resume", "network_weights"]


# --------------------------------------------------------------------------- 1. capture the training command
def capture(script: Path, work: Path) -> dict:
    """Run the launch script with `accelerate` replaced by a recorder (a bash function wins over the venv's PATH)."""
    out = work / "cmd.txt"
    envf = work / "env.bin"
    cwdf = work / "cwd.txt"
    shim = (f'accelerate() {{ printf "%s\\0" "$@" > {shlex.quote(str(out))}; pwd > {shlex.quote(str(cwdf))}; '
            f'env -0 > {shlex.quote(str(envf))}; echo "[speedtest] training command recorded"; exit 0; }}; '
            f'export -f accelerate; exec bash {shlex.quote(str(script))}')
    print(f"[speedtest] preparing: running {script.name} up to its training command (no training) ...")
    p = subprocess.run(["bash", "-c", shim], cwd=script.parent, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    log = p.stdout or ""
    if not out.is_file():
        tail = "\n".join(log.splitlines()[-25:])
        sys.exit(f"[speedtest] the launch script stopped before its training command:\n{tail}")
    args = [a for a in out.read_bytes().decode("utf-8", "replace").split("\0") if a != ""]
    env = {}
    for item in envf.read_bytes().decode("utf-8", "replace").split("\0"):
        if "=" in item and not item.startswith("BASH_FUNC_"):
            k, v = item.split("=", 1)
            env[k] = v
    if args and args[0] == "launch":
        args = args[1:]
    i = next((n for n, a in enumerate(args) if a.endswith(".py")), None)
    if i is None:
        sys.exit(f"[speedtest] no training script in the command: {' '.join(args)[:300]}")
    return {"launch": args[:i], "train": args[i], "args": args[i + 1:], "cwd": cwdf.read_text().strip(),
            "env": env, "log": log}


# --------------------------------------------------------------------------- the option list
def opt_index(args: list, name: str) -> int:
    return next((n for n, a in enumerate(args) if a == f"--{name}" or a.startswith(f"--{name}=")), -1)


def has(args, name) -> bool:
    return opt_index(args, name) >= 0


def value(args, name):
    i = opt_index(args, name)
    if i < 0:
        return None
    if "=" in args[i]:
        return args[i].split("=", 1)[1]
    return args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""


def drop(args, name) -> list:
    out, skip = [], False
    for n, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == f"--{name}":
            if n + 1 < len(args) and not args[n + 1].startswith("--"):
                skip = True
            continue
        if a.startswith(f"--{name}="):
            continue
        out.append(a)
    return out


def opts_of(args: list) -> dict:
    """{option: value or None} of an argument list."""
    out, n = {}, 0
    while n < len(args):
        a = args[n]
        if a.startswith("--"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
            else:
                k = a[2:]
                v = args[n + 1] if n + 1 < len(args) and not args[n + 1].startswith("--") else None
                n += 1 if v is not None else 0
            out[k] = v
        n += 1
    return out


def changes(old: list, new: list) -> list[tuple]:
    """[(option, old value or MISSING, new value or MISSING)] between two argument lists."""
    a, b = opts_of(old), opts_of(new)
    return [(k, a.get(k, MISSING), b.get(k, MISSING)) for k in list(a) + [k for k in b if k not in a]
            if a.get(k, MISSING) != b.get(k, MISSING)]


MISSING = object()


def apply_changes(args: list, ch: list) -> list:
    for k, _, v in ch:
        args = drop(args, k) if v is MISSING else setopt(args, k, v)
    return args


def setopt(args, name, val=None) -> list:
    args = drop(args, name)
    return args + ([f"--{name}"] if val is None else [f"--{name}", str(val)])


def supported(cap: dict) -> set:
    """The options the training script knows (from its --help)."""
    env = cap["env"]
    py = shutil.which("python", path=env.get("PATH")) or sys.executable
    try:
        p = subprocess.run([py, cap["train"], "--help"], cwd=cap["cwd"], env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        return set(re.findall(r"--([a-z0-9_]+)", p.stdout or ""))
    except Exception as e:                               # noqa: BLE001
        print(f"[speedtest] could not read the trainer's options ({e}); trying everything")
        return set()


# --------------------------------------------------------------------------- the variants
def variants(args: list, opts: set, quick: bool) -> list[dict]:
    ok = (lambda n: not opts or n in opts)
    v = [{"name": "baseline", "why": "your script as it is", "args": args, "env": {}}]
    swap = value(args, "blocks_to_swap")
    if swap not in (None, ""):
        b = int(float(swap))
        for nb in sorted({max(0, b - 4), max(0, b - 8), 0} - {b}, reverse=True):
            v.append({"name": f"swap{nb}", "why": f"--blocks_to_swap {nb} instead of {b} (less moving between "
                      "system memory and the card per step)", "args": setopt(args, "blocks_to_swap", nb) if nb else
                      drop(args, "blocks_to_swap"), "env": {}})
        if ok("use_pinned_memory_for_block_swap") and not has(args, "use_pinned_memory_for_block_swap"):
            v.append({"name": "pinned", "why": "--use_pinned_memory_for_block_swap (faster block swapping)",
                      "args": setopt(args, "use_pinned_memory_for_block_swap"), "env": {}})
    if has(args, "gradient_checkpointing"):
        v.append({"name": "no_checkpointing", "why": "without --gradient_checkpointing (faster, needs more memory)",
                  "args": drop(args, "gradient_checkpointing"), "env": {}})
    cur = next((a for a in ATTN if has(args, a)), None)
    for a in ATTN:
        if a != cur and (ok(a) if opts else a in ("sdpa", "flash_attn")):
            if quick and a not in ("sdpa", "flash_attn", "sage_attn"):
                continue
            na = args
            for b in ATTN:
                na = drop(na, b)
            v.append({"name": a, "why": f"--{a} attention instead of {('--' + cur) if cur else 'the default'}",
                      "args": na + [f"--{a}"], "env": {}})
    if ok("fp8_base") and not has(args, "fp8_base"):
        extra = ["--fp8_scaled"] if "fp8_scaled" in opts else []
        v.append({"name": "fp8", "why": "--fp8_base" + (" --fp8_scaled" if extra else "") + " (the frozen model in "
                  "8 bit: less memory; watch the loss)", "args": setopt(args, "fp8_base") + extra, "env": {}})
    if not quick:
        if ok("compile") and "compile" in opts and not has(args, "compile"):
            v.append({"name": "compile", "why": "--compile (torch.compile: the first steps are slow while it compiles)",
                      "args": setopt(args, "compile"), "env": {}})
        elif ok("torch_compile") and "torch_compile" in opts and not has(args, "torch_compile"):
            v.append({"name": "compile", "why": "--torch_compile (the first steps are slow while it compiles)",
                      "args": setopt(args, "torch_compile"), "env": {}})
    for key, (why, env) in ROCM_ENV.items():
        if quick and key == "miopen_fast":
            continue
        v.append({"name": key, "why": why, "args": args, "env": env})
    return v


# --------------------------------------------------------------------------- 3. running one variant
class VRAM:
    """Peak video memory while a variant runs (AMD: sysfs, NVIDIA: nvidia-smi)."""

    def __init__(self):
        self.peak, self.total, self._stop = 0, 0, threading.Event()
        cards = []
        for d in glob.glob("/sys/class/drm/card*/device"):
            try:
                cards.append((int(Path(d, "mem_info_vram_total").read_text()), d))
            except (OSError, ValueError):
                pass
        self.amd = max(cards)[1] if cards else None
        self.total = max(cards)[0] if cards else 0
        self.nv = shutil.which("nvidia-smi") if not self.amd else None

    def used(self) -> int:
        try:
            if self.amd:
                return int(Path(self.amd, "mem_info_vram_used").read_text())
            if self.nv:
                o = subprocess.run([self.nv, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.split("\n")[0].split(",")
                self.total = int(o[1]) * 2 ** 20
                return int(o[0]) * 2 ** 20
        except Exception:                                # noqa: BLE001
            pass
        return 0

    def start(self):
        self.peak = 0
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                self.peak = max(self.peak, self.used())
                time.sleep(0.5)
        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> int:
        self._stop.set()
        return self.peak


STEP = re.compile(r"(?:steps|it)?[: ]*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
OOM = re.compile(r"out of memory|OutOfMemoryError|HIP error: out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.I)
NAN = re.compile(r"avr_loss=nan|loss=nan|NaN detected", re.I)


def run_variant(cap, var, steps, warm, out_dir, timeout, vram: VRAM, logdir: Path, say=print, stop=None) -> dict:
    args = list(var["args"])
    for d in DROP_ALWAYS:
        args = drop(args, d)
    args = setopt(args, "max_train_steps", steps)
    args = setopt(args, "output_dir", out_dir)
    args = setopt(args, "output_name", "lsspeedtest")
    env = {**cap["env"], **var["env"]}
    acc = shutil.which("accelerate", path=env.get("PATH")) or "accelerate"
    cmd = [acc, "launch", *cap["launch"], cap["train"], *args]
    log = logdir / f"{var['name']}.log"
    times, last, status, buf = [], -1, "ok", ""
    t0 = time.time()
    vram.start()
    with open(log, "w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        p = subprocess.Popen(cmd, cwd=cap["cwd"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            while True:
                chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", "replace")
                lf.write(text)
                buf = (buf + text)[-4000:]
                for m in STEP.finditer(text):
                    n = int(m.group(1))
                    if n > last:
                        times.append((n, time.time()))
                        last = n
                        say(f"  {var['name']}: step {n}/{steps}", True)
                if OOM.search(text):
                    status = "out of memory"
                if NAN.search(text) and status == "ok":
                    status = "NaN loss"
                if time.time() - t0 > timeout:
                    status = "too slow (timeout)"
                    break
                if stop is not None and stop.is_set():
                    status = "stopped"
                    break
        finally:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            peak = vram.stop()
    if p.returncode not in (0, None) and status == "ok":
        status = "failed" + (": option not available" if re.search(r"unrecognized arguments|No module named",
                                                                    buf) else "")
    pairs = [(a, b) for a, b in zip(times, times[1:]) if a[0] >= warm and b[0] > a[0]]
    per = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in pairs]
    sps = round(statistics.median(per), 2) if per else None
    return {"name": var["name"], "why": var["why"], "status": status, "s_per_step": sps,
            "steps": last, "vram": peak, "seconds": round(time.time() - t0), "log": str(log),
            "args": var["args"], "env": var["env"]}


# --------------------------------------------------------------------------- 4. the report
def gb(b):
    return f"{b / 2 ** 30:.1f} GB" if b else "?"


def report(script: Path, cap, results, vram: VRAM, steps) -> Path:
    base = next((r for r in results if r["name"] == "baseline"), None)
    b = base["s_per_step"] if base else None
    lines = [f"# Speed test: {script.name}", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · trainer `{Path(cap['train']).name}` · {steps} steps per variant · "
             f"video memory {gb(vram.total)}", "",
             "| Variant | s/step | vs. now | Peak memory | Result | What changed |", "|---|---|---|---|---|---|"]
    for r in results:
        d = (f"{(r['s_per_step'] / b - 1) * 100:+.0f}%" if b and r["s_per_step"] else "")
        lines.append(f"| {r['name']} | {r['s_per_step'] or '-'} | {d} | {gb(r['vram'])} | {r['status']} | {r['why']} |")
    good = [r for r in results if r["status"] == "ok" and r["s_per_step"]]
    best = min(good, key=lambda r: r["s_per_step"]) if good else None
    lines += [""]
    if best and b and best["name"] != "baseline":
        lines += [f"**Fastest that fits: {best['name']}**, {best['s_per_step']} s/step instead of {b} "
                  f"({(1 - best['s_per_step'] / b) * 100:.0f}% less time per step).", "",
                  "To use it, change these in the launch script (a text editor, or LoRA Studio › step 9 › The script text):", ""]
        for k, old, new in changes(base["args"], best["args"]):
            if new is MISSING:
                lines.append(f"- remove `--{k}{(' ' + old) if old else ''}`")
            elif old is MISSING:
                lines.append(f"- add `--{k}{(' ' + new) if new else ''}`")
            else:
                lines.append(f"- change `--{k} {old}` to `--{k} {new}`")
        for k, v in best["env"].items():
            lines.append(f"- add `export {k}={v}` near the other `export` lines")
    elif b:
        lines.append("Your current settings are already the fastest of the ones tried.")
    lines += ["", "Notes:", "- s/step is the median after the first warm-up steps; the first steps of tunableop and "
              "compile are slow while they tune, so a longer run can gain more than shown.",
              "- fp8 and attention changes can change the result a little: check the loss and a test picture.",
              f"- logs of every variant: `{Path(results[0]['log']).parent}`"]
    out = script.with_name(f"speedtest_{script.stem}_{time.strftime('%Y%m%d_%H%M')}.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _say(line: str, progress: bool = False):
    print(("\r[speedtest] " + line + "   ") if progress else ("[speedtest] " + line), end="" if progress else "\n",
          flush=True)


def change_list(base_args, best_args) -> list[dict]:
    out = []
    for k, old, new in changes(base_args, best_args):
        out.append({"opt": k, "op": "remove" if new is MISSING else "add" if old is MISSING else "change",
                    "old": None if old is MISSING else old, "new": None if new is MISSING else new})
    return out


def test_script(script: Path, depth: str = "quick", steps: int = 15, warm: int = 4, timeout: int = 1200,
                only: str = "", combo: bool = True, say=_say, stop=None, report_dir: Path | None = None) -> dict:
    """The whole test of one launch script. depth: 'now' (only how fast it is now), 'quick' or 'full'.
    Returns {'report', 'baseline', 'best', 'best_s', 'changes', 'env', 'results', 'status'}."""
    script = Path(script).resolve()
    work = Path(tempfile.mkdtemp(prefix="lsspeedtest_"))
    say(f"preparing: running {script.name} up to its training command (no training) ...")
    cap = capture(script, work)
    opts = supported(cap) if depth != "now" else set()
    vs = variants(cap["args"], opts, depth != "full")
    if depth == "now":
        vs = vs[:1]
    if only:
        keep = {x.strip() for x in only.split(",")} | {"baseline"}
        vs = [v for v in vs if v["name"] in keep]
    vram = VRAM()
    if vram.used() > 2 * 2 ** 30:
        say(f"note: {gb(vram.used())} of video memory is already in use (ComfyUI? Ollama?). "
            "Results are better with it free.")
    steps = max(steps, warm + 4)
    results = []
    out_dir = work / "out"

    def one(v):
        out_dir.mkdir(exist_ok=True)
        r = run_variant(cap, v, steps, warm, out_dir, timeout, vram, work, say, stop)
        shutil.rmtree(out_dir, ignore_errors=True)
        say(f"  -> {r['s_per_step'] or '-'} s/step, peak {gb(r['vram'])}, {r['status']}")
        results.append(r)
        return r

    try:
        for n, v in enumerate(vs, 1):
            if stop is not None and stop.is_set():
                break
            say(f"{n}/{len(vs)} {v['name']}: {v['why']}")
            one(v)
        base = next((r for r in results if r["name"] == "baseline" and r["s_per_step"]), None)
        wins = [r for r in results if base and r["name"] != "baseline" and r["status"] == "ok" and r["s_per_step"]
                and r["s_per_step"] < base["s_per_step"] * 0.97]
        if len(wins) >= 2 and combo and not (stop is not None and stop.is_set()):
            args, env, names = list(cap["args"]), {}, []
            swaps = [r for r in wins if r["name"].startswith("swap")]
            for r in sorted(wins, key=lambda r: r["s_per_step"]):
                if r["name"].startswith("swap") and r is not min(swaps, key=lambda x: x["s_per_step"]):
                    continue
                if r["name"] in ATTN and any(n in ATTN for n in names):
                    continue
                if r["name"] == "pinned" and "swap0" in names:
                    continue                            # nothing is swapped any more
                args = apply_changes(args, changes(cap["args"], r["args"]))
                env.update(r["env"])
                names.append(r["name"])
            say(f"together: {'+'.join(names)}")
            r = one({"name": "+".join(names), "why": "the faster ones together", "args": args, "env": env})
            if r["status"] == "out of memory" and "no_checkpointing" in names and len(names) > 2:
                names2 = [x for x in names if x != "no_checkpointing"]
                say(f"together without the memory-hungry one: {'+'.join(names2)}")
                one({"name": "+".join(names2), "why": "the faster ones together, keeping gradient checkpointing",
                     "args": setopt(args, "gradient_checkpointing"), "env": env})
    except KeyboardInterrupt:
        say("stopped; writing the report of what ran")
    if not results:
        return {"status": "nothing ran", "results": []}
    out = report(script, cap, results, vram, steps)
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        out = Path(shutil.move(str(out), Path(report_dir) / out.name))
    base = next((r for r in results if r["name"] == "baseline"), None)
    good = [r for r in results if r["status"] == "ok" and r["s_per_step"]]
    best = min(good, key=lambda r: r["s_per_step"]) if good else None
    return {"status": "done" if not (stop is not None and stop.is_set()) else "stopped", "report": str(out),
            "baseline": base and base["s_per_step"], "best": best and best["name"],
            "best_s": best and best["s_per_step"], "vram_total": vram.total, "depth": depth,
            "changes": change_list(base["args"], best["args"]) if base and best and best is not base else [],
            "env": dict(best["env"]) if best and best is not base else {},
            "results": [{k: r[k] for k in ("name", "why", "status", "s_per_step", "vram")} for r in results]}


def main():
    ap = argparse.ArgumentParser(description="Speed test for a LoRA Studio launch script.")
    ap.add_argument("script", type=Path)
    ap.add_argument("--steps", type=int, default=15, help="training steps per variant (default 15)")
    ap.add_argument("--warmup", type=int, default=4, help="steps not measured at the start (default 4)")
    ap.add_argument("--quick", action="store_true", help="fewer variants")
    ap.add_argument("--now", action="store_true", help="only measure how fast it is now")
    ap.add_argument("--only", default="", help="comma-separated variant names (see --list)")
    ap.add_argument("--list", action="store_true", help="show the variants and stop")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per variant before it is stopped")
    ap.add_argument("--no-combo", action="store_true", help="do not try the winners together")
    a = ap.parse_args()
    script = a.script.resolve()
    if not script.is_file():
        sys.exit(f"no such launch script: {script}")
    if a.list:
        cap = capture(script, Path(tempfile.mkdtemp(prefix="lsspeedtest_")))
        for v in variants(cap["args"], supported(cap), a.quick):
            print(f"  {v['name']:18s} {v['why']}")
        return
    r = test_script(script, "now" if a.now else "quick" if a.quick else "full", a.steps, a.warmup, a.timeout,
                    a.only, not a.no_combo)
    if r.get("report"):
        print(f"\n[speedtest] report: {r['report']}\n")
        print(Path(r["report"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
