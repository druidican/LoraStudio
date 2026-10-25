# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The machine check: does LoRA Studio work on THIS computer, with your folders, trainers, GPU and model files?

    ./venv/bin/python selftest.py --machine            (all checks, then a 10-step training on made-up pictures)
    ./venv/bin/python selftest.py --machine --no-train (the checks only)

or Settings › Version › Check this machine. It reads your settings (never changes them), looks at the GPU, the
trainers and their venv, the model files of every LoRA type, Ollama, ComfyUI and the picture tools, and then trains a
tiny LoRA (10 steps at 512 px, 8 made-up pictures, in a temporary folder that is deleted afterwards) with the same
launch script LoRA Studio would write, to see that a real run starts, uses the GPU, logs a loss and saves a file.
The report is written to lorastudio_check_<time>.md in the program folder: no token, your home folder shown as ~,
no dataset names (only how many there are). Upload that file when something is not right.
"""
from __future__ import annotations
__version__ = "1.251026h"  # LoRA Studio version, see version.py

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = str(Path.home())
TRAIN_STEPS = 10
TRAIN_RES = "512,512"
TRAIN_TIMEOUT = 30 * 60
# SDXL-family types first: the cheapest real training; then any type whose model files are all there
TRAIN_ORDER = ["illustrious", "sdxl", "pony", "noobai", "sd15"]

GPU_TEST = r"""
import json, time
r = {}
try:
    import torch
    r["torch"] = torch.__version__
    r["hip"] = getattr(torch.version, "hip", None)
    r["cuda"] = torch.version.cuda
    r["gpu_ok"] = torch.cuda.is_available()
    if r["gpu_ok"]:
        p = torch.cuda.get_device_properties(0)
        r["name"] = torch.cuda.get_device_name(0)
        r["arch"] = getattr(p, "gcnArchName", "")
        r["vram_gb"] = round(p.total_memory / 1024 ** 3, 1)
        r["bf16"] = bool(torch.cuda.is_bf16_supported())
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20):
            b = a @ a
        torch.cuda.synchronize()
        r["fp16_tflops"] = round(20 * 2 * 2048 ** 3 / (time.time() - t) / 1e12, 1)
except Exception as e:
    r["error"] = repr(e)[:300]
for m in ("accelerate", "transformers", "diffusers", "safetensors", "bitsandbytes", "xformers", "tensorboard"):
    try:
        r[m] = getattr(__import__(m), "__version__", "?")
    except Exception:
        r[m] = "missing"
try:                                     # the ROCm pip parts (rocm-sdk-*), which must all match torch's +rocmX.Y.Z
    from importlib.metadata import distributions
    r["rocm_parts"] = sorted({(d.metadata["Name"] or "") + " " + d.version for d in distributions()
                              if (d.metadata["Name"] or "").lower().startswith("rocm")
                              and (d.metadata["Name"] or "").lower() != "rocm-bootstrap"})
except Exception:
    r["rocm_parts"] = []
print("##GPU " + json.dumps(r))
"""


def _tilde(s) -> str:
    s = str(s)
    return s.replace(HOME, "~") if HOME and HOME != "/" else s


class Report:
    def __init__(self, say):
        self.say = say                     # callback(line) for live output
        self.rows = []                     # (section, state, title, detail)
        self.sec = ""
        self.extra = []

    def section(self, name):
        self.sec = name
        self.say(f"\n== {name}")

    def add(self, state, title, detail=""):
        detail = _tilde(detail)
        self.rows.append((self.sec, state, title, detail))
        mark = {"ok": "✓", "warn": "!", "bad": "✕", "info": "·"}[state]
        self.say(f"  {mark} {title}" + (f": {detail}" if detail else ""))

    def counts(self):
        c = {"ok": 0, "warn": 0, "bad": 0}
        for r in self.rows:
            if r[1] in c:
                c[r[1]] += 1
        return c

    def markdown(self, version) -> str:
        c = self.counts()
        out = [f"# LoRA Studio machine check", "",
               f"- LoRA Studio {version}, {time.strftime('%Y-%m-%d %H:%M')}",
               f"- Result: {c['ok']} fine, {c['warn']} to look at, {c['bad']} not working", ""]
        sec = None
        for s, state, title, detail in self.rows:
            if s != sec:
                out += ["", f"## {s}", "", "| | Check | Detail |", "|---|---|---|"]
                sec = s
            mark = {"ok": "✓", "warn": "!", "bad": "✕", "info": "·"}[state]
            out.append(f"| {mark} | {title} | {detail.replace('|', '/')} |")
        for title, text in self.extra:
            out += ["", f"## {title}", "", "```", _tilde(text).rstrip(), "```"]
        return "\n".join(out) + "\n"


def _run(cmd, timeout=60, env=None, cwd=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:                               # noqa: BLE001
        return -1, repr(e)


def _git(folder: Path) -> str:
    if not (folder / ".git").exists() or not shutil.which("git"):
        return ""
    code, out = _run(["git", "-C", str(folder), "log", "-1", "--format=%h %cs"], 15)
    return out.strip() if code == 0 else ""


def _free_gb(p) -> float | None:
    try:
        return round(shutil.disk_usage(p).free / 1024 ** 3, 1)
    except OSError:
        return None


# ------------------------------------------------------------------------------------------------ setup like the app
def configure():
    """Point sdgui at your folders exactly as lora_studio.py does at start (nothing is written)."""
    sys.path.insert(0, str(HERE))
    import lora_library as ll
    import lora_studio  # noqa: F401  (apply_folders, use_own_config_folder)
    import sdgui
    s = ll.LIB.settings
    lora_studio.apply_folders(s)
    lora_studio.use_own_config_folder()
    sdgui.DEFAULT_VENV = sdgui.detect_venv()
    sdgui.SCAN_SCRIPT = sdgui.find_scan_script(None)
    sdgui.apply_gfx_env()
    return sdgui, ll, s


# ------------------------------------------------------------------------------------------------ the checks
def check_program(R, version):
    R.section("Program")
    R.add("info", "LoRA Studio", f"{version.VERSION} in {HERE}")
    bad = version.problems()
    R.add("ok" if not bad else "warn", "Program files",
          f"all {len(version.PARTS)} are the ones of {version.VERSION}" if not bad else
          ", ".join(f"{p['name']} ({p['state']} {p['version']})" for p in bad))
    R.add("info", "System", f"{platform.platform()} · Python {platform.python_version()}")
    try:
        import gradio
        R.add("info", "Gradio", gradio.__version__)
    except Exception as e:                               # noqa: BLE001
        R.add("bad", "Gradio", repr(e))


def check_folders(R, G, s):
    R.section("Folders")
    ds = [Path(os.path.expanduser(d)) for d in s.get("dataset_dirs") or []]
    if not ds or not ds[0].is_dir():
        R.add("bad", "Dataset folder", "not set or missing (Settings › Folders)")
    else:
        n = len(G.list_lora_dirs())
        R.add("ok", "Dataset folder", f"{ds[0]} · {n} datasets · {_free_gb(ds[0])} GB free")
    lds = [Path(os.path.expanduser(d)) for d in s.get("lora_dirs") or []]
    if not lds:
        R.add("warn", "LoRA folder", "not set: finished LoRAs cannot be copied to ComfyUI")
    for d in lds:
        R.add("ok" if d.is_dir() else "warn", "LoRA folder", f"{d}" + ("" if d.is_dir() else " does not exist"))
    R.add("ok" if os.access(HERE, os.W_OK) else "bad", "Program folder writable", f"{HERE} · {_free_gb(HERE)} GB free")


def check_trainers(R, G):
    R.section("Trainers")
    import features as FT
    if not FT.have("sd_scripts") and not FT.have("musubi"):
        R.add("info", "Trainers", "not installed here (your choice): launch scripts are made, not run")
        return ""
    sd = G.sd_root()
    ok_sd = (sd / "sdxl_train_network.py").is_file()
    R.add("ok" if ok_sd else "bad", "sd-scripts", f"{sd}" + (f" · {_git(sd)}" if _git(sd) else "")
          + ("" if ok_sd else " · sdxl_train_network.py not found"))
    mu = G.musubi_root()
    if (mu / "src" / "musubi_tuner").is_dir():
        R.add("ok", "musubi-tuner", f"{mu}" + (f" · {_git(mu)}" if _git(mu) else ""))
    else:
        R.add("info", "musubi-tuner", "not installed (only needed for Z-Image, Qwen-Image, FLUX.2)")
    wd, venv, _ = G.runner_settings()
    act = Path(os.path.expanduser(venv)) / "bin" / "activate" if venv else None
    R.add("ok" if act and act.is_file() else "bad", "Training venv", f"{venv or 'not set'}"
          + ("" if act and act.is_file() else " · no bin/activate (Settings › Trainers and GPU)"))
    env = {k: v for k, v in G.env_saved().items()} if hasattr(G, "env_saved") else {}
    R.add("info", "GPU settings", ", ".join(f"{k}={v}" for k, v in env.items()) or "none saved")
    return venv if act and act.is_file() else ""


def check_gpu(R, G, venv):
    R.section("GPU")
    try:
        import statusbar
        gs = statusbar.machine()["gpus"]
        for g in gs:
            R.add("ok", "GPU (rocm-smi / nvidia-smi)", f"{g['name']} · {g['vram_total'] / 1024 ** 3:.0f} GB"
                  f" · {g['vram_used'] / 1024 ** 3:.1f} GB used now")
        if not gs:
            R.add("warn", "GPU (rocm-smi / nvidia-smi)", "not found")
    except Exception as e:                               # noqa: BLE001
        R.add("warn", "GPU (rocm-smi / nvidia-smi)", repr(e))
    try:
        v, msg = G.detect_gfx()
        R.add("info", "GPU architecture", msg or str(v))
    except Exception:                                    # noqa: BLE001
        pass
    if not venv:
        R.add("bad", "PyTorch in the training venv", "no venv to test")
        return {}
    py = Path(os.path.expanduser(venv)) / "bin" / "python"
    R.say("  … starting PyTorch in the training venv (can take a minute)")
    code, out = _run([str(py), "-c", GPU_TEST], timeout=300, env=os.environ.copy())
    m = re.search(r"^##GPU (.*)$", out, re.M)
    if not m:
        R.add("bad", "PyTorch in the training venv", out.strip()[-400:])
        return {}
    r = json.loads(m.group(1))
    if r.get("error"):
        R.add("bad", "PyTorch", r["error"])
    elif not r.get("gpu_ok"):
        R.add("bad", "PyTorch sees the GPU", f"torch {r.get('torch')} · no GPU (wrong torch build, or "
                                              "HSA_OVERRIDE_GFX_VERSION needed: Settings › Trainers and GPU)")
    else:
        R.add("ok", "PyTorch sees the GPU", f"{r['name']} {r.get('arch', '')} · {r['vram_gb']} GB · torch {r['torch']}"
              + (f" · ROCm/HIP {r['hip']}" if r.get("hip") else f" · CUDA {r.get('cuda')}"))
        R.add("ok" if r.get("bf16") else "warn", "bf16", "supported" if r.get("bf16") else
              "not supported: use fp16 mixed precision")
        R.add("info", "Speed (fp16 matrix test)", f"{r['fp16_tflops']} TFLOPS")
    want = (r.get("torch") or "").partition("+rocm")[2]
    if want and r.get("rocm_parts"):
        odd = [x for x in r["rocm_parts"] if x.split(" ")[-1] != want]
        if odd:
            R.add("bad", "ROCm parts match PyTorch", f"torch is built for ROCm {want}, but: {', '.join(odd)}. Mixed "
                  "versions crash trainings ('Memory access fault by GPU'): ./install.sh puts them back")
        else:
            R.add("ok", "ROCm parts match PyTorch", f"all {len(r['rocm_parts'])} at {want}")
    for k in ("accelerate", "transformers", "diffusers", "safetensors", "bitsandbytes", "tensorboard"):
        v = r.get(k)
        need = k in ("accelerate", "transformers", "diffusers", "safetensors")
        R.add("ok" if v != "missing" else ("bad" if need else "info"), k, v)
    return r


def check_models(R, G):
    R.section("Model files per LoRA type")
    sc, presets = G.merged_presets()
    if sc is None:
        R.add("bad", "Lora_Datatypes.py", G.PRESETS_ERROR["msg"])
        return {}
    usable = {}
    for tkey in sorted(presets):
        files = {f.lstrip("-"): v for f, v in presets[tkey].get("model_args", [])}
        files.update(G.settings_models(tkey))                 # your choices in Settings › Model files win
        files = {k: v for k, v in files.items() if v}
        label = presets[tkey].get("label", tkey)
        if not files or all(v.startswith(G._PH) for v in files.values()):
            R.add("info", label, "no model files chosen")
            continue
        missing = [f"{k}: {Path(v).name}" + (" (not chosen)" if v.startswith(G._PH) else "")
                   for k, v in files.items() if v.startswith(G._PH) or not Path(os.path.expanduser(v)).exists()]
        if missing:
            R.add("bad", label, "missing " + ", ".join(missing))
        else:
            R.add("ok", label, ", ".join(Path(v).name for v in files.values()))
            usable[tkey] = files
    return usable


def check_helpers(R):
    R.section("Helpers")
    import features as FT
    try:
        if not FT.have("ollama"):
            raise LookupError("not installed (your choice)")
        import ollamactl
        url, model = ollamactl.settings()
        ms = ollamactl.installed(url)
        vis = [m["name"] for m in ms if m.get("vision")]
        R.add("ok" if vis else "warn", "Ollama", f"{url} · {len(ms)} models, {len(vis)} that see pictures"
              + (f" · chosen: {model}" if model else ""))
    except LookupError as e:
        R.add("info", "Ollama", str(e))
    except Exception as e:                               # noqa: BLE001
        R.add("warn", "Ollama", f"not reachable ({type(e).__name__}): captions need it")
    try:
        import comfytest as CT
        cs = CT.settings()
        alive = CT.alive(cs["url"])
        fams = [f for f, wf in (cs.get("families") or {}).items() if wf]
        R.add("ok" if alive else "warn", "ComfyUI", f"{cs['url']} " + ("answers" if alive else "does not answer")
              + f" · workflows read: {', '.join(fams) or 'none'}")
        if alive:
            R.add("info", "ComfyUI LoRA list", f"{len(CT.lora_names(cs['url']))} LoRAs")
    except Exception as e:                               # noqa: BLE001
        R.add("warn", "ComfyUI", repr(e))
    try:
        import pictools
        st = pictools.status(fresh=True)
        R.add("ok" if st["ok"] else "warn", "Picture tools", st["text"])
    except Exception as e:                               # noqa: BLE001
        R.add("warn", "Picture tools", repr(e))


# ------------------------------------------------------------------------------------------------ the tiny training
def _pictures(folder: Path, n=8):
    from PIL import Image, ImageDraw
    folder.mkdir(parents=True)
    for i in range(n):
        im = Image.new("RGB", (512, 640), (40 + 20 * i, 90, 160 - 10 * i))
        d = ImageDraw.Draw(im)
        d.ellipse((150, 100 + 5 * i, 360, 420), fill=(220, 180, 140))
        d.rectangle((190, 430, 320, 620), fill=(30, 30, 30))
        im.save(folder / f"pic_{i:02d}.png")
        (folder / f"pic_{i:02d}.txt").write_text(f"lscheck, a person, plain background, picture {i}\n")


def tiny_training(R, G, usable, venv):
    R.section("Tiny training")
    if not venv:
        R.add("bad", "Training", "skipped: no training venv")
        return
    if getattr(G, "RUN", None) is not None and G.RUN.running():
        R.add("warn", "Training", "skipped: a training run is active in LoRA Studio")
        return
    _, presets = G.merged_presets()

    def rank(t):
        c = presets.get(t, {})
        base = c.get("base") or t
        if t in TRAIN_ORDER:
            return TRAIN_ORDER.index(t)
        if base in TRAIN_ORDER:
            return len(TRAIN_ORDER) + TRAIN_ORDER.index(base)
        return 99 if not G.is_musubi_cfg(c) else 999
    tkey = min(usable, key=rank) if usable else None
    if tkey and rank(tkey) >= 999:
        tkey = None
    if not tkey:
        R.add("warn", "Training", "skipped: no SDXL-family or SD 1.5 type has all its model files "
                                  "(Settings › Model files)")
        return
    tmp = Path(tempfile.mkdtemp(prefix="._lscheck_", dir=HERE))
    old = G.IMAGES_DIR
    try:
        ds = "lscheck"
        _pictures(tmp / ds / f"1_{ds}")
        G.IMAGES_DIR = tmp
        p = G.make_plan(ds, tkey, hf=False, samples=False)
        text = p["script_text"]
        vals = G.read_form(text)
        names = [n for n, _ in G.FORM_FIELDS]
        want = {"max_train_epochs": "1", "resolution": TRAIN_RES, "save_every_n_epochs": "1",
                "train_batch_size": "1", "gradient_accumulation_steps": "1", "max_data_loader_n_workers": "0"}
        vals = [want.get(n, v) for n, v in zip(names, vals)]
        text, _, failed = G.apply_form(text, vals)
        text, ok = G.append_args(text, f"--max_train_steps {TRAIN_STEPS}")
        if failed or not ok:
            R.add("bad", "Training", f"could not shorten the launch script ({', '.join(failed) or 'max_train_steps'})")
            return
        script = tmp / ds / "lscheck.sh"
        script.write_text(text)
        act = Path(os.path.expanduser(venv)) / "bin" / "activate"
        lines = text.split("\n")
        i = 1 if lines and lines[0].startswith("#!") else 0
        lines[i:i] = [f"source '{act}'"]
        run = tmp / ds / ".run.sh"
        run.write_text("\n".join(lines) + "\n")
        wd, _, _ = G.runner_settings()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = os.pathsep.join(x for x in env.get("PATH", "").split(os.pathsep) if x != venv_bin)
        env.pop("VIRTUAL_ENV", None)
        try:
            import ollamactl
            ollamactl.stop_all()                 # free the video memory, as a real run does
        except Exception:                        # noqa: BLE001
            pass
        R.say(f"  … training {TRAIN_STEPS} steps of {p['filename']} ({tkey}, {TRAIN_RES} px); "
              "the first start loads the model and can take a few minutes")
        t0 = time.time()
        log = []
        proc = subprocess.Popen(["bash", str(run)], cwd=wd or str(tmp), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
        last = 0.0
        try:
            for line in proc.stdout:
                log.append(line.rstrip("\n"))
                if time.time() - last > 15 and G.PROG_RE.search(line):
                    R.say("    " + G.PROG_RE.search(line).group(1))
                    last = time.time()
                if time.time() - t0 > TRAIN_TIMEOUT:
                    proc.kill()
                    log.append(f"### stopped after {TRAIN_TIMEOUT // 60} minutes")
                    break
            proc.wait(timeout=60)
        except Exception as e:                   # noqa: BLE001
            proc.kill()
            log.append(f"### {e!r}")
        dt = time.time() - t0
        whole = "\n".join(log)
        made = list((tmp / ds / "model").rglob("*.safetensors"))
        prog = G.PROG_RE.findall(whole)
        loss = re.findall(r"avr_loss=([0-9.]+|nan)", whole)
        oom = re.search(r"out of memory|HIP error|CUDA error|hipErrorOutOfMemory", whole, re.I)
        if made and proc.returncode == 0:
            R.add("ok", "Training ran", f"{p['filename']}: {TRAIN_STEPS} steps in {dt:.0f} s, saved "
                                        f"{made[0].name} ({made[0].stat().st_size / 1e6:.0f} MB)")
        else:
            R.add("bad", "Training ran", f"exit code {proc.returncode} after {dt:.0f} s"
                  + (f" · {oom.group(0)}" if oom else "") + " · see the log at the end of the report")
        if prog:
            m = re.search(r"([0-9.]+)(s/it|it/s)", prog[-1])
            R.add("info", "Speed", prog[-1].strip() if not m else
                  f"{m.group(1)} {m.group(2)} at {TRAIN_RES} px, batch 1 (a full-size run is about 3-4 x slower per step)")
        if loss:
            R.add("ok" if loss[-1] != "nan" else "bad", "Loss", f"last {loss[-1]}"
                  + (" · NaN: lower the learning rate or use fp16/bf16 correctly" if loss[-1] == "nan" else ""))
        elif made:
            R.add("warn", "Loss", "no avr_loss in the log")
        tail = "\n".join(log[-60:])
        if not (made and proc.returncode == 0):
            R.extra.append(("Training log (last 60 lines)", tail))
        R.extra.append(("The launch script used", text))
    except Exception as e:                       # noqa: BLE001
        R.add("bad", "Training", repr(e))
    finally:
        G.IMAGES_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


def run(train=True, say=print) -> Path:
    import version
    R = Report(say)
    say(f"LoRA Studio machine check · {version.VERSION}")
    check_program(R, version)
    G, ll, s = configure()
    check_folders(R, G, s)
    venv = check_trainers(R, G)
    check_gpu(R, G, venv)
    usable = check_models(R, G)
    check_helpers(R)
    if train:
        tiny_training(R, G, usable, venv)
    out = HERE / f"lorastudio_check_{time.strftime('%Y%m%d_%H%M')}.md"
    out.write_text(R.markdown(version.VERSION), encoding="utf-8")
    c = R.counts()
    say(f"\n{c['ok']} fine, {c['warn']} to look at, {c['bad']} not working")
    say(f"Report: {out}")
    return out
