# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Tags with the WD tagger (SmilingWolf/wd-eva02-large-tagger-v3) instead of Ollama, for the tag-style caption sets
(Illustrious, Pony, SDXL anime, Anima: booru tags).

It runs sd-scripts' own tagger (finetune/tag_images_by_wd14_tagger.py --onnx) in the training venv, on a temporary
folder of links to the pictures, so nothing else in the dataset is touched. Only the general tags are kept: no
character, copyright, artist, meta or rating tags, so the trigger word stands for the character, as usual. Each line
then goes through the same cleaning as Ollama's tags (captioner.clean_tags): the trigger word first, the "Never
describe" traits taken out, at most the tag count of the Length setting. The consistency check works on the result
like on any caption set.

Needs onnx + onnxruntime in the training venv: installed by itself the first time in the venv LoRA Studio made
(onnxruntime-gpu on NVIDIA; on AMD it runs on the CPU, about a second or two per picture); for your own venv the
command is shown. The model (about 1.2 GB) is downloaded the first time into userdata/wd_tagger/.
"""
from __future__ import annotations
__version__ = "1.261026r"  # LoRA Studio version, see version.py

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import lspaths as _LP

REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_DIR = _LP.USER / "wd_tagger"
SCRIPT = Path("finetune") / "tag_images_by_wd14_tagger.py"
THRESH = 0.35                                    # general tags with at least this confidence
OFF = "1.1"                                      # above 1 = the category is never added
EXT = ".wdtags"                                  # the tagger's own files, in the temporary folder only


def _sg():
    return sys.modules.get("sdgui")


def training_python() -> tuple[Path | None, bool]:
    try:
        import venvcheck
        return venvcheck.training_python()
    except Exception:                            # noqa: BLE001
        return None, False


def status() -> dict:
    """{ok, text, script, py, managed, need_install}"""
    sg = _sg()
    sd = Path(sg.sd_root()) if sg is not None else _LP.ROOT / "sd-scripts"
    script = sd / SCRIPT
    py, managed = training_python()
    if not script.is_file():
        return {"ok": False, "text": f"sd-scripts' tagger is not there ({script})", "script": script, "py": py,
                "managed": managed, "need_install": False}
    if not py:
        return {"ok": False, "text": "no training venv found", "script": script, "py": py, "managed": managed,
                "need_install": False}
    r = subprocess.run([str(py), "-c", "import onnx, onnxruntime"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return {"ok": False, "text": "onnx / onnxruntime are not installed in the training venv", "script": script,
                "py": py, "managed": managed, "need_install": True}
    return {"ok": True, "text": f"ready ({REPO})", "script": script, "py": py, "managed": managed,
            "need_install": False}


def _packages() -> list[str]:
    try:
        import features
        nv = features.gpu_kind() == "nvidia"
    except Exception:                            # noqa: BLE001
        nv = False
    return ["onnx", "onnxruntime-gpu" if nv else "onnxruntime"]


def install_command(py) -> str:
    return f"{py} -m pip install {' '.join(_packages())}"


def _install(py: Path):
    """Generator: pip install onnx + onnxruntime into the venv LoRA Studio made (within constraints.txt)."""
    cons = _LP.ROOT / "constraints.txt"
    cmd = [str(py), "-m", "pip", "install", "-q", *_packages()] + (["-c", str(cons)] if cons.is_file() else [])
    yield f"⏳ Installing the WD tagger packages ({', '.join(_packages())}) into the training venv, once …"
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        yield "❌ " + ((r.stdout + r.stderr).strip().splitlines() or ["pip failed"])[-1]
        return False
    return True


def tag(images: list[Path], trigger: str, leave_out: str, max_tags: int, write, stop=None):
    """Generator of progress lines. `write(img, line)` stores one finished caption. Returns nothing; the last line
    says how it went."""
    import captioner as CAP
    st = status()
    if not st["ok"] and st["need_install"]:
        if not st["managed"]:
            yield (f"❌ The WD tagger needs onnx and onnxruntime in your training venv. Run once:\n\n"
                   f"`{install_command(st['py'])}`")
            return
        ok = yield from _install(st["py"])
        if ok is False:
            return
        st = status()
    if not st["ok"]:
        yield f"❌ The WD tagger cannot run: {st['text']}."
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="wdtag_", dir=_LP.USER))
    try:
        links = {}
        for i, p in enumerate(images):
            ln = tmp / f"{i:05d}{p.suffix.lower()}"
            try:
                os.symlink(p.resolve(), ln)
            except OSError:
                shutil.copy2(p, ln)
            links[ln.stem] = p
        first = not any(MODEL_DIR.rglob("*.onnx"))
        yield (f"⏳ WD tagger: loading {REPO}" + (" (first time: downloading about 1.2 GB) …" if first else " …"))
        sd = st["script"].parent.parent
        cmd = [str(st["py"]), "-u", str(st["script"]), str(tmp), "--onnx", "--repo_id", REPO,
               "--model_dir", str(MODEL_DIR), "--batch_size", "4", "--caption_extension", EXT,
               "--general_threshold", str(THRESH), "--character_threshold", OFF, "--copyright_threshold", OFF,
               "--artist_threshold", OFF, "--meta_threshold", OFF, "--model_threshold", OFF, "--remove_underscore",
               "--max_data_loader_n_workers", "2"]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        tok = ""
        sg = _sg()
        try:
            tok = sg.read_hf_token() if sg is not None else ""
        except Exception:                        # noqa: BLE001
            pass
        if tok:
            env["HF_TOKEN"] = tok
        proc = subprocess.Popen(cmd, cwd=str(sd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=env, errors="replace")
        tail, t0, last = [], time.time(), 0.0
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-12:]
            if stop is not None and stop.is_set():
                proc.terminate()
                break
            if time.time() - last > 2:
                last = time.time()
                done = len(list(tmp.glob(f"*{EXT}")))
                yield f"⏳ WD tagger: {done}/{len(images)} pictures tagged"
        proc.wait()
        n, empty = 0, 0
        for f in sorted(tmp.glob(f"*{EXT}")):
            img = links.get(f.stem)
            if img is None:
                continue
            raw = f.read_text(encoding="utf-8", errors="ignore")
            line, _ = CAP.clean_tags(raw, trigger, leave_out, max_tags)
            if line.count(",") < 1:
                empty += 1
            write(img, line)
            n += 1
        if stop is not None and stop.is_set():
            yield f"■ Stopped: {n} of {len(images)} pictures tagged by the WD tagger."
        elif proc.returncode != 0 and n == 0:
            yield "❌ The WD tagger stopped:\n\n```\n" + "\n".join(tail[-8:]) + "\n```"
        else:
            yield (f"✔ Done: {n} of {len(images)} pictures tagged by the WD tagger ({REPO}), "
                   f"in {int(time.time() - t0)} s" + (f"; {empty} got almost no tags" if empty else "") + ".")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
