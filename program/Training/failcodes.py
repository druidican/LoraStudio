# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""What went wrong in a failed training run, in a few words instead of a long debug log.

explain(log_text, exit_code) -> {"code", "title", "what", "fix", "line"} (or None when nothing is recognised).
The patterns are checked in order; the first one that matches wins, so the specific ones come first. "line" is the
log line that matched (shown under the description, for the curious).
"""
from __future__ import annotations
__version__ = "1.251026o"  # LoRA Studio version, see version.py

import re

# (code, regex, title, what happened, what to do)
RULES = [
    ("OOM", r"out of memory|OutOfMemoryError|hipErrorOutOfMemory|HIP out of memory|CUDA out of memory|"
            r"Tried to allocate [\d.]+ [GM]iB",
     "Out of video memory (OOM)",
     "The GPU ran out of video memory while training: the model, the pictures at this resolution and the batch did "
     "not fit in its memory together.",
     "Lower the batch size to 1, or the resolution (1024 → 896 or 768); turn on gradient checkpointing; use a smaller "
     "network dim; close other programs on the GPU (ComfyUI, Ollama: LoRA Studio stops Ollama, not ComfyUI)."),
    ("RAM", r"Killed\b|MemoryError|Cannot allocate memory|oom-kill|exit code -9|exited with code 137",
     "Out of normal memory (RAM)",
     "The computer ran out of RAM and the system stopped the trainer (often while loading the model or caching).",
     "Close other programs; use fewer data loader workers (0 or 1); cache latents to disk; add swap space."),
    ("NAN", r"avr_loss=nan|loss is nan|loss: nan|NaN detected|nan or inf",
     "The loss became NaN (training broke down)",
     "The numbers in the training exploded: usually a learning rate that is too high for this model, or fp16 where "
     "the model needs bf16.",
     "Lower the learning rate (for example 1e-4 → 5e-5); use bf16 mixed precision; check a few pictures are not "
     "broken files."),
    ("MODEL", r"(No such file or directory|FileNotFoundError|does not exist|not found).{0,200}\.(safetensors|ckpt|pt|pth|bin)"
              r"|safetensors_rust\.SafetensorError|Error while deserializing header|HeaderTooLarge|invalid load key",
     "A model file is missing or broken",
     "The base model, VAE or text encoder in the launch script was not found, or the file is damaged (often a "
     "download that stopped early).",
     "Check the model files in Settings › Model files and in the launch script; download a broken file again."),
    ("WRONGMODEL", r"size mismatch for|Missing key\(s\) in state_dict|Unexpected key\(s\) in state_dict|"
                   r"is not a valid model|unknown model type|Could not detect model type",
     "The model does not fit this LoRA type",
     "The base model file is of another kind than the LoRA type expects (for example an SD 1.5 model in an SDXL "
     "script, or a VAE where the checkpoint should be).",
     "Pick the right base model for the LoRA type in Settings › Model files, then make the launch script again."),
    ("NODATA", r"No data found|no images found|num train images\s*\*\s*repeats:\s*0\b|total images.*: 0\b|"
               r"ZeroDivisionError|dataset is empty|no <N>_name folders with images",
     "No pictures to train on",
     "The trainer found no pictures: the repeat folder is empty, has another name than the script expects, or the "
     "pictures are in a folder the trainer does not read.",
     "Open the dataset in Datasets › Edit existing: the repeat folder must be <N>_<dataset name> with pictures in it."),
    ("CAPTION", r"caption file not found|captions? (are|is) missing|UnicodeDecodeError",
     "A caption file could not be read",
     "A caption is missing where the script needs one, or a caption file has characters the trainer cannot read.",
     "Datasets › 8 Review lists pictures without a caption; save broken caption files again as UTF-8."),
    ("IMAGE", r"cannot identify image file|PIL\.UnidentifiedImageError|image file is truncated|broken data stream|"
              r"Decompressed Data Too Large",
     "A picture is broken",
     "One of the pictures could not be opened (damaged, half downloaded, or not really a picture).",
     "Run Datasets › 2 Usable scan: it finds broken files; move them to _reserve or delete them."),
    ("GPUFAULT", r"HSA_STATUS_ERROR_MEMORY_FAULT|Memory access fault by GPU|Memory Fault Error|"
                 r"illegal memory access was encountered|hipErrorIllegalAddress",
     "The GPU crashed (memory access fault)",
     "A calculation on the graphics card read memory it may not use, and the driver stopped the run. This is a fault in "
     "the GPU libraries (ROCm, hipBLASLt, PyTorch) or the driver, not in your dataset or settings; it often comes from "
     "mixed ROCm versions in the training venv.",
     "Settings › Version › Check this machine shows mixed ROCm versions; ./install.sh puts them back. Else try "
     "export TORCH_BLAS_PREFER_HIPBLASLT=0 in the launch script, or AOTriton experimental 0 (Settings › Trainers and GPU)."),
    ("GPU", r"no CUDA GPUs are available|Torch not compiled with CUDA|HIP error: invalid device function|"
            r"hipErrorNoBinaryForGpu|No HIP GPUs are available|torch\.cuda\.is_available\(\) is False|"
            r"Found no NVIDIA driver|amdgpu.*not found",
     "The GPU could not be used",
     "PyTorch did not find or could not use the graphics card: the wrong PyTorch build, a missing driver, or (AMD) "
     "a GPU version that needs HSA_OVERRIDE_GFX_VERSION.",
     "Settings › Version › Check this machine says what is missing; set the GPU value in Settings › Trainers and GPU."),
    ("IMPORT", r"ModuleNotFoundError: No module named '([\w.]+)'|ImportError: cannot import name|"
               r"command not found|accelerate: not found",
     "A program or Python package is missing",
     "The trainer's venv lacks a package it needs, or the launch script runs a program that is not installed "
     "(for example accelerate when the venv is not activated).",
     "Settings › Trainers and GPU: check the venv; Settings › Updates can install the trainers' requirements again."),
    ("ARGS", r"error: unrecognized arguments|error: argument --|invalid choice:|the following arguments are required",
     "The trainer does not know an option in the launch script",
     "The launch script uses an option the installed sd-scripts or musubi-tuner does not have (it is older or newer "
     "than the script expects), or an option has a wrong value.",
     "Datasets › 9 › Check flags against my trainer marks the option; update the trainer or remove the option."),
    ("DISK", r"No space left on device|Disk quota exceeded|OSError: \[Errno 28\]",
     "The disk is full",
     "There was no room left to save a file (save points, latent caches or the log).",
     "Free some space: Training › Results › Clean up removes epoch files you did not copy, saved states and caches."),
    ("PERM", r"Permission denied|PermissionError|Read-only file system",
     "No permission to read or write a file",
     "The trainer could not read a picture or model, or not write into the output folder.",
     "Check that the dataset and output folders belong to you (not root) and are not on a read-only drive."),
    ("NET", r"ConnectionError|Max retries exceeded|Temporary failure in name resolution|HTTPError: 40[13]|"
            r"401 Client Error|Repository Not Found|gated repo|Could not reach|timed out",
     "A download failed",
     "The trainer tried to download something (a tokenizer or model part from Hugging Face) and could not.",
     "Check the internet connection; for gated models set the Hugging Face token in Settings › AI helpers."),
    ("STOPPED", r"KeyboardInterrupt|--- stopped by the user ---|Terminated|exit code -15|exited with code -15|"
                r"exited with code 143",
     "Stopped",
     "The run was stopped (Stop now, a restart of the computer or the program, or another program ended it).",
     "Training › Run: a saved state shows as Resume from epoch N; else run it again."),
    ("SEGV", r"Segmentation fault|core dumped|exited with code -11|exited with code 139|Fatal Python error",
     "The trainer crashed",
     "The trainer stopped with a crash inside a library (often the GPU driver or PyTorch, sometimes bitsandbytes).",
     "Run it again once; if it happens again, Settings › Version › Check this machine, and try another optimizer "
     "than AdamW8bit."),
]
RULES_RE = [(c, re.compile(rx, re.I), t, w, f) for c, rx, t, w, f in RULES]
TOO_MANY = 25               # this many Traceback / Error lines without a known cause: "many errors"


def explain(text: str, exit_code=None) -> dict | None:
    """The cause of a failed run from its log (the end matters most: only the last 300 000 characters are read)."""
    t = (text or "")[-300_000:]
    if exit_code is None:
        m = re.findall(r"--- exited with code (-?\d+) ---", t)
        exit_code = m[-1] if m else None
    for code, rx, title, what, fix in RULES_RE:
        m = rx.search(t)
        if m:
            line = t[t.rfind("\n", 0, m.start()) + 1: t.find("\n", m.end()) if t.find("\n", m.end()) > 0 else None]
            return {"code": code, "title": title, "what": what, "fix": fix, "line": line.strip()[:300]}
    errors = re.findall(r"^.*(Traceback \(most recent call last\)|\bError\b:).*$", t, re.M)
    last = ""
    m = list(re.finditer(r"^(\w*(Error|Exception)\b.*)$", t, re.M))
    if m:
        last = m[-1].group(1).strip()[:300]
    if len(errors) >= TOO_MANY:
        return {"code": "MANY", "title": "Many errors in a row",
                "what": f"The log has {len(errors)} errors; the first one usually caused the others.",
                "fix": "Send the report (Report for Claude) or look at the first Traceback in the log.", "line": last}
    if last:
        name = last.split(":")[0]
        return {"code": "ERROR", "title": f"Python error: {name}",
                "what": "The trainer stopped with an error LoRA Studio does not know by name.",
                "fix": "Report for Claude makes a file with the log end and the settings, to ask about it.",
                "line": last}
    if exit_code not in (None, 0, "0"):
        return {"code": "EXIT", "title": f"Stopped with exit code {exit_code}",
                "what": "The trainer ended with an error code but wrote no error message.",
                "fix": "Look at the end of the log; Report for Claude collects it.", "line": ""}
    return None


def short(text: str, exit_code=None) -> str:
    e = explain(text, exit_code)
    return e["title"] if e else ""


def for_log(path, exit_code=None) -> dict | None:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 300_000))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    if exit_code is None:
        m = re.findall(r"--- exited with code (-?\d+) ---", text)
        exit_code = m[-1] if m else None
    return explain(text, exit_code)
