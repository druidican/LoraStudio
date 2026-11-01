#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
Training_Recommendations.py - the tuned settings LoRA Studio uses for new launch scripts.

Lora_Datatypes.py says HOW each LoRA type is trained (trainer, required flags, model files).
This file says WHICH VALUES to use for your goal, and is meant to be replaced as a whole when it is renewed:

  * per model family (SDXL/Illustrious/Pony, SD 1.5, FLUX.1, Chroma, Anima, SD3, Lumina, HunyuanImage, Z-Image Turbo)
  * per purpose   : character, style, concept/object
  * per priority  : balanced, quality (best quality), fast (fastest)
  * scaled by the number of images, with checks for image size and the model files chosen

How the step target is found (per family + purpose, see RECS):
    steps = views_per_image x number_of_images, kept between steps_min and steps_max
    quality: views, minimum and maximum x 1.25; fast: x 0.7
LoRA Studio's plan (Lora_Datatypes.py) then picks batch, gradient accumulation, repeats and epochs for that target.

Renewing: in LoRA Studio press "Export report for Claude" (Create Data Launch Script tab), send the report together
with this file, and put the returned file in place of this one. It is read fresh every time (no restart needed).
The version and date below are shown in every recommendation, so you can see which one a script came from.

Run it on its own to print the table:  python Training_Recommendations.py
"""

from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import os
import re

VERSION = "1.1"
UPDATED = "2026-10-15"
HARDWARE = "AMD Radeon RX 9070 XT, 16 GB VRAM, ROCm (PyTorch 2.13 + ROCm 10), 32 GB RAM"

SOURCES = [
    "sd-scripts docs: flux_train_network.md (FLUX.1 + Chroma), sd3_train_network.md, lumina_train_network.md, "
    "hunyuan_image_train_network.md, anima_train_network.md, train_network.md (github.com/kohya-ss/sd-scripts)",
    "Chroma1-HD model card (huggingface.co/lodestones/Chroma1-HD): T5-XXL + FLUX AE, T5 padding masked, CFG ~3-4",
    "Anima model card (huggingface.co/circlestone-labs/Anima): train on Anima-Base, keep the LLM adapter frozen, "
    "low LR (2e-5 suggested at rank 32), tags + natural language captions",
    "Illustrious style guide (civarchive.com/articles/25645): ~50 images, ~3000 steps, dim 32, noise offset 0.03-0.05",
    "Illustrious character guide (digitalcreativeai.net): 1024 px, AdamW8bit, cosine, unet 1e-4, ~1600 steps",
    "FLUX.1 style LoRA study (modal.com/blog/fine-tuning-flux-style-lora): styles want a higher rank than characters, "
    "20-25 varied images, best result rank 16 @ 2e-4",
    "Z-Image / musubi-tuner docs (github.com/kohya-ss/musubi-tuner, docs/zimage.md): shift timestep sampling, flow shift 2.0, "
    "network_alpha must be set explicitly (default 1); ostris zimage_turbo_training_adapter_v2 for training on Turbo",
    "Your own tested settings from Lora_Datatypes.py (SDXL 2500 steps, Anima 1e-4 / dim 16 / sigmoid, attn torch)",
]

PURPOSES = {"character": "Character / person", "style": "Art style", "concept": "Concept / object / pose"}
PRIORITIES = {"balanced": "Balanced", "quality": "Best quality", "fast": "Fastest"}

# LoRA type -> family (subtypes such as Pony use their base type)
FAMILY = {"sdxl": "sdxl", "illustrious": "sdxl", "sd15": "sd15", "flux": "flux", "chroma": "chroma",
          "anima": "anima", "sd3": "sd3", "lumina": "lumina", "hunyuan": "hunyuan", "zimage": "zimage", "qwenimage": "qwenimage", "flux2": "flux2"}

# ---------------------------------------------------------------------------------------------------------------
# Per family and purpose. views = views per image over the whole run; steps_min/steps_max bound the step target;
# epochs = how many checkpoints to compare; dim/alpha = LoRA rank and alpha; lr = learning rate.
# ---------------------------------------------------------------------------------------------------------------
RECS = {
    "sdxl": {
        "character": dict(views=50, steps_min=1600, steps_max=3000, epochs=12, lr="1e-4", dim=32, alpha=16, noise_offset=0.0357,
                          notes=["Caption with booru tags, trigger word first; drop tags for permanent traits (hair/eye "
                                 "colour, horns) so they bind to the trigger, keep clothes, pose, background, expression.",
                                 "Illustrious and Pony are trained the same way; generate with clip skip 2 as usual for them."]),
        "style": dict(views=30, steps_min=2000, steps_max=4000, epochs=10, lr="1e-4", dim=32, alpha=32, noise_offset=0.04,
                      notes=["Style LoRAs need more varied images (30-100+: many subjects, scenes and compositions) and "
                             "a higher rank than characters.",
                             "Caption what is IN each picture (subjects, objects, setting) so the style is what stays "
                             "unexplained and binds to the trigger; do not tag the style itself.",
                             "Noise offset 0.04 by default; 0.05 for heavy, high-contrast styles, 0.03 for light ones."]),
        "concept": dict(views=40, steps_min=1500, steps_max=3000, epochs=12, lr="1e-4", dim=16, alpha=8, noise_offset=0.0357,
                        notes=["Caption everything EXCEPT the concept, and put the trigger word for the concept first; "
                               "show the concept on many different subjects and backgrounds."]),
    },
    "sd15": {
        "character": dict(views=50, steps_min=1200, steps_max=2400, epochs=12, lr="1e-4", dim=32, alpha=16, notes=[]),
        "style": dict(views=30, steps_min=1500, steps_max=3200, epochs=10, lr="1e-4", dim=32, alpha=32,
                      notes=["Caption the content of each image, not the style; 30-100 varied images."]),
        "concept": dict(views=40, steps_min=1200, steps_max=2400, epochs=12, lr="1e-4", dim=16, alpha=8, notes=[]),
    },
    "flux": {
        "character": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16,
                          notes=["Natural-language captions starting with the trigger word work best (T5)."]),
        "style": dict(views=35, steps_min=1500, steps_max=3000, epochs=10, lr="1e-4", dim=32, alpha=32,
                      notes=["Styles want a higher rank than characters (16 for simple styles, 32+ for detailed ones); "
                             "20-40 varied images; describe the content of each image in sentences."]),
        "concept": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16,
                        notes=["Describe everything except the concept, which the trigger word stands for."]),
    },
    "anima": {
        "character": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16,
                          notes=["Your tested setting (1e-4, dim 16 = alpha). The model's author suggests a lighter touch "
                                 "(about 2e-5 at rank 32): if results look over-baked or stiff, try 5e-5."]),
        "style": dict(views=30, steps_min=1500, steps_max=3000, epochs=10, lr="5e-5", dim=32, alpha=32,
                      notes=["Anima already knows many styles; a lower LR (5e-5) keeps it from losing its general ability. "
                             "Raise to 1e-4 if the style is too weak at the last epoch."]),
        "concept": dict(views=35, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
    },
    "sd3": {
        "character": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
        "style": dict(views=30, steps_min=1500, steps_max=3000, epochs=10, lr="1e-4", dim=32, alpha=32, notes=[]),
        "concept": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
    },
    "lumina": {
        "character": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
        "style": dict(views=30, steps_min=1500, steps_max=3000, epochs=10, lr="1e-4", dim=32, alpha=32, notes=[]),
        "concept": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
    },
    "hunyuan": {
        "character": dict(views=35, steps_min=1100, steps_max=2000, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
        "style": dict(views=30, steps_min=1400, steps_max=2800, epochs=10, lr="1e-4", dim=32, alpha=32, notes=[]),
        "concept": dict(views=35, steps_min=1100, steps_max=2000, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
    },
}
RECS["zimage"] = {
    # Z-Image Turbo through musubi-tuner + the ostris training adapter. 6B model, so a lower LR than the tiny-rank
    # defaults of the SDXL family is not needed; 1e-4 at rank 16 with AdamW is the usual starting point.
    "character": dict(views=50, steps_min=1600, steps_max=2800, epochs=10, lr="1e-4", dim=32, alpha=32,
                      notes=["Z-Image Turbo needs the training adapter (--base_weights); the finished LoRA is used WITHOUT it.",
                             "Captions: short natural sentences starting with the trigger word (Qwen3 reads them); leave out "
                             "the face, hair colour, freckles and tattoos so the trigger word learns them.",
                             "Raised in v1.1 from rank 16 / 1200-2200 steps / 768 px: that gave weak likeness for a real person."]),
    "style": dict(views=30, steps_min=1500, steps_max=3000, epochs=10, lr="1e-4", dim=32, alpha=32,
                  notes=["Styles want more rank than characters; describe what is IN each picture, not the style."]),
    "concept": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=16, alpha=16, notes=[]),
}
RECS["qwenimage"] = {
    # Qwen-Image through musubi-tuner (docs/qwen_image.md example: dim 16, LR 5e-5, shift 2.2). 20B: slow, so fewer steps.
    "character": dict(views=35, steps_min=1000, steps_max=2000, epochs=10, lr="5e-5", dim=16, alpha=16,
                      notes=["Documented example values (dim 16, LR 5e-5); musubi says optimal settings are not known yet. "
                             "If likeness is weak at the last epoch, raise the LR to 1e-4 before adding epochs.",
                             "Natural-language captions starting with the trigger word (Qwen2.5-VL reads them)."]),
    "style": dict(views=30, steps_min=1200, steps_max=2500, epochs=10, lr="5e-5", dim=32, alpha=32, notes=[]),
    "concept": dict(views=35, steps_min=1000, steps_max=2000, epochs=10, lr="5e-5", dim=16, alpha=16, notes=[]),
}
RECS["flux2"] = {
    # FLUX.2 klein base through musubi-tuner (docs/flux_2.md example: dim 32, LR 1e-4, flux2_shift).
    "character": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=32, alpha=32,
                      notes=["Documented example values (dim 32, LR 1e-4); no klein-specific advice exists yet.",
                             "Train on klein-base, not the distilled klein: the distilled one is for inference."]),
    "style": dict(views=30, steps_min=1500, steps_max=3000, epochs=10, lr="1e-4", dim=32, alpha=32, notes=[]),
    "concept": dict(views=40, steps_min=1200, steps_max=2200, epochs=10, lr="1e-4", dim=32, alpha=32, notes=[]),
}
RECS["chroma"] = RECS["flux"]          # same family, same numbers; Chroma-specific flags live in Lora_Datatypes.py

# Training resolution per family and priority (None = keep the preset's)
RESOLUTION = {
    "sdxl": {"balanced": None, "quality": None, "fast": None},      # SDXL below 1024 loses quality: keep 1024
    "sd15": {"balanced": None, "quality": None, "fast": None},
    "flux": {"balanced": None, "quality": 1024, "fast": 768},
    "chroma": {"balanced": None, "quality": 1024, "fast": 768},
    "anima": {"balanced": None, "quality": 1024, "fast": 768},
    "sd3": {"balanced": None, "quality": 1024, "fast": 768},
    "lumina": {"balanced": None, "quality": 1024, "fast": 768},
    "hunyuan": {"balanced": None, "quality": None, "fast": None},   # trained at 2K only; multi-resolution is discouraged
    "qwenimage": {"balanced": None, "quality": 1328, "fast": 768},   # 1328 is Qwen-Image's native size; musubi: dataset.toml
    "flux2": {"balanced": None, "quality": None, "fast": 768},
    "zimage": {"balanced": None, "quality": None, "fast": 768},     # preset is 1024; musubi: written into dataset.toml
}

# Images smaller than about side x side pixels get upscaled by bucketing (the analysis warns about them)
MIN_SIDE = {"sdxl": 896, "sd15": 448, "flux": 896, "chroma": 896, "anima": 768, "sd3": 896, "lumina": 896,
            "hunyuan": 1536, "zimage": 896, "qwenimage": 896, "flux2": 896}
# Caption style the analysis checks for ("natural" = warn when captions are only short tag lists)
CAPTIONS = {"flux": "natural", "chroma": "natural", "sd3": "natural", "lumina": "natural", "hunyuan": "natural",
            "zimage": "natural", "qwenimage": "natural", "flux2": "natural"}

# Flags to add or remove per family (on top of Lora_Datatypes.py), and why
FLAGS = {
    "lumina": {"add": ['--system_prompt "You are an assistant designed to generate high-quality images based on user prompts."'],
               "why": "Lumina 2 was trained with a system prompt in front of every caption; sd-scripts recommends this one."},
    "hunyuan": {"remove": ["--fp8_base"], "add": ["--fp8_vl", "--model_prediction_type raw"],
                "why": "sd-scripts does not support --fp8_base for HunyuanImage (it uses --fp8_scaled); --fp8_vl keeps "
                       "the 7B text encoder in fp8 while caching, and raw prediction is the recommended default."},
}

PRIORITY_NOTES = {
    "balanced": "Balanced: the usual views per image and resolution for this family.",
    "quality": "Best quality: about 25% more training (views per image), full resolution, more checkpoints to compare. "
               "Takes longer; pick the best epoch rather than the last.",
    "fast": "Fastest: about 30% fewer steps and, where the model allows it, 768 px instead of 1024 (roughly half the time "
            "per step). Good for testing a dataset; train the final LoRA with Balanced or Best quality.",
}


# ---------------------------------------------------------------------------------------------------------------
def family_of(tkey: str, base: str = "") -> str:
    return FAMILY.get(tkey) or FAMILY.get(base or "", "")


def step_target(fam: str, purpose: str, priority: str, n_images: int) -> tuple[int, int, int, int]:
    """(steps, steps_min, steps_max, views) for this many images."""
    r = RECS[fam][purpose]
    views, lo, hi = r["views"], r["steps_min"], r["steps_max"]
    f = {"quality": 1.25, "fast": 0.7}.get(priority, 1.0)
    views, lo, hi = round(views * f), round(lo * f), round(hi * f)
    steps = max(lo, min(hi, views * max(1, n_images)))
    return int(round(steps / 50) * 50), lo, hi, views


def _sub(flags: list, pattern: str, repl: str) -> list:
    return [re.sub(pattern, repl, f) for f in flags]


def apply(tkey: str, cfg: dict, purpose: str = "character", priority: str = "balanced", n_images: int = 0,
          model_files: dict | None = None, identify=None) -> tuple[dict, list]:
    """Return (tuned copy of the preset, notes explaining what was changed and why)."""
    fam = family_of(tkey, cfg.get("base", ""))
    if fam not in RECS:
        return dict(cfg), [f"No tuned recommendations for this type yet (Training_Recommendations.py {VERSION})."]
    purpose = purpose if purpose in PURPOSES else "character"
    priority = priority if priority in PRIORITIES else "balanced"
    r = RECS[fam][purpose]
    out = dict(cfg)
    flags = list(cfg.get("flags", []))
    notes = [f"**{PURPOSES[purpose]}, {PRIORITIES[priority]}** (Training_Recommendations.py v{VERSION}, {UPDATED})."]

    steps, lo, hi, views = step_target(fam, purpose, priority, n_images)
    out.update(steps=steps, views=views, targets=(max(lo, round(steps * 0.75)), steps, min(round(hi * 1.1), round(steps * 1.3))))
    epochs = r["epochs"] + (3 if priority == "quality" else -2 if priority == "fast" else 0)
    out["epochs"] = max(6, epochs)
    notes.append(f"Step target {steps} ({views} views per image × {n_images} images, kept within {lo}-{hi} for "
                 f"{PURPOSES[purpose].lower()} LoRAs), over about {out['epochs']} epochs.")

    out["lr"] = r["lr"]
    flags = _sub(flags, r"--network_dim[ =]\S+", f"--network_dim {r['dim']}")
    flags = _sub(flags, r"--network_alpha[ =]\S+", f"--network_alpha {r['alpha']}")
    if priority == "quality" and purpose == "style":
        flags = _sub(flags, r"--network_dim[ =]\S+", f"--network_dim {r['dim'] * 2}")
        notes.append(f"Network dim {r['dim'] * 2}, alpha {r['alpha']}: extra capacity for the detail of a style.")
    else:
        notes.append(f"Network dim {r['dim']}, alpha {r['alpha']}, LR {r['lr']}.")
    if "noise_offset" in r:
        flags = _sub(flags, r"--noise_offset[ =]\S+", f"--noise_offset {r['noise_offset']}")

    res = RESOLUTION.get(fam, {}).get(priority)
    if res:
        flags = _sub(flags, r"--resolution[ =]\S+", f"--resolution {res},{res}")
        lo_b, hi_b = max(256, res // 2), min(2048, res + res // 2)
        flags = _sub(flags, r"--min_bucket_reso[ =]\S+", f"--min_bucket_reso {lo_b}")
        flags = _sub(flags, r"--max_bucket_reso[ =]\S+", f"--max_bucket_reso {hi_b}")
        notes.append(f"Resolution {res} px (buckets {lo_b}-{hi_b}).")

    fx = FLAGS.get(fam, {})
    for f in fx.get("remove", []):
        flags = [re.sub(rf"(^|\s){re.escape(f)}(?=\s|$)", " ", x).strip() for x in flags]
        flags = [x for x in flags if x]
    for f in fx.get("add", []):
        name = f.split()[0]
        if not any(re.search(rf"(^|\s){re.escape(name)}(?=\s|=|$)", x) for x in flags):
            flags.append(f)
    if fx.get("why"):
        notes.append(fx["why"])
    if fam == "hunyuan" and not any("--blocks_to_swap" in x for x in flags):
        flags.append("--blocks_to_swap 22")
    out["flags"] = flags
    out["min_side"] = MIN_SIDE.get(fam, out.get("min_side"))
    if CAPTIONS.get(fam):
        out["captions"] = CAPTIONS[fam]

    # model files
    mf = {k.lstrip("-"): v for k, v in (model_files or {}).items() if v and "/PATH/TO" not in str(v)}
    if identify:
        base = mf.get("pretrained_model_name_or_path")
        if base and os.path.isfile(os.path.expanduser(base)):
            info = identify(base)
            size_gb = (info.get("size") or 0) / 1024 ** 3
            if fam == "sd3" and size_gb > 10 and not any("--blocks_to_swap" in x for x in flags):
                out["flags"] = flags + ["--blocks_to_swap 24"]
                notes.append(f"The base model is {size_gb:.0f} GB (SD3.5 Large): added --blocks_to_swap 24 to fit 16 GB.")
            if str(info.get("dtype", "")).startswith("fp8") and not info.get("scaled"):
                notes.append("The base model is already plain fp8: it loads faster, and --fp8_base keeps it as it is.")
        t5 = mf.get("t5xxl")
        if t5 and os.path.isfile(os.path.expanduser(t5)):
            if str(identify(t5).get("dtype", "")).startswith("fp8"):
                notes.append("T5-XXL is fp8: fine, because its outputs are cached once; fp16 is a little more precise.")

    notes.append(PRIORITY_NOTES[priority])
    notes += r.get("notes", [])
    if fam == "anima":
        notes.append("Train on Anima-Base (not Aesthetic or Turbo) and keep the LLM adapter frozen (the default). If your "
                     "sd-scripts has --qwen_image_vae_2d, adding it makes latent caching about 2x faster with less VRAM.")
    if fam in ("flux", "chroma") and priority != "fast":
        notes.append("Speed: if the GPU status shows 3+ GB VRAM free, leave --blocks_to_swap out (it is the slowest part).")
    out["rec_notes"] = notes
    return out, notes


if __name__ == "__main__":
    print(f"Training_Recommendations.py v{VERSION} ({UPDATED}) for {HARDWARE}\n")
    for fam, by in RECS.items():
        for purpose, r in by.items():
            row = [f"{fam:8s} {purpose:10s}"]
            for pr in PRIORITIES:
                s, _, _, v = step_target(fam, purpose, pr, 50)
                row.append(f"{pr}: {s} steps")
            print("  ".join(row), f"| dim {r['dim']}/{r['alpha']} lr {r['lr']}  (for 50 images)")
