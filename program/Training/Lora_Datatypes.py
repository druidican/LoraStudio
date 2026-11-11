#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
Lora_Datatypes.py

ALL LoRA types (architectures) used by LoRA Studio live here, in PRESETS: SDXL, Illustrious, Anima,
SD 1.5, SD3 / SD3.5, FLUX.1, Chroma1-HD, Lumina Image 2.0, HunyuanImage-2.1 and Z-Image Turbo
(the one type trained with musubi-tuner instead of sd-scripts: "trainer": "musubi"). LoRA Studio's
"Create Data Launch Script" tab reads this file fresh every time, so edits apply without a restart.
To add a type, add an entry to PRESETS (copy one that trains the same way and change it).

It also still works on its own from a terminal:
Give it the name of a LoRA folder; it looks in ~/sd-scripts/images/<name>,
scans it and all subfolders for "<N>_name" repeat folders, asks which LoRA type
you are training, then prints a step/epoch plan and a ready-to-run launch command.

It also WRITES two files into the LoRA folder:
  * <n><Type>training.sh   e.g. PersonSDXLtraining.sh  (chmod +x, ready to run)
  * SDXLprompts.txt / ANIMAprompts.txt   (by type) sample prompts built from the most frequent tags in your
                           .txt captions (rendered every epoch during training)

Usage:
    python Lora_Datatypes.py                      # asks for everything
    python Lora_Datatypes.py MyChar            # asks for type/batch size
    python Lora_Datatypes.py MyChar --type anima   # accepts suggested batch/accum/repeats
    python Lora_Datatypes.py MyChar --type sdxl --lr 5e-5
    python Lora_Datatypes.py MyChar --type anima --batch_size 1 --grad_accum 4
    python Lora_Datatypes.py MyChar --base /other/images/dir
    python Lora_Datatypes.py MyChar --type sdxl --steps 3000   # change the step target
    python Lora_Datatypes.py MyChar --type sdxl --dry_run      # show, write nothing

How the plan is calculated (per LoRA type, see PRESETS):
  Each type has a target step count ("steps"), a target number of epochs ("epochs")
  and a target number of views per image ("views"). From those:

      effective_batch = views * images / steps        (bigger dataset -> bigger batch)
      batch_size      = effective_batch, capped by VRAM (max_batch)
      grad_accum      = whatever is left to reach the effective batch
      repeats         = steps * effective_batch / (epochs * images)

  Example, 81 images:  SDXL -> batch 2, repeats 4, 15 epochs = ~2430 steps
                       Anima -> batch 2, repeats 4, 10 epochs = ~1600 steps
  Repeats are kept low and epochs high so you get many checkpoints to compare.

How sd-scripts counts (this script mirrors it):
  * Only folders named "<N>_<name>" are dataset folders (N = repeats).
    sd-scripts only looks at the DIRECT children of --train_data_dir.
  * steps_per_epoch = ceil(sum(images * repeats) / (batch_size * grad_accum))
    (approximate: sd-scripts buckets images, so the real number can be a bit higher;
     check the step count it prints at startup.)
"""
__version__ = "1.111126e"  # LoRA Studio version, see version.py

import argparse
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_SD_DIR = (Path(__file__).resolve().parents[2] / "sd-scripts"   # install.sh layout: next to this file
                  if (Path(__file__).resolve().parents[2] / "sd-scripts").is_dir()
                  else Path.home() / "sd-scripts")   # where venv/ and the *_train_network.py files live
DEFAULT_BASE = DEFAULT_SD_DIR / "images"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CAPTION_EXTS = {".txt", ".caption"}
REPEAT_RE = re.compile(r"^(\d+)_")
# Folders that are never dataset folders (outputs etc.)
IGNORE_DIRS = {"model", "models", "output", "outputs", "samples", "sample", "logs",
               "loramanager", "_set_aside"}   # LoraManager = Personal sync; _set_aside = Check dataset pictures

# Exported at the top of every generated training .sh
ROCM_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "12.0.1",
    "OMP_NUM_THREADS": "12",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
}

# Sample prompt generation (prompts.txt)
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, jpeg artifacts, watermark, blurry"
SAMPLE_SEED = 42  # fixed so the same prompt is comparable from epoch to epoch
TAG_BUCKETS = {   # first match wins, so more specific buckets come first
    "view": ("from side", "from behind", "from above", "from below", "front view", "profile",
             "close-up", "upper body", "full body", "cowboy shot", "portrait",
             "looking at viewer", "looking back", "looking away", "looking to the side"),
    "expression": ("smile", "smiling", "grin", "open mouth", "closed mouth", "blush", "laughing",
                   "angry", "sad", "surprised", "serious", "smirk", "pout", "frown", "tongue",
                   "wink", "crying", "closed eyes", "parted lips"),
    "pose": ("standing", "sitting", "lying", "kneeling", "walking", "running", "leaning",
             "crouching", "squatting", "arms up", "arms behind back", "arms crossed", "hand on",
             "hands on", "jumping", "stretching", "on back", "on side", "all fours", "hand up"),
    "clothes": ("dress", "shirt", "hoodie", "jacket", "skirt", "pants", "shorts", "jeans", "coat",
                "sweater", "bikini", "swimsuit", "t-shirt", "top", "uniform", "kimono", "suit",
                "socks", "boots", "shoes", "hat", "gloves", "armor", "bodysuit", "scarf", "cape"),
    "background": ("background", "outdoors", "indoors", "room", "bed", "street", "beach", "forest",
                   "city", "sky", "window", "kitchen", "bathroom", "couch", "chair", "tree",
                   "night", "sunset", "studio"),
}

# ---------------------------------------------------------------------------
# EDIT THESE to match your machine. Paths are the ones from your launch scripts.
#
#   steps / epochs / views : targets that drive the batch, repeats and epoch plan
#   targets                : the three step counts shown in the epoch plan
#                            (the middle one is used for the launch command)
#   lr / warmup            : learning rate (override with --lr) and warmup steps
# ---------------------------------------------------------------------------
PRESETS = {
    "sdxl": {
        "label": "SDXL",
        "suffix": "SDXL",
        "max_batch": 6,    # largest batch size that fits your 16GB (you have run 6)
        "steps": 2500,     # target total steps for a character LoRA (override: --steps N)
        "epochs": 15,
        "views": 60,       # ~views per image over the whole run
        "targets": (1800, 2500, 3200),   # middle one is used for the .sh
        "lr": "1e-4",
        "optimizer": "AdamW8bit",
        "scheduler": "cosine_with_restarts",
        "warmup": 80,
        "script": "sdxl_train_network.py",
        "sample_args": "--w 1024 --h 1024 --s 28 --l 6",
        "prompts_name": "SDXLprompts.txt",
        "script_name": "{name}SDXLtraining.sh",   # generated file name ({name} = LoRA folder)
        "model_args": [
            ('--pretrained_model_name_or_path',
             "/PATH/TO/sd_xl_base_1.0.safetensors"),
        ],
        "flags": [
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora",
            "--network_dim 32 --network_alpha 32",
            "--mixed_precision bf16 --save_precision bf16",
            "--sdpa --no_half_vae --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--network_train_unet_only",
            # Text encoder outputs are NOT cached so tags can be shuffled;
            # keep_tokens 1 pins the trigger word at the front.
            "--shuffle_caption --keep_tokens 1",
            "--max_grad_norm 1.0",
            "--min_snr_gamma 5",
            "--noise_offset 0.0357",
        ],
        "sample_extra": "--sample_sampler euler_a",
        "notes": [
            "LR 1e-4 is the standard for dim 32 / alpha 32 with AdamW8bit; values around "
            "1e-5 to 3e-5 usually undertrain (weak likeness).",
            "NaN mid-run? Retry with --lr 5e-5, then 1e-5, or Adafactor with --optimizer_args "
            "scale_parameter=False relative_step=False warmup_init=False.",
            "Captions: drop tags for permanent traits (hair/eye color, skin, body type) so they "
            "bind to the trigger word; keep clothes, pose, background and expression tags.",
        ],
    },
    "anima": {
        "label": "Anima (DiT)",
        "suffix": "Anima",
        "max_batch": 2,    # ~8.3GB used at batch 2 with gradient checkpointing
        "steps": 1600,
        "epochs": 10,
        "views": 40,
        "targets": (1200, 1600, 2200),
        "lr": "1e-4",
        "optimizer": "AdamW8bit",
        "scheduler": "cosine",
        "warmup": 80,
        "script": "anima_train_network.py",
        "prompts_name": "ANIMAprompts.txt",
        "script_name": "{name}ANIMA_lora.sh",
        "env": {"PYTORCH_ROCM_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8"},
        "save_every": 2,                                       # checkpoint every N epochs
        "save_flags": "--save_last_n_epochs_state 2 --save_state",
        "workers": 4,
        "model_args": [
            ("--pretrained_model_name_or_path",
             "/PATH/TO/anima-base.safetensors"),
            ("--qwen3", "/PATH/TO/qwen_3_06b_base.safetensors"),
            ("--vae", "/PATH/TO/qwen_image_vae.safetensors"),
        ],
        "flags": [
            "--resolution 1024,1024",
            "--network_module networks.lora_anima",
            "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16",
            "--attn_mode torch --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            # Text encoder cached -> no caption shuffling; keep tag order consistent,
            # trigger word first.
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only",
            "--max_grad_norm 1.0",
            "--timestep_sampling sigmoid --discrete_flow_shift 1.0 --weighting_scheme uniform",
        ],
        "sample_extra": "",
        "notes": [
            "Keep --attn_mode torch for training: sageattn produced a corrupted LoRA in your test.",
            "Keep --gradient_checkpointing on: dropping it caused the VRAM-exhaustion crash.",
            "Underfit at the last epoch? Retrain at --lr 2e-4 instead of adding epochs.",
        ],
    },
}

# Illustrious = SDXL architecture, so it reuses the SDXL preset (same trainer, flags, 2500-step
# plan) with its own checkpoint, names and notes. Override any key here if it needs different settings.
PRESETS["illustrious"] = {
    **PRESETS["sdxl"],
    "label": "Illustrious (SDXL)",
    "suffix": "ILL",
    "model_args": [
        ("--pretrained_model_name_or_path",
         "/PATH/TO/illustriousXL.safetensors"),
    ],
    "sample_args": "--w 1024 --h 1024 --s 28 --l 5",
    "prompts_name": "ILLprompts.txt",
    "script_name": "{name}ILLtraining.sh",
    "notes": [
        "Illustrious is booru-tag based: keep captions as comma-separated tags with the trigger "
        "word first (--keep_tokens 1 pins it).",
        "Same LR guidance as SDXL: 1e-4 with AdamW8bit; NaN mid-run -> retry with --lr 5e-5.",
    ],
}

# Shared values for the templates below
_BIG_ENV = {"PYTORCH_ROCM_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8"}
_PH = "/PATH/TO/"          # placeholder for model files this file cannot know

# More architectures sd-scripts can train LoRAs for (moved here from sdgui.py). These are TEMPLATES written
# from the sd-scripts documentation ("template": True marks them in the type list): set the model paths in
# LoRA Studio's Settings > Training, and use "Check flags" to confirm your sd-scripts accepts every option.
# Once a template works for you, remove its "template": True line.
PRESETS.update({
    "sd15": {
        "label": "SD 1.5 (for SD 2.x add --v2)", "suffix": "SD15", "max_batch": 8,
        "steps": 1800, "epochs": 15, "views": 60, "targets": (1300, 1800, 2400),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine_with_restarts", "warmup": 80,
        "script": "train_network.py", "sample_args": "--w 512 --h 512 --s 28 --l 7",
        "prompts_name": "SD15prompts.txt", "script_name": "{name}SD15training.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "sd15-checkpoint.safetensors")],
        "flags": [
            "--resolution 512,512",
            "--enable_bucket --min_bucket_reso 256 --max_bucket_reso 1024 --bucket_reso_steps 64",
            "--network_module networks.lora", "--network_dim 32 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa",
            "--cache_latents --cache_latents_to_disk",
            "--text_encoder_lr 5e-5", "--clip_skip 2",
            "--shuffle_caption --keep_tokens 1", "--max_grad_norm 1.0",
            "--min_snr_gamma 5", "--noise_offset 0.0357",
        ],
        "sample_extra": "--sample_sampler euler_a",
        "notes": ["--clip_skip 2 is right for anime-derived checkpoints; use 1 for photographic SD 1.5 models.",
                  "Add --network_train_unet_only if you want to freeze the text encoder (less overfit to caption wording)."],
        "template": True,
    },
    "sd3": {
        "label": "SD3 / SD3.5", "suffix": "SD3", "max_batch": 2,
        "steps": 1600, "epochs": 10, "views": 40, "targets": (1200, 1600, 2200),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "sd3_train_network.py", "sample_args": "--w 1024 --h 1024 --s 28 --l 4.5",
        "prompts_name": "SD3prompts.txt", "script_name": "{name}SD3training.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "sd3.5_medium.safetensors"),
                       ("--clip_l", _PH + "clip_l.safetensors"), ("--clip_g", _PH + "clip_g.safetensors"),
                       ("--t5xxl", _PH + "t5xxl_fp16.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora_sd3", "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only", "--max_grad_norm 1.0",
        ],
        "sample_extra": "", "env": _BIG_ENV,
        "notes": ["SD3.5 Large (8B) will not fit 16 GB as is: add --fp8_base and --blocks_to_swap N.",
                  "SD3.5 Medium (2.5B) fits comfortably with gradient checkpointing."],
        "template": True,
    },
    "flux": {
        "label": "FLUX.1 (dev / schnell)", "suffix": "FLUX", "max_batch": 1,
        "steps": 2200, "epochs": 10, "views": 50, "targets": (1600, 2200, 3000),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "flux_train_network.py", "sample_args": "--w 1024 --h 1024 --s 20 --g 3.5",
        "prompts_name": "FLUXprompts.txt", "script_name": "{name}FLUXtraining.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "flux1-dev.safetensors"),
                       ("--clip_l", _PH + "clip_l.safetensors"), ("--t5xxl", _PH + "t5xxl_fp16.safetensors"),
                       ("--ae", _PH + "ae.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora_flux", "--network_dim 32 --network_alpha 32",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--fp8_base", "--network_train_unet_only",
            "--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0",
        ],
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "notes": ["FLUX.1 is a 12B model: --fp8_base is what makes it fit 16 GB. If you still run out of memory, "
                  "add --blocks_to_swap 8 (raise as needed).",
                  "Captions can be natural language instead of tags; the trigger word still goes first."],
        "template": True,
    },
    "lumina": {
        "label": "Lumina Image 2.0", "suffix": "LUM", "max_batch": 2,
        "steps": 2200, "epochs": 10, "views": 50, "targets": (1600, 2200, 3000),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "lumina_train_network.py", "sample_args": "--w 1024 --h 1024 --s 36 --l 4.0",
        "prompts_name": "LUMprompts.txt", "script_name": "{name}LUMtraining.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "lumina_2_model_bf16.safetensors"),
                       ("--gemma2", _PH + "gemma_2_2b_fp16.safetensors"), ("--ae", _PH + "ae.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora_lumina", "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only",
            "--timestep_sampling nextdit_shift --discrete_flow_shift 6.0 --model_prediction_type raw",
        ],
        "sample_extra": "", "env": _BIG_ENV,
        "notes": ["Lumina 2.0 is a 2B model and fits 16 GB at batch 2; it uses the Gemma-2 2B text encoder."],
        "template": True,
    },
    "hunyuan": {
        "label": "HunyuanImage-2.1", "suffix": "HYI", "max_batch": 1,
        "steps": 1500, "epochs": 10, "views": 35, "targets": (1100, 1500, 2000),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "hunyuan_image_train_network.py", "sample_args": "--w 2048 --h 2048 --s 50 --g 3.5",
        "prompts_name": "HYIprompts.txt", "script_name": "{name}HYItraining.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "hunyuanimage2.1.safetensors"),
                       ("--text_encoder", _PH + "qwen_2.5_vl_7b.safetensors"),
                       ("--byt5", _PH + "byt5_small_glyphxl_fp16.safetensors"),
                       ("--vae", _PH + "hunyuan_image_2.1_vae_fp16.safetensors")],
        "flags": [
            "--resolution 2048,2048",
            "--enable_bucket --min_bucket_reso 1024 --max_bucket_reso 2560 --bucket_reso_steps 64",
            "--network_module networks.lora_hunyuan_image", "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--fp8_scaled --fp8_vl --blocks_to_swap 22", "--network_train_unet_only",
            "--timestep_sampling shift --discrete_flow_shift 5.0 --model_prediction_type raw",
        ],
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "notes": ["A 17B-parameter model at 2K resolution: on 16 GB it only works with fp8 plus heavy block "
                  "swapping, and will be slow. 24 GB+ is the comfortable minimum. sd-scripts does not support "
                  "--fp8_base here; it uses --fp8_scaled (and --fp8_vl for the text encoder). Its guide suggests "
                  "--blocks_to_swap 9 at 24 GB and 32 at 12 GB.",
                  "This is the least-tested template here: run 'Check flags' before training."],
        "template": True,
    },
    # Chroma1-HD (lodestones): 8.9B, derived from FLUX.1-schnell, T5-XXL only (no CLIP-L), FLUX AE, trained at 1024.
    # sd-scripts trains it with the FLUX trainer in Chroma mode (docs/flux_train_network.md, "Chroma"):
    # --model_type chroma, --apply_t5_attn_mask (required), --guidance_scale 0.0 (required), no --clip_l,
    # --timestep_sampling sigmoid + --model_prediction_type raw (recommended); samples need a negative prompt
    # and real CFG (--l) with embedded guidance off (--g 0.0).
    "chroma": {
        "label": "Chroma1-HD (FLUX.1-schnell based)", "suffix": "CHROMA", "max_batch": 1,
        "steps": 2200, "epochs": 10, "views": 50, "targets": (1600, 2200, 3000),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "flux_train_network.py",
        "detect": r"--model_type[ =]\"?chroma",     # tells a Chroma script from a FLUX.1 one (same trainer)
        "sample_args": "--w 1024 --h 1024 --s 30 --l 4.0 --g 0.0",
        "negative": "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, "
                    "restricted palette, flat colors",
        "prompts_name": "CHROMAprompts.txt", "script_name": "{name}CHROMAtraining.sh",
        "model_args": [("--pretrained_model_name_or_path", _PH + "chroma_v10HD.safetensors"),
                       ("--t5xxl", _PH + "t5xxl_fp16.safetensors"),
                       ("--ae", _PH + "ae.safetensors")],
        "flags": [
            "--model_type chroma",
            "--resolution 1024,1024",
            "--enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 --bucket_reso_steps 64",
            "--network_module networks.lora_flux", "--network_dim 32 --network_alpha 32",
            "--mixed_precision bf16 --save_precision bf16", "--sdpa --gradient_checkpointing",
            "--cache_latents --cache_latents_to_disk",
            # T5 outputs cached -> the 9 GB text encoder is not in VRAM while training; no caption shuffling
            "--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk",
            "--apply_t5_attn_mask",
            "--fp8_base", "--network_train_unet_only", "--max_grad_norm 1.0",
            "--timestep_sampling sigmoid --sigmoid_scale 1.0 --model_prediction_type raw --guidance_scale 0.0",
        ],
        "save_flags": "--save_last_n_epochs_state 2 --save_state",
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "captions": "natural",      # analysis: T5 understands sentences; warn when captions are only short tags
        "min_side": 896,            # analysis: trained at 1024, warn about images with fewer pixels than 896x896
        "notes": [
            "Base model: Chroma1-HD (chroma_v10HD.safetensors). Text encoder: T5-XXL (t5xxl_fp16, or the fp8 "
            "version to save RAM while caching). AE: the FLUX.1 ae.safetensors. There is no CLIP-L: leave that slot empty.",
            "Needs an sd-scripts version that knows --model_type (2025 or newer). If Check flags reports "
            "--model_type or --apply_t5_attn_mask as unknown, update sd-scripts.",
            "--apply_t5_attn_mask and --guidance_scale 0.0 are required for Chroma (it was trained with T5 padding "
            "masked and has no distilled guidance). Do not remove them.",
            "8.9B parameters: --fp8_base plus gradient checkpointing is what fits 16 GB at batch 1 (accumulation "
            "gives the larger effective batch). Out of memory? Add --blocks_to_swap 8 and raise it step by step (max 35).",
            "Captions: Chroma reads natural language through T5. Write short sentences that start with the trigger "
            "word (\"mychar, a woman with short red hair, sitting on a chair, smiling\"); plain booru tags also work but "
            "teach less. Text-encoder outputs are cached, so the order stays as written.",
            "Sample images use real CFG (--l 4.0) with embedded guidance off (--g 0.0) and need a negative prompt, "
            "which the prompts file includes. In ComfyUI use CFG 3-4.5, ~30-40 steps, with a negative prompt.",
            "Rank 32 / alpha 32 since LoRA Studio 1.091126 (rank 16 learned faces poorly). Weak likeness at the last "
            "epoch: retrain at --learning_rate 2e-4, or rank 64 / alpha 64 with 50+ pictures, instead of adding many "
            "epochs. Keep alpha = dim: a lower alpha makes the LoRA learn at a fraction of its strength. NaN loss: halve the LR.",
            "A Chroma LoRA works on Chroma models (HD, Base); it is not interchangeable with FLUX.1-dev LoRAs. "
            "Test the first saved epoch in ComfyUI before a long run.",
        ],
        "template": True,
    },
    # Z-Image Turbo (Tongyi-MAI), 6B single-stream DiT, Qwen3-4B text encoder, FLUX AE. Trained with kohya-ss/musubi-tuner
    # (NOT sd-scripts): "trainer": "musubi" makes LoRA Studio write ONE all-in-one script that checks the files, builds the
    # dataset config, caches latents and text-encoder outputs, trains, and converts the LoRA for ComfyUI. The script prints
    # "##LS_STAGE i/n name" lines (and "##LS_FAIL ..." on an error) so LoRA Studio can show which stage is running.
    # Turbo is a distilled model: training on it needs the ostris training adapter (--base_weights), otherwise the
    # distillation breaks down quickly. No sample images (musubi uses another prompt format and they are not wanted).
    "zimage": {
        "label": "Z-Image Turbo (musubi-tuner)", "suffix": "ZIT", "trainer": "musubi", "max_batch": 1,
        "convert": "networks/convert_z_image_lora_to_comfy.py",
        # --base_weights must be a Z-Image LoRA (the ostris training adapter): musubi merges by layer name, so a LoRA
        # for another model merges NOTHING without an error, and Turbo then trains without its adapter (blurry results)
        "base_weights_check": r"layers[._]\d+[._](attention|feed_forward|adaln)",    # musubi stage 6 (path inside src/musubi_tuner); None = the LoRA is used as saved
        "steps": 2400, "epochs": 10, "views": 50, "targets": (1600, 2400, 3000),   # a face needs more than a style
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "zimage_train_network.py",
        "prompts_name": "ZITprompts.txt", "script_name": "{name}ZIT_lora.sh",
        "model_args": [("--dit", _PH + "z_image_turbo_bf16.safetensors"),
                       ("--vae", _PH + "ae.safetensors"),
                       ("--text_encoder", _PH + "qwen_3_4b.safetensors"),
                       ("--base_weights", _PH + "zimage_turbo_training_adapter_v2.safetensors")],
        "flags": [
            "--resolution 1024,1024",        # not a musubi option: LoRA Studio writes it into the dataset config
            "--network_module networks.lora_zimage",
            "--network_dim 32 --network_alpha 32",     # musubi's --network_alpha defaults to 1, so it is always set
            "--mixed_precision bf16",
            "--sdpa --gradient_checkpointing",
            "--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0",
            "--blocks_to_swap 10",           # block swapping instead of fp8 (fp8 was a NaN suspect on this machine)
            "--max_grad_norm 1.0",
        ],
        "save_flags": "--save_last_n_epochs_state 2 --save_state",
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "captions": "natural",
        "min_side": 896,
        "notes": [
            "Trained with musubi-tuner (set its folder in Settings > Training: 'musubi-tuner path'), not sd-scripts. "
            "One run does everything: check files, dataset config, cache latents, cache text encoder, train, convert for ComfyUI.",
            "The training adapter (zimage_turbo_training_adapter_v2.safetensors, --base_weights) is required for Turbo. "
            "The converted LoRA is used in ComfyUI WITHOUT the adapter, on the normal Z-Image Turbo model (8 steps, CFG 1).",
            "Resolution and batch size are written into the dataset config (dataset.toml, rebuilt on every run from your "
            "<N>_name folders). The cache is kept in <musubi-tuner>/cache/<name>, outside the dataset folder.",
            "1024 px / rank 32 / ~2200 steps since v2 (768 / 16 / 1600 gave weak likeness for a real person). Out of memory? Raise --blocks_to_swap (max 28) or lower the resolution to 768. VRAM to spare? Lower it to speed up.",
            "NaN loss: musubi does not stop by itself. LoRA Studio marks the run as failed and shows the step where the loss "
            "first became NaN; retrain at --learning_rate 5e-5.",
            "Captions: short natural sentences that start with the trigger word (Qwen3 reads sentences, not booru tags). For a "
            "person, describe clothes, pose, place and light, never the face, hair colour, freckles or tattoos: those go into "
            "the trigger word. Recaption with Ollama (this tab) writes such captions. Text-encoder outputs are cached, so "
            "caption order is not shuffled.",
        ],
        "template": True,
    },
})

# Z-Image Base: the undistilled model, trained the same way as Turbo by musubi-tuner but WITHOUT the training adapter.
# A LoRA trained on Base keeps likeness better and also works on Turbo (often at a strength of 1.0-1.5).
PRESETS["zimagebase"] = {
    **PRESETS["zimage"],
    "label": "Z-Image Base (musubi-tuner)", "suffix": "ZIB",
    "base_weights_check": None,
    "prompts_name": "ZIBprompts.txt", "script_name": "{name}ZIB_lora.sh",
    "model_args": [("--dit", _PH + "z_image_bf16.safetensors"),
                   ("--vae", _PH + "ae.safetensors"),
                   ("--text_encoder", _PH + "qwen_3_4b.safetensors")],
    "notes": [
        "Z-Image Base is the undistilled model: no training adapter is needed. Trained with musubi-tuner, like Turbo "
        "(the same one-run script: check files, dataset config, cache, train, convert for ComfyUI).",
        "Model files: z_image_bf16.safetensors (Comfy-Org/z_image), and the same Qwen3 4B text encoder and AE as Turbo.",
        "Use the LoRA on Base (about 30 steps, CFG 3.5-5) or on Turbo (8 steps, CFG 1); on Turbo it may need a strength "
        "of 1.2-1.5 to look as strong.",
        "Captions: short natural sentences that start with the trigger word and the class word; never the face, hair "
        "colour, freckles or tattoos of a person: those go into the trigger word.",
        "NaN loss: retrain at --learning_rate 5e-5. Out of memory? Raise --blocks_to_swap (max 28) or lower the resolution to 768.",
    ],
    "template": True,
}
PRESETS.update({
    # Qwen-Image (20B MMDiT, Qwen2.5-VL-7B text encoder, Qwen-Image VAE) with musubi-tuner (docs/qwen_image.md).
    # 20B does not fit 16 GB / 32 GB RAM in bf16: fp8_scaled (applied at load time to the bf16 file) plus block swapping.
    "qwenimage": {
        "label": "Qwen-Image (musubi-tuner)", "suffix": "QWI", "trainer": "musubi", "max_batch": 1,
        "steps": 1600, "epochs": 10, "views": 40, "targets": (1200, 1600, 2200),
        "lr": "5e-5", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "qwen_image_train_network.py", "model_version": "original",
        "te_args": "--fp8_vl --batch_size 1",           # the 7B text encoder only has to fit while caching
        "convert": None,
        "prompts_name": "QWIprompts.txt", "script_name": "{name}QWI_lora.sh",
        "model_args": [("--dit", _PH + "qwen_image_bf16.safetensors"),
                       ("--vae", _PH + "qwen_image_vae.safetensors"),
                       ("--text_encoder", _PH + "qwen_2.5_vl_7b.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--network_module networks.lora_qwen_image",
            "--network_dim 16 --network_alpha 16",
            "--mixed_precision bf16",
            "--sdpa --gradient_checkpointing",
            "--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.2",
            "--fp8_base --fp8_scaled --blocks_to_swap 45",
            "--max_grad_norm 1.0",
        ],
        "save_flags": "--save_last_n_epochs_state 2 --save_state",
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "captions": "natural", "min_side": 896,
        "notes": [
            "Trained with musubi-tuner (Settings > Training > musubi-tuner path). Same all-in-one run as Z-Image, minus the "
            "conversion stage: musubi's documentation names no ComfyUI conversion for Qwen-Image LoRAs, so the saved files are used as they are.",
            "Model files (Comfy-Org/Qwen-Image_ComfyUI split_files): qwen_image_bf16, qwen_2.5_vl_7b, qwen_image_vae. Use the plain bf16 "
            "files: musubi cannot use fp8_scaled / fp8_e4m3fn files; --fp8_scaled in the script quantises the bf16 DiT while loading.",
            "20B parameters: --fp8_base --fp8_scaled --blocks_to_swap 45 is the documented route to about 12 GB VRAM. Block swapping "
            "keeps the rest of the model in system RAM, so 32 GB RAM is tight (musubi recommends 64 GB): close other programs.",
            "fp8 was a NaN suspect on this machine. If the loss turns NaN, try --blocks_to_swap 50 without --fp8_base --fp8_scaled "
            "only if you have the RAM, otherwise lower the learning rate first.",
            "LR 5e-5 and dim 16 are the documented example values (musubi says optimal settings are not known yet).",
            "Test the first saved epoch in ComfyUI before a long run; if it does not load, tell me and I will add a conversion stage.",
        ],
        "template": True,
    },
    # FLUX.2 klein (Black Forest Labs) with musubi-tuner (docs/flux_2.md). The documentation recommends the BASE models for
    # training: klein-base-4b (fits 16 GB without swapping) or klein-base-9b (needs fp8 + swapping). Change MODEL_VERSION,
    # the DiT and the text encoder together.
    "flux2": {
        "label": "FLUX.2 klein (musubi-tuner)", "suffix": "F2K", "trainer": "musubi", "max_batch": 1,
        "steps": 1600, "epochs": 10, "views": 40, "targets": (1200, 1600, 2200),
        "lr": "1e-4", "optimizer": "AdamW8bit", "scheduler": "cosine", "warmup": 80,
        "script": "flux_2_train_network.py", "model_version": "klein-base-4b",
        "te_args": "--batch_size 8",
        "convert": None,
        "prompts_name": "F2Kprompts.txt", "script_name": "{name}F2K_lora.sh",
        "model_args": [("--dit", _PH + "flux2-klein-base-4b.safetensors"),
                       ("--vae", _PH + "ae.safetensors"),
                       ("--text_encoder", _PH + "qwen3_4b_klein-00001-of-00002.safetensors")],
        "flags": [
            "--resolution 1024,1024",
            "--network_module networks.lora_flux_2",
            "--network_dim 32 --network_alpha 32",
            "--mixed_precision bf16",
            "--sdpa --gradient_checkpointing",
            "--timestep_sampling flux2_shift --weighting_scheme none",
            "--max_grad_norm 1.0",
        ],
        "save_flags": "--save_last_n_epochs_state 2 --save_state",
        "sample_extra": "", "env": _BIG_ENV, "workers": 2,
        "captions": "natural", "min_side": 896,
        "notes": [
            "Trained with musubi-tuner (Settings > Training > musubi-tuner path). Same all-in-one run as Z-Image, minus the conversion "
            "stage: musubi's documentation names no ComfyUI conversion for FLUX.2 LoRAs, so the saved files are used as they are.",
            "Train on a BASE model (flux2-klein-base-4b, default, or klein-base-9b): the distilled klein models are meant for inference. "
            "The line MODEL_VERSION at the top of the script, the DiT file and the text encoder must belong together "
            "(4b: Qwen3-4B encoder, 9b: Qwen3-8B encoder).",
            "The text encoder is split into several files: pick the FIRST one (...-00001-of-0000N.safetensors) and keep all parts in the same folder. "
            "A Qwen3-4B file you already have for Z-Image also works for the 4B models.",
            "The AE is the FLUX.2 ae.safetensors from the FLUX.2-dev repo, not the Diffusers-format vae/diffusion_pytorch_model.safetensors.",
            "9B on 16 GB: add --fp8_base --fp8_scaled --blocks_to_swap 12 (at most 16 for klein-9b, 13 for klein-4b) and --fp8_text_encoder "
            "for the caching stage; fp8 was a NaN suspect here, so try 4B first.",
            "Rank 32 / LR 1e-4 are the documented example values; musubi gives no klein-specific recommendation.",
            "Test the first saved epoch in ComfyUI before a long run; if it does not load, tell me and I will add a conversion stage.",
        ],
        "template": True,
    },
})


VIEWS_LOW, VIEWS_HIGH = 20, 100  # rough guide only, not a rule

# ------------------------------- musubi-tuner support -------------------------------
_HERE = Path(__file__).resolve().parents[2]                                 # install.sh layout: trainers next to this file
DEFAULT_MUSUBI_DIR = _HERE / "musubi-tuner" if (_HERE / "musubi-tuner").is_dir() else Path.home() / "musubi-tuner"
MUSUBI_SRC = "src/musubi_tuner"        # where musubi's scripts live inside its folder


def is_musubi(cfg):
    """True for LoRA types that are trained with musubi-tuner instead of sd-scripts."""
    return (cfg or {}).get("trainer") == "musubi"


def musubi_var(flag):
    """--text_encoder -> TEXT_ENCODER (the shell variable that holds that model file in a musubi script)."""
    return flag.lstrip("-").upper()


def split_resolution(flags, default="768,768"):
    """(resolution 'W,H', flags without it): musubi has no --resolution option, it belongs in the dataset config."""
    res, out = default, []
    for f in flags:
        m = re.search(r"--resolution[ =](\d+)(?:,(\d+))?", f)
        if m:
            res = f"{m.group(1)},{m.group(2) or m.group(1)}"
            f = re.sub(r"\s*--resolution[ =]\S+", "", f).strip()
        if f:
            out.append(f)
    return res, out


# ------------------------------- input helpers -------------------------------

def ask(prompt, default=None):
    if not sys.stdin.isatty():
        if default is None:
            sys.exit(f"Missing value and no terminal to ask: {prompt}")
        return default
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def ask_int(prompt, default):
    while True:
        val = ask(prompt, str(default))
        try:
            n = int(val)
            if n >= 1:
                return n
        except ValueError:
            pass
        print("  Please enter a whole number >= 1.")


def choose(prompt, options):
    """options: list of (label, value). Returns value."""
    if not sys.stdin.isatty():
        sys.exit(f"Cannot ask '{prompt}' without a terminal; pass it as a flag.")
    print(prompt)
    for i, (label, _) in enumerate(options, 1):
        print(f"  {i}) {label}")
    while True:
        val = input("Choose number: ").strip()
        if val.isdigit() and 1 <= int(val) <= len(options):
            return options[int(val) - 1][1]
        print("  Invalid choice.")


def resolve_lora_dir(base: Path, name):
    """Find base/<name>. Accepts exact, case-insensitive, or unique partial match."""
    if name and (os.sep in name or name.startswith("~")):
        p = Path(os.path.expanduser(name)).resolve()
        if p.is_dir():
            return p
        sys.exit(f"Not a directory: {p}")

    if not base.is_dir():
        sys.exit(f"Base directory does not exist: {base}\n(use --base to point elsewhere)")
    dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if not dirs:
        sys.exit(f"No folders inside {base}")

    if name:
        low = name.lower()
        matches = ([d for d in dirs if d.name == name]
                   or [d for d in dirs if d.name.lower() == low]
                   or [d for d in dirs if low in d.name.lower()])
        if len(matches) == 1:
            return matches[0]
        if matches:
            print(f"'{name}' matches several folders.")
            return choose("Which one?", [(d.name, d) for d in matches])
        print(f"No folder named '{name}' in {base}.")

    return choose(f"LoRA folders in {base}:", [(d.name, d) for d in dirs])


# --------------------------------- scanning ----------------------------------

def repeats_from_name(path: Path):
    m = REPEAT_RE.match(path.name)
    return int(m.group(1)) if m else None


def find_repeat_folders(root: Path):
    """Walk root and all subfolders. Returns (repeat_folders, loose_image_dirs)."""
    r = repeats_from_name(root)
    if r is not None:
        return [(root, r)], {}

    found, loose = [], {}
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if p != root and re.match(r"^\d+_loramanager$", p.name, re.I):
            dirnames[:] = []  # a renamed LoraManager folder: previews, not training images
            continue
        if p != root and repeats_from_name(p) is not None:
            found.append((p, repeats_from_name(p)))
            dirnames[:] = []  # sd-scripts does not recurse into repeat folders
            continue
        dirnames[:] = sorted(d for d in dirnames
                             if d.lower() not in IGNORE_DIRS and not d.startswith("."))
        n = sum(1 for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS)
        if n:
            loose[p] = n
    return found, loose


def scan_folder(folder: Path):
    files = [f for f in folder.iterdir() if f.is_file()]
    imgs = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
    caps = {f.stem for f in files if f.suffix.lower() in CAPTION_EXTS}
    stems = [f.stem for f in imgs]
    dup = {s for s in stems if stems.count(s) > 1}
    return len(imgs), set(stems), caps, dup


# -------------------------------- calculations -------------------------------

def suggest_repeats(num_images, batch_size, grad_accum, epochs, steps):
    """Repeats giving ~`steps` over ~`epochs`. Repeats vs epochs is mostly about
    checkpoint granularity: fewer repeats + more epochs = more checkpoints."""
    if num_images <= 0:
        return 1
    return max(1, min(20, round(steps * batch_size * grad_accum / (epochs * num_images))))


def suggest_settings(num_images, cfg):
    """Suggest (batch_size, grad_accum, repeats) from the image count and the
    preset's step / epoch / views targets.

      effective_batch = views * images / steps
      batch_size      = effective_batch, capped by VRAM (max_batch)
      grad_accum      = whatever is left to reach the effective batch
      repeats         = steps * effective_batch / (epochs * images)
    """
    steps, epochs, views = cfg["steps"], cfg["epochs"], cfg["views"]
    eff = max(1, round(views * num_images / steps))
    bs = max(1, min(cfg["max_batch"], eff))
    ga = math.ceil(eff / bs)
    repeats = suggest_repeats(num_images, bs, ga, epochs, steps)
    return bs, ga, repeats


def build_command(cfg, name, train_dir, out_dir, bs, ga, epochs, save_every,
                  lr, sample_prompts=None):
    if not cfg["script"]:
        return None
    if is_musubi(cfg):
        return build_musubi_command(cfg, ga, epochs, save_every, lr)
    q = lambda s: f'"{s}"'
    parts = [f"accelerate launch --num_cpu_threads_per_process 2 {cfg['script']}"]
    for flag, path in cfg["model_args"]:
        parts.append(f"{flag} {q(path)}")
    parts += [
        f"--train_data_dir {q(train_dir)}",
        f"--output_dir {q(out_dir)}",
        f"--output_name {q(name + cfg['suffix'])}",
        f"--train_batch_size {bs}",
    ]
    if ga > 1:
        parts.append(f"--gradient_accumulation_steps {ga}")
    parts.append(f"--max_train_epochs {epochs}")
    parts += cfg["flags"]
    parts += [
        f"--optimizer_type {cfg['optimizer']} --learning_rate {lr}",
        f"--lr_scheduler {cfg['scheduler']} --lr_warmup_steps {cfg['warmup']}",
        f"--save_every_n_epochs {save_every}",
        "--save_model_as safetensors",
        f'--caption_extension "{cfg.get("caption_ext") or ".txt"}"',
        f"--max_data_loader_n_workers {cfg.get('workers', 2)} --persistent_data_loader_workers",
        "--seed 42",
    ]
    if cfg.get("save_flags"):
        parts.append(cfg["save_flags"])
    if sample_prompts:
        sample = f"--sample_every_n_epochs 1 --sample_prompts {q(sample_prompts)}"
        if cfg["sample_extra"]:
            sample += f" {cfg['sample_extra']}"
        parts.append(sample)
    return " \\\n  ".join(parts)


def build_musubi_command(cfg, ga, epochs, save_every, lr):
    """The training command of a musubi-tuner type. Model files, dataset config, output folder and output name are shell
    variables set at the top of the script; resolution and batch size are in the dataset config, not on the command line."""
    _, flags = split_resolution(cfg["flags"])
    parts = [f"accelerate launch --num_cpu_threads_per_process 2 {MUSUBI_SRC}/{cfg['script']}"]
    for flag, _path in cfg["model_args"]:
        parts.append(f'{flag} "${{{musubi_var(flag)}}}"')
    if cfg.get("model_version"):
        parts.append('--model_version "${MODEL_VERSION}"')
    parts += [
        '--dataset_config "${DATASET_TOML}"',
        '--output_dir "${OUTPUT_DIR}"',
        '--output_name "${OUTPUT_NAME}"',
    ]
    if ga > 1:
        parts.append(f"--gradient_accumulation_steps {ga}")
    parts.append(f"--max_train_epochs {epochs}")
    parts += flags
    parts += [
        f"--optimizer_type {cfg['optimizer']} --learning_rate {lr}",
        f"--lr_scheduler {cfg['scheduler']} --lr_warmup_steps {cfg['warmup']}",
        f"--save_every_n_epochs {save_every}",
        f"--max_data_loader_n_workers {cfg.get('workers', 2)} --persistent_data_loader_workers",
        "--seed 42",
    ]
    if cfg.get("save_flags"):
        parts.append(cfg["save_flags"])
    return " \\\n  ".join(parts)


# ------------------------------ sample prompts -------------------------------

def read_captions(folders):
    """All .txt captions in the repeat folders, as lists of comma-separated tags."""
    caps = []
    for folder, _ in folders:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() == ".txt":
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                tags = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
                if tags:
                    caps.append(tags)
    return caps


def classify_tag(tag):
    for bucket, words in TAG_BUCKETS.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\b", tag):
                return bucket
    return None


def build_prompts_text(caps, fallback_trigger, name, cfg):
    """Build prompts.txt from the most frequent caption tags.
    Returns (text, trigger, trigger_from_captions, top_tags)."""
    sample_args = cfg.get("sample_args", "--w 1024 --h 1024 --s 28 --l 5")

    # Trigger word = the first tag, if (nearly) every caption starts with the same one
    trigger, from_caps = fallback_trigger, False
    if caps:
        first, n = Counter(c[0] for c in caps).most_common(1)[0]
        if n / len(caps) >= 0.8:
            trigger, from_caps = first, True

    counts = Counter(t for tags in caps for t in {x.lower() for x in tags})
    buckets = {b: [] for b in TAG_BUCKETS}
    base = []  # generic tags in most captions (1girl, solo, ...)
    for tag, n in counts.most_common():
        if tag == trigger.lower():
            continue
        b = classify_tag(tag)
        if b:
            buckets[b].append(tag)
        elif n / len(caps) >= 0.5 and len(base) < 3:
            base.append(tag)
    if not caps:
        buckets.update(expression=["smile"], pose=["standing", "sitting"],
                       view=["looking at viewer", "from side"], background=["simple background"])

    def pick(bucket, i):
        lst = buckets[bucket]
        return lst[i % len(lst)] if lst else None

    lines = []

    pre, suf = cfg.get("quality_words") or ("", "")      # Pony: score_9, … first; Illustrious: masterpiece, … last

    def add(parts):
        tags = []
        for t in [trigger] + base + parts:
            if t and t not in tags:
                tags.append(t)
        text = ", ".join([x for x in (pre, ", ".join(tags), suf) if x])
        lines.append(f"{text} --n {cfg.get('negative') or NEGATIVE_PROMPT} {sample_args} --d {SAMPLE_SEED}")

    add([pick("view", 0) or "looking at viewer", pick("expression", 0)])      # plain baseline
    for i in range(4):                                                         # varied, from your tags
        add([pick("expression", i), pick("pose", i), pick("view", i + 1),
             pick("clothes", i), pick("background", i)])
    add(["smile", "wearing a red jacket", "outdoors", "city street"])        # prompt-control test

    top = [f"{t} ({n})" for t, n in counts.most_common(25)]
    header = [
        f"# Sample prompts for {name} (generated by Lora_Datatypes.py)",
        f"# Trigger word: {trigger}"
        + ("" if from_caps else "   <- guessed from the folder name, check it"),
        "# Most frequent caption tags: " + (", ".join(top) if top else "(no captions found)"),
        f"# Edit freely. '#' lines are ignored. --d {SAMPLE_SEED} fixes the seed so epochs compare.",
        "# Last line tests that clothes/background still follow the prompt (not baked in).",
        "",
    ]
    return "\n".join(header + lines) + "\n", trigger, from_caps, top


# ------------------------------ training script ------------------------------

def next_version(model_dir, base):
    """Next free version number for LoRA output name `base` (e.g. PersonSDXL).
    Looks in the model folder for anything named <base>V<n>... and returns highest n + 1; 1 if none."""
    top = 0
    if model_dir.is_dir():
        pat = re.compile(r"^" + re.escape(base) + r"V(\d+)")
        for f in model_dir.iterdir():
            m = pat.match(f.name)
            if m:
                top = max(top, int(m.group(1)))
    return top + 1


def under(path, anchor, anchor_sh):
    """Shell expression for `path`: anchor_sh/relative if it sits inside `anchor`, else the plain path."""
    try:
        rel = path.resolve().relative_to(anchor.resolve())
    except ValueError:
        return str(path)
    return anchor_sh if str(rel) == "." else f"{anchor_sh}/{rel}"


def render_training_script(sd_dir_sh, train_dir_sh, prompts_sh, plan_line, cmd, env, cfg=None, bs=1):
    """`sd_dir_sh` is the trainer folder (sd-scripts, or musubi-tuner for a musubi type). `cfg` (the preset with the versioned
    suffix) and `bs` are only needed for musubi types, whose script is the all-in-one pipeline."""
    if is_musubi(cfg):
        return render_musubi_script(sd_dir_sh, train_dir_sh, plan_line, cmd, env, cfg, bs)
    out = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Generated by Lora_Datatypes.py",
        f"# {plan_line}",
        "# Run this from inside the character's images folder. It calls sd-scripts by path,",
        "# so nothing here is tied to a user name.",
        "",
        "# This script lives in the character's images folder",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        f'SD_SCRIPTS_DIR="{sd_dir_sh}"',
        "",
        "# Character / dataset name = name of this folder",
        'CHARACTER_NAME="$(basename "${SCRIPT_DIR}")"',
        f'TRAIN_DIR="{train_dir_sh}"',
        'OUTPUT_DIR="${TRAIN_DIR}/model"',
    ]
    if prompts_sh:
        out.append(f'PROMPTS_FILE="{prompts_sh}"')
    out += ["", "# Environment variables for execution & ROCm"]
    for k, v in env.items():
        out.append(f'export {k}="{v}"' if re.search(r"[^\w.\-]", v) else f"export {k}={v}")
    out += ["", "# Hugging Face token (fill in if needed)", 'export HF_TOKEN=""']
    out += ["", "# Training code and venv live in sd-scripts",
            'cd "${SD_SCRIPTS_DIR}"', "source venv/bin/activate", "", cmd, ""]
    return "\n".join(out)


STAGES_MUSUBI = ["Check files", "Prepare dataset", "Cache latents", "Cache text encoder", "Train LoRA",
                 "Convert for ComfyUI"]


def musubi_stages(cfg):
    """Stage names of a musubi type: the last one (conversion) only exists for types that have a converter."""
    return STAGES_MUSUBI if cfg.get("convert") else STAGES_MUSUBI[:-1]


def render_musubi_script(musubi_dir_sh, train_dir_sh, plan_line, cmd, env, cfg, bs):
    """One bash script that runs the whole musubi-tuner pipeline. Every stage announces itself with a
    '##LS_STAGE i/n name' line and a failure prints '##LS_FAIL i/n name exit=N': LoRA Studio shows them."""
    res, _ = split_resolution(cfg["flags"])
    ST = musubi_stages(cfg)
    n = len(ST)
    prefix = cfg["script"][:-len("_train_network.py")]          # zimage / qwen_image / flux_2
    mv = ' --model_version "${MODEL_VERSION}"' if cfg.get("model_version") else ""
    out = [
        "#!/bin/bash",
        f"# Generated by Lora_Datatypes.py: {cfg['label']}, all in one run",
        f"# {plan_line}",
        "# Stages: " + " > ".join(ST),
        "# Resolution, batch size and the model files are the variables below; dataset.toml is rebuilt on every run",
        "# from the <N>_name folders in the images folder, so renaming a repeat folder needs no change here.",
        "set -eE -o pipefail",
        "",
        "# This script lives in the character's images folder",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        f'MUSUBI_DIR="{musubi_dir_sh}"',
        'MUSUBI_VENV="${MUSUBI_DIR}/venv"',
        "",
        "# Character / dataset name = name of this folder",
        'CHARACTER_NAME="$(basename "${SCRIPT_DIR}")"',
        f'TRAIN_DIR="{train_dir_sh}"',
        'OUTPUT_DIR="${TRAIN_DIR}/model"',
        f'OUTPUT_NAME="${{CHARACTER_NAME}}{cfg["suffix"]}"',
        "",
        "# Cache and dataset config live outside the dataset (Studio's library scans .safetensors files)",
        'CACHE_DIR="${MUSUBI_DIR}/cache/${CHARACTER_NAME}"',
        'DATASET_TOML="${CACHE_DIR}/dataset.toml"',
        f'RESOLUTION="{res}"',
        f'BATCH_SIZE="{bs}"',
        *([f'MODEL_VERSION="{cfg["model_version"]}"'] if cfg.get("model_version") else []),
        "",
        "# Model files",
    ]
    for flag, path in cfg["model_args"]:
        out.append(f'{musubi_var(flag)}="{path}"')
    out += ["", "# Environment variables for execution & ROCm"]
    for k, v in env.items():
        out.append(f'export {k}="{v}"' if re.search(r"[^\w.\-]", v) else f"export {k}={v}")
    out += ["", "# Hugging Face token (fill in if needed)", 'export HF_TOKEN=""']
    out += [
        "",
        "# ---- stage markers: LoRA Studio reads these lines ----",
        f"STAGE_TOTAL={n}",
        "STAGE_NO=0",
        'STAGE_NAME="Start"',
        'stage() { STAGE_NO=$((STAGE_NO+1)); STAGE_NAME="$1"; echo; echo "##LS_STAGE ${STAGE_NO}/${STAGE_TOTAL} ${STAGE_NAME}"; }',
        'fail_stage() { echo "$1"; echo "##LS_FAIL ${STAGE_NO}/${STAGE_TOTAL} ${STAGE_NAME} $1"; exit 1; }',
        "trap 'rc=$?; echo; echo \"##LS_FAIL ${STAGE_NO}/${STAGE_TOTAL} ${STAGE_NAME} exit=${rc}\"' ERR",
        "",
        f'stage "{ST[0]}"',
        '[ -d "${MUSUBI_DIR}" ] || fail_stage "musubi-tuner folder not found: ${MUSUBI_DIR} (set it in LoRA Studio: Settings > Training)"',
        '[ -f "${MUSUBI_VENV}/bin/activate" ] || fail_stage "musubi-tuner venv not found: ${MUSUBI_VENV}"',
        'cd "${MUSUBI_DIR}"',
        'source "${MUSUBI_VENV}/bin/activate"',
        # (script names without ".py" here: LoRA Studio finds the training command by looking for *train*.py)
        f'for n in {prefix}_cache_latents {prefix}_cache_text_encoder_outputs {cfg["script"][:-3]}; do',
        f'  [ -f "${{MUSUBI_DIR}}/{MUSUBI_SRC}/${{n}}.py" ] || fail_stage "musubi-tuner has no ${{n}}.py: update musubi-tuner (git pull)"',
        "done",
        'command -v accelerate >/dev/null || fail_stage "accelerate is not installed in ${MUSUBI_VENV}"',
        *([f'python -c "import bitsandbytes" >/dev/null 2>&1 || fail_stage "bitsandbytes does not load in ${{MUSUBI_VENV}} '
           f'(needed for --optimizer_type {cfg["optimizer"]}): pip install bitsandbytes, or use AdamW"']
          if str(cfg.get("optimizer", "")).lower().endswith("8bit") else []),
        *([f"""[ ! -f "${{BASE_WEIGHTS}}" ] || python - "${{BASE_WEIGHTS}}" <<'PYCHECK' || fail_stage "--base_weights is not a Z-Image training adapter: ${{BASE_WEIGHTS}} (its layers belong to another model, so musubi would merge nothing and Turbo would train without its adapter). Use zimage_turbo_training_adapter_v2.safetensors"
import json, re, struct, sys
with open(sys.argv[1], "rb") as fh:
    keys = list(json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0])))
ok = any(re.search(r"{cfg['base_weights_check']}", k) for k in keys)
print(f"base weights: {{len(keys)}} tensors, Z-Image layers: {{'yes' if ok else 'NO'}}")
sys.exit(0 if ok else 1)
PYCHECK"""] if cfg.get("base_weights_check") and any(fl == "--base_weights" for fl, _ in cfg["model_args"]) else []),
        "for f in " + " ".join(f'"${{{musubi_var(fl)}}}"' for fl, _ in cfg["model_args"]) + "; do",
        '  [ -f "$f" ] || fail_stage "model file not found: $f"',
        "done",
        "",
        f'stage "{ST[1]}"',
        '[ -n "${CHARACTER_NAME}" ] || fail_stage "empty character name"',
        'rm -rf "${CACHE_DIR}"',
        'mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}"',
        # musubi pairs a caption with its picture by cutting off ONE extension, so IMG.sentences.txt never matches
        # IMG.png: the pictures of each repeat folder are linked into the cache with their captions as IMG.txt
        f'CAPTION_EXT="{cfg.get("caption_ext") or ".txt"}"',
        '{',
        '  echo "[general]"',
        '  echo "resolution = [${RESOLUTION}]"',
        '  echo "caption_extension = \\".txt\\""',
        '  echo "batch_size = ${BATCH_SIZE}"',
        '  echo "enable_bucket = true"',
        '  echo "bucket_no_upscale = false"',
        '} > "${DATASET_TOML}"',
        "count=0",
        'for d in "${TRAIN_DIR}"/[0-9]*_*/; do',
        '  [ -d "$d" ] || continue',
        '  d="${d%/}"; b="$(basename "$d")"; rep="${b%%_*}"',
        # LoraManager = LoRA Studio's Personal sync folder (previews / example images), never training data
        '  case "${b,,}" in *_loramanager) echo "skipped: ${b} (LoRA previews, not training images)"; continue;; esac',
        '  src="${CACHE_DIR}/images/${b}"; mkdir -p "${src}"',
        '  imgs=0; nocap=0',
        "  while IFS= read -r -d '' f; do",
        '    base="${f%.*}"; name="$(basename "${base}")"',
        '    if [ -f "${base}${CAPTION_EXT}" ]; then',
        '      ln -s "$f" "${src}/$(basename "$f")" && cp "${base}${CAPTION_EXT}" "${src}/${name}.txt" && imgs=$((imgs+1))',
        '    else nocap=$((nocap+1)); fi',
        "  done < <(find \"$d\" -maxdepth 1 -type f \\( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \\) -print0)",
        '  [ "${nocap}" -eq 0 ] || echo "WARNING: ${nocap} pictures in ${b} have no caption (*${CAPTION_EXT}) and are left out"',
        '  [ "${imgs}" -gt 0 ] || { [ "${nocap}" -eq 0 ] || fail_stage "no captions (*${CAPTION_EXT}) for the pictures in ${b}: write them in Datasets › 7 Tags (this LoRA type uses that caption set)"; continue; }',
        '  {',
        '    echo ""',
        '    echo "[[datasets]]"',
        '    echo "image_directory = \\"${src}\\""',
        '    echo "cache_directory = \\"${CACHE_DIR}/${b}\\""',
        '    echo "num_repeats = ${rep}"',
        '  } >> "${DATASET_TOML}"',
        '  echo "dataset: ${b}: ${imgs} images x ${rep} repeats"',
        "  count=$((count+1))",
        "done",
        '[ "${count}" -gt 0 ] || fail_stage "no <N>_name folders with images found in ${TRAIN_DIR}"',
        'echo "dataset config: ${DATASET_TOML} (${RESOLUTION}, batch ${BATCH_SIZE})"',
        "",
        f'stage "{ST[2]}"',
        f'python {MUSUBI_SRC}/{prefix}_cache_latents.py --dataset_config "${{DATASET_TOML}}" --vae "${{VAE}}"{mv}',
        "",
        f'stage "{ST[3]}"',
        f'python {MUSUBI_SRC}/{prefix}_cache_text_encoder_outputs.py --dataset_config "${{DATASET_TOML}}" --text_encoder "${{TEXT_ENCODER}}" {cfg.get("te_args", "--batch_size 8")}{mv}',
        "",
        f'stage "{ST[4]}"',
        cmd,
        "",
    ]
    if cfg.get("convert"):
        out += [
        f'stage "{ST[5]}"',
        "# musubi saves LoRAs in its own key format; ComfyUI needs the converted one. Each file is converted once",
        "# (a list of converted names is kept in the model folder), so running this stage again is safe.",
        f'CONVERTER="${{MUSUBI_DIR}}/{MUSUBI_SRC}/{cfg["convert"]}"',
        f'[ -f "${{CONVERTER}}" ] || fail_stage "musubi-tuner has no {cfg["convert"]}: update musubi-tuner (git pull)"',
        'DONE_LIST="${OUTPUT_DIR}/.comfy_converted"',
        'touch "${DONE_LIST}"',
        "shopt -s nullglob",
        "converted=0",
        'for f in "${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors" "${OUTPUT_DIR}/${OUTPUT_NAME}"-[0-9]*.safetensors; do',
        '  [ -f "$f" ] || continue',
        '  b="$(basename "$f")"',
        '  grep -qxF "$b" "${DONE_LIST}" && continue',
        '  python "${CONVERTER}" "$f" "${f%.safetensors}.tmp.safetensors"',
        '  mv -f "${f%.safetensors}.tmp.safetensors" "$f"',
        '  echo "$b" >> "${DONE_LIST}"',
        '  echo "converted: $b"',
        "  converted=$((converted+1))",
        "done",
        'echo "${converted} file(s) converted for ComfyUI"',
        "",
        ]
    out += [
        'echo "##LS_DONE ${STAGE_TOTAL}/${STAGE_TOTAL}"',
        "",
    ]
    return "\n".join(out)


# ---------------------------------- report -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Scan a LoRA dataset and plan epochs/steps.")
    ap.add_argument("name", nargs="?", help="LoRA folder name inside the base dir (or a full path)")
    ap.add_argument("--base", default=str(DEFAULT_BASE), help=f"base dir (default {DEFAULT_BASE})")
    ap.add_argument("--type", choices=list(PRESETS), help="LoRA type (asked if omitted)")
    ap.add_argument("--batch_size", type=int, help="override the suggested batch size")
    ap.add_argument("--grad_accum", type=int, help="override the suggested grad accumulation")
    ap.add_argument("--repeats", type=int, help="plan with this repeat count instead of the suggestion")
    ap.add_argument("--lr", help="override the preset learning rate (e.g. 5e-5)")
    ap.add_argument("--sample_prompts",
                    help="use this sample prompts file instead of generating <lora folder>/SDXLprompts.txt or ANIMAprompts.txt")
    ap.add_argument("--steps", type=int,
                    help="target total steps (default: the preset's, 2500 for SDXL)")
    ap.add_argument("--sd_dir", default=str(DEFAULT_SD_DIR),
                    help=f"sd-scripts dir with venv/ (default {DEFAULT_SD_DIR})")
    ap.add_argument("--musubi_dir", default=str(DEFAULT_MUSUBI_DIR),
                    help=f"musubi-tuner dir with venv/ for musubi types such as zimage (default {DEFAULT_MUSUBI_DIR})")
    ap.add_argument("--dry_run", action="store_true",
                    help="show what would be written, but write no files")
    args = ap.parse_args()

    base = Path(os.path.expanduser(args.base)).resolve()
    lora_dir = resolve_lora_dir(base, args.name)
    print(f"\nScanning {lora_dir} (including subfolders)...")

    folders, loose = find_repeat_folders(lora_dir)

    if not folders:
        print("\nNo '<N>_name' repeat folders found.")
        if loose:
            total = sum(loose.values())
            print(f"Found {total} images outside repeat folders; sd-scripts would IGNORE them:")
            for d, n in loose.items():
                print(f"  {d}  ({n} images)")
            sug = suggest_repeats(total, args.batch_size or 2, args.grad_accum or 1, 10, 1600)
            print(f"\nMove them into a folder like  {lora_dir.name}/{sug}_{lora_dir.name.lower()}/"
                  "  and run this again.")
        sys.exit(1)

    # Scan each repeat folder
    data = []
    for folder, repeats in folders:
        n, img_stems, cap_stems, dup = scan_folder(folder)
        data.append(dict(folder=folder, repeats=repeats, n=n,
                         matched=len(img_stems & cap_stems),
                         missing=img_stems - cap_stems,
                         orphans=cap_stems - img_stems, dup=dup))

    total_images = sum(d["n"] for d in data)
    if total_images == 0:
        sys.exit("Repeat folders found, but no images inside them.")

    # LoRA type
    type_key = args.type or choose(
        "\nWhich LoRA type?", [(v["label"], k) for k, v in PRESETS.items()])
    cfg = dict(PRESETS[type_key])
    if args.steps:
        cfg["steps"] = args.steps
        cfg["targets"] = (round(args.steps * 0.7), args.steps, round(args.steps * 1.3))
    lr = args.lr or cfg["lr"]

    # Suggest batch size / grad accum / repeats from the image count
    sug_bs, sug_ga, sug_rep = suggest_settings(total_images, cfg)
    print(f"\nFound {total_images} images. Suggested for {cfg['label']}: "
          f"batch size {sug_bs}, grad accum {sug_ga}, repeats {sug_rep}")
    print(f"  (aims for ~{cfg['steps']} steps, ~{cfg['views']} views/image, "
          f"~{cfg['epochs']} epochs; batch size capped at {cfg['max_batch']} by VRAM)")
    bs = args.batch_size or ask_int("Batch size", sug_bs)
    ga = args.grad_accum or ask_int("Gradient accumulation steps", sug_ga)
    eff = bs * ga

    # Repeats (single repeat folder only; multi-folder setups keep their own values)
    rename_cmd = None
    if len(data) == 1:
        cur = data[0]["repeats"]
        want = args.repeats or suggest_repeats(total_images, bs, ga, cfg["epochs"], cfg["steps"])
        if want != cur:
            print(f"\nFolder '{data[0]['folder'].name}' uses {cur} repeats; "
                  f"best for these settings is {want}.")
            use = True if args.repeats else ask(f"Plan with {want} repeats instead? (Y/n)", "y")
            if use is True or str(use).lower().startswith("y"):
                data[0]["repeats"] = want
                old = data[0]["folder"]
                new_name = REPEAT_RE.sub(f"{want}_", old.name, count=1)
                rename_cmd = f'mv "{old}" "{old.parent / new_name}"'

    samples = sum(d["n"] * d["repeats"] for d in data)
    spe = max(1, math.ceil(samples / eff))

    def short(lst):
        lst = sorted(lst)
        return ", ".join(lst[:5]) + (" ..." if len(lst) > 5 else "")

    print("\n" + "=" * 78)
    print(f" DATASET: {lora_dir.name}   TYPE: {cfg['label']}")
    print("=" * 78)
    for d in data:
        rel = d["folder"].relative_to(lora_dir) if d["folder"] != lora_dir else Path(".")
        print(f" {str(rel):<34} {d['n']:>4} images x {d['repeats']:<3} repeats ({d['matched']} captioned)")
        if d["missing"]:
            print(f"   ! images without caption ({len(d['missing'])}): {short(d['missing'])}")
        if d["orphans"]:
            print(f"   ! captions without image ({len(d['orphans'])}): {short(d['orphans'])}")
        if d["dup"]:
            print(f"   ! same filename, different extension ({len(d['dup'])}): {short(d['dup'])}")
    if loose:
        print(" ! images outside repeat folders (ignored by sd-scripts):")
        for p, n in loose.items():
            print(f"     {p}  ({n})")

    parents = sorted({f.parent for f, _ in folders})
    if len(parents) == 1:
        train_dir = parents[0]
        print(f"\n --train_data_dir should be: {train_dir}")
    else:
        train_dir = parents[0]
        print("\n ! Repeat folders sit in different parents. sd-scripts only reads the direct")
        print("   children of --train_data_dir, so move them under one parent:")
        for p in parents:
            print(f"     {p}")

    print("-" * 78)
    print(f" Total images:        {total_images}")
    print(f" Samples per epoch:   {samples}")
    print(f" Batch x accum:       {bs} x {ga} = {eff}")
    print(f" Steps per epoch:     ~{spe}")
    print(f" Learning rate:       {lr}  ({cfg['optimizer']}, {cfg['scheduler']}, "
          f"warmup {cfg['warmup']})")
    if rename_cmd:
        print(f" Repeats (planned):   {data[0]['repeats']}  <- rename the folder to apply it")
    print("\n EPOCH PLAN (use --max_train_epochs OR --max_train_steps, not both)")
    plan = {}
    main_target = cfg["targets"][len(cfg["targets"]) // 2]
    for target in cfg["targets"]:
        ep = max(1, round(target / spe))
        total = ep * spe
        views = total * eff / total_images
        plan[target] = ep
        flag = ""
        if views < VIEWS_LOW:
            flag = "  <- few views/image, may undertrain"
        elif views > VIEWS_HIGH:
            flag = "  <- many views/image, overfit risk"
        mark = "  <- used for launch command" if target == main_target else ""
        print(f"  ~{target:<5} steps: {ep:>3} epochs = {total:<5} steps, ~{views:.0f} views/image"
              f"{flag}{mark}")

    epochs = plan[main_target]
    lo, hi = max(1, round(epochs * 0.6)), max(1, round(epochs * 0.8))
    print(f"  Compare the saved checkpoints: the best one is usually around "
          f"epoch {lo}-{hi} of {epochs}, not the last.")
    print("  Likeness weak at the end -> raise LR / add epochs. Clothes or poses bleeding in "
          "-> use an earlier epoch or tighten captions.")

    print(f"\n {cfg['label']} notes")
    for n in cfg["notes"]:
        print(f"  - {n}")

    if rename_cmd:
        print("\n RENAME FIRST (sd-scripts reads repeats from the folder name):")
        print(f"   {rename_cmd}")

    # Sample prompts: generated from the most frequent caption tags
    write_prompts = False
    if is_musubi(cfg):
        sp = None                    # musubi types: no sample images
    elif args.sample_prompts:
        sp = Path(os.path.expanduser(args.sample_prompts))
        if not sp.is_file():
            print(f"\n ! --sample_prompts file not found: {sp}")
    else:
        sp = lora_dir / cfg["prompts_name"]
        caps = read_captions(folders)
        text, trigger, from_caps, top = build_prompts_text(
            caps, lora_dir.name.lower(), lora_dir.name, cfg)
        print(f"\n SAMPLE PROMPTS  (trigger word: {trigger}"
              + ("" if from_caps else " - guessed from folder name, check it") + ")")
        print(f"  Read {len(caps)} captions. Top tags: {', '.join(top[:10])}")
        write_prompts = True
        if sp.is_file():
            print(f"  {sp.name} already exists.")
            ans = ask("  Overwrite it with new suggestions? (y/N)", "n")
            write_prompts = str(ans).lower().startswith("y")
        if write_prompts:
            if args.dry_run:
                print("  [dry run] would write " + str(sp) + ":\n")
                print("".join("    " + l + "\n" for l in text.splitlines()))
            else:
                sp.write_text(text, encoding="utf-8")
                print(f"  Wrote {sp}  ({text.count(chr(10)) - 6} prompts, edit as you like)")
    sample_prompts = str(sp) if (sp is not None and (sp.is_file() or (write_prompts and args.dry_run))) else None

    save_every = cfg.get("save_every", 1)  # checkpoint every N epochs; samples are every epoch
    out_dir = train_dir / "model"
    # The .sh uses shell variables, so edit CHARACTER_NAME / paths at the top of the file.
    # Version the LoRA name: PersonSDXL -> PersonSDXLV1, V2, ... based on what is in model/
    base_name = lora_dir.name + cfg["suffix"]
    version = next_version(out_dir, base_name)
    out_name = f"{base_name}V{version}"
    run_cfg = {**cfg, "suffix": f"{cfg['suffix']}V{version}"}
    cmd = build_command(run_cfg, "${CHARACTER_NAME}", "${TRAIN_DIR}", "${OUTPUT_DIR}",
                        bs, ga, epochs, save_every, lr,
                        "${PROMPTS_FILE}" if sample_prompts else None)
    print("\n" + "=" * 78)
    if not cmd:
        print(" No launch template for this type; edit PRESETS to add one.")
        print("=" * 78 + "\n")
        return

    if is_musubi(cfg):
        sd_dir = Path(os.path.expanduser(args.musubi_dir)).resolve()
        if not (sd_dir / "venv").is_dir() or not (sd_dir / MUSUBI_SRC / cfg["script"]).is_file():
            print(f" ! {sd_dir} has no venv/ or {MUSUBI_SRC}/{cfg['script']} - check --musubi_dir")
    else:
        sd_dir = Path(os.path.expanduser(args.sd_dir)).resolve()
        if not (sd_dir / "venv").is_dir() or not (sd_dir / cfg["script"]).is_file():
            print(f" ! {sd_dir} has no venv/ or {cfg['script']} - check --sd_dir")
    print(f" LoRA output name: {out_name}"
          + ("  (no earlier versions in model/)" if version == 1
             else f"  (found V{version - 1} in model/)"))
    plan_line = (f"Output {out_name}. Plan: {total_images} images x {data[0]['repeats'] if len(data) == 1 else 'mixed'} "
                 f"repeats, batch {bs} x accum {ga}, ~{spe} steps/epoch, {epochs} epochs "
                 f"= ~{epochs * spe} steps (target {main_target})")
    prompts_sh = under(Path(sample_prompts), lora_dir, "${SCRIPT_DIR}") if sample_prompts else None
    script_text = render_training_script(under(sd_dir, Path.home(), "${HOME}"),
                                         under(train_dir, lora_dir, "${SCRIPT_DIR}"),
                                         prompts_sh, plan_line, cmd,
                                         {**ROCM_ENV, **cfg.get("env", {})}, cfg=run_cfg, bs=bs)
    sh_path = lora_dir / cfg["script_name"].format(name=lora_dir.name)

    if args.dry_run:
        print(f" [dry run] would write {sh_path}")
        print("=" * 78)
        print(script_text)
    else:
        if sh_path.exists():
            bak = sh_path.with_suffix(".sh.bak")
            sh_path.replace(bak)
            print(f" Existing script moved to {bak.name}")
        sh_path.write_text(script_text, encoding="utf-8")
        sh_path.chmod(sh_path.stat().st_mode | 0o755)
        print(f" TRAINING SCRIPT WRITTEN (~{main_target}-step plan, {epochs} epochs)")
        print("=" * 78)
        print(f" {sh_path}")
        if rename_cmd:
            print(" Rename the repeat folder first (see above), then:")
        print(f"\n   {sh_path}\n")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
