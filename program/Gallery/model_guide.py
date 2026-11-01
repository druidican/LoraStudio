# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
model_guide.py - which model files belong together, for LoRA Studio's Settings > Training.

1. identify(path): reads only the .safetensors header (tensor names, shapes, number types; no weights) and says
   what the file is: a Chroma / FLUX.1 / SDXL / SD3 / Lumina / ... diffusion model, a T5-XXL / CLIP-L / CLIP-G /
   Qwen3 / Gemma2 / ByT5 text encoder, a FLUX AE / SD VAE / Qwen-Image VAE, and in which form (bf16, fp16, plain
   fp8, ComfyUI fp8_scaled, or a full checkpoint with several parts bundled).
2. suggest(...): for a LoRA type and the chosen base model, ranks the files found in your model folders for every
   other slot (best first) and explains why other files do not fit.
3. lookup_online(...): fetches the base model's Hugging Face model card(s) and Civitai pages and pulls out the
   lines about text encoders, VAE and settings, plus links (and a Reddit search link).

Results of identify() are cached in model_ids.json (keyed by path, size and modification time).
"""

from __future__ import annotations
__version__ = "1.011126"  # LoRA Studio version, see version.py

import html
import json
import os
import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
CACHE_FILE = _LP.USER / "model_ids.json"
IDENT_VERSION = 5      # bump when identify() learns new kinds, so cached answers for old files are redone
_CACHE: dict | None = None

# What each kind is called in the UI
KIND_LABEL = {
    "chroma": "Chroma diffusion model", "flux_dev": "FLUX.1-dev diffusion model",
    "flux_schnell": "FLUX.1-schnell diffusion model", "sdxl_ckpt": "SDXL checkpoint (SDXL / Illustrious / Pony / NoobAI)",
    "sd15_ckpt": "SD 1.5 checkpoint", "sd2_ckpt": "SD 2.x checkpoint", "sd3": "SD3 / SD3.5 model",
    "lumina": "Lumina Image 2.0 model", "hunyuan": "HunyuanImage model", "anima": "Anima model",
    "t5xxl": "T5-XXL text encoder", "clip_l": "CLIP-L text encoder", "clip_g": "CLIP-G text encoder",
    "qwen_image": "Qwen-Image model", "flux2": "FLUX.2 model", "flux2_vae": "FLUX.2 AE (32-channel VAE)",
    "qwen3_8b": "Qwen3 8B text encoder (FLUX.2 klein 9B)",
    "zimage": "Z-Image Turbo model", "qwen3_4b": "Qwen3 4B text encoder (Z-Image)",
    "zimage_adapter": "Z-Image Turbo training adapter", "lora": "LoRA for another model (not a Z-Image adapter)",
    "zimage_lora": "Z-Image LoRA (a character/style LoRA, not the training adapter)",
    "qwen3": "Qwen3 text encoder", "gemma2": "Gemma-2 text encoder", "qwen25vl": "Qwen2.5-VL text encoder",
    "byt5": "ByT5 text encoder", "flux_ae": "FLUX AE (16-channel VAE)", "sd3_vae": "SD3 / SD3.5 VAE (not the FLUX AE)", "sd_vae": "SD / SDXL VAE (4-channel)",
    "qwen_vae": "Qwen-Image / Wan VAE (video-style)", "hunyuan_vae": "HunyuanImage VAE", "unknown": "unknown",
}

# Per LoRA type: slot -> accepted kinds (first = best). Subtypes (Pony, ...) use their base type.
REQUIREMENTS = {
    "sdxl": {"pretrained_model_name_or_path": ["sdxl_ckpt"], "vae": ["sd_vae"]},
    "illustrious": {"pretrained_model_name_or_path": ["sdxl_ckpt"], "vae": ["sd_vae"]},
    "sd15": {"pretrained_model_name_or_path": ["sd15_ckpt", "sd2_ckpt"], "vae": ["sd_vae"]},
    "sd3": {"pretrained_model_name_or_path": ["sd3"], "clip_l": ["clip_l"], "clip_g": ["clip_g"], "t5xxl": ["t5xxl"]},
    "flux": {"pretrained_model_name_or_path": ["flux_dev", "flux_schnell"], "clip_l": ["clip_l"],
             "t5xxl": ["t5xxl"], "ae": ["flux_ae"]},
    "chroma": {"pretrained_model_name_or_path": ["chroma"], "t5xxl": ["t5xxl"], "ae": ["flux_ae"]},
    "lumina": {"pretrained_model_name_or_path": ["lumina"], "gemma2": ["gemma2"], "ae": ["flux_ae"]},
    "anima": {"pretrained_model_name_or_path": ["anima"], "qwen3": ["qwen3"], "vae": ["qwen_vae"]},
    "hunyuan": {"pretrained_model_name_or_path": ["hunyuan"], "text_encoder": ["qwen25vl"], "byt5": ["byt5"],
                "vae": ["hunyuan_vae"]},
    # Z-Image Turbo (musubi-tuner): DiT, FLUX AE, Qwen3-4B, and the ostris training adapter (a LoRA file)
    "zimage": {"dit": ["zimage"], "vae": ["flux_ae"], "text_encoder": ["qwen3_4b"],
               "base_weights": ["zimage_adapter"]},     # ONLY the ostris adapter: any other LoRA merges nothing
}
REQUIREMENTS["qwenimage"] = {"dit": ["qwen_image"], "vae": ["qwen_vae"], "text_encoder": ["qwen25vl"]}
REQUIREMENTS["flux2"] = {"dit": ["flux2"], "vae": ["flux2_vae"], "text_encoder": ["qwen3_4b", "qwen3_8b"]}
BASE_FLAGS = ("pretrained_model_name_or_path", "dit")      # the slot that holds the diffusion model
# Which LoRA type to pick when the base model is of another family
KIND_TO_TYPE = {"chroma": "chroma", "flux_dev": "flux", "flux_schnell": "flux", "sdxl_ckpt": "sdxl",
                "sd15_ckpt": "sd15", "sd2_ckpt": "sd15", "sd3": "sd3", "lumina": "lumina", "anima": "anima",
                "hunyuan": "hunyuan", "zimage": "zimage", "qwen_image": "qwenimage", "flux2": "flux2"}

# Official model pages, used by the online lookup next to a search for the file's own name
OFFICIAL_HF = {
    "chroma": ["lodestones/Chroma1-HD"], "flux_dev": ["black-forest-labs/FLUX.1-dev"],
    "flux_schnell": ["black-forest-labs/FLUX.1-schnell"], "sdxl_ckpt": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "sd15_ckpt": ["stable-diffusion-v1-5/stable-diffusion-v1-5"], "sd3": ["stabilityai/stable-diffusion-3.5-medium"],
    "lumina": ["Alpha-VLLM/Lumina-Image-2.0"], "hunyuan": ["tencent/HunyuanImage-2.1"],
    "qwen_image": ["Comfy-Org/Qwen-Image_ComfyUI", "Qwen/Qwen-Image"],
    "flux2": ["black-forest-labs/FLUX.2-klein-base-4B", "black-forest-labs/FLUX.2-klein-base-9B", "black-forest-labs/FLUX.2-dev"],
    "zimage": ["Tongyi-MAI/Z-Image-Turbo", "Comfy-Org/z_image_turbo", "ostris/zimage_turbo_training_adapter"],
}
SD_SCRIPTS_DOCS = {
    "flux": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/flux_train_network.md",
    "chroma": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/flux_train_network.md",
    "sd3": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/sd3_train_network.md",
    "lumina": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/lumina_train_network.md",
    "sdxl": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md",
    "illustrious": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md",
    "sd15": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md",
    "hunyuan": "https://github.com/kohya-ss/sd-scripts/blob/main/docs/hunyuan_image_train_network.md",
    "zimage": "https://github.com/kohya-ss/musubi-tuner/blob/main/docs/zimage.md",
    "qwenimage": "https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md",
    "flux2": "https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md",
}

SKIP_DIRS = {"loras", "lora", "lycoris", "controlnet", "upscale_models", "embeddings", "clip_vision", "ipadapter",
             "sams", "ultralytics", "insightface", "facerestore_models", "style_models", "photomaker",
             "animatediff_models", "gligen", "hypernetworks", "__pycache__", ".git", "output", "outputs", "temp"}
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "first_stage_model.", "vae.")


# --------------------------------------------------------------------------- identify
def read_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n <= 0 or n > 300_000_000:
            raise ValueError("not a safetensors file")
        return json.loads(fh.read(n))


def _dtype_summary(head: dict) -> str:
    count: dict = {}
    for k, v in head.items():
        if k == "__metadata__" or not isinstance(v, dict) or k.endswith(("scale_weight", "scale_input")):
            continue
        count[v.get("dtype", "?")] = count.get(v.get("dtype", "?"), 0) + 1
    if not count:
        return "?"
    names = {"F16": "fp16", "BF16": "bf16", "F32": "fp32", "F8_E4M3": "fp8", "F8_E5M2": "fp8 (e5m2)", "U8": "uint8"}
    main = sorted(count.items(), key=lambda kv: -kv[1])
    first = names.get(main[0][0], main[0][0])
    fp8 = sum(c for d, c in count.items() if d.startswith("F8"))
    if fp8 and fp8 < sum(count.values()) * 0.9 and not first.startswith("fp8"):
        return f"{first} + fp8 (mixed)"
    if first.startswith("fp8") and len(main) > 1 and main[1][1] > 20:
        return "fp8 (mixed)"
    return first


def _shape(head, key):
    v = head.get(key)
    return v.get("shape") if isinstance(v, dict) else None


def _identify_head(head: dict, name: str) -> dict:
    keys = [k for k in head if k != "__metadata__"]
    meta = head.get("__metadata__") or {}
    out = {"kind": "unknown", "dtype": _dtype_summary(head), "scaled": False, "bundle": [], "note": ""}
    out["scaled"] = any(k.endswith((".scale_weight", ".scale_input")) or k == "scaled_fp8" for k in keys)
    parts = []
    if any(k.startswith(("model.diffusion_model.", "diffusion_model.")) for k in keys):
        parts.append("diffusion model")
    if any(k.startswith(("first_stage_model.", "vae.")) for k in keys):
        parts.append("VAE")
    if any(k.startswith(("conditioner.", "cond_stage_model.", "text_encoders.")) for k in keys):
        parts.append("text encoder(s)")
    out["bundle"] = parts if len(parts) > 1 else []

    def has(s):
        return any(s in k for k in keys)

    def strip(k):
        for p in PREFIXES:
            if k.startswith(p):
                return k[len(p):]
        return k
    sk = {strip(k) for k in keys}
    low = name.lower()

    # --- diffusion models / checkpoints
    flat0 = low.replace("_", "").replace("-", "")
    if has("double_stream_modulation") or (("flux2" in flat0 or "klein" in flat0)
                                            and any(k.startswith(("double_blocks.", "single_blocks.")) for k in sk)):
        out["kind"] = "flux2"                           # FLUX.2: shared modulation layers instead of per-block ones
        return out
    if any(k.startswith("transformer_blocks.") for k in sk) and (has("img_mod") or has("txt_norm") or "qwenimage" in flat0):
        out["kind"] = "qwen_image"
        return out
    if "double_blocks.0.img_attn.qkv.weight" in sk or any(k.startswith("double_blocks.") for k in sk):
        if has("distilled_guidance_layer") or "chroma" in low or (
                "time_in.in_layer.weight" not in sk and "vector_in.in_layer.weight" not in sk and not has("byt5_in")):
            out["kind"] = "chroma"
        elif has("byt5_in") or "hunyuan" in low:
            out["kind"] = "hunyuan"
        elif "guidance_in.in_layer.weight" in sk:
            out["kind"] = "flux_dev"
        else:
            out["kind"] = "flux_schnell" if "schnell" in low or "guidance_in.in_layer.weight" not in sk else "flux_dev"
        return out
    if any(k.startswith("joint_blocks.") for k in sk):
        out["kind"] = "sd3"
        return out
    if any(k.startswith(("cap_embedder.", "noise_refiner.", "context_refiner.")) for k in sk):
        def shp(n):
            return next((_shape(head, k) for k in keys if strip(k) == n), None)
        cap = shp("cap_embedder.1.weight") or shp("cap_embedder.0.weight")
        width = cap[-1] if cap else 0
        flat = low.replace("_", "").replace("-", "")
        if width == 2560 or "zimage" in flat:
            out["kind"] = "zimage"                      # Z-Image shares Lumina 2's layout but reads Qwen3-4B (2560 wide)
        else:
            out["kind"] = "lumina"
        return out
    if any(("lora_A" in k or "lora_down" in k) for k in keys[:400]):
        flat = low.replace("_", "").replace("-", "")
        # Z-Image layout: layers.N.attention / layers.N.feed_forward (other models use blocks / transformer_blocks)
        zlike = any(re.search(r"(^|[._])layers[._]\d+[._](attention|feed_forward|adaln)", k) for k in keys[:2000]) \
            and not any(re.search(r"transformer_blocks|double_blocks|single_blocks|joint_blocks|input_blocks", k)
                        for k in keys[:2000])
        if zlike and ("adapter" in flat or "deturbo" in flat):
            out["kind"] = "zimage_adapter"
        elif zlike:
            out["kind"] = "zimage_lora"
        else:
            out["kind"] = "lora"
            if "adapter" in flat:
                out["note"] = "named like an adapter, but its layers are not Z-Image layers"
        return out
    if any(k.startswith("conditioner.embedders.1") for k in keys) or (
            any(k.startswith("model.diffusion_model.input_blocks") for k in keys)
            and any(k.startswith("conditioner.") for k in keys)):
        out["kind"] = "sdxl_ckpt"
        return out
    if any(k.startswith("cond_stage_model.transformer") for k in keys):
        out["kind"] = "sd15_ckpt"
        return out
    if any(k.startswith("cond_stage_model.model.") for k in keys):
        out["kind"] = "sd2_ckpt"
        return out
    if any(k.startswith("input_blocks.") for k in sk) and any(k.startswith("label_emb") for k in sk):
        out["kind"] = "sdxl_ckpt"
        out["note"] = "UNet only"
        return out
    if "anima" in low and (any(k.startswith(("blocks.", "net.")) for k in sk) or len(keys) > 200):
        out["kind"] = "anima"
        out["note"] = "recognised by its name"
        return out

    # --- text encoders
    t5_shared = _shape(head, "shared.weight") or _shape(head, "encoder.embed_tokens.weight")
    wi = _shape(head, "encoder.block.0.layer.1.DenseReluDense.wi_0.weight") or _shape(
        head, "encoder.block.0.layer.1.DenseReluDense.wi.weight")
    if any(k.startswith("encoder.block.") for k in keys):
        width = (t5_shared or [0, 0])[-1] or (wi or [0, 0])[-1]
        if (t5_shared and t5_shared[0] == 384) or width in (1472,):
            out["kind"] = "byt5"
        elif width == 4096:
            out["kind"] = "t5xxl"
        else:
            out["kind"] = "t5xxl" if "t5" in low else "unknown"
            out["note"] = "" if out["kind"] == "t5xxl" else f"a T5-family encoder of width {width}"
        return out
    tok = _shape(head, "text_model.embeddings.token_embedding.weight") or _shape(
        head, "token_embedding.weight") or _shape(head, "transformer.text_model.embeddings.token_embedding.weight")
    if tok or any(k.startswith(("text_model.encoder.layers.", "transformer.resblocks.")) for k in keys):
        width = (tok or [0, 0])[-1]
        if width == 768:
            out["kind"] = "clip_l"
        elif width == 1280 or any(k.startswith("transformer.resblocks.31") for k in keys):
            out["kind"] = "clip_g"
        elif any(k.startswith("text_model.encoder.layers.31") for k in keys):
            out["kind"] = "clip_g"
        else:
            out["kind"] = "clip_l" if "clip_l" in low else ("clip_g" if "clip_g" in low else "unknown")
        return out
    emb = (_shape(head, "model.embed_tokens.weight") or _shape(head, "embed_tokens.weight")
           or _shape(head, "model.language_model.embed_tokens.weight"))
    if emb or any(k.startswith(("model.layers.", "layers.")) and "self_attn" in k for k in keys):
        vocab, width = (emb or [0, 0])[0], (emb or [0, 0])[-1]
        if vocab == 256000 or "gemma" in low:
            out["kind"] = "gemma2"
        elif width == 4096 and ("qwen" in low or "klein" in low):
            out["kind"] = "qwen3_8b"
        elif width == 2560 or "qwen_3_4b" in low or "qwen3_4b" in low or "qwen3-4b" in low:
            out["kind"] = "qwen3_4b"
        elif has("visual.") or width == 3584 or "qwen2.5" in low or "qwen_2.5" in low:
            out["kind"] = "qwen25vl"
        elif width == 1024 or "qwen3" in low or "qwen_3" in low:
            out["kind"] = "qwen3"
        else:
            out["note"] = f"a language-model text encoder (width {width})"
        return out

    # --- VAEs
    cin = _shape(head, "decoder.conv_in.weight") or _shape(head, "first_stage_model.decoder.conv_in.weight") \
        or _shape(head, "vae.decoder.conv_in.weight") or _shape(head, "decoder.conv1.weight")
    if cin or has("decoder.up_blocks") or has("decoder.mid"):
        ch = cin[1] if cin and len(cin) > 1 else 0
        if cin and len(cin) == 5:
            out["kind"] = "qwen_vae"
        elif ch == 16 and re.search(r"sd3|sd_3|3\.5|35", low.replace("-", "_")) and "flux" not in low:
            out["kind"] = "sd3_vae"                     # same shape as the FLUX AE, other weights: only the name tells
            out["note"] = "identified by its name: SD3's VAE has the same layout as the FLUX AE"
        elif ch == 16:
            out["kind"] = "flux_ae"
        elif ch == 4:
            out["kind"] = "sd_vae"
        elif ch == 32 and "hunyuan" not in low:
            out["kind"] = "flux2_vae"
        elif ch in (32, 64):
            out["kind"] = "hunyuan_vae"
        else:
            out["kind"] = "unknown"
            out["note"] = f"a VAE with {ch} latent channels"
        return out
    if meta.get("modelspec.architecture"):
        out["note"] = f"modelspec: {meta['modelspec.architecture']}"
    return out


def _cache() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _CACHE = {}
    return _CACHE


def save_cache():
    try:
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache()), encoding="utf-8")
        tmp.replace(CACHE_FILE)
    except OSError:
        pass


def identify(path) -> dict:
    """{'kind', 'dtype', 'scaled', 'bundle', 'note', 'size'} for a model file (cached)."""
    p = Path(os.path.expanduser(str(path)))
    try:
        st = p.stat()
    except OSError:
        return {"kind": "missing", "dtype": "", "scaled": False, "bundle": [], "note": "file not found", "size": 0}
    key = str(p)
    hit = _cache().get(key)
    if hit and hit.get("_s") == st.st_size and hit.get("_m") == st.st_mtime and hit.get("_v") == IDENT_VERSION:
        return hit
    if p.suffix.lower() not in (".safetensors", ".sft"):
        info = {"kind": "unknown", "dtype": "", "scaled": False, "bundle": [], "note": f"{p.suffix} files are not inspected"}
    else:
        try:
            info = _identify_head(read_header(p), p.name)
        except Exception as e:
            info = {"kind": "unknown", "dtype": "", "scaled": False, "bundle": [], "note": f"could not read: {e}"}
    info.update(size=st.st_size, _s=st.st_size, _m=st.st_mtime, _v=IDENT_VERSION)
    _cache()[key] = info
    return info


def describe(info: dict) -> str:
    """'T5-XXL text encoder, fp16, 9.1 GB' (+ problems)."""
    bits = [KIND_LABEL.get(info.get("kind"), info.get("kind", "?"))]
    if info.get("dtype"):
        bits.append(info["dtype"])
    if info.get("size"):
        gb = info["size"] / 1024 ** 3
        bits.append(f"{gb:.1f} GB" if gb >= 1 else f"{info['size'] / 1024 ** 2:.0f} MB")
    s = ", ".join(bits)
    if info.get("scaled"):
        s += " · **ComfyUI fp8_scaled**"
    if info.get("bundle"):
        s += " · contains " + " + ".join(info["bundle"])
    if info.get("note"):
        s += f" · {info['note']}"
    return s


# --------------------------------------------------------------------------- scan folders
def model_roots(dirs) -> list:
    """The folders to search: every given folder, plus the 'models' folder above it (where the other kinds live)."""
    roots = []
    for d in dirs:
        if not d:
            continue
        p = Path(os.path.expanduser(str(d)))
        if not p.is_dir():
            continue
        roots.append(p)
        for anc in list(p.parents)[:4]:
            if anc.name.lower() in ("models", "model"):
                roots.append(anc)
                break
    # the folder all slot folders share (e.g. /data/models for .../diffusion_models/..., .../vae/...)
    if len(roots) > 1:
        try:
            common = Path(os.path.commonpath([str(r) for r in roots]))
            if len(common.parts) >= 3 and common not in (Path.home(), Path("/mnt"), Path("/home"), Path("/media")):
                roots.append(common)
        except ValueError:
            pass
    out = []
    for r in sorted(set(roots), key=lambda x: len(x.parts)):
        if not any(r != o and o in r.parents for o in out):
            out.append(r)
    return out


def scan(dirs, limit=2000) -> list:
    """[(path, info)] for every .safetensors file under the model roots (skips LoRA, ControlNet, ... folders)."""
    files = []
    for root in model_roots(dirs):
        base_depth = len(root.parts)
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d.lower() not in SKIP_DIRS and not d.startswith(".")]
            if len(Path(dp).parts) - base_depth >= 4:
                dns[:] = []
            files += [Path(dp) / f for f in fns if f.lower().endswith((".safetensors", ".sft"))]
            if len(files) >= limit:
                break
    out = []
    for f in dict.fromkeys(files):
        out.append((f, identify(f)))
    save_cache()
    return out


# --------------------------------------------------------------------------- suggest
def _rank(info: dict, slot: str) -> tuple:
    """Lower is better."""
    dt = info.get("dtype", "")
    if slot in BASE_FLAGS:
        order = {"bf16": 0, "fp16": 1, "fp32": 3}
        r = order.get(dt, 2 if dt.startswith("fp8") else 4)
    else:
        order = {"fp16": 0, "bf16": 0, "fp32": 2}
        r = order.get(dt, 1 if dt.startswith("fp8") else 3)
    return (r, -info.get("size", 0))


def problems(info: dict, slot: str, kinds: list) -> str:
    """Why a file cannot be used in this slot ('' = usable)."""
    if info.get("kind") not in kinds:
        return f"is a {KIND_LABEL.get(info.get('kind'), 'unknown file')}, not a {KIND_LABEL.get(kinds[0], kinds[0])}"
    if info.get("scaled"):
        return "is a ComfyUI fp8_scaled file: the trainer ignores its scales and would train on wrong weights"
    if info.get("bundle") and slot not in BASE_FLAGS:
        return "is a full checkpoint with other parts bundled, not a standalone file"
    return ""


def requirements_for(tkey: str, base_type: str = "") -> dict:
    return REQUIREMENTS.get(tkey) or REQUIREMENTS.get(base_type or "", {})


def suggest(tkey: str, base_type: str, slots: list, current: dict, dirs) -> dict:
    """slots: [flag, ...] of this LoRA type. Returns {'md': markdown, 'picks': {flag: path}}."""
    req = requirements_for(tkey, base_type)
    if not req:
        return {"md": "No compatibility table for this LoRA type yet.", "picks": {}}
    found = scan(dirs)
    base_flag = next((f for f in BASE_FLAGS if f in req), "pretrained_model_name_or_path")
    L, picks = [], {}
    base = (current.get(base_flag) or "").strip()
    if base and "/PATH/TO" not in base:
        bi = identify(base)
        L.append(f"**Base model:** `{Path(base).name}`: {describe(bi)}.")
        bad = problems(bi, base_flag, req.get(base_flag, []))
        if bi.get("kind") == "missing":
            L.append("- ⚠ The file does not exist.")
        elif bad:
            other = KIND_TO_TYPE.get(bi.get("kind"))
            L.append(f"- ⚠ This file {bad}." + (f" Pick the LoRA type for it (`{other}`) or another base model."
                                                if other and other != tkey and other != base_type else ""))
        elif bi.get("bundle") and bi.get("kind") not in ("sdxl_ckpt", "sd15_ckpt", "sd2_ckpt", "sd3"):
            L.append("- This file also contains other parts; a diffusion-model-only file is the safer choice for sd-scripts.")
        else:
            L.append("- ✓ Fits this LoRA type.")
    else:
        L.append("**Base model:** not chosen yet. Suggestions for it are listed below.")
    L.append("")
    if not found:
        L.append("No model files found in the slot folders (or a `models` folder above them). Set a folder for a slot first.")
    for flag in slots:
        kinds = req.get(flag)
        if not kinds:
            continue
        ok = [(p, i) for p, i in found if not problems(i, flag, kinds)]
        ok.sort(key=lambda pi: (kinds.index(pi[1]["kind"]), *_rank(pi[1], flag)))
        near = [(p, i, problems(i, flag, kinds)) for p, i in found
                if i.get("kind") in kinds and problems(i, flag, kinds)]
        label = "Base model" if flag in BASE_FLAGS else KIND_LABEL.get(kinds[0], flag)
        L.append(f"**{label}** (`--{flag}`, needs a {KIND_LABEL.get(kinds[0], kinds[0])}):")
        if flag == base_flag and base:
            others = [(p, i) for p, i in ok if p != Path(os.path.expanduser(base))]
            bi = identify(base)
            if others and bi.get("kind") in kinds and problems(bi, flag, kinds):
                picks[flag] = str(others[0][0])        # same family, but this file is unusable: offer the good one
                L.append(f"- ★ use instead: `{others[0][0].name}`: {describe(others[0][1])}")
                others = others[1:]
            if others:
                L.append("- other fitting files: " + ", ".join(f"`{p.name}`" for p, _ in others[:4]))
            L.append("")
            continue
        if ok:
            picks[flag] = str(ok[0][0])
            cur = (current.get(flag) or "").strip()
            for n, (p, i) in enumerate(ok[:4]):
                mark = "★ best" if n == 0 else "alternative"
                now = " (selected now)" if cur and Path(os.path.expanduser(cur)) == p else ""
                L.append(f"- {mark}: `{p.name}`{now}: {describe(i)}")
        else:
            L.append(f"- none found. Download a {KIND_LABEL.get(kinds[0], kinds[0])} (see **Look up online**).")
        cur = (current.get(flag) or "").strip()
        if cur and "/PATH/TO" not in cur and not any(Path(os.path.expanduser(cur)) == p for p, _ in ok):
            ci = identify(cur)
            why = problems(ci, flag, kinds) if ci.get("kind") != "missing" else "does not exist"
            if why:
                L.append(f"- ⚠ selected now: `{Path(cur).name}` {why}.")
        for p, i, why in near[:3]:
            L.append(f"- ✗ `{p.name}` {why}.")
        L.append("")
    L.append(f"_Checked {len(found)} model files in: " + ", ".join(f"`{r}`" for r in model_roots(dirs)) + "._")
    return {"md": "\n".join(L), "picks": picks}


# --------------------------------------------------------------------------- online lookup
UA = {"User-Agent": "LoRA-Studio/1.0 (model guide)"}
KEYWORDS = re.compile(r"text[ _-]?encoder|\bt5|clip[ _-]?[lg]?\b|\bvae\b|ae\.safetensors|\bqwen|\bgemma|byt5|"
                      r"\bcfg\b|guidance|\bsteps?\b|sampler|scheduler|resolution|negative prompt|fp8|bf16|fp16",
                      re.I)


def _get(url: str, token: str = "", timeout: int = 12):
    headers = dict(UA)
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _clean_lines(text: str, limit: int = 10) -> list:
    text = re.sub(r"</?(strong|b|i|em|code|a|span|u|small|sup|sub)\b[^>]*>", "", text or "", flags=re.I)
    text = re.sub(r"<[^>]+>", "\n", html.unescape(text))
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    out, seen = [], set()
    for raw in re.split(r"\n|(?<=[.!?])\s+(?=[A-Z])", text):
        ln = re.sub(r"(^|\s)[#>]+\s*|[*`|]+|(?<!\w)_+|_+(?!\w)", " ", raw)
        ln = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", ln)
        ln = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", ln)
        ln = re.sub(r"\s+", " ", ln).strip(" -:")
        if 25 <= len(ln) and KEYWORDS.search(ln) and ln.lower() not in seen:
            seen.add(ln.lower())
            out.append(ln[:240] + ("..." if len(ln) > 240 else ""))
        if len(out) >= limit:
            break
    return out


def search_name(path: str) -> str:
    """'Chroma1-HD-fp8_scaled_original_hybrid.safetensors' -> 'Chroma1 HD'."""
    stem = Path(path).stem
    stem = re.sub(r"(?i)[_\-. ](fp8|fp16|bf16|fp32|e4m3fn|e5m2|scaled|original|hybrid|pruned|emaonly|ema|"
                  r"fp8e4m3fn|q\d_k\w*|gguf|full|model|base)(?=[_\-. ]|$).*$", "", stem)
    return re.sub(r"[_\-]+", " ", stem).strip() or Path(path).stem


def lookup_online(tkey: str, base_path: str, hf_token: str = "") -> str:
    info = identify(base_path) if base_path else {"kind": "unknown"}
    kind = info.get("kind", "unknown")
    name = search_name(base_path) if base_path else tkey
    L = [f"### Online: {name}" + (f" ({KIND_LABEL.get(kind, kind)})" if kind not in ("unknown", "missing") else "")]
    errors = []

    # Hugging Face: official page(s) + a search for this file's name
    repos = list(OFFICIAL_HF.get(kind, []))
    try:
        found = json.loads(_get("https://huggingface.co/api/models?" + urllib.parse.urlencode(
            {"search": name, "limit": 5, "sort": "downloads", "direction": -1}), hf_token))
        repos += [m["id"] for m in found if m.get("id") and m["id"] not in repos][:2]
    except Exception as e:
        errors.append(f"Hugging Face search: {e}")
    for repo in repos[:3]:
        url = f"https://huggingface.co/{repo}"
        try:
            readme = _get(f"https://huggingface.co/{repo}/raw/main/README.md", hf_token)
            lines = _clean_lines(readme)
            try:
                files = json.loads(_get(f"https://huggingface.co/api/models/{repo}", hf_token)).get("siblings", [])
                comp = [f["rfilename"] for f in files if re.search(
                    r"(text_encoder|t5|clip|vae|ae\.safetensors|qwen|gemma|byt5)", f.get("rfilename", ""), re.I)
                    and f["rfilename"].endswith((".safetensors", ".json"))][:8]
            except Exception:
                comp = []
            L.append(f"\n**Hugging Face: [{repo}]({url})**")
            if comp:
                L.append("- files in the repo: " + ", ".join(f"`{c}`" for c in comp))
            L += [f"- {ln}" for ln in lines] or ["- (no lines about encoders / VAE / settings in the model card)"]
        except Exception as e:
            errors.append(f"{repo}: {e}")

    # Civitai: model pages found by name
    try:
        data = json.loads(_get("https://civitai.com/api/v1/models?" + urllib.parse.urlencode(
            {"query": name, "types": "Checkpoint", "limit": 3, "sort": "Highest Rated"})))
        for m in data.get("items", [])[:3]:
            ver = (m.get("modelVersions") or [{}])[0]
            url = f"https://civitai.com/models/{m.get('id')}"
            L.append(f"\n**Civitai: [{m.get('name', '?')}]({url})** · base model: {ver.get('baseModel', '?')}")
            text = (m.get("description") or "") + "\n" + (ver.get("description") or "")
            L += [f"- {ln}" for ln in _clean_lines(text, 8)] or ["- (no lines about encoders / VAE / settings)"]
        if not data.get("items"):
            L.append(f"\n**Civitai:** nothing found for “{name}”.")
    except Exception as e:
        errors.append(f"Civitai: {e}")

    q = urllib.parse.quote(f"{name} text encoder vae")
    L.append("\n**More:**")
    L.append(f"- [Reddit r/StableDiffusion search](https://www.reddit.com/r/StableDiffusion/search/?q={q}&restrict_sr=1)")
    doc = SD_SCRIPTS_DOCS.get(tkey) or SD_SCRIPTS_DOCS.get(KIND_TO_TYPE.get(kind, ""), "")
    if doc:
        L.append(f"- [sd-scripts training guide for this model family]({doc})")
    if errors:
        L.append("\n_Could not reach: " + "; ".join(errors[:4]) + "_")
    L.append("\n_Lines are quoted from the pages as they are; the suggestions above (from your files) are the checked part._")
    return "\n".join(L)
