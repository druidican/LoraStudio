# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Downloads missing model files from Hugging Face, on a click, with your token (huggingtoken.txt).

KNOWN lists the usual file for every model slot of the built-in LoRA types, taken from the sd-scripts and
musubi-tuner documentation (and the ComfyUI repackaged repos they point to). Any other Hugging Face file can be
fetched by pasting its link (…/blob/main/<file> or …/resolve/main/<file>) for a slot.

Before a download the file is looked up on Hugging Face (size, gated or not). It is saved in your models folder
(Settings › Folders) in the usual ComfyUI subfolder for its kind (checkpoints, diffusion_models, text_encoders, vae,
loras), or in the slot's own folder when one is set. Text encoders split in several files are fetched whole. A
stopped download goes on from where it stopped next time (<file>.part). Afterwards the slot is set and saved in
training_settings.txt, like pressing Save on Settings › Model files.

Gated repos (FLUX.1-dev, FLUX.2-dev, SD 3.5): accept the licence once on the repo's page while logged in to
huggingface.co, with the account the token belongs to.
"""
from __future__ import annotations
__version__ = "1.111126h"  # LoRA Studio version, see version.py

import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 4 * 1024 * 1024
STOP = threading.Event()

# flag (without --) -> (repo, path in the repo, number of parts). Parts > 1: path is the first part,
# "...-00001-of-0000N...", and all N are fetched.
_AE1 = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors", 1)               # the FLUX.1 AE, not gated
_T5 = ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors", 1)
_CLIPL = ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", 1)
_SDXL_VAE = ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors", 1)
KNOWN: dict[str, dict[str, tuple]] = {
    "sdxl": {"pretrained_model_name_or_path": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", 1),
             "vae": _SDXL_VAE},
    "illustrious": {"pretrained_model_name_or_path": ("OnomaAIResearch/Illustrious-xl-early-release-v0",
                                                      "Illustrious-XL-v0.1.safetensors", 1),
                    "vae": _SDXL_VAE},
    "sd15": {"pretrained_model_name_or_path": ("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                               "v1-5-pruned-emaonly.safetensors", 1)},
    "sd3": {"pretrained_model_name_or_path": ("stabilityai/stable-diffusion-3.5-medium", "sd3.5_medium.safetensors", 1),
            "clip_l": ("stabilityai/stable-diffusion-3.5-medium", "text_encoders/clip_l.safetensors", 1),
            "clip_g": ("stabilityai/stable-diffusion-3.5-medium", "text_encoders/clip_g.safetensors", 1),
            "t5xxl": ("stabilityai/stable-diffusion-3.5-medium", "text_encoders/t5xxl_fp16.safetensors", 1)},
    "flux": {"pretrained_model_name_or_path": ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", 1),
             "clip_l": _CLIPL, "t5xxl": _T5, "ae": _AE1},
    "chroma": {"pretrained_model_name_or_path": ("lodestones/Chroma1-HD", "Chroma1-HD.safetensors", 1),
               "t5xxl": _T5, "ae": _AE1},
    "lumina": {"pretrained_model_name_or_path": ("Comfy-Org/Lumina_Image_2.0_Repackaged",
                                                 "split_files/diffusion_models/lumina_2_model_bf16.safetensors", 1),
               "gemma2": ("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/text_encoders/gemma_2_2b_fp16.safetensors", 1),
               "ae": ("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/vae/ae.safetensors", 1)},
    "hunyuan": {"pretrained_model_name_or_path": ("tencent/HunyuanImage-2.1", "dit/hunyuanimage2.1.safetensors", 1),
                "text_encoder": ("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", 1),
                "byt5": ("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", 1),
                "vae": ("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors", 1)},
    "anima": {"pretrained_model_name_or_path": ("circlestone-labs/Anima",
                                                "split_files/diffusion_models/anima-base-v1.0.safetensors", 1),
              "qwen3": ("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors", 1),
              "vae": ("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", 1)},
    "zimage": {"dit": ("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors", 1),
               "text_encoder": ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", 1),
               "vae": ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", 1),
               "base_weights": ("ostris/zimage_turbo_training_adapter", "zimage_turbo_training_adapter_v2.safetensors", 1)},
    "zimagebase": {"dit": ("Comfy-Org/z_image", "split_files/diffusion_models/z_image_bf16.safetensors", 1),
                   "text_encoder": ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", 1),
                   "vae": ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", 1)},
    "qwenimage": {"dit": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_bf16.safetensors", 1),
                  "text_encoder": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", 1),
                  "vae": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", 1)},
    "flux2": {"dit": ("black-forest-labs/FLUX.2-klein-base-4B", "flux-2-klein-base-4b.safetensors", 1),
              "vae": ("black-forest-labs/FLUX.2-dev", "ae.safetensors", 1),
              "text_encoder": ("black-forest-labs/FLUX.2-klein-base-4B", "text_encoder/model-00001-of-00002.safetensors", 2)},
}

# the ComfyUI subfolder for each kind of slot
KIND = {"vae": "vae", "ae": "vae", "clip_l": "text_encoders", "clip_g": "text_encoders", "t5xxl": "text_encoders",
        "qwen3": "text_encoders", "gemma2": "text_encoders", "text_encoder": "text_encoders", "byt5": "text_encoders",
        "base_weights": "loras", "dit": "diffusion_models"}
CHECKPOINT_TYPES = {"sdxl", "illustrious", "sd15", "sd3"}
GENERIC = re.compile(r"^(ae|model|diffusion_pytorch_model)([-.]|$)", re.I)    # names that need their own subfolder


# --------------------------------------------------------------------------- the list
def _base_of(tkey: str) -> str:
    try:
        import sd_script
        sub = sd_script.custom_subtypes().get(tkey)
        if sub:
            return sub["base"]
    except Exception:                                  # noqa: BLE001
        pass
    return tkey


def known(tkey: str) -> dict:
    """{flag: (repo, path, parts)} for a LoRA type. A subtype (Pony …) gets its base type's files except the base
    model, which is its own."""
    if tkey in KNOWN:
        return dict(KNOWN[tkey])
    base = _base_of(tkey)
    out = dict(KNOWN.get(base, {}))
    out.pop("pretrained_model_name_or_path", None)
    out.pop("dit", None)
    return out


def part_paths(path: str, parts: int) -> list[str]:
    if parts <= 1:
        return [path]
    m = re.search(r"(\d+)-of-(\d+)", path)
    if not m:
        return [path]
    w, total = len(m.group(1)), m.group(2)
    return [path[:m.start()] + f"{i:0{w}d}-of-{total}" + path[m.end():] for i in range(1, parts + 1)]


def parse_link(text: str):
    """A Hugging Face file link -> (repo, revision, path), else None. Takes …/blob/<rev>/<path>,
    …/resolve/<rev>/<path> and owner/repo/<path>."""
    t = (text or "").strip().strip("\"'<>")
    if not t:
        return None
    t = t.split("?", 1)[0].split("#", 1)[0]
    t = re.sub(r"^(https?://)?(www\.)?(huggingface\.co|hf\.co)/", "", t)
    if t.startswith(("http://", "https://")):
        return None
    parts = [urllib.parse.unquote(p) for p in t.split("/") if p]
    if len(parts) >= 5 and parts[2] in ("blob", "resolve", "raw"):
        return "/".join(parts[:2]), parts[3], "/".join(parts[4:])
    if len(parts) >= 3 and parts[2] not in ("tree", "blob", "resolve"):
        return "/".join(parts[:2]), "main", "/".join(parts[2:])
    return None


def models_root() -> Path | None:
    sg = sys.modules.get("sdgui")
    d = (getattr(sg, "DEFAULT_MODEL_DIR", "") if sg else "") or ""
    if not d:
        try:
            import lora_library as ll
            d = ll.load_settings().get("models_dir", "") or ""
        except Exception:                              # noqa: BLE001
            d = ""
    return Path(os.path.expanduser(d)) if d else None


def kind_folder(flag: str, tkey: str) -> str:
    if flag in KIND:
        return KIND[flag]
    return "checkpoints" if _base_of(tkey) in CHECKPOINT_TYPES else "diffusion_models"


def dest_for(flag: str, tkey: str, repo: str, path: str, slot_dir: str = "") -> Path:
    """Where the file goes: the slot's folder if it exists, else <models>/<kind>/; generic names (ae.safetensors,
    model-00001-of-00002…) in a subfolder named after the repo, so two different ae files never collide."""
    sd = Path(os.path.expanduser(slot_dir.strip())) if (slot_dir or "").strip() else None
    root = models_root()
    if sd is not None and sd.is_dir() and not (root is not None and sd.resolve() == root.resolve()):
        base = sd                                      # a folder chosen for this slot (not just the models folder)
    else:
        if root is None:
            raise RuntimeError("no models folder is set (Settings › Folders › Models), and the slot has no folder")
        base = root / kind_folder(flag, tkey)
    name = Path(path).name
    if GENERIC.match(name):
        base = base / re.sub(r"[^A-Za-z0-9._-]+", "_", repo.split("/")[-1])
    return base / name


# --------------------------------------------------------------------------- talking to Hugging Face
def _token() -> str:
    try:
        import sd_settings
        return sd_settings.read_hf_token() or ""
    except Exception:                                  # noqa: BLE001
        pass
    sg = sys.modules.get("sdgui")
    try:
        return sg.read_hf_token() if sg is not None else ""
    except Exception:                                  # noqa: BLE001
        return ""


def url_of(repo: str, path: str, rev: str = "main") -> str:
    return f"{ENDPOINT}/{repo}/resolve/{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(path)}"


def page_of(repo: str) -> str:
    return f"{ENDPOINT}/{repo}"


class _Head(urllib.request.HTTPRedirectHandler):
    """HEAD without following the redirect to the file store: the size is on the first answer."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _req(url: str, token: str, method="GET", extra=None):
    h = {"User-Agent": "LoRA-Studio"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    h.update(extra or {})
    return urllib.request.Request(url, headers=h, method=method)


def _why(code: int, repo: str, token: str) -> str:
    if code in (401, 403):
        if not token:
            return (f"needs a Hugging Face token: put yours in Settings › Hugging Face token (and accept the licence on "
                    f"{page_of(repo)} if it asks).")
        return (f"is gated: open {page_of(repo)} while logged in to huggingface.co with the account of your token, "
                "accept the licence (agree / request access), then try again.")
    if code == 404:
        return "is not there (moved or renamed): find the file on huggingface.co and paste its link below."
    return f"answered HTTP {code}."


def probe(repo: str, path: str, rev: str = "main", token: str | None = None) -> dict:
    """{ok, size, code, text}: is the file there and may this token fetch it?"""
    token = _token() if token is None else token
    opener = urllib.request.build_opener(_Head)
    try:
        r = opener.open(_req(url_of(repo, path, rev), token, "HEAD"), timeout=30)
        code, hd = r.status, r.headers
    except urllib.error.HTTPError as e:              # 302 (not followed) lands here too
        code, hd = e.code, e.headers
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "size": 0, "code": 0, "text": f"Hugging Face cannot be reached ({getattr(e, 'reason', e)})."}
    if code in (200, 301, 302, 303, 307, 308):
        size = int(hd.get("X-Linked-Size") or hd.get("Content-Length") or 0)
        return {"ok": True, "size": size, "code": code, "text": ""}
    return {"ok": False, "size": 0, "code": code, "text": f"`{repo}/{path}` " + _why(code, repo, token)}


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def download(repo: str, path: str, dest: Path, rev: str = "main", token: str | None = None, size: int = 0,
             stop: threading.Event | None = None, label: str = ""):
    """Generator of progress lines; returns True when the whole file is in place. Goes on from <dest>.part."""
    token = _token() if token is None else token
    stop = stop or STOP
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if size and have >= size:
        os.replace(part, dest)
        return True
    hdr = {"Range": f"bytes={have}-"} if have else {}
    try:
        r = urllib.request.urlopen(_req(url_of(repo, path, rev), token, extra=hdr), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:                      # already complete
            os.replace(part, dest)
            return True
        yield f"❌ `{Path(path).name}` " + _why(e.code, repo, token)
        return False
    except (urllib.error.URLError, OSError) as e:
        yield f"❌ Hugging Face cannot be reached ({getattr(e, 'reason', e)})."
        return False
    if have and r.status != 206:                        # the server ignored the range: start again
        have = 0
    total = size or (have + int(r.headers.get("Content-Length") or 0))
    t0, last, got = time.time(), 0.0, 0
    name = label or Path(path).name
    try:
        with open(part, "ab" if have else "wb") as f:
            while True:
                if stop.is_set():
                    yield f"■ Stopped `{name}` at {gb(have + got)} of {gb(total)}: the next download goes on from there."
                    return False
                b = r.read(CHUNK)
                if not b:
                    break
                f.write(b)
                got += len(b)
                if time.time() - last > 1.0:
                    last = time.time()
                    rate = got / max(0.1, last - t0)
                    pct = f"{100 * (have + got) / total:.0f} %" if total else gb(have + got)
                    eta = f", about {int((total - have - got) / rate / 60) + 1} min left" if total and rate > 0 else ""
                    yield f"⏳ `{name}`: {pct} of {gb(total)} ({rate / 1e6:.0f} MB/s{eta})"
    except OSError as e:
        yield f"❌ Writing `{part}` failed: {e}"
        return False
    finally:
        r.close()
    if total and part.stat().st_size < total:
        yield f"❌ `{name}` stopped early ({gb(part.stat().st_size)} of {gb(total)}): press Download again to go on."
        return False
    os.replace(part, dest)
    return True


# --------------------------------------------------------------------------- one slot
def _save_slot(tkey: str, flag: str, path: Path):
    """The slot's file and folder into training_settings.txt (only these two keys of this type change)."""
    import sd_settings as SS
    data = SS.read_settings()
    sec = data.setdefault(tkey, {})
    sec[flag] = str(path)
    sec[flag + ".dir"] = str(path.parent if not GENERIC.match(path.name) else path.parent.parent)
    SS.write_settings(data)


def fetch(tkey: str, flag: str, slot_dir: str = "", link: str = "", stop=None, save=True):
    """Generator of Markdown lines for one slot; the last one says how it went. Returns the file's path or None.
    `link` (any Hugging Face file link) replaces the built-in entry."""
    stop = stop or STOP
    if link:
        p = parse_link(link)
        if p is None:
            yield "❌ That is not a Hugging Face file link (it should look like `https://huggingface.co/<owner>/<repo>/blob/main/<file>`)."
            return None
        repo, rev, path = p
        m = re.search(r"\d+-of-(\d+)", Path(path).name)
        parts = int(m.group(1)) if m else 1             # one part of a split file pasted: all parts are fetched
        if parts > 1:
            path = part_paths(path, parts)[0]
    else:
        ent = known(tkey).get(flag)
        if not ent:
            yield f"No built-in file for `--{flag}` of this type: paste a Hugging Face link for it below."
            return None
        (repo, path, parts), rev = ent, "main"
    files = part_paths(path, parts)
    token = _token()
    infos = []
    yield f"⏳ Looking up `{repo}` on Hugging Face …"
    for fp in files:
        info = probe(repo, fp, rev, token)
        if not info["ok"]:
            yield "❌ " + info["text"]
            return None
        infos.append(info)
    try:
        dests = [dest_for(flag, tkey, repo, fp, slot_dir) for fp in files]
    except RuntimeError as e:
        yield f"❌ Cannot download: {e}."
        return None
    need = sum(i["size"] for d, i in zip(dests, infos) if not (d.is_file() and (not i["size"] or d.stat().st_size == i["size"])))
    part_have = sum(d.with_name(d.name + ".part").stat().st_size for d in dests if d.with_name(d.name + ".part").is_file())
    if need:
        free = shutil.disk_usage(_existing(dests[0].parent)).free
        if free < need - part_have + 512 * 1024 * 1024:
            yield f"❌ Not enough free space in `{_existing(dests[0].parent)}`: {gb(need - part_have)} needed, {gb(free)} free."
            return None
    stop.clear()
    total = sum(i["size"] for i in infos)
    yield (f"⏳ `{Path(path).name}` from `{repo}` ({gb(total)}" + (f", {parts} parts" if parts > 1 else "")
           + f") into `{dests[0].parent}` …")
    for k, (fp, d, info) in enumerate(zip(files, dests, infos), 1):
        if d.is_file() and (not info["size"] or d.stat().st_size == info["size"]):
            continue
        ok = yield from download(repo, fp, d, rev, token, info["size"], stop,
                                 label=d.name + (f" (part {k} of {parts})" if parts > 1 else ""))
        if not ok:
            return None
    if save:
        try:
            _save_slot(tkey, flag, dests[0])
        except Exception as e:                          # noqa: BLE001
            yield f"✔ Downloaded `{dests[0]}`, but saving the slot failed ({e}): pick the file and press Save."
            return dests[0]
    yield f"✔ `--{flag}` = `{dests[0]}` ({gb(total)}), saved for this LoRA type."
    return dests[0]


def _existing(p: Path) -> Path:
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


# --------------------------------------------------------------------------- what is missing
def current_files(tkey: str) -> dict:
    """{flag: path} as a new launch script would use it: saved in settings, else the preset default."""
    import sd_settings as SS
    saved = SS.read_settings().get(tkey, {})
    return {k: (saved.get(k) or d or "") for k, d in SS.type_slots(tkey)}


def is_missing(path: str) -> bool:
    p = (path or "").strip()
    return not p or "/PATH/TO" in p or not Path(os.path.expanduser(p)).is_file()


def missing(tkey: str, include_optional=False) -> list[str]:
    """Slots of this type whose file is not on this computer and that have a built-in download."""
    if not tkey:
        return []
    k = known(tkey)
    try:
        cur = current_files(tkey)
    except Exception:                                  # noqa: BLE001
        return []
    out = []
    for flag, p in cur.items():
        if flag not in k or not is_missing(p):
            continue
        if flag == "vae" and not p and not include_optional and _base_of(tkey) in ("sdxl", "illustrious", "sd15"):
            continue                                    # the optional external VAE of the SDXL family
        out.append(flag)
    return out


def fetch_missing(tkey: str, stop=None):
    """Generator: every missing slot with a built-in file, one after another."""
    stop = stop or STOP
    todo = missing(tkey)
    if not todo:
        yield "✔ Every model file of this LoRA type is on this computer."
        return []
    done, log = [], []
    for flag in todo:
        last = ""
        for line in fetch(tkey, flag, stop=stop):
            last = line
            yield "\n\n".join(log + [line])
        log.append(last)
        if last.startswith("✔"):
            done.append(flag)
        if stop.is_set():
            break
    yield "\n\n".join(log + [f"**{len(done)} of {len(todo)} files downloaded.**"])
    return done


# --------------------------------------------------------------------------- downloads in the background
BG = {"running": False, "line": "", "types": []}


def start_background(tkeys) -> str:
    """Download every missing model file of these LoRA types, one after another, while you go on working (Datasets ›
    2 Setup and 9 Train start it; the line in BG says how far it is)."""
    if BG["running"]:
        return "A download is already running."
    todo = [t for t in dict.fromkeys(tkeys or []) if missing(t)]
    if not todo:
        return "✔ Every model file of these LoRA types is on this computer."
    BG.update(running=True, line="⏳ starting …", types=todo)
    STOP.clear()

    def work():
        try:
            for t in todo:
                for line in fetch_missing(t):
                    BG["line"] = line.strip().splitlines()[-1] if line.strip() else BG["line"]
                if STOP.is_set():
                    BG["line"] = "Stopped: the next download goes on from where it stopped."
                    return
            BG["line"] = "✔ Downloads finished." if not any(missing(t) for t in todo) else BG["line"]
        except Exception as e:                           # noqa: BLE001
            BG["line"] = f"❌ Download failed: {e}"
        finally:
            BG["running"] = False
    threading.Thread(target=work, daemon=True, name="hf-background").start()
    return "⏳ Downloading in the background: you can go on with the next steps."

