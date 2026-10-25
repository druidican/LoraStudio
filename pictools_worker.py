# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""LoRA Studio picture tools: runs inside the sd-scripts venv (it has PyTorch for your GPU), started by pictools.py.

One JSON command per line on stdin, one JSON answer per line on stdout. Models load on first use and stay loaded
until the worker is told to quit (LoRA Studio quits it before a training run and when it has been idle a while).

  ping                                  device and what is installed
  analyze   paths                       per picture: quality score (LAION aesthetic predictor on CLIP ViT-L/14),
                                        photo / drawing, CLIP embedding (for look-alikes), person boxes
  inpaint   src dst boxes               paint over boxes with LaMa (big-lama), the rest of the picture untouched
  upscale   src dst target_short kind   Real-ESRGAN (x2 for photos, anime 6B for drawings), then down to target

Model files are downloaded on first use into the models folder given as the first argument.
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import base64
import json
import os
import sys
import traceback
import urllib.request
from pathlib import Path

MODELS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "pictools_models"
URLS = {
    "aesthetic.pth": "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
    "big-lama.pt": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}
CLIP_ID = "openai/clip-vit-large-patch14"
KIND_TEXT = {"photo": ["a photo", "a photograph of a person", "a photo taken with a phone"],
             "drawing": ["a drawing", "an anime illustration", "a digital painting", "a cartoon", "a 3D render"]}
M: dict = {}


def say(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def fetch(name: str) -> Path:
    p = MODELS / name
    if p.is_file() and p.stat().st_size > 1000:
        return p
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with urllib.request.urlopen(URLS[name], timeout=600) as r, open(tmp, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
    tmp.replace(p)
    return p


def torch_dev():
    import torch
    if "dev" not in M:
        M["dev"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return M["dev"]


# --------------------------------------------------------------------------- analyze
def clip():
    if "clip" not in M:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = torch_dev()
        dt = torch.float16 if dev.type == "cuda" else torch.float32
        model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=dt).to(dev).eval()
        proc = CLIPProcessor.from_pretrained(CLIP_ID)
        with torch.no_grad():
            kinds = {}
            for k, texts in KIND_TEXT.items():
                t = proc(text=texts, return_tensors="pt", padding=True).to(dev)
                e = model.get_text_features(**t).float()
                e = e / e.norm(dim=-1, keepdim=True)
                kinds[k] = e.mean(0)
                kinds[k] = kinds[k] / kinds[k].norm()
        M["clip"] = (model, proc, dt, kinds)
    return M["clip"]


def aesthetic():
    if "aes" not in M:
        import torch
        from torch import nn
        layers = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                               nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
        sd = torch.load(fetch("aesthetic.pth"), map_location="cpu")
        sd = {k.replace("layers.", "", 1): v for k, v in sd.items()}
        layers.load_state_dict(sd)
        M["aes"] = layers.to(torch_dev()).eval()
    return M["aes"]


def detector():
    if "det" not in M:
        import torchvision
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.6)
        M["det"] = m.to(torch_dev()).eval()
    return M["det"]


def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def analyze(paths):
    import torch
    import torchvision.transforms.functional as TF
    model, proc, dt, kinds = clip()
    aes = aesthetic()
    det = detector()
    dev = torch_dev()
    out = {}
    for p in paths:
        try:
            im = load_rgb(p)
        except Exception as e:                           # noqa: BLE001
            out[p] = {"error": str(e)}
            continue
        with torch.no_grad():
            px = proc(images=im, return_tensors="pt")["pixel_values"].to(dev, dt)
            emb = model.get_image_features(pixel_values=px).float()[0]
            emb = emb / emb.norm()
            score = float(aes(emb[None])[0, 0])
            sims = {k: float(emb @ v) for k, v in kinds.items()}
            small = im.copy()
            small.thumbnail((1024, 1024))
            f = im.size[0] / small.size[0]
            r = det([TF.to_tensor(small).to(dev)])[0]
            persons = [[round(float(b[0]) * f), round(float(b[1]) * f), round(float(b[2]) * f), round(float(b[3]) * f),
                        round(float(s), 3)]
                       for b, l, s in zip(r["boxes"], r["labels"], r["scores"]) if int(l) == 1]
        e16 = emb.half().cpu().numpy().tobytes()
        out[p] = {"aes": round(score, 3), "kind": max(sims, key=sims.get), "persons": persons[:4],
                  "emb": base64.b64encode(e16).decode()}
    return out


# --------------------------------------------------------------------------- inpaint (LaMa)
def lama():
    if "lama" not in M:
        import torch
        M["lama"] = torch.jit.load(str(fetch("big-lama.pt")), map_location=torch_dev()).eval()
    return M["lama"]


def inpaint(src, dst, boxes, grow=0.15, mask_path=None):
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    im = load_rgb(src)
    W, H = im.size
    if mask_path:                                         # painted by hand: exactly that, a little wider
        mask = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST).point(lambda v: 255 if v > 20 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(9))
        bb = mask.getbbox()
        if not bb:
            return {"ok": False, "error": "nothing was painted"}
        region = list(bb)
    else:
        mask = Image.new("L", (W, H), 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(mask)
        region = [W, H, 0, 0]
        for b in boxes:
            x1, y1, x2, y2 = b[:4]
            gx, gy = max(6, (x2 - x1) * grow), max(6, (y2 - y1) * grow)
            bx = [max(0, x1 - gx), max(0, y1 - gy), min(W, x2 + gx), min(H, y2 + gy)]
            d.rectangle(bx, fill=255)
            region = [min(region[0], bx[0]), min(region[1], bx[1]), max(region[2], bx[2]), max(region[3], bx[3])]
    # work on the area around the marks (with enough context), not the whole picture
    rw, rh = region[2] - region[0], region[3] - region[1]
    ctx = max(96, int(max(rw, rh)))
    cx0, cy0 = int(max(0, region[0] - ctx)), int(max(0, region[1] - ctx))
    cx1, cy1 = int(min(W, region[2] + ctx)), int(min(H, region[3] + ctx))
    crop, mcrop = im.crop((cx0, cy0, cx1, cy1)), mask.crop((cx0, cy0, cx1, cy1))
    cw, ch = crop.size
    f = min(1.0, 1024 / max(cw, ch))
    sw, sh = max(8, int(cw * f)), max(8, int(ch * f))
    small, msmall = crop.resize((sw, sh), Image.LANCZOS), mcrop.resize((sw, sh), Image.NEAREST)
    pw, ph = (8 - sw % 8) % 8, (8 - sh % 8) % 8
    a = np.asarray(small).astype("float32") / 255.0
    m = (np.asarray(msmall) > 127).astype("float32")
    a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m = np.pad(m, ((0, ph), (0, pw)), mode="reflect")
    dev = torch_dev()
    with torch.no_grad():
        t = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)
        tm = torch.from_numpy(m)[None, None].to(dev)
        r = lama()(t, tm)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    r = Image.fromarray((r[:sh, :sw] * 255).astype("uint8")).resize((cw, ch), Image.LANCZOS)
    soft = mcrop.filter(ImageFilter.GaussianBlur(3))
    patched = Image.composite(r, crop, soft)
    out = im.copy()
    out.paste(patched, (cx0, cy0))
    save_like(out, src, dst)
    return {"ok": True, "area": round(float((np.asarray(mask) > 0).mean()), 4)}


def save_like(im, src, dst):
    from PIL import Image
    ext = Path(dst).suffix.lower()
    try:
        with Image.open(src) as s:
            icc = s.info.get("icc_profile")
    except Exception:                                    # noqa: BLE001
        icc = None
    kw = {"icc_profile": icc} if icc else {}
    if ext in (".jpg", ".jpeg"):
        im.save(dst, "JPEG", quality=95, subsampling=0, **kw)
    elif ext == ".webp":
        im.save(dst, "WEBP", quality=95, **kw)
    else:
        im.save(dst, **kw)


# --------------------------------------------------------------------------- upscale (Real-ESRGAN through spandrel)
def esrgan(kind):
    key = "esr_flat" if kind == "flat" else "esr_x2"
    if key not in M:
        from spandrel import ModelLoader
        # x2plus for photos and drawings alike: tested on semi-realistic illustrations it keeps skin, freckles and fine
        # lines, where the anime model flattens them. The anime model only for flat cel-shaded art ("flat").
        name = "RealESRGAN_x4plus_anime_6B.pth" if kind == "flat" else "RealESRGAN_x2plus.pth"
        d = ModelLoader().load_from_file(str(fetch(name)))
        M[key] = d.to(torch_dev()).eval()
    return M[key]


def tiled(m, a, tile=384, pad=24):
    """Run the upscaler in overlapping tiles, so a big picture does not run out of video memory."""
    import numpy as np
    import torch
    H, W = a.shape[:2]
    scale = int(getattr(m, "scale", 0) or 0)
    dev = torch_dev()
    out = None
    with torch.no_grad():
        for y in range(0, H, tile):
            for x in range(0, W, tile):
                y0, x0 = max(0, y - pad), max(0, x - pad)
                y1, x1 = min(H, y + tile + pad), min(W, x + tile + pad)
                t = torch.from_numpy(np.ascontiguousarray(a[y0:y1, x0:x1])).permute(2, 0, 1)[None].to(dev)
                r = m(t)[0].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
                if out is None:
                    scale = scale or round(r.shape[0] / (y1 - y0))
                    out = np.zeros((H * scale, W * scale, 3), dtype="float32")
                ty1, tx1 = min(H, y + tile), min(W, x + tile)
                out[y * scale:ty1 * scale, x * scale:tx1 * scale] = \
                    r[(y - y0) * scale:(ty1 - y0) * scale, (x - x0) * scale:(tx1 - x0) * scale]
    return out


def upscale(src, dst, target_short, kind="photo"):
    import numpy as np
    from PIL import Image
    im = load_rgb(src)
    m = esrgan(kind)
    a = np.asarray(im).astype("float32") / 255.0
    r = tiled(m, a)
    big = Image.fromarray((r * 255).round().astype("uint8"))
    f = target_short / min(big.size)
    if f != 1:                                            # down to the size asked for (or up, past what x2 gives)
        big = big.resize((max(1, round(big.size[0] * f)), max(1, round(big.size[1] * f))), Image.LANCZOS)
    save_like(big, src, dst)
    return {"ok": True, "size": list(big.size)}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            cmd = c.get("cmd")
            if cmd == "quit":
                say({"ok": True})
                return
            if cmd == "ping":
                import torch
                say({"ok": True, "device": str(torch_dev()), "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""})
            elif cmd == "analyze":
                say({"ok": True, "result": analyze(c["paths"])})
            elif cmd == "inpaint":
                say(inpaint(c["src"], c["dst"], c.get("boxes") or [], mask_path=c.get("mask")))
            elif cmd == "upscale":
                say(upscale(c["src"], c["dst"], int(c["target_short"]), c.get("kind", "photo")))
            else:
                say({"ok": False, "error": f"unknown command {cmd}"})
        except Exception as e:                           # noqa: BLE001
            say({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
