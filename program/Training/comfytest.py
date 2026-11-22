# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Test pictures with ComfyUI: the same prompts with each save point of a run, side by side, to pick the best epoch.

Your workflows (Settings › Test pictures, exported from ComfyUI) are read for the settings you use: checkpoint or
diffusion model, text encoder, VAE, CLIP skip, sampler, scheduler, steps, CFG, size and negative prompt. LoRA Studio
then sends ComfyUI a small graph of ComfyUI's own nodes with those settings (loader, LoRA loader, prompt, sampler,
decode, save), so it works without the custom nodes of your workflow. Upscaling and face detailers of the workflow are
left out: the test pictures compare epochs, not finished pictures.

  prompts   8 captions of the dataset (picked once per run, kept in model/compare/<name>/prompts.json), each with the
            fixed tags you set per model family in front; the same seed per prompt for every epoch.
  epochs    the save point to try first and the ones next to it, plus the final file.
  LoRA      each save point is copied for a moment into <LoRA folder>/_lorastudio_test/ so ComfyUI can load it,
            and removed afterwards.
  pictures  <dataset>/model/compare/<name>/e<epoch>_p<n>.png
"""
from __future__ import annotations
__version__ = "1.221126"  # LoRA Studio version, see version.py

import contextlib
import json
import re
import random
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import lspaths as _LP  # noqa: E402  (program/lspaths.py: ROOT, PROG, USER)
APP_DIR = _LP.USER
FILE = APP_DIR / "comfy_settings.json"
WF_DIR = APP_DIR / "comfy_workflows"
TEST_DIR = "_lorastudio_test"
# Each model family's own advice for pictures (from the model cards; Training/workflows/ holds LoRA Studio's workflow
# with these settings, tools/make_workflows.py writes them): words in front of every test prompt and the negative
# prompt are in those files; POSITIVE is what Settings › Test pictures starts the "fixed tags" with.
POSITIVE = {
    "sdxl": "",
    "illustrious": "masterpiece, best quality",
    "pony": "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_anime",
    "noobai": "masterpiece, best quality, newest, absurdres, highres, safe",
    "vpred": "masterpiece, best quality, newest, absurdres, highres, safe",
    "zimage": "",
}
NOOB_POSITIVE = POSITIVE["noobai"]
BUILTIN = {"sdxl": "sdxl.json", "illustrious": "illustrious.json", "pony": "pony.json",
           "noobai": "noobai_eps.json", "vpred": "noobai_vpred.json", "zimage": ""}   # zimage: made in code
BUILTIN_NOTE = {
    "sdxl": "SDXL 1.0: no quality words, negative 'worst quality, low quality, blurry', DPM++ 2M Karras, CFG 7, "
            "28 steps",
    "illustrious": "Illustrious-XL: masterpiece, best quality in front, its own negative prompt, Euler a, CFG 6, 28 steps",
    "pony": "Pony Diffusion V6 XL: the score_ words in front, no negative prompt (the model card: not needed), "
            "Euler a, CFG 7, 25 steps, CLIP skip 2",
    "noobai": "NoobAI-XL EPS: its words in front, its negative prompt, Euler a, CFG 5.5, 28 steps",
    "vpred": "NoobAI-XL V-pred: its words in front, its negative prompt, Euler, CFG 5, 28 steps, v-prediction with "
             "zero terminal SNR",
    "zimage": "Z-Image Turbo: the Turbo model (int8 / fp8 when ComfyUI has one), Qwen3 4B, the FLUX AE, res_multistep, "
              "CFG 1, 8 steps, shift 3, no negative prompt (CFG 1 ignores it). A Z-Image Base LoRA is tested on Turbo too",
}
TYPE_OF = {"sdxl": "sdxl", "illustrious": "illustrious", "pony": "", "noobai": "noobai", "vpred": "noobai_vpred",
           "zimage": "zimage"}
FAMILIES = {"sdxl": "SDXL", "illustrious": "Illustrious", "pony": "Pony", "noobai": "NoobAI-XL EPS",
            "vpred": "NoobAI-XL V-pred", "anima": "Anima", "zimage": "Z-Image"}
FAMILY_OF = {"illustrious": "illustrious", "sdxl": "sdxl", "pony": "pony", "noobai_vpred": "vpred", "noobai": "noobai",
             "anima": "anima", "zimage": "zimage"}      # zimagebase starts with zimage: Base LoRAs are tested on Turbo
# a family without a workflow of its own uses this one's (before 1.171126d SDXL, Illustrious, Pony and NoobAI shared one)
FALLBACK = {"illustrious": "sdxl", "pony": "sdxl"}
# NoobAI has no fallback: its model and its advice differ too much; without a workflow of its own LoRA Studio's is set
# up the first time a NoobAI LoRA needs test pictures (workflow())
DEFAULTS = {"url": "http://127.0.0.1:8188", "lora_dir": "", "prompts": 8, "auto": True, "around": 2,
            "fixed": {**POSITIVE, "anima": ""}, "families": {}}
JOBS: dict = {}            # log name -> {"state", "text", "done", "total"}
WHICH: dict = {}           # log name -> which save points the next pictures use (pick_epochs `which`)
BATCH_FILE = APP_DIR / "last_batch.json"      # the runs of the last finished queue (Home › Just trained)
PENDING: list = []         # log names waiting for ComfyUI (made when it answers)
_LOCK = threading.Lock()


def sg():
    return sys.modules.get("sdgui")


# --------------------------------------------------------------------------- settings
def settings() -> dict:
    try:
        d = json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        d = {}
    out = json.loads(json.dumps(DEFAULTS))
    out.update({k: v for k, v in d.items() if k in DEFAULTS})
    out["fixed"] = {**DEFAULTS["fixed"], **(d.get("fixed") or {})}
    return out


def save_settings(**kw) -> dict:
    s = settings()
    for k, v in kw.items():
        if k in DEFAULTS:
            s[k] = v
    FILE.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def workflow(cfg: dict, fam: str | None) -> tuple[dict | None, str]:
    """(the workflow settings for a family, the family whose fixed tags go with them): its own, else the one it used
    to share (a Pony LoRA keeps using your SDXL workflow until Pony has one of its own)."""
    fams = cfg.get("families") or {}
    if fam and fams.get(fam):
        return fams[fam], fam
    if fam in ("noobai", "vpred", "zimage"):
        try:
            wf, _ = use_builtin(fam)
            return wf, fam
        except Exception:                                # noqa: BLE001
            return None, fam
    fb = FALLBACK.get(fam or "")
    if fb and fams.get(fb):
        return fams[fb], fb
    return None, fam or ""


# The LoRA strength of the test pictures (and the LoRA page's "How to use it") per LoRA type: a Turbo LoRA is trained
# with the training adapter, which is not in the finished file, so it comes through weaker (owner 2026-11-22: 1.0 was
# 40 % there, 1.3 about 90 %).
STRENGTH = {"zimage": 1.3}
STRENGTH_WHY = {"zimage": "a Z-Image Turbo LoRA comes through weaker at 1.0 (it was trained with the training adapter, "
                          "which is not in the finished file)."}


def lora_strength(tkey: str) -> float:
    return STRENGTH.get((tkey or "").lower(), 1.0)


def family(tkey: str) -> str | None:
    t = (tkey or "").lower()
    return FAMILY_OF.get(t) or next((f for k, f in FAMILY_OF.items() if t.startswith(k)), None)


# --------------------------------------------------------------------------- reading your workflow (UI format)
def _named(n: dict) -> dict:
    w = n.get("widgets_values_named")
    if isinstance(w, dict):
        return w
    return {}


def _source(nodes: dict, links: dict, node: dict, input_name: str):
    """The node and output name feeding `input_name` of `node` (follows Reroute nodes)."""
    for inp in node.get("inputs") or []:
        if inp.get("name") == input_name and inp.get("link") is not None:
            ln = links.get(inp["link"])
            if not ln:
                return None, None
            src = nodes.get(ln[1])
            seen = 0
            while src and src.get("type") == "Reroute" and seen < 10:
                rin = (src.get("inputs") or [{}])[0].get("link")
                ln = links.get(rin)
                src = nodes.get(ln[1]) if ln else None
                seen += 1
            if not src:
                return None, None
            outs = src.get("outputs") or []
            oname = outs[ln[2]]["name"] if ln and ln[2] < len(outs) else ""
            return src, oname
    return None, None


def _value(nodes, links, node, name):
    """A widget value of a node, also when it comes in through a link from a value node (selectors, literals)."""
    src, oname = _source(nodes, links, node, name)
    if src is not None:
        w = _named(src)
        for k in (oname, name, "sampler_name", "scheduler", "int", "cfg", "seed", "float", "text", "string"):
            if k in w and not isinstance(w[k], dict):
                return w[k]
        vals = [v for v in w.values() if not isinstance(v, (dict, list))]
        return vals[0] if vals else None
    return _named(node).get(name)


def _text_into(nodes, links, node, name, depth=0):
    """The prompt text that reaches `name` of `node` (through text nodes like 'easy positive')."""
    src, _ = _source(nodes, links, node, name)
    if src is None:
        return str(_named(node).get(name) or "")
    w = _named(src)
    for k in ("text", "positive", "negative", "text_g", "string", "prompt"):
        if isinstance(w.get(k), str) and w[k].strip():
            return w[k]
    if depth < 3:
        for k in ("text", "positive", "negative", "string"):
            t = _text_into(nodes, links, src, k, depth + 1)
            if t:
                return t
    return ""


def read_workflow(d: dict) -> dict:
    """The settings of a ComfyUI workflow (the normal 'Save' format, not API) that matter for test pictures."""
    nodes = {n["id"]: n for n in d.get("nodes") or []}
    links = {}
    for ln in d.get("links") or []:
        if isinstance(ln, list) and len(ln) >= 5:
            links[ln[0]] = ln                               # [id, from node, from slot, to node, to slot, type]
    out = {"kind": "", "ckpt": "", "unet": "", "clip": "", "clip_type": "stable_diffusion", "vae": "", "clip_skip": 0,
           "steps": 20, "cfg": 6.0, "sampler": "euler", "scheduler": "normal", "width": 1024, "height": 1024,
           "negative": "", "fixed": "", "vpred": False, "zsnr": False, "shift": 0.0,
           "unet_node": "UNETLoader", "clip_node": "CLIPLoader"}
    for n in nodes.values():
        t, w = n.get("type", ""), _named(n)
        if n.get("mode") in (2, 4):                         # muted / bypassed
            continue
        if "ckpt_name" in w and not out["ckpt"]:
            out["ckpt"] = w["ckpt_name"]
        if "unet_name" in w and not out["unet"]:              # UNETLoader, or a GGUF loader (ComfyUI-GGUF)
            out["unet"], out["unet_node"] = w["unet_name"], t or "UNETLoader"
        if t in ("CLIPLoader", "CLIPLoaderGGUF") and w.get("clip_name"):
            out["clip"], out["clip_type"], out["clip_node"] = w["clip_name"], w.get("type") or "stable_diffusion", t
        if t in ("ModelSamplingAuraFlow", "ModelSamplingSD3") and w.get("shift") not in (None, ""):
            out["shift"] = float(w["shift"])
        if t == "VAELoader" and w.get("vae_name"):
            out["vae"] = w["vae_name"]
        if t == "ModelSamplingDiscrete" and str(w.get("sampling") or "").startswith("v_pred"):
            out["vpred"], out["zsnr"] = True, bool(w.get("zsnr"))
        if t == "CLIPSetLastLayer":
            out["clip_skip"] = int(w.get("stop_at_clip_layer") or 0)
        if isinstance(w.get("resolution_ui"), dict):
            r = w["resolution_ui"]
            out["width"], out["height"] = int(r.get("w") or 1024), int(r.get("h") or 1024)
        elif t in ("EmptyLatentImage", "EmptySD3LatentImage") and w.get("width") and not any(
                isinstance(_named(x).get("resolution_ui"), dict) for x in nodes.values()):
            out["width"], out["height"] = int(w["width"]), int(w["height"])
    # the main sampler: the KSampler with full denoise (the others refine)
    ks = [n for n in nodes.values() if n.get("type") in ("KSampler", "KSamplerAdvanced") and n.get("mode") not in (2, 4)]
    main = next((n for n in ks if float(_named(n).get("denoise", 1) or 1) >= 0.99), ks[0] if ks else None)
    if main:
        for k, name in (("steps", "steps"), ("cfg", "cfg"), ("sampler", "sampler_name"), ("scheduler", "scheduler")):
            v = _value(nodes, links, main, name)
            if v not in (None, ""):
                out[k] = v
        out["negative"] = _text_into(nodes, links, *_cond_node(nodes, links, main, "negative"))
        pos = _text_into(nodes, links, *_cond_node(nodes, links, main, "positive"))
        first = pos.strip().split("\n\n")[0] if pos.strip() else ""
        if any(q in first.lower() for q in ("score_", "masterpiece", "best quality", "quality")):
            out["fixed"] = first.strip().rstrip(",")
    try:
        out["steps"], out["cfg"] = int(out["steps"]), float(out["cfg"])
    except (TypeError, ValueError):
        out["steps"], out["cfg"] = 20, 6.0
    out["kind"] = "unet" if out["unet"] and not out["ckpt"] else "ckpt"
    return out


def _cond_node(nodes, links, ks, name):
    """(text encode node, its text input name) feeding the positive / negative of a sampler."""
    src, _ = _source(nodes, links, ks, name)
    if src is None:
        return ks, name
    for k in ("text", "text_g"):
        if any(i.get("name") == k for i in src.get("inputs") or []) or k in _named(src):
            return src, k
    return src, "text"


def import_workflow(fam: str, path) -> tuple[dict, str]:
    """Read an exported workflow file for a model family, keep a copy and its settings. Returns (settings, note)."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if "nodes" not in d:
        raise ValueError("This is not a ComfyUI workflow in the normal format (use Workflow › Save or Export).")
    wf = read_workflow(d)
    WF_DIR.mkdir(exist_ok=True)
    shutil.copy2(p, WF_DIR / f"{fam}.json")
    s = settings()
    s["families"][fam] = wf
    if wf.get("fixed") and not s["fixed"].get(fam):
        s["fixed"][fam] = wf["fixed"]
    save_settings(families=s["families"], fixed=s["fixed"])
    model = wf["ckpt"] or wf["unet"] or "?"
    return wf, (f"✔ Read: model `{model}`, {wf['sampler']} / {wf['scheduler']}, {wf['steps']} steps, CFG {wf['cfg']}, "
                f"{wf['width']}×{wf['height']}" + (f", CLIP skip {abs(wf['clip_skip'])}" if wf["clip_skip"] else "")
                + (f", text encoder `{wf['clip']}`, VAE `{wf['vae']}`" if wf["kind"] == "unet" else "") + ".")


def _pick(names: list[str], must: tuple, prefer: tuple) -> str:
    """The first name containing all of must (letters and digits only), the preferred kinds first."""
    ok = [n for n in names if all(m in _key(n) for m in must)]
    for p in prefer:
        hit = next((n for n in ok if p in _key(n)), "")
        if hit:
            return hit
    return ok[0] if ok else ""


def zimage_builtin() -> tuple[dict, str]:
    """LoRA Studio's Z-Image Turbo test workflow, made from the files ComfyUI has: the Turbo model (int8 / fp8 first:
    the same pictures in half the memory), Qwen3 4B (fp8 first), the FLUX AE; res_multistep, simple, 8 steps, CFG 1,
    shift 3 (the Comfy-Org example workflow)."""
    url = settings()["url"]
    unets, clips, vaes = (model_names(url, k) for k in ("unet", "clip", "vae"))
    unet = _pick(unets, ("zimage", "turbo"), ("int8", "fp8", "bf16")) or "z_image_turbo_bf16.safetensors"
    clip = _pick(clips, ("qwen", "34b"), ("fp8", "")) or _pick(clips, ("qwen3", "4b"), ("fp8", "")) \
        or "qwen_3_4b.safetensors"
    vae = next((v for v in vaes if Path(v).name.lower() in ("ae.safetensors", "flux_ae.safetensors")), "") \
        or _pick(vaes, ("flux", "ae"), ("",)) or "ae.safetensors"
    wf = {"kind": "unet", "ckpt": "", "unet": unet, "unet_node": "UNETLoader", "clip": clip, "clip_node": "CLIPLoader",
          "clip_type": "lumina2", "vae": vae, "clip_skip": 0, "steps": 8, "cfg": 1.0, "sampler": "res_multistep",
          "scheduler": "simple", "width": 832, "height": 1216, "negative": "", "fixed": "", "vpred": False,
          "zsnr": False, "shift": 3.0}
    found = bool(unets)
    if found:                                            # kept only when made from ComfyUI's own files
        s = settings()
        s["families"]["zimage"] = wf
        s["fixed"]["zimage"] = ""
        save_settings(families=s["families"], fixed=s["fixed"])
    if not found:
        return wf, ("❌ ComfyUI did not list its models (is it running?). Start it and press the button again: the "
                    "workflow is made from the files your ComfyUI has.")
    return wf, f"✔ Z-Image Turbo workflow set: `{unet}`, `{clip}`, `{vae}`."


def use_builtin(fam: str) -> tuple[dict, str]:
    if fam == "zimage":
        return zimage_builtin()
    return _use_file(fam)


def _use_file(fam: str) -> tuple[dict, str]:
    """LoRA Studio's own NoobAI workflow for a family (NoobAI's advice: prompt words, negative prompt, sampler, CFG; V-pred
    with Model Sampling Discrete v_prediction + zsnr), with the checkpoint of the LoRA type from Settings › Training
    when one is set there (only its file name: ComfyUI knows its models by name)."""
    src = Path(__file__).with_name("workflows") / BUILTIN[fam]
    d = json.loads(src.read_text(encoding="utf-8"))
    try:
        import sd_settings as SS
        keys = [TYPE_OF[fam]] if fam != "pony" else [k for k in SS.read_settings() if "pony" in k.lower()]
        base = next((v for k in keys if (v := (SS.settings_models(k) or {}).get("pretrained_model_name_or_path"))), "")
    except Exception:                                    # noqa: BLE001
        base = ""
    for n in d["nodes"]:
        if n.get("type") == "CheckpointLoaderSimple":
            want = Path(base).name if base else n["widgets_values"][0]
            n["widgets_values"][0] = n["widgets_values_named"]["ckpt_name"] = comfy_name(want)
    tmp = WF_DIR / f"_{BUILTIN[fam]}"
    WF_DIR.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    try:
        wf, note = import_workflow(fam, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    s = settings()
    s["fixed"][fam] = POSITIVE[fam]
    save_settings(fixed=s["fixed"])
    return wf, note + ("" if base else " The checkpoint name is the model's usual file name: if yours is called "
                                       "differently, set it in Settings › Training first and press the button again.")


# --------------------------------------------------------------------------- talking to ComfyUI
def _req(url, path, data=None, timeout=30):
    full = url.rstrip("/") + path
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(full, data=body, headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
    return raw


def alive(url=None) -> bool:
    try:
        _req(url or settings()["url"], "/system_stats", timeout=3)
        return True
    except Exception:                                    # noqa: BLE001
        return False


def lora_names(url) -> list[str]:
    try:
        d = json.loads(_req(url, "/object_info/LoraLoader", timeout=20))
        return list(d["LoraLoader"]["input"]["required"]["lora_name"][0])
    except Exception:                                    # noqa: BLE001
        return []


LOADERS = {"ckpt": ("CheckpointLoaderSimple", "ckpt_name"), "unet": ("UNETLoader", "unet_name"),
           "vae": ("VAELoader", "vae_name"), "clip": ("CLIPLoader", "clip_name")}


def model_names(url, what: str = "ckpt", node: str = "") -> list[str]:
    """The file names ComfyUI offers for a loader (with their subfolders, e.g. Anima/furry/x.safetensors); node: the
    workflow's own loader (a GGUF one) instead of the usual."""
    node, inp = node or LOADERS[what][0], LOADERS[what][1]
    try:
        d = json.loads(_req(url, f"/object_info/{node}", timeout=20))
        return list(d[node]["input"]["required"][inp][0])
    except Exception:                                    # noqa: BLE001
        return []


def _key(name: str) -> str:
    """A file name without folder, extension, case and punctuation: novaFurryAM_v30 and NovaFurryAMV30 are one."""
    return re.sub(r"[^a-z0-9]", "", Path(name.replace("\\", "/")).stem.lower())


def match_name(want: str, names: list[str]) -> str:
    """ComfyUI's own name for a file: the same name, else the same file name in another subfolder, else the same name
    written differently (case, _ - and spaces). '' when ComfyUI has no such file."""
    if not want or not names or want in names:
        return want if (want in names or not names) else ""
    base = Path(want.replace("\\", "/")).name.lower()
    hit = [n for n in names if Path(n.replace("\\", "/")).name.lower() == base]
    if not hit:
        k = _key(want)
        hit = [n for n in names if _key(n) == k]
    return hit[0] if len(hit) == 1 else ""


def comfy_name(want: str, url=None) -> str:
    """ComfyUI's own name for a checkpoint file (see match_name); else as is."""
    return match_name(want, model_names(url or settings()["url"], "ckpt")) or want


def check_models(wf: dict, url: str) -> tuple[dict, list[str]]:
    """The workflow's model files under the names ComfyUI has now (a file renamed or moved to another subfolder since
    the workflow was read is found again). Returns (workflow, [what is missing])."""
    keys = ("unet", "clip", "vae") if wf.get("kind") == "unet" else ("ckpt", "vae")
    out, missing = dict(wf), []
    for k in keys:
        want = wf.get(k) or ""
        if not want:
            continue
        names = model_names(url, k, wf.get(f"{k}_node") or "")
        if not names:                                    # this ComfyUI does not list them: leave it to ComfyUI
            continue
        got = match_name(want, names)
        if got:
            out[k] = got
        else:
            missing.append(f"{LOADERS[k][1]} `{want}`")
    return out, missing


def graph(wf: dict, lora_name: str, positive: str, seed: int, prefix: str) -> dict:
    """The small API graph: ComfyUI's own nodes only."""
    g = {}
    if wf["kind"] == "unet":
        un, cn = wf.get("unet_node") or "UNETLoader", wf.get("clip_node") or "CLIPLoader"
        g["1"] = {"class_type": un, "inputs": {"unet_name": wf["unet"],
                                               **({"weight_dtype": "default"} if un == "UNETLoader" else {})}}
        g["2"] = {"class_type": cn, "inputs": {"clip_name": wf["clip"], "type": wf.get("clip_type") or
                                               "stable_diffusion"}}
        g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": wf["vae"]}}
        model, clip, vae = ["1", 0], ["2", 0], ["3", 0]
    else:
        g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": wf["ckpt"]}}
        model, clip, vae = ["1", 0], ["1", 1], ["1", 2]
        if wf.get("vae"):
            g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": wf["vae"]}}
            vae = ["3", 0]
    if wf.get("shift"):                                 # flow models (Z-Image, Anima …): the workflow's own shift
        g["1s"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": float(wf["shift"])}}
        model = ["1s", 0]
    if wf.get("vpred"):                                 # a v-prediction model (NoobAI-XL V-pred): tell the sampler
        g["2v"] = {"class_type": "ModelSamplingDiscrete", "inputs": {"model": model, "sampling": "v_prediction",
                                                                     "zsnr": bool(wf.get("zsnr", True))}}
        model = ["2v", 0]
    if wf.get("clip_skip"):
        g["4"] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": clip, "stop_at_clip_layer": int(wf["clip_skip"])}}
        clip = ["4", 0]
    g["5"] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": lora_name,
                                                     "strength_model": float(wf.get("lora_strength") or 1.0),
                                                     "strength_clip": float(wf.get("lora_strength") or 1.0)}}
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 1], "text": positive}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 1], "text": wf.get("negative") or ""}}
    g["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": int(wf["width"]), "height": int(wf["height"]),
                                                           "batch_size": 1}}
    g["9"] = {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0],
                                                   "latent_image": ["8", 0], "seed": int(seed), "steps": int(wf["steps"]),
                                                   "cfg": float(wf["cfg"]), "sampler_name": wf["sampler"],
                                                   "scheduler": wf["scheduler"], "denoise": 1.0}}
    g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae}}
    g["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}}
    return g


def _run_one(url, g, dest: Path, timeout=900):
    r = json.loads(_req(url, "/prompt", {"prompt": g, "client_id": "lorastudio"}))
    if r.get("node_errors"):
        raise RuntimeError("ComfyUI refused the graph: " + json.dumps(r["node_errors"])[:300])
    pid = r["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.loads(_req(url, f"/history/{pid}"))
        if pid in h:
            st = (h[pid].get("status") or {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages") or [] if m and m[0] == "execution_error"]
                raise RuntimeError("ComfyUI: " + (str(msgs[0][1].get("exception_message"))[:300] if msgs else "error"))
            for out in (h[pid].get("outputs") or {}).values():
                for im in out.get("images") or []:
                    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                "type": im.get("type", "output")})
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(_req(url, f"/view?{q}", timeout=120))
                    return dest
            if st.get("completed"):
                raise RuntimeError("ComfyUI finished without a picture")
        time.sleep(1.0)
    raise RuntimeError("ComfyUI took too long")


# --------------------------------------------------------------------------- one run
def _run_facts(logname: str):
    import training_ui as T
    f = sg().LOG_DIR / logname
    script = T.log_script(f)
    if not script or not Path(script).is_file():
        return None
    info = T.script_info(script)
    final, eps = T._final_and_epochs(info)
    a = T._advice(script)
    return {"script": script, "info": info, "final": final, "eps": eps, "best": a["best"] if a else None}


def pick_epochs(eps: dict, final, best, around: int = 2, which: str = "around") -> list[tuple[str, Path]]:
    """[(label, file)]. which = "around": the save point to try first and `around` on each side, plus the final file;
    "best+final": the save point to try first and the final file (one LoRA just trained); "final": the final file
    only (several LoRAs just trained: one picture set each, side by side)."""
    keys = sorted(eps)
    if which == "final":
        return [("final", final)] if final else []
    if which == "best+final":
        out = [(f"epoch {best}", eps[best])] if best in eps else []
        return out + ([("final", final)] if final else [])
    out = []
    if keys:
        b = best if best in eps else keys[len(keys) // 2]
        i = keys.index(b)
        for k in keys[max(0, i - around):i + around + 1]:
            out.append((f"epoch {k}", eps[k]))
    if final:
        out.append(("final", final))
    return out


def compare_dir(info) -> Path:
    return info["out_dir"] / "compare" / info["out_name"]


def prompts_for(info, n: int) -> list[dict]:
    """The prompts of a run (picked once from the captions, then kept): [{"text", "seed", "from"}]."""
    d = compare_dir(info)
    pf = d / "prompts.json"
    try:
        old = json.loads(pf.read_text(encoding="utf-8"))
        if len(old) >= n:
            return old[:n]
    except Exception:                                    # noqa: BLE001
        pass
    S = sg()
    caps = []
    imgs = []
    try:
        for row in S.dataset_stats(info["dataset"])["rows"]:      # the dataset's own repeat folder only
            imgs += S.list_images(row["folder"])
    except Exception:                                    # noqa: BLE001
        imgs = []
    for im in imgs:
        c = S.read_caption(im) if hasattr(S, "read_caption") else ""
        if c and c.strip():
            caps.append((im.name, c.strip()))
    rnd = random.Random(info["dataset"])                # the same prompts for every version of a dataset
    rnd.shuffle(caps)
    out = [{"text": c, "seed": rnd.randint(1, 2 ** 31 - 1), "from": name} for name, c in caps[:n]]
    if not out:
        out = [{"text": info["dataset"], "seed": rnd.randint(1, 2 ** 31 - 1), "from": ""}]
    d.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def results(info) -> dict:
    """{"labels": [..], "prompts": [..], "pics": {label: [path or None]}} of the test pictures made so far."""
    d = compare_dir(info)
    try:
        meta = json.loads((d / "grid.json").read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}
    for lab, files in meta.get("pics", {}).items():
        meta["pics"][lab] = [f if f and Path(f).is_file() else None for f in files]
    return meta


def _lora_root():
    cfg = settings()
    if cfg.get("lora_dir"):
        return Path(cfg["lora_dir"]).expanduser()
    dirs = [Path(p).expanduser() for p in sg().lora_dirs_setting()]
    return dirs[0] if dirs else None


def _stage(f: Path, url: str, root: Path) -> tuple[Path, str]:
    """Copy a LoRA file into <LoRA folder>/_lorastudio_test/ and return (the copy, the name ComfyUI knows it by)."""
    tdir = root / TEST_DIR
    tdir.mkdir(parents=True, exist_ok=True)
    tmp = tdir / f.name
    shutil.copy2(f, tmp)
    name = f"{TEST_DIR}/{f.name}"
    names = lora_names(url)
    if names and name not in names:
        alt = next((x for x in names if x.replace("\\", "/").endswith(name)), None)
        if not alt:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ComfyUI does not see `{tdir}`: set the LoRA folder ComfyUI reads in Settings › "
                               "Test pictures.")
        name = alt
    return tmp, name


@contextlib.contextmanager
def staged_lora(f: Path, url: str):
    """with staged_lora(file, url) as name: the LoRA is in ComfyUI's LoRA folder for the time of the block."""
    root = _lora_root()
    if root is None:
        raise RuntimeError("No LoRA folder is set (Settings › Folders, or Settings › Test pictures).")
    tmp, name = _stage(Path(f), url, root)
    try:
        yield name
    finally:
        tmp.unlink(missing_ok=True)
        try:
            if tmp.parent.is_dir() and not any(tmp.parent.iterdir()):
                tmp.parent.rmdir()
        except OSError:
            pass


def prepare(dataset: str, tkey: str, type_label: str = "") -> tuple:
    """(settings, workflow, url, LoRA folder, fixed words) for test pictures of a LoRA of this dataset and type; the
    look of the result (Datasets › 7) adds its words. RuntimeError with the reason when they cannot be made now."""
    S = sg()
    fam = family(tkey)
    cfg = settings()
    wf, ffam = workflow(cfg, fam)
    if not fam or not wf:
        raise RuntimeError(f"No workflow for {type_label or tkey} yet: add one in Settings › Test pictures." if fam else
                           f"Test pictures are not made for {type_label or tkey} yet (SDXL, Illustrious, Pony, Anima).")
    url = cfg["url"]
    if not alive(url):
        raise RuntimeError(f"ComfyUI does not answer at {url}. Start it; the test pictures are made when it answers.")
    wf = {**wf, "lora_strength": lora_strength(tkey)}
    fixed_wf, missing = check_models(wf, url)
    if missing:
        raise RuntimeError("ComfyUI does not have the model file(s) of your workflow any more: " + ", ".join(missing)
                           + ". Open the workflow in ComfyUI, pick the model again, save it and read it again in "
                           "Settings › Test pictures.")
    if fixed_wf != wf:                                  # renamed or moved: remember ComfyUI's name for next time
        wf = fixed_wf
        if (cfg.get("families") or {}).get(ffam):
            cfg["families"][ffam] = {**cfg["families"][ffam], **{k: wf[k] for k in LOADERS if k in wf}}
            save_settings(families=cfg["families"])
    roots = [Path(cfg["lora_dir"]).expanduser()] if cfg.get("lora_dir") else [Path(p).expanduser()
                                                                              for p in S.lora_dirs_setting()]
    if not roots:
        raise RuntimeError("No LoRA folder is set (Settings › Folders, or Settings › Test pictures).")
    fixed = (cfg["fixed"].get(ffam) or "").strip().rstrip(",")
    try:                                                # the look of the result (Datasets › 7): style words
        import lean
        l_pos, l_neg = lean.prompt_words(dataset, tkey)
    except Exception:                                    # noqa: BLE001
        l_pos, l_neg = "", ""
    if l_pos and "source_" in l_pos:                    # Pony: the lean's source word replaces the fixed one
        fixed = ", ".join(w for w in (x.strip() for x in fixed.split(",")) if w and not w.startswith("source_"))
    fixed = ", ".join(x for x in (fixed, l_pos) if x)
    if l_neg:
        wf = {**wf, "negative": ", ".join(x for x in (l_neg, wf.get("negative") or "") if x)}
    if fam == "vpred" and not wf.get("vpred"):          # a V-pred LoRA is always shown as v-prediction
        wf = {**wf, "vpred": True, "zsnr": True}
    return cfg, wf, url, roots[0], fixed


def make(logname: str, say=None) -> str:
    """Make the test pictures of one run. Returns a message (starting with ✔ or ❌)."""
    say = say or (lambda t: None)
    S = sg()
    if S.RUN.running():
        return "❌ A training run is using the GPU: the test pictures are made after the queue."
    facts = _run_facts(logname)
    if not facts:
        return "❌ The launch script of this run is gone."
    info = facts["info"]
    try:
        cfg, wf, url, root, fixed = prepare(info["dataset"], info["tkey"], info["type"])
    except RuntimeError as e:
        return f"❌ {e}"
    eps = pick_epochs(facts["eps"], facts["final"], facts["best"], int(cfg.get("around") or 2),
                      WHICH.pop(logname, "around"))
    if not eps:
        return "❌ This run has no LoRA files."
    prompts = prompts_for(info, int(cfg.get("prompts") or 8))
    d = compare_dir(info)
    grid = {"labels": [lab for lab, _ in eps], "files": {lab: str(f) for lab, f in eps},
            "prompts": prompts, "pics": {lab: [None] * len(prompts) for lab, _ in eps}, "at": time.strftime("%Y-%m-%d %H:%M")}
    total, done = len(eps) * len(prompts), 0
    JOBS[logname] = {"state": "running", "done": 0, "total": total, "text": ""}
    tdir = root / TEST_DIR
    try:
        for lab, f in eps:
            try:
                tmp, name = _stage(f, url, root)
            except RuntimeError as e:
                return f"❌ {e}"
            for i, pr in enumerate(prompts):
                text = (fixed + ", " if fixed else "") + pr["text"]
                dest = d / f"{lab.replace(' ', '')}_p{i + 1}.png"
                say(f"⏳ {lab}: picture {i + 1} of {len(prompts)} ({done + 1} of {total})")
                JOBS[logname].update(done=done, text=f"{lab}, picture {i + 1}")
                try:
                    _run_one(url, graph(wf, name, text, pr["seed"], f"lorastudio/{info['out_name']}"), dest)
                    grid["pics"][lab][i] = str(dest)
                except Exception as e:                   # noqa: BLE001
                    JOBS[logname]["text"] = str(e)
                    if i == 0:
                        tmp.unlink(missing_ok=True)
                        _save_grid(d, grid)
                        return f"❌ {e}"
                done += 1
                _save_grid(d, grid)
            tmp.unlink(missing_ok=True)
    finally:
        JOBS[logname]["state"] = "done"
        try:
            if tdir.is_dir() and not any(tdir.iterdir()):
                tdir.rmdir()
        except OSError:
            pass
    return f"✔ {done} test pictures made with {len(eps)} save points, in `{d}`."


def _save_grid(d: Path, grid: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "grid.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")


def make_in_background(logname: str) -> str:
    if JOBS.get(logname, {}).get("state") == "running":
        return "The test pictures of this run are being made."
    JOBS[logname] = {"state": "running", "done": 0, "total": 0, "text": "starting"}

    def work():
        msg = make(logname)
        JOBS[logname] = {"state": "done", "text": msg, "done": 0, "total": 0}
        if msg.startswith("❌ ComfyUI does not answer"):
            with _LOCK:
                if logname not in PENDING:
                    PENDING.append(logname)
    threading.Thread(target=work, daemon=True, name="comfy-test").start()
    return "Making the test pictures in ComfyUI …"


def share_prompts(lognames: list[str]) -> str:
    """Compare runs fairly: give every run the prompts (and seeds) of the first one, and make the test pictures of
    the runs that have none or had other prompts (one after the other, when ComfyUI answers). The pictures made
    with other prompts are kept as grid_before.json."""
    facts = [(ln, _run_facts(ln)) for ln in lognames]
    facts = [(ln, f) for ln, f in facts if f]
    if len(facts) < 2:
        return "❌ Pick at least two runs whose launch scripts still exist."
    n = int(settings().get("prompts") or 8)
    first = prompts_for(facts[0][1]["info"], n)
    todo = []
    for ln, f in facts:
        info = f["info"]
        d = compare_dir(info)
        cur = None
        try:
            cur = json.loads((d / "prompts.json").read_text(encoding="utf-8"))[:n]
        except Exception:                                # noqa: BLE001
            pass
        has = bool(results(info))
        if cur != first:
            d.mkdir(parents=True, exist_ok=True)
            (d / "prompts.json").write_text(json.dumps(first, indent=1), encoding="utf-8")
            if (d / "grid.json").is_file():
                (d / "grid.json").replace(d / "grid_before.json")
            todo.append(ln)
        elif not has:
            todo.append(ln)
    if not todo:
        return "✔ All these runs have test pictures with the same prompts."
    with _LOCK:
        for ln in todo:
            if ln not in PENDING and JOBS.get(ln, {}).get("state") != "running":
                PENDING.append(ln)
    _kick()
    return (f"⏳ {len(todo)} run(s) get test pictures with the same {len(first)} prompts"
            + ("" if alive() else " as soon as ComfyUI answers") + ".")


# --------------------------------------------------------------------------- after the queue
def free_memory(url=None):
    """Ask ComfyUI to unload its models, so the next training gets the whole GPU."""
    try:
        _req(url or settings()["url"], "/free", {"unload_models": True, "free_memory": True}, timeout=30)
    except Exception:                                    # noqa: BLE001
        pass


NOW: dict = {}      # what is being made right now (the Queue page and the status strip show it)


def now_text() -> str:
    """'Making test pictures for pipSDXLV3: picture 3 of 8' while _finish runs, else ''."""
    if not NOW:
        return ""
    return f"Making test pictures for {NOW.get('name', '')}" + (f": {NOW['text']}" if NOW.get("text") else " …")


def _finish(logname: str, say=None) -> str:
    """After a run: its test pictures (the save point to try first and the final file), ComfyUI's memory freed, and
    the LoRA card (LoRA Manager data, the first test picture as preview)."""
    import re
    name = re.sub(r"^\d{8}_\d{6}_|\.log$", "", logname)
    NOW.clear()
    NOW.update(log=logname, name=name, text="", done=0, total=0, t=time.time())

    def tell(t: str):
        m = re.search(r"\((\d+) of (\d+)\)", t)
        if m:
            NOW.update(done=int(m.group(1)), total=int(m.group(2)), text=f"picture {m.group(1)} of {m.group(2)}")
        if say:
            say(t)
    try:
        msg = make(logname, tell)
        NOW.update(text="freeing ComfyUI's memory")
        free_memory()
        NOW.update(text="writing the LoRA card")
        try:
            import lorapage
            msg += " " + lorapage.write_card_for_run(logname)
        except Exception as e:                           # noqa: BLE001
            msg += f" (LoRA card: {e})"
        return msg
    finally:
        NOW.clear()


def after_run(logname: str, say=None) -> str:
    """Called by the queue right after a run finished well, before the next one starts."""
    say = say or (lambda t: None)
    try:                                                # a run of a comparison: its own pictures come after the queue
        f = _run_facts(logname)
        if f and Path(f["script"]).parent.name == "_compare":
            return "A run of a comparison: its test pictures are made when all its runs are done."
    except Exception:                                    # noqa: BLE001
        pass
    cfg = settings()
    if not cfg.get("auto"):
        try:
            import lorapage
            return lorapage.write_card_for_run(logname)
        except Exception as e:                           # noqa: BLE001
            return f"LoRA card: {e}"
    WHICH[logname] = "best+final"
    if not alive(cfg["url"]):
        with _LOCK:
            if logname not in PENDING:
                PENDING.append(logname)
        _kick()
        return "ComfyUI does not answer: the test pictures are made when it does."
    JOBS[logname] = {"state": "running", "done": 0, "total": 0, "text": "starting"}
    msg = _finish(logname, say)
    JOBS[logname] = {"state": "done", "text": msg, "done": 0, "total": 0}
    return msg


def after_queue(lognames: list[str]):
    """Called when the training queue has finished: remember its runs for Home › Just trained (their test pictures
    were made after each run, see after_run), and make the pictures of comparisons whose runs are all done."""
    try:
        BATCH_FILE.write_text(json.dumps({"logs": list(lognames), "at": time.time()}), encoding="utf-8")
    except OSError:
        pass
    try:
        import compare
        compare.after_queue()
    except Exception as e:                               # noqa: BLE001
        print(f"[compare] {e}")


def _kick():
    def work():
        while True:
            with _LOCK:
                if not PENDING:
                    return
            S = sg()
            if S is not None and S.RUN.running():
                time.sleep(60)
                continue
            if not alive():
                time.sleep(60)                           # ComfyUI not started yet: try again in a minute
                continue
            with _LOCK:
                ln = PENDING.pop(0) if PENDING else None
            if ln:
                JOBS[ln] = {"state": "running", "done": 0, "total": 0, "text": "starting"}
                msg = _finish(ln)
                JOBS[ln] = {"state": "done", "text": msg, "done": 0, "total": 0}
    if not any(t.name == "comfy-pending" for t in threading.enumerate()):
        threading.Thread(target=work, daemon=True, name="comfy-pending").start()


def last_batch() -> dict:
    """{"logs": [...], "at": time} of the last finished queue, {} when none."""
    try:
        d = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
        return d if d.get("logs") else {}
    except (OSError, ValueError):
        return {}


def batch_view(logs: list[str]) -> list[dict]:
    """What Home › Just trained shows per run: [{"log", "dataset", "type", "version", "state", "labels", "pics"}]."""
    out = []
    for ln in logs:
        f = _run_facts(ln)
        if not f:
            continue
        info = f["info"]
        job = JOBS.get(ln) or {}
        state = ("making" if job.get("state") == "running" else "waiting" if ln in PENDING else
                 job.get("text", "") if job.get("text", "").startswith("❌") else "")
        g = results(info) or {}
        out.append({"log": ln, "dataset": info["dataset"], "type": info["type"], "version": info.get("version", ""),
                    "state": state, "labels": g.get("labels") or [], "pics": g.get("pics") or {},
                    "prompts": g.get("prompts") or [], "best": f["best"]})
    return out


# --------------------------------------------------------------------------- choosing one
