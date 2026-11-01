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
__version__ = "1.011126"  # LoRA Studio version, see version.py

import json
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
FAMILIES = {"sdxl": "SDXL / Illustrious / Pony", "anima": "Anima"}
FAMILY_OF = {"illustrious": "sdxl", "sdxl": "sdxl", "pony": "sdxl", "noobai": "sdxl", "anima": "anima"}
DEFAULTS = {"url": "http://127.0.0.1:8188", "lora_dir": "", "prompts": 8, "auto": True, "around": 2,
            "fixed": {"sdxl": "", "anima": ""}, "families": {}}
JOBS: dict = {}            # log name -> {"state", "text", "done", "total"}
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
           "negative": "", "fixed": ""}
    for n in nodes.values():
        t, w = n.get("type", ""), _named(n)
        if n.get("mode") in (2, 4):                         # muted / bypassed
            continue
        if "ckpt_name" in w and not out["ckpt"]:
            out["ckpt"] = w["ckpt_name"]
        if "unet_name" in w and not out["unet"]:
            out["unet"] = w["unet_name"]
        if t in ("CLIPLoader",) and w.get("clip_name"):
            out["clip"], out["clip_type"] = w["clip_name"], w.get("type") or "stable_diffusion"
        if t == "VAELoader" and w.get("vae_name"):
            out["vae"] = w["vae_name"]
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


def graph(wf: dict, lora_name: str, positive: str, seed: int, prefix: str) -> dict:
    """The small API graph: ComfyUI's own nodes only."""
    g = {}
    if wf["kind"] == "unet":
        g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": wf["unet"], "weight_dtype": "default"}}
        g["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": wf["clip"], "type": wf.get("clip_type") or
                                                         "stable_diffusion"}}
        g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": wf["vae"]}}
        model, clip, vae = ["1", 0], ["2", 0], ["3", 0]
    else:
        g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": wf["ckpt"]}}
        model, clip, vae = ["1", 0], ["1", 1], ["1", 2]
        if wf.get("vae"):
            g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": wf["vae"]}}
            vae = ["3", 0]
    if wf.get("clip_skip"):
        g["4"] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": clip, "stop_at_clip_layer": int(wf["clip_skip"])}}
        clip = ["4", 0]
    g["5"] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": lora_name,
                                                     "strength_model": 1.0, "strength_clip": 1.0}}
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


def pick_epochs(eps: dict, final, best, around: int = 2) -> list[tuple[str, Path]]:
    """[(label, file)]: the save point to try first and `around` on each side, plus the final file."""
    keys = sorted(eps)
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
    fam = family(info["tkey"])
    cfg = settings()
    wf = cfg["families"].get(fam or "")
    if not fam or not wf:
        return (f"❌ No workflow for {info['type']} yet: add one in Settings › Test pictures."
                if fam else f"❌ Test pictures are not made for {info['type']} yet (SDXL, Illustrious, Pony, Anima).")
    url = cfg["url"]
    if not alive(url):
        return f"❌ ComfyUI does not answer at {url}. Start it; the test pictures are made when it answers."
    roots = [Path(cfg["lora_dir"]).expanduser()] if cfg.get("lora_dir") else [Path(p).expanduser()
                                                                              for p in S.lora_dirs_setting()]
    if not roots:
        return "❌ No LoRA folder is set (Settings › Folders, or Settings › Test pictures)."
    root = roots[0]
    eps = pick_epochs(facts["eps"], facts["final"], facts["best"], int(cfg.get("around") or 2))
    if not eps:
        return "❌ This run has no LoRA files."
    prompts = prompts_for(info, int(cfg.get("prompts") or 8))
    fixed = (cfg["fixed"].get(fam) or "").strip().rstrip(",")
    d = compare_dir(info)
    grid = {"labels": [lab for lab, _ in eps], "files": {lab: str(f) for lab, f in eps},
            "prompts": prompts, "pics": {lab: [None] * len(prompts) for lab, _ in eps}, "at": time.strftime("%Y-%m-%d %H:%M")}
    total, done = len(eps) * len(prompts), 0
    JOBS[logname] = {"state": "running", "done": 0, "total": total, "text": ""}
    tdir = root / TEST_DIR
    try:
        tdir.mkdir(parents=True, exist_ok=True)
        for lab, f in eps:
            tmp = tdir / f.name
            shutil.copy2(f, tmp)
            name = f"{TEST_DIR}/{f.name}"
            names = lora_names(url)
            if names and name not in names:
                alt = next((x for x in names if x.replace("\\", "/").endswith(name)), None)
                if not alt:
                    tmp.unlink(missing_ok=True)
                    return (f"❌ ComfyUI does not see `{tdir}`: set the LoRA folder ComfyUI reads in Settings › "
                            "Test pictures.")
                name = alt
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
def after_queue(lognames: list[str]):
    """Called when the training queue has finished: test pictures for its successful runs (when switched on)."""
    if not settings().get("auto"):
        return
    with _LOCK:
        for ln in lognames:
            if ln not in PENDING:
                PENDING.append(ln)
    _kick()


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
                msg = make(ln)
                JOBS[ln] = {"state": "done", "text": msg, "done": 0, "total": 0}
    if not any(t.name == "comfy-pending" for t in threading.enumerate()):
        threading.Thread(target=work, daemon=True, name="comfy-pending").start()


# --------------------------------------------------------------------------- choosing one
def use(logname: str, label: str, dest_dir: str, state: dict, overwrite: bool = False) -> str:
    """Copy the save point `label` to the LoRA folder (with its LoRA Manager data) and use its first test picture as
    the LoRA Manager preview."""
    import training_ui as T
    facts = _run_facts(logname)
    if not facts:
        return "❌ The launch script of this run is gone."
    grid = results(facts["info"])
    f = Path((grid.get("files") or {}).get(label) or "")
    if not f.is_file():
        return "❌ That save point file is gone."
    msg = T.copy_file(state, f.name, dest_dir, overwrite)
    if not msg.startswith("✔"):
        return msg
    copied_name = f.name
    m = __import__("re").search(r"-(\d{6})$", f.stem)
    if m and f.stem[:-7] == state.get("out"):
        copied_name = f"{state['out']}-e{int(m.group(1))}.safetensors"
    dest = Path(dest_dir).expanduser() / copied_name
    pics = [p for p in (grid.get("pics") or {}).get(label) or [] if p]
    if pics:
        prev = dest.with_name(dest.stem + ".preview.png")
        shutil.copy2(pics[0], prev)
        sc = dest.with_name(dest.stem + ".metadata.json")
        try:
            data = json.loads(sc.read_text(encoding="utf-8"))
            if not data.get("preview_url"):
                data["preview_url"] = str(prev)
                import lora_library as ll
                ll.write_sidecar(sc, data)
        except Exception:                                # noqa: BLE001
            pass
        msg += f" · preview `{prev.name}`"
    return msg
