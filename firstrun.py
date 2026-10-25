# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""LoRA Studio setup page: shown in the browser on the first start (no settings yet), or with
./start.sh --setup.

Step by step, every step can be skipped (and changed later in Settings):
  0  Welcome: start the setup, or **import a backup** (Settings > Backup) and skip everything else
  1  Trainers: sd-scripts and musubi-tuner (install.sh puts them in the LoRA Studio folder)
  2  Datasets: where your datasets are / new ones are made
  3  Models: base models, text encoders, VAEs (+ the Z-Image Turbo training adapter, optional download)
  4  ComfyUI: its LoRA folder(s) and the folder it saves pictures in (Gallery tab)
  5  Hugging Face token (only for gated models)
  6  Ollama: for captions and the picture check (optional)
  7  Backup folder
  8  Summary -> Finish: the settings are written and LoRA Studio restarts into the normal program

Folders are suggested from what is found on this computer (a ComfyUI install, its extra_model_paths.yaml,
Ollama running, ...). Nothing is moved or deleted.
"""
from __future__ import annotations
__version__ = "1.251026g"  # LoRA Studio version, see version.py

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
HOME = Path.home()
DONE = APP_DIR / ".setup_done"
LIB_SETTINGS = APP_DIR / "library_settings.txt"
TRAIN_SETTINGS = APP_DIR / "training_settings.txt"
TOKEN_FILE = APP_DIR / "huggingtoken.txt"
ADAPTER_REPO = "ostris/zimage_turbo_training_adapter"
ADAPTER_FILE = "zimage_turbo_training_adapter_v2.safetensors"
TOKEN_HEADER = "# Put your Hugging Face token (hf_...) on its own line below. Lines starting with # are ignored.\n"
TRAIN_HEADER = ("# sdgui settings: model, VAE and text-encoder files for each LoRA type.\n"
                "# Written by sdgui.py (Settings tab); safe to edit by hand.\n")
SKIP = {"venv", ".venv", "node_modules", "__pycache__", "site-packages", ".git", ".cache", ".local", "snap",
        "proc", "sys", "dev", "tmp", ".Trash-1000"}


def needs_setup() -> bool:
    """First start: no library settings yet and the setup page never finished."""
    return not LIB_SETTINGS.exists() and not DONE.exists()


def tilde(p) -> str:
    p, home = str(p), str(HOME)
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p).strip().strip('"').strip("'"))))


def lines(text: str) -> list[str]:
    return [l.strip() for l in (text or "").replace(";", "\n").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Looking around this computer
# --------------------------------------------------------------------------- #

def _walk_dirs(root: Path, depth: int, budget: int = 6000):
    """Folders under root up to `depth` levels (breadth first, hidden / system folders skipped)."""
    level, n = [root], 0
    for _ in range(depth):
        nxt = []
        for d in level:
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False) and not e.name.startswith(".") and e.name not in SKIP:
                            nxt.append(Path(e.path))
                            n += 1
                            yield Path(e.path)
                            if n > budget:
                                return
            except OSError:
                continue
        level = nxt


def find_comfyui() -> list[Path]:
    roots = [(HOME, 4)] + [(Path(r), 3) for r in ("/mnt", "/media", f"/run/media/{os.environ.get('USER', '')}", "/opt")]
    out = []
    for root, depth in roots:
        if not root.is_dir():
            continue
        for d in _walk_dirs(root, depth):
            if d.name.lower() == "comfyui" and ((d / "main.py").exists() or (d / "models").is_dir()):
                if d not in out:
                    out.append(d)
    return out


def extra_model_paths(comfy: Path) -> list[dict]:
    """base_path + loras / checkpoints folders from ComfyUI's extra_model_paths.yaml (simple reader, no YAML lib)."""
    f = comfy / "extra_model_paths.yaml"
    if not f.is_file():
        return []
    out, cur = [], None
    for raw in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith((" ", "\t")) and raw.rstrip().endswith(":"):
            cur = {"name": raw.strip()[:-1], "base": "", "loras": [], "checkpoints": []}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+(\w+):\s*\|?\s*(.*)$", raw)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if key == "base_path":
                cur["base"] = val
            elif key in ("loras", "checkpoints") and val:
                cur[key].append(val)
            cur["_last"] = key
        elif cur.get("_last") in ("loras", "checkpoints") and raw.strip():
            cur[cur["_last"]].append(raw.strip())
    res = []
    for c in out:
        base = expand(c["base"]) if c["base"] else comfy
        if c["base"] and not base.is_absolute():
            base = (comfy / base).resolve()
        res.append({"base": base, "loras": [base / l for l in c["loras"]], "checkpoints": [base / x for x in c["checkpoints"]]})
    return res


def count_files(d: Path, exts=(".safetensors", ".ckpt", ".pt"), limit: int = 5000) -> int:
    n = 0
    try:
        for _root, _dirs, files in os.walk(d):
            n += sum(1 for f in files if f.lower().endswith(exts))
            if n >= limit:
                break
    except OSError:
        pass
    return n


def count_datasets(d: Path) -> int:
    try:
        return sum(1 for c in d.iterdir() if c.is_dir() and any(re.match(r"^\d+_", g.name) for g in c.iterdir() if g.is_dir()))
    except OSError:
        return 0


def ollama_models(url: str) -> tuple[list[str], str]:
    try:
        import captioner
        ms = captioner.list_models(url)
        vis = [m["name"] for m in ms if m.get("vision")]
        other = [m["name"] for m in ms if not m.get("vision")]
        return vis + other, f"✔ Ollama answers at `{url}`: {len(ms)} model(s), {len(vis)} that can see pictures"
    except Exception as e:
        return [], f"Ollama does not answer at `{url}` ({str(e).splitlines()[0][:120]})"


def suggestions() -> dict:
    """Starting values for the steps, from what is on this computer."""
    s = {"sd": APP_DIR / "sd-scripts", "mu": APP_DIR / "musubi-tuner", "datasets": HOME / "LoraStudio-data" / "datasets",
         "models": "", "loras": [], "pictures": "", "comfy": [], "ollama": "http://127.0.0.1:11434",
         "backup": HOME / "LoraStudio-backups"}
    for alt in (HOME / "sd-scripts",):
        if not (s["sd"] / "sdxl_train_network.py").exists() and (alt / "sdxl_train_network.py").exists():
            s["sd"] = alt
    if not (s["mu"] / "src" / "musubi_tuner").exists() and (HOME / "musubi-tuner" / "src" / "musubi_tuner").exists():
        s["mu"] = HOME / "musubi-tuner"
    comfy = find_comfyui()
    s["comfy"] = comfy
    for c in comfy:
        extra = extra_model_paths(c)
        for e in extra:
            for l in e["loras"]:
                if l.is_dir() and l not in s["loras"]:
                    s["loras"].append(l)
            if not s["models"] and any(x.is_dir() for x in e["checkpoints"]):
                s["models"] = e["base"]
        if (c / "models" / "loras").is_dir() and (c / "models" / "loras") not in s["loras"]:
            s["loras"].append(c / "models" / "loras")
        if not s["models"] and (c / "models").is_dir():
            s["models"] = c / "models"
        if not s["pictures"] and (c / "output").is_dir():
            s["pictures"] = c / "output"
    return s


# --------------------------------------------------------------------------- #
# Writing the settings
# --------------------------------------------------------------------------- #

def set_keys(path: Path, section: str, values: dict):
    """key = value inside [section], keeping every other line."""
    text = path.read_text(encoding="utf-8") if path.is_file() else TRAIN_HEADER
    ls = text.splitlines()
    start = next((i for i, l in enumerate(ls) if l.strip().lower() == f"[{section.lower()}]"), None)
    if start is None:
        ls += ["", f"[{section}]"]
        start = len(ls) - 1
    end = next((i for i in range(start + 1, len(ls)) if ls[i].strip().startswith("[")), len(ls))
    while end > start + 1 and not ls[end - 1].strip():
        end -= 1
    for k, v in values.items():
        hit = next((i for i in range(start + 1, end) if re.match(rf"\s*{re.escape(k)}\s*=", ls[i])), None)
        if hit is None:
            ls.insert(end, f"{k} = {v}")
            end += 1
        else:
            ls[hit] = f"{k} = {v}"
    path.write_text("\n".join(ls) + "\n", encoding="utf-8")


def write_all(v: dict, log: list) -> None:
    import lora_library as ll
    vals = ll.default_settings_values()
    if LIB_SETTINGS.exists():                          # --setup on an existing install: start from what is there
        cur = ll.load_settings()
        vals.update({k: cur[k] for k in cur if k in vals})
    vals.update({
        "sd_scripts_dir": str(expand(v["sd"])) if v["sd"] else vals["sd_scripts_dir"],
        "dataset_dirs": [str(expand(d)) for d in lines(v["datasets"])] or vals["dataset_dirs"],
        "lora_dirs": [str(expand(d)) for d in lines(v["loras"])],
        "models_dir": str(expand(v["models"])) if v["models"] else "",
        "pictures_dir": str(expand(v["pictures"])) if v["pictures"] else "",
        "backup_dir": str(expand(v["backup"])) if v["backup"] else str(HOME / "LoraStudio-backups"),
    })
    if v.get("make_datasets"):
        for d in vals["dataset_dirs"]:
            Path(d).mkdir(parents=True, exist_ok=True)
    LIB_SETTINGS.write_text(ll.render_settings(vals), encoding="utf-8")
    log.append(f"✔ Folders saved in `{LIB_SETTINGS.name}`")

    general = {}
    if v["sd"]:
        general["SDGUI_SD_SCRIPTS"] = tilde(expand(v["sd"]))
    if v["mu"]:
        general["SDGUI_MUSUBI"] = tilde(expand(v["mu"]))
    if v.get("ollama_url"):
        general["OLLAMA_URL"] = v["ollama_url"].strip()
    if v.get("ollama_model"):
        general["OLLAMA_MODEL"] = v["ollama_model"]
    if general or not TRAIN_SETTINGS.exists():
        set_keys(TRAIN_SETTINGS, "general", general)
        log.append(f"✔ Trainers{' and Ollama' if v.get('ollama_model') else ''} saved in `{TRAIN_SETTINGS.name}`")

    tok = (v.get("token") or "").strip()
    if tok:
        TOKEN_FILE.write_text(TOKEN_HEADER + tok + "\n", encoding="utf-8")
        os.chmod(TOKEN_FILE, 0o600)
        log.append("✔ Hugging Face token saved (readable only by you)")
    elif not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(TOKEN_HEADER, encoding="utf-8")
        os.chmod(TOKEN_FILE, 0o600)


def download_adapter(models_dir: str, log: list) -> None:
    base = expand(models_dir) if models_dir else APP_DIR / "models"
    dest = base / "loras" / "zimage"
    f = dest / ADAPTER_FILE
    if not f.exists():
        try:
            from huggingface_hub import hf_hub_download
            dest.mkdir(parents=True, exist_ok=True)
            hf_hub_download(ADAPTER_REPO, ADAPTER_FILE, local_dir=str(dest))
        except Exception as e:
            log.append(f"⚠ Z-Image adapter download failed ({type(e).__name__}): get it later from "
                       f"https://huggingface.co/{ADAPTER_REPO}")
            return
    set_keys(TRAIN_SETTINGS, "zimage", {"base_weights": str(f), "base_weights.dir": str(dest)})
    log.append(f"✔ Z-Image Turbo training adapter: `{tilde(f)}`")


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

CSS = """
#fr_wrap { max-width: 980px; margin: 0 auto; }
.fr-steps { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 14px; }
.fr-step { padding:3px 11px; border-radius:14px; font-size:13px; border:1px solid var(--border-color-primary); opacity:.65; }
.fr-step.cur { background:#a85a08; color:#fff; opacity:1; border-color:#e8a33d; }
.fr-step.done { opacity:.95; }
.fr-note { font-size:13px; opacity:.85; }
"""
STEPS = ["Welcome", "Trainers", "Datasets", "Models", "ComfyUI", "Token", "Ollama", "Backups", "Finish"]


def build(hint: dict):
    import gradio as gr
    import backup as BK

    def bar(i):
        return '<div class="fr-steps">' + "".join(
            f'<span class="fr-step{" cur" if k == i else " done" if k < i else ""}">{k}. {n}</span>' if k else
            f'<span class="fr-step{" cur" if k == i else " done" if k < i else ""}">{n}</span>'
            for k, n in enumerate(STEPS)) + "</div>"

    import inspect
    bk = {} if "css" in inspect.signature(gr.Blocks.launch).parameters else {"css": CSS}
    with gr.Blocks(title="LoRA Studio setup", **bk) as app:
        with gr.Column(elem_id="fr_wrap"):
            gr.Markdown("# LoRA Studio setup")
            steps_html = gr.HTML(bar(0))
            step = gr.State(0)
            cols = []

            # 0 welcome
            with gr.Column(visible=True) as c0:
                gr.Markdown("Welcome! A few questions so LoRA Studio knows where your things are. **Every step can "
                            "be skipped** (press *Skip*) and everything can be changed later in **Settings**.\n\n"
                            "Moving from another computer or install? **Import a backup** (made with Settings > "
                            "Backup) and skip all the steps.")
                with gr.Accordion("Import from a backup", open=bool(hint["backups"])):
                    bk_pick = gr.Dropdown(label="Backups found on this computer", choices=hint["backups"],
                                          value=hint["backups"][0][1] if hint["backups"] else None, interactive=True)
                    bk_path = gr.Textbox(label="Or the path of a backup .zip", placeholder="~/LoraStudio-backups/LoraStudio-backup-....zip")
                    bk_info = gr.Markdown(BK.describe(hint["backups"][0][1]) if hint["backups"] else "")
                    bk_go = gr.Button("Import this backup and start LoRA Studio", variant="primary")
                    bk_msg = gr.Markdown()
            cols.append(c0)

            # 1 trainers
            with gr.Column(visible=False) as c1:
                gr.Markdown("### 1. Trainers\nThe training programs. `install.sh` puts both in the LoRA Studio folder; "
                            "only change these if yours are somewhere else.")
                sd = gr.Textbox(label="sd-scripts folder (SDXL, Illustrious, Pony, Anima, Flux, ...)", value=tilde(hint["sd"]))
                mu = gr.Textbox(label="musubi-tuner folder (Z-Image, Qwen-Image, FLUX.2)", value=tilde(hint["mu"]))
                t_note = gr.Markdown()
            cols.append(c1)

            # 2 datasets
            with gr.Column(visible=False) as c2:
                gr.Markdown("### 2. Datasets\nWhere your datasets are: one folder per LoRA, with its pictures in a repeat "
                            "folder like `10_mychar`. New datasets are made in the first folder. Several folders: one per line.")
                ds = gr.Textbox(label="Dataset folder(s)", value="\n".join(tilde(x) for x in str(hint["datasets"]).splitlines()), lines=2)
                ds_make = gr.Checkbox(label="Create the folder if it does not exist", value=True)
                d_note = gr.Markdown()
            cols.append(c2)

            # 3 models
            with gr.Column(visible=False) as c3:
                gr.Markdown("### 3. Models\nThe folder with your base models, text encoders and VAEs (for example ComfyUI's "
                            "`models` folder, with `checkpoints/`, `vae/`, `text_encoders/` ...). The file lists in "
                            "Settings > Training start here; the files themselves are picked per LoRA type later.")
                md = gr.Textbox(label="Model folder", value=tilde(hint["models"]) if hint["models"] else "",
                                placeholder="e.g. ~/ComfyUI/models")
                md_adapter = gr.Checkbox(label=f"Download the Z-Image Turbo training adapter ({ADAPTER_FILE}, about 0.3 GB) "
                                               "into <model folder>/loras/zimage (only needed to train Z-Image Turbo)",
                                         value=False)
                m_note = gr.Markdown()
            cols.append(c3)

            # 4 comfyui
            with gr.Column(visible=False) as c4:
                found = ", ".join(f"`{tilde(c)}`" for c in hint["comfy"]) or "none found"
                gr.Markdown("### 4. ComfyUI\nYour finished LoRAs (with the LoRA Manager files) and the folder ComfyUI "
                            f"saves pictures in, for the **Gallery** tab. ComfyUI installs found: {found}.")
                lo = gr.Textbox(label="LoRA folder(s), one per line", value="\n".join(tilde(l) for l in hint["loras"]), lines=2)
                pics = gr.Textbox(label="ComfyUI picture folder (its output folder)",
                                  value=tilde(hint["pictures"]) if hint["pictures"] else "")
                c_note = gr.Markdown()
            cols.append(c4)

            # 5 token
            with gr.Column(visible=False) as c5:
                gr.Markdown("### 5. Hugging Face token\nOnly needed for gated models (for example FLUX.1-dev). Make one at "
                            "https://huggingface.co/settings/tokens (\"Read\" is enough). Saved in `huggingtoken.txt`, "
                            "readable only by you.")
                tok = gr.Textbox(label="Token (hf_...)", type="password")
                tok_test = gr.Button("Test the token", size="sm")
                k_note = gr.Markdown()
            cols.append(c5)

            # 6 ollama
            with gr.Column(visible=False) as c6:
                gr.Markdown("### 6. Ollama (optional)\nA local Ollama with a vision model (for example `qwen2.5vl:7b`) "
                            "writes captions and checks dataset pictures. Without it, everything else works.")
                with gr.Row(equal_height=True):
                    ol_url = gr.Textbox(label="Ollama address", value=hint["ollama"], scale=3)
                    ol_test = gr.Button("Look for models", scale=1)
                ol_model = gr.Dropdown(label="Vision model for captions", choices=hint["ollama_models"],
                                       value=hint["ollama_models"][0] if hint["ollama_models"] else None,
                                       allow_custom_value=True, interactive=True)
                o_note = gr.Markdown(hint["ollama_note"])
            cols.append(c6)

            # 7 backups
            with gr.Column(visible=False) as c7:
                gr.Markdown("### 7. Backups\nWhere **Settings > Backup** saves backups of your settings and LoRA Studio's "
                            "databases (not your datasets or models). Pick a folder that is copied off this computer, "
                            "for example a synced folder or a second drive.")
                bk_dir = gr.Textbox(label="Backup folder", value=tilde(hint["backup"]))
            cols.append(c7)

            # 8 finish
            with gr.Column(visible=False) as c8:
                gr.Markdown("### Summary")
                summary = gr.Markdown()
                fin = gr.Button("Finish and start LoRA Studio", variant="primary")
                fin_msg = gr.Markdown()
            cols.append(c8)

            with gr.Row():
                back = gr.Button("◀ Back", visible=False)
                skip = gr.Button("Skip this step", visible=False)
                nxt = gr.Button("Start the setup ▶", variant="primary")

        fields = [sd, mu, ds, ds_make, md, md_adapter, lo, pics, tok, ol_url, ol_model, bk_dir]

        # ---- notes per step
        def note_trainers(a, b):
            out = []
            for name, p, probe in (("sd-scripts", a, "sdxl_train_network.py"), ("musubi-tuner", b, "src/musubi_tuner")):
                d = expand(p) if p else None
                if not d:
                    out.append(f"- {name}: skipped (set it later in Settings > Training)")
                elif (d / probe).exists():
                    out.append(f"- {name}: ✔ found in `{tilde(d)}`")
                else:
                    out.append(f"- {name}: ✖ not found in `{tilde(d)}` (run `./install.sh`, or enter the right folder)")
            return "\n".join(out)

        def note_datasets(t, make):
            out = []
            for d in lines(t):
                p = expand(d)
                if p.is_dir():
                    n = count_datasets(p)
                    out.append(f"- `{tilde(p)}`: ✔ {n} dataset{'s' if n != 1 else ''} with repeat folders")
                else:
                    out.append(f"- `{tilde(p)}`: does not exist yet" + (" (it will be created)" if make else ""))
            return "\n".join(out) or "Skipped: `~/LoraStudio-data/datasets` is used."

        def note_models(t):
            if not t:
                return "Skipped: you pick the model files per LoRA type in Settings > Training."
            p = expand(t)
            if not p.is_dir():
                return f"✖ `{tilde(p)}` does not exist."
            subs = [n for n in ("checkpoints", "diffusion_models", "unet", "vae", "text_encoders", "clip", "loras") if (p / n).is_dir()]
            return f"✔ {count_files(p)} model files in `{tilde(p)}`" + (f" (folders: {', '.join(subs)})" if subs else "")

        def note_comfy(l, pc):
            out = []
            for d in lines(l):
                p = expand(d)
                out.append(f"- LoRAs `{tilde(p)}`: " + (f"✔ {count_files(p, ('.safetensors',))} LoRA files" if p.is_dir() else "✖ does not exist"))
            if pc:
                p = expand(pc)
                out.append(f"- Pictures `{tilde(p)}`: " + ("✔ found" if p.is_dir() else "✖ does not exist"))
            return "\n".join(out) or "Skipped: the Library and Gallery work with your datasets only; set these later in Settings > Folders."

        sd.change(note_trainers, [sd, mu], t_note)
        mu.change(note_trainers, [sd, mu], t_note)
        ds.change(note_datasets, [ds, ds_make], d_note)
        ds_make.change(note_datasets, [ds, ds_make], d_note)
        md.change(note_models, md, m_note)
        lo.change(note_comfy, [lo, pics], c_note)
        pics.change(note_comfy, [lo, pics], c_note)

        def test_token(t):
            t = (t or "").strip()
            if not t:
                return "No token entered."
            try:
                from huggingface_hub import whoami
                return f"✔ The token works: Hugging Face user **{whoami(token=t).get('name', '?')}**"
            except Exception as e:
                return f"✖ Could not check it ({type(e).__name__}); it is saved anyway when you continue."

        tok_test.click(test_token, tok, k_note)

        def look_ollama(url):
            ms, note = ollama_models(url.strip())
            return gr.update(choices=ms, value=ms[0] if ms else None), note

        ol_test.click(look_ollama, ol_url, [ol_model, o_note])

        def make_summary(*v):
            (a, b, d, dmk, m, ad, l, pc, t, ou, om, bd) = v
            row = lambda k, x: f"| {k} | {x} |"
            rows = ["| | |", "|---|---|",
                    row("sd-scripts", f"`{a}`" if a else "*skipped*"),
                    row("musubi-tuner", f"`{b}`" if b else "*skipped*"),
                    row("Datasets", ", ".join(f"`{x}`" for x in lines(d)) or "`~/LoraStudio-data/datasets`"),
                    row("Models", (f"`{m}`" if m else "*skipped*") + (" + Z-Image adapter download" if ad else "")),
                    row("LoRA folders", ", ".join(f"`{x}`" for x in lines(l)) or "*skipped*"),
                    row("ComfyUI pictures", f"`{pc}`" if pc else "*skipped*"),
                    row("Hugging Face token", "entered" if (t or "").strip() else "*skipped*"),
                    row("Ollama", f"`{om}` at `{ou}`" if om else "*skipped*"),
                    row("Backups", f"`{bd}`" if bd else "`~/LoraStudio-backups`")]
            return "\n".join(rows) + "\n\nEverything can be changed later in **Settings**."

        # ---- navigation
        def show(i, *v):
            i = max(0, min(len(cols) - 1, i))
            vis = [gr.update(visible=(k == i)) for k in range(len(cols))]
            label = "Start the setup ▶" if i == 0 else "Next ▶"
            extra = [make_summary(*v) if i == len(cols) - 1 else gr.update()]
            notes = [note_trainers(v[0], v[1]) if i == 1 else gr.update(),
                     note_datasets(v[2], v[3]) if i == 2 else gr.update(),
                     note_models(v[4]) if i == 3 else gr.update(),
                     note_comfy(v[6], v[7]) if i == 4 else gr.update()]
            return [i, bar(i)] + vis + [gr.update(visible=i > 0), gr.update(visible=0 < i < len(cols) - 1),
                                        gr.update(visible=i < len(cols) - 1, value=label)] + extra + notes

        nav_out = [step, steps_html] + cols + [back, skip, nxt, summary, t_note, d_note, m_note, c_note]
        clear_by_step = {1: [sd, mu], 2: [ds], 3: [md], 4: [lo, pics], 5: [tok], 6: [ol_model], 7: [bk_dir]}

        nxt.click(lambda i, *v: show(i + 1, *v), [step] + fields, nav_out)
        back.click(lambda i, *v: show(i - 1, *v), [step] + fields, nav_out)

        def do_skip(i, *v):
            """Skip = leave this step's answers empty, then go on."""
            v = list(v)
            upd = [gr.update() for _ in fields]
            for comp in clear_by_step.get(i, []):
                k = fields.index(comp)
                v[k] = None if comp is ol_model else ""
                upd[k] = gr.update(value=v[k])
            if i == 3:
                k = fields.index(md_adapter)
                v[k] = False
                upd[k] = gr.update(value=False)
            return show(i + 1, *v) + upd

        skip.click(do_skip, [step] + fields, nav_out + fields)

        # ---- import a backup
        bk_pick.change(lambda p: BK.describe(p) if p else "", bk_pick, bk_info)
        bk_path.change(lambda t: BK.describe(t.strip()) if t and t.strip() else "", bk_path, bk_info)

        def do_import(p, typed):
            src = (typed or "").strip() or p
            if not src:
                return "Pick a backup or enter the path of one."
            try:
                text = BK.restore(src)
            except Exception as e:
                return f"❌ {e}"
            BK.restart_program()
            return text + "\n\n**Starting LoRA Studio** with your restored settings; this page reloads by itself."

        # ---- finish
        def do_finish(*v):
            (a, b, d, dmk, m, ad, l, pc, t, ou, om, bd) = v
            log = []
            vals = {"sd": a, "mu": b, "datasets": d, "make_datasets": dmk, "models": m, "loras": l, "pictures": pc,
                    "token": t, "ollama_url": ou if om else "", "ollama_model": om, "backup": bd}
            try:
                write_all(vals, log)
                yield "  \n".join(log)
                if ad:
                    log.append("Downloading the Z-Image Turbo training adapter ...")
                    yield "  \n".join(log)
                    log.pop()
                    download_adapter(m, log)
                DONE.write_text(f"setup page finished {time.strftime('%Y-%m-%d %H:%M')}\n")
            except Exception as e:
                log.append(f"❌ {e!r}")
                yield "  \n".join(log)
                return
            log.append("\n**Starting LoRA Studio** ... this page reloads by itself in a few seconds.")
            BK.restart_program()
            yield "  \n".join(log)

        # after the server answered: wait for the restarted program, then reload the page
        RELOAD_JS = """() => { var mine = (window.gradio_config || {}).app_id, n = 0;
          var t = setInterval(function(){
            fetch('gradio_api/app_id', {cache:'no-store'}).then(r => r.ok ? r.json() : null).then(function(x){
              if (x && x.app_id !== undefined && x.app_id !== mine) { clearInterval(t); window.location.href = window.location.pathname; }
            }).catch(function(){});
            if (++n > 200) clearInterval(t);
          }, 1500); }"""
        bk_go.click(do_import, [bk_pick, bk_path], bk_msg).then(None, None, None, js=RELOAD_JS)
        fin.click(do_finish, fields, fin_msg).then(None, None, None, js=RELOAD_JS)
    return app


def run(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True, auth=None):
    import gradio as gr
    import backup as BK
    print("[LoRA Studio] setup page: open it in the browser "
          f"(http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port})", flush=True)
    print("[LoRA Studio] looking for ComfyUI, models and Ollama on this computer ...", flush=True)
    hint = suggestions()
    if LIB_SETTINGS.exists():                       # ./start.sh --setup on an existing install: start from its settings
        try:
            import lora_library as ll
            cur = ll.load_settings()
            hint.update({k: v for k, v in {
                "sd": cur.get("sd_scripts_dir"), "datasets": "\n".join(cur.get("dataset_dirs") or []),
                "models": cur.get("models_dir"), "pictures": cur.get("pictures_dir"),
                "backup": cur.get("backup_dir")}.items() if v})
            if cur.get("lora_dirs"):
                hint["loras"] = [Path(d) for d in cur["lora_dirs"]]
            g = {}
            for raw in TRAIN_SETTINGS.read_text(encoding="utf-8").splitlines() if TRAIN_SETTINGS.exists() else []:
                if "=" in raw and not raw.strip().startswith("#"):
                    k, _, val = raw.partition("=")
                    g.setdefault(k.strip(), val.strip())
            if g.get("SDGUI_MUSUBI"):
                hint["mu"] = expand(g["SDGUI_MUSUBI"])
            if g.get("OLLAMA_URL"):
                hint["ollama"] = g["OLLAMA_URL"]
        except Exception as e:                      # noqa: BLE001
            print(f"[LoRA Studio] could not read the current settings for the setup page: {e!r}")
    hint["backups"] = BK.find_backups()
    hint["ollama_models"], hint["ollama_note"] = ollama_models(hint["ollama"])
    if not hint["ollama_models"]:
        hint["ollama_note"] += ". Skip this step, or start Ollama and press *Look for models*."
    app = build(hint)
    kw = {"auth": auth} if auth else {}
    if auth:
        try:
            import lora_studio as _ls
            kw["auth_message"] = _ls.login_html()          # the same login page as the program
        except Exception:                                  # noqa: BLE001
            pass
    if "css" in __import__("inspect").signature(gr.Blocks.launch).parameters:
        kw["css"] = CSS
    app.launch(server_name=host, server_port=port, inbrowser=open_browser, **kw)
