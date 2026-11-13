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
__version__ = "1.011126"  # LoRA Studio version, see version.py

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lspaths as _LP  # noqa: E402
_LP.bootstrap()
APP_DIR = _LP.USER
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
    s = {"sd": _LP.ROOT / "sd-scripts", "mu": _LP.ROOT / "musubi-tuner", "datasets": HOME / "LoraStudio-data" / "datasets",
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
    base = expand(models_dir) if models_dir else _LP.ROOT / "models"
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

STEPS = ["Welcome", "Trainers", "Datasets", "Models", "ComfyUI", "Token", "Ollama", "Backups", "Finish"]


def run(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True, auth=None):
    """The setup page on its own (nothing else of the program is loaded); Finish starts LoRA Studio again."""
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
    from nicegui import app, ui
    import login
    import setup_page
    if auth:
        login.install(*auth)

    @ui.page("/")
    def index():
        if auth and not app.storage.user.get("ok"):
            return ui.navigate.to("/login")
        ui.add_css((_LP.PROG / "UI" / "theme.css").read_text(encoding="utf-8"))
        setup_page.build(hint)
    fav = _LP.PROG / "logo" / "favicon.png"
    ui.run(host=host, port=port, title="LoRA Studio setup", favicon=fav if fav.is_file() else None, show=open_browser,
           reload=False, storage_secret=login.storage_secret(), show_welcome_message=False)
