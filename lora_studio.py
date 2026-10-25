#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""
LoRA Studio
===========
One window for your LoRA collection, your datasets and training:

  Collection  - Browse (Pictures or Status), the LoRA page (Overview, Pictures, Training, Edit), Maintenance
  Datasets    - New dataset, Edit existing, Improvements; nine steps from a folder of pictures to the launch script
  Training    - Run (launch scripts, the queue, the live run) and Results (every finished run)
  Settings    - a grouped list that opens on a Health check

A status strip on every page shows the training run and the machine (GPU, VRAM, CPU, RAM).

This program's folder (LoraStudio, put it anywhere) holds all of its own files: library_settings.txt,
training_settings.txt, launch_settings.json, huggingtoken.txt, caches and the venv. sd-scripts, your datasets and
your LoRAs can each be anywhere else; link them in Settings.

Run:  ./start.sh            (or: python lora_studio.py)   -> http://127.0.0.1:7860
Options typed on the command line (--port, --host, --auth, --no-browser) win over Settings > Launch options.
"""

from __future__ import annotations
__version__ = "1.251026n"  # LoRA Studio version, see version.py

import argparse
import inspect
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gradio as gr  # noqa: E402
import version  # noqa: E402


def login_html() -> str:
    """The login page: the painted logo as a card behind the login fields (not the whole screen). Gradio shows this
    as HTML above the fields and loads no other style there, so the look comes along inline."""
    import base64
    f = HERE / "logo" / "lorastudio_login.jpg"
    if not f.is_file():
        return "LoRA Studio"
    url = "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()
    return (f'<span class="ls-login">LoRA Studio</span><style>{LOGIN_CSS.replace("@IMG@", url)}</style>')


LOGIN_CSS = """
html:has(.ls-login), body:has(.ls-login), body:has(.ls-login) gradio-app, body:has(.ls-login) .gradio-container,
body:has(.ls-login) .main, body:has(.ls-login) .contain {
  background:#1b2130 !important; border:none !important; box-shadow:none !important; }
.wrap:has(.ls-login) { --lw:min(780px, calc(100vw - 32px)); display:flex; align-items:center; justify-content:center;
  min-height:100vh; box-sizing:border-box; padding:16px; border:none !important;
  background:#1b2130 url(@IMG@) center / var(--lw) auto no-repeat !important; }
.wrap:has(.ls-login) > * { flex:none !important; min-width:0 !important; width:min(300px, calc(var(--lw) * .42)) !important; box-sizing:border-box; gap:6px !important;
  margin-bottom:calc(var(--lw) * .2); padding:18px 22px 20px !important; border-radius:14px; border:none !important;
  background:rgba(22,28,42,.74) !important; backdrop-filter:blur(3px); box-shadow:0 8px 28px rgba(0,0,0,.4); }
.wrap:has(.ls-login) h2 { color:#f6efe2 !important; margin:0 !important; font-size:19px !important; text-align:center; }
.ls-login { display:block; text-align:center; color:#e8c98f; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; font-size:11px; }
.wrap:has(.ls-login) p.auth { margin:0 !important; }
.wrap:has(.ls-login) p.creds { color:#ffb4a2 !important; margin:4px 0 0 !important; }
.wrap:has(.ls-login) label span, .wrap:has(.ls-login) label { color:#f3ead9 !important; }
.wrap:has(.ls-login) input { background:rgba(255,250,240,.95) !important; color:#1d2433 !important; }
.wrap:has(.ls-login) .block { padding:4px 0 !important; }
.wrap:has(.ls-login) .block, .wrap:has(.ls-login) .form { background:transparent !important; border:none !important;
  box-shadow:none !important; gap:0 !important; }
.wrap:has(.ls-login) button { background:#b5462f !important; color:#fff !important; border:none !important;
  margin-top:6px; }
.wrap:has(.ls-login) button:hover { background:#c9553b !important; }
@media (max-width: 600px) {
  .wrap:has(.ls-login) { background-size:auto min(560px, 92vh) !important; }
  .wrap:has(.ls-login) > * { width:calc(100vw - 64px) !important; margin-bottom:0; }
}
"""


def logo_img(px: int = 40) -> str:
    """The logo as a small inline picture (logo/lorastudio_logo_80.png, the middle of the painted logo)."""
    import base64
    f = HERE / "logo" / ("lorastudio_logo_80.png" if px <= 40 else "lorastudio_logo_128.png")
    if not f.is_file():
        return ""
    return (f'<img class="ls-logo" alt="" width="{px}" height="{px}" '
            f'src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}">')


def _setup_page_first():
    """First start (no settings yet) or --setup: show the setup page instead of the program. It runs BEFORE
    lora_library / sdgui are imported, because those create default settings files when there are none.
    When the setup page finishes, it starts this program again (same port) with the new settings."""
    import firstrun
    if "--setup" not in sys.argv and not firstrun.needs_setup():
        return
    import launchopts
    lo = launchopts.load()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--port", type=int, default=lo["port"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--auth")
    ap.add_argument("--no-browser", action="store_true")
    a, _ = ap.parse_known_args()
    auth = a.auth or os.environ.get("SDGUI_AUTH")
    auth = tuple(auth.split(":", 1)) if auth and ":" in auth else None
    firstrun.run(a.host, a.port, not a.no_browser, auth)
    sys.exit(0)


if __name__ == "__main__":
    _setup_page_first()

import lora_library as ll  # noqa: E402
import sdgui  # noqa: E402
import ticker  # noqa: E402


def apply_folders(s: dict):
    """Point sdgui at the folders from library_settings.txt (also called after saving Settings)."""
    sdgui.ROOT = Path(os.path.expanduser(s["sd_scripts_dir"])).absolute()
    datasets = [Path(os.path.expanduser(d)).absolute() for d in s["dataset_dirs"]]
    sdgui.IMAGES_DIR = datasets[0] if datasets else None
    extra = datasets[1:] + [Path(os.path.expanduser(d)).absolute() for d in s["script_dirs"]]
    sdgui.EXTRA_SCRIPT_DIRS[:] = extra
    sdgui.DEFAULT_MODEL_DIR = os.path.expanduser(s.get("models_dir") or "")


TRAINING_SETTINGS = "training_settings.txt"


def use_own_config_folder():
    """Keep the training settings and the Hugging Face token in this program's folder (next to
    library_settings.txt) instead of inside sd-scripts. Existing files are imported by setup.py
    (first start, or ./start.sh --setup), never copied silently."""
    sdgui.CONFIG_DIR = HERE
    sdgui.SETTINGS_FILE = TRAINING_SETTINGS


STUDIO_JS = """
<script>
window.lsSend = function(id, v){
  var el = document.querySelector('#' + id + ' textarea, #' + id + ' input'); if (!el) return;
  var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
  setter.call(el, v + '|' + Date.now());
  el.dispatchEvent(new Event('input', {bubbles:true}));
};
%s
window.lsClickTab = function(label){
  var b = Array.from(document.querySelectorAll('button[role="tab"]')).find(function(x){ return x.textContent.trim() === label; });
  if (b) { b.click(); return true; } return false;
};
window.lsGoQueue = function(){
  window.lsClickTab('Training');
  var n = 0; (function t(){ if (!window.lsClickTab('Run') && ++n < 30) setTimeout(t, 100); })();
};
(function(){
  if (!%s || !window.Notification) return;
  var last = null;
  function ask(){ if (Notification.permission === 'default') Notification.requestPermission(); }
  document.addEventListener('click', function once(){ ask(); document.removeEventListener('click', once); });
  setInterval(function(){
    var el = document.querySelector('.ls-strip'); if (!el) return;
    var id = el.dataset.done || '';
    if (last === null) { last = id; return; }
    if (id && id !== last) {
      last = id;
      if (Notification.permission === 'granted') new Notification('LoRA Studio', {body: el.dataset.doneText || 'A run finished'});
    }
  }, 3000);
})();
</script>
"""


if os.environ.get("LS_NO_TIMERS"):           # diagnostics: build the page without its timers
    _T = gr.Timer
    gr.Timer = lambda *a, **k: _T(*a, **{**k, "active": False})


def _profile(app):
    LIM = float(os.environ.get("LS_PROFILE_MIN", "0.2"))
    import functools
    import inspect
    import time as _t

    def wrap(fn, name):
        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def g(*a, **k):
                t0 = _t.time()
                yield from fn(*a, **k)
                d = _t.time() - t0
                if d >= LIM:
                    print(f"[Profile] {_t.strftime('%H:%M:%S')} {d:6.2f} s  {name}", flush=True)
            return g

        @functools.wraps(fn)
        def f(*a, **k):
            t0 = _t.time()
            try:
                return fn(*a, **k)
            finally:
                d = _t.time() - t0
                if d >= LIM:
                    print(f"[Profile] {_t.strftime('%H:%M:%S')} {d:6.2f} s  {name}", flush=True)
        return f
    fns = app.fns.values() if isinstance(app.fns, dict) else app.fns
    for bf in fns:
        if bf.fn is not None:
            nm = getattr(bf.fn, "__qualname__", "?")
            mod = getattr(bf.fn, "__module__", "?")
            bf.fn = wrap(bf.fn, f"{mod}.{nm}")


def main():
    import launchopts
    import statusbar
    lo = launchopts.load()
    ap = argparse.ArgumentParser(description="LoRA Studio: your LoRA collection, datasets and training",
                                 epilog="Defaults come from Settings > Launch options (launch_settings.json).")
    ap.add_argument("--port", type=int, default=lo["port"])
    ap.add_argument("--host", default="0.0.0.0" if lo["network"] else "127.0.0.1",
                    help="127.0.0.1 = this computer only; 0.0.0.0 = whole network")
    ap.add_argument("--auth", help="login as user:password (recommended with --host 0.0.0.0); or set SDGUI_AUTH")
    ap.add_argument("--venv", help="venv to pre-fill for the queue (default: <sd-scripts>/venv)")
    ap.add_argument("--datatypes", "--scan-script", dest="scan_script",
                    help="path to Lora_Datatypes.py, the file with all LoRA types (default: next to this file, in sd-scripts, in ~)")
    ap.add_argument("--allow-path", action="append", default=[],
                    help="extra folder the galleries may show images from (repeatable)")
    ap.add_argument("--no-browser", action="store_true", default=not lo["browser"], help="don't open a browser tab")
    ap.add_argument("--setup", action="store_true", help="show the setup page again (folders, token, backup import)")
    args = ap.parse_args()

    s = ll.LIB.settings
    apply_folders(s)
    ll.ON_SETTINGS_SAVED.append(apply_folders)
    use_own_config_folder()
    sdgui.DEFAULT_VENV = os.path.expanduser(args.venv) if args.venv else sdgui.detect_venv()
    sdgui.SCAN_SCRIPT = sdgui.find_scan_script(args.scan_script)
    sdgui.apply_gfx_env()
    sdgui.BIND_HOST = args.host

    def sd_dir_saved(path):                 # Settings > Trainers + GPU is the one place for the sd-scripts folder
        v = ll.load_settings()
        if os.path.expanduser(v.get("sd_scripts_dir") or "") != os.path.expanduser(path):
            v["sd_scripts_dir"] = path
            ll._write_values(v)
            ll.LIB.settings["sd_scripts_dir"] = path
    sdgui.ON_SD_DIR.append(sd_dir_saved)
    sample_dir = s["sample_dir"] or str(sdgui.ROOT / "output")

    theme = ll.make_theme()
    import settings_ui
    import studio_css
    import datasets_ui
    import manual
    head = ll.head_html(s["style"]) + STUDIO_JS % (settings_ui.NAV_JS + datasets_ui.GRID_JS + manual.JS,
                                                   "true" if lo["notify"] else "false")
    css = sdgui.CSS + "\n" + ll.CSS + "\n" + studio_css.CSS
    style_in_launch = "css" in inspect.signature(gr.Blocks.launch).parameters
    blocks_kwargs = {"title": f"LoRA Studio {version.VERSION}"}
    if not style_in_launch:
        blocks_kwargs.update(css=css, head=head, theme=theme)
    start = lo["start_page"] if lo["start_page"] in {v for _, v in launchopts.START_PAGES} else "collection"

    ticker.ACTIVE["on"] = True                 # the parts register their refreshes; one timer does them all
    with gr.Blocks(**blocks_kwargs) as app:
        import notice                       # the very first page, confirmed once: who owns sd-scripts / musubi-tuner
        notice.build_gate(app, notice.logo_data_url())
        with gr.Row(elem_id="ls_top"):
            gr.HTML(f'<div class="ls-title">{logo_img(40)}<h1>LoRA Studio</h1><span class="ls-ver">{version.VERSION}</span></div>')
            gr.Button("Dark / Normal", size="sm", scale=0).click(None, None, None, js=ll.TOGGLE_JS)
        strip = gr.HTML(statusbar.strip_html(), elem_id="ls_strip")
        sdgui.build_ui(sample_dir, parent=app, detach_settings=True)      # the training pages, placed below
        B, UI = sdgui.BLOCKS, sdgui.UI

        def show(key):
            try:
                B[key].render()
            except Exception as e:              # noqa: BLE001
                gr.Markdown(f"This page could not be built: {e!r}")

        with gr.Tabs(selected=start, elem_id="ls_tabs") as top:
            with gr.Tab("Collection", id="collection"):
                ll.build_library_ui(app, detach_settings=True)
            with gr.Tab("Datasets", id="datasets") as t_ds:
                import datasets_ui
                with gr.Column(visible=False):
                    show("prepare")             # shared controls of the steps and the full picture check (hidden)
                DS = datasets_ui.build(app, show)
            with gr.Tab("Training", id="training"):
                import training_ui
                with gr.Tabs(selected="tr_run") as tr_tabs:
                    import features as FT
                    _run_lab = "Run" if (FT.have("sd_scripts") or FT.have("musubi")) else "Run (trainers missing)"
                    with gr.Tab(_run_lab, id="tr_run") as t_run:
                        TR = training_ui.build_run(app)
                    with gr.Tab("Results", id="tr_results") as t_res:
                        RS = training_ui.build_results(app)
                t_run.select(*TR["refresh"], show_progress="hidden")
                t_res.select(*RS["refresh"], show_progress="hidden")
            with gr.Tab("Manual", id="manual") as t_man:
                MN = manual.build(app)
                t_man.select(*MN["load"], show_progress="hidden")
            with gr.Tab("Settings", id="settings"):
                def _backup():
                    import backup
                    backup.build_ui()

                def _updates():
                    import updater
                    import features as FT_
                    settings_ui.selfupdate_panel(app)
                    if FT_.have("sd_scripts") or FT_.have("musubi"):
                        settings_ui.venv_panel(app)
                    import features as FT
                    if FT.have("sd_scripts") or FT.have("musubi"):
                        gr.Markdown("### The trainers")
                        updater.build_ui(app)
                    else:
                        gr.Markdown("### The trainers (sd-scripts and musubi-tuner missing)",
                                    elem_classes=["ls-missing"])

                def _ollama_note():
                    import features as FT
                    if not FT.have("ollama"):
                        gr.Markdown("**Ollama is not installed** (Settings › Start and network › What is installed): the "
                                    "detailed sort, writing captions and the consistency check by Ollama are hidden. "
                                    f"Get it from [{FT.ollama_guide()}]({FT.ollama_guide()}), tick it there, restart.",
                                    elem_classes=["ls-missing-note"])

                def _look():
                    gr.Markdown("How the Collection looks and how it tells which base model a LoRA was trained on.",
                                elem_classes=["ls-note"])
                    ll.LOOK_BLOCK.render()
                settings_ui.build(app, {"start": lambda: (launchopts.build_ui(), settings_ui.features_panel(app)), "folders": ll.FOLDERS_BLOCK.render,
                                        "models": lambda: show("models"), "trainers": lambda: show("trainers"),
                                        "ai": lambda: (_ollama_note(), show("ai")), "look": _look, "backup": _backup,
                                        "updates": _updates, "batch": lambda: settings_ui.batch_panel(app),
                                        "comfy": lambda: settings_ui.comfy_panel(app),
                                        "version": lambda: settings_ui.version_panel(app)})

        # the status strip, and moving between pages
        ticker.timer(2.0, statusbar.strip_html, None, strip, show_progress="hidden")
        def _busy():
            import dsflow
            return sdgui.RUN.running() or dsflow.BATCH.get("running") or bool(sdgui.QUEUE.get("running"))
        ticker.BUSY.append(_busy)
        try:
            import comfytest
            ticker.BUSY.append(lambda: any(j.get("state") == "running" for j in comfytest.JOBS.values()))
        except Exception:                      # noqa: BLE001
            pass
        try:
            import ollamactl
            ticker.BUSY.append(lambda: bool(ollamactl.PULL.get("active")) or
                               (bool(ollamactl.PULL.get("name")) and not ollamactl.PULL.get("done")))
        except Exception:                      # noqa: BLE001
            pass
        ticker.build()                          # the one timer for the whole page (see ticker.py)

        # "Train a new version": Datasets › Edit existing › 9 Launch script of that dataset
        def to_step9(cur, ds):
            if not ds:
                return cur, gr.update()
            return DS["go"](cur, view="flow", mode="edit", ds=ds, step=9), gr.Tabs(selected="datasets")

        def ds_of_run(st):
            return Path(st["script"]).parent.name if st and st.get("script") else ""

        def ds_of_lora(key):
            l = ll.LIB.loras.get(key)
            return l.dataset.name if l is not None and l.dataset is not None else ""
        DS["after_nav"](RS["newv"].click(lambda cur, st: to_step9(cur, ds_of_run(st)), [DS["state"], RS["state"]],
                                         [DS["state"], top]))
        if "lp_newv" in ll.UI_HOOKS:
            btn, key = ll.UI_HOOKS["lp_newv"]
            DS["after_nav"](btn.click(lambda cur, k: to_step9(cur, ds_of_lora(k)), [DS["state"], key], [DS["state"], top]))

    allowed = [str(sdgui.ROOT), str(sdgui.images_dir()), str(sdgui.CACHE_DIR)]
    allowed += [str(d) for d in sdgui.EXTRA_SCRIPT_DIRS] + s["dataset_dirs"] + s["lora_dirs"]
    allowed += [os.path.expanduser(p) for p in args.allow_path]
    ll.THUMB_DIR.mkdir(exist_ok=True)
    allowed.append(str(ll.THUMB_DIR))      # Gallery / Library thumbnails
    allowed += [str(HERE / "reports"), str(HERE / "manual"), str(HERE / "logo")]     # reports, the manual, the logo
    try:                                   # the pictures on the Datasets pages: links to small files, not inline data
        import datacheck
        datacheck.URL_DIR["dir"] = ll.THUMB_DIR / "pages"
        import time as _t
        for f in (ll.THUMB_DIR / "pages").glob("*.jpg"):          # old page pictures: made again when needed
            if _t.time() - f.stat().st_mtime > 30 * 86400:
                f.unlink(missing_ok=True)
    except Exception:                      # noqa: BLE001
        pass

    kw = {"css": css, "head": head, "theme": theme} if style_in_launch else {}
    auth = args.auth or os.environ.get("SDGUI_AUTH") or (f"{lo['user']}:{lo['password']}"
                                                           if lo["user"] and lo["password"] else None)
    if auth:
        user, sep, pw = auth.partition(":")
        if not sep or not user or not pw:
            ap.error("--auth must look like user:password")
        kw["auth"] = (user, pw)
        kw["auth_message"] = login_html()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n*** Listening on {args.host}:{args.port}: anyone on your network can open the program and run "
              f"shell commands through it. {'A login is required.' if auth else 'No login is set: use --auth user:password.'} ***\n")

    # Load the saved scan (or scan) right away, so the first page does not have to wait for it
    import threading
    threading.Thread(target=lambda: (ll.LIB.ensure_loaded(), ll.warm_thumbnails(ll.LIB)),
                     daemon=True, name="library-load").start()
    print(version.banner())
    note = sdgui.queue_restore(start=lo["resume_queue"])
    if note:
        print(f"[Queue] {note}")
    n_old = sdgui.archive_all_datasets()
    if n_old:
        print(f"[Scripts] {n_old} older launch script(s) moved to {sdgui.OLD_RUNS}/ (one script per LoRA type is shown)")
    try:                                   # an update from before 1.251026d: bring the manual and the logo
        import selfupdate
        selfupdate.repair_dirs()
    except Exception as e:                 # noqa: BLE001
        print(f"[Updates] {e!r}")
    try:                                   # PyTorch and its ROCm / CUDA parts in the training venv (no download)
        import venvcheck
        venvcheck.start_check()
    except Exception as e:                 # noqa: BLE001
        print(f"[Training venv] {e!r}")
    if lo["check_updates"]:
        try:                               # quiet check for sd-scripts / musubi-tuner updates (never installs)
            import updater
            updater.start_background_check()
        except Exception as e:             # noqa: BLE001
            print(f"[Updates] not available: {e!r}")

    if os.environ.get("LS_PROFILE"):          # LS_PROFILE=1 ./start.sh: every event that takes 0.2 s or more is printed
        _profile(app)
    fav = HERE / "logo" / "favicon.png"
    if fav.is_file() and "favicon_path" in inspect.signature(gr.Blocks.launch).parameters:
        kw["favicon_path"] = str(fav)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        gz = _gzip_middleware()                # over the network: the 3-4 MB page goes compressed (about 10 times smaller)
        if gz:
            kw["app_kwargs"] = {"middleware": [gz]}
    app.launch(server_name=args.host, server_port=args.port, allowed_paths=allowed,
               inbrowser=not args.no_browser, **kw)


def _gzip_middleware():
    """Starlette's GZip, only when it leaves the live event stream (text/event-stream) alone, as newer versions do."""
    try:
        from starlette.middleware import Middleware
        from starlette.middleware import gzip as _gz
        if "text/event-stream" not in getattr(_gz, "DEFAULT_EXCLUDED_CONTENT_TYPES", ()):
            return None
        if "app_kwargs" not in inspect.signature(gr.Blocks.launch).parameters:
            return None
        return Middleware(_gz.GZipMiddleware, minimum_size=2000)
    except Exception:                          # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
