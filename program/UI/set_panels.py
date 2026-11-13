# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The Settings panels, one function per group (panel_<key>), built when the group is opened. Each uses the logic
of its own module (launchopts, features, lora_library, backup, selfupdate, updater, sd_settings, ollamactl, …)."""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import os
from pathlib import Path

from nicegui import run, ui

import lora_library as LL
import settings_ui as SU
from widgets import confirm, note

LIB = LL.LIB


def card():
    return ui.card().props("flat bordered").classes("ls-card w-full gap-2")


def restart_soon(msg_el, text: str):
    """Restart LoRA Studio; the page reloads itself when the program answers again."""
    msg_el.content = text
    ui.run_javascript("setTimeout(function wait(){ fetch(location.href, {cache: 'no-store'}).then(function(r){ "
                      "if (r.ok) location.reload(); else setTimeout(wait, 2000); }).catch(function(){ "
                      "setTimeout(wait, 2000); }); }, 6000);")


# --------------------------------------------------------------------------- Health check
def panel_health(page):
    html_ = ui.html(SU.health_html(), sanitize=False).classes("w-full")
    msg = ui.markdown("")

    def setup_again():
        try:
            import launchopts
            launchopts.restart(extra=["--setup"])
            restart_soon(msg, "Restarting on the setup page … this page reloads by itself.")
        except Exception as e:                           # noqa: BLE001
            msg.content = f"Could not restart: {e!r}. Run ./start.sh --setup."
    with ui.row().classes("gap-2"):
        ui.button("Check again", on_click=lambda: setattr(html_, "content", SU.health_html())).props("outline dense no-caps")
        ui.button("Run the setup page again", on_click=setup_again).props("flat dense no-caps")
    note("Run the setup page again walks through folders, trainers, models and AI helpers in order; LoRA Studio "
         "restarts on the setup page.")


# --------------------------------------------------------------------------- Start and network
def panel_start(page):
    import features as FT
    import launchopts as LO
    v = LO.load()
    note("How LoRA Studio starts. Saved in launch_settings.json (readable only by you); used by ./start.sh and the "
         "desktop shortcut from the next start. An option typed on the command line still wins.")
    with card():
        with ui.grid(columns=2).classes("w-full gap-3"):
            net = ui.checkbox("Reachable on the network (other computers can open LoRA Studio; needs a login)",
                              value=v["network"])
            port = ui.number("Port", value=v["port"], min=1024, max=65535, precision=0).props("outlined dense")
            user = ui.input("Login: user name", value=v["user"]).props("outlined dense")
            pw = ui.input("Login: password", value=v["password"], password=True, password_toggle_button=True) \
                .props("outlined dense")
            browser = ui.checkbox("Open the browser at start", value=v["browser"])
            start = ui.select(dict((k, lab) for lab, k in LO.START_PAGES), value=v["start_page"], label="Start page") \
                .props("outlined dense")
            upd = ui.checkbox("Check for trainer updates at start (only looks)", value=v["check_updates"])
            resume = ui.checkbox("Resume the queue after a restart", value=v["resume_queue"])
            notify_ = ui.checkbox("Notice when a run finishes (a browser notification)", value=v["notify"])
        msg = ui.markdown("")

        def collect():
            return {"network": bool(net.value), "port": int(port.value or 7860), "browser": bool(browser.value),
                    "user": (user.value or "").strip(), "password": pw.value or "", "start_page": start.value or "home",
                    "check_updates": bool(upd.value), "resume_queue": bool(resume.value), "notify": bool(notify_.value),
                    "queue_autostart": LO.load().get("queue_autostart", True)}

        def check(d) -> str:
            if d["network"] and not (d["user"] and d["password"]):
                return ("❌ Not saved: reachable on the network needs a user name and password, otherwise anyone on "
                        "the network could run commands through LoRA Studio.")
            if bool(d["user"]) != bool(d["password"]):
                return "❌ Not saved: give both a user name and a password, or neither."
            return ""

        def save(and_restart: bool):
            d = collect()
            err = check(d)
            if err:
                msg.content = err
                return
            LO.save(d)
            if and_restart:
                LO.restart()
                restart_soon(msg, f"✔ Saved. Restarting at port {d['port']} … this page reloads by itself when the port "
                                  "stays the same; after a port change, open the new address.")
            else:
                msg.content = "✔ Saved. Used from the next start."
        with ui.row().classes("gap-2"):
            ui.button("Save", on_click=lambda: save(False)).props("outline dense no-caps")
            ui.button("Save and restart", on_click=lambda: save(True)).props("unelevated dense no-caps color=primary")
    with card():
        ui.label("What is installed").classes("font-semibold")
        note("A part that is not installed has its buttons hidden, so nothing can be started by mistake; its steps and "
             "pages stay, greyed out with (… missing). Change it here when you add or remove a part, then restart.")
        boxes = {k: ui.checkbox(f"{name} is installed (needed for {FT.WHAT[k]})", value=FT.have(k))
                 for k, name in FT.PARTS.items()}
        note(f"Ollama: get it from {FT.ollama_guide()}" + (
            " (on Windows: install it in Windows; LoRA Studio in WSL reaches it at the address in Settings › AI helpers)."
            if FT.is_wsl() else "; or on Linux: curl -fsSL https://ollama.com/install.sh | sh"))
        fmsg = ui.markdown("")

        def fsave():
            cur = FT.load()
            new = {}
            for key, b in boxes.items():
                old = cur.get(key, "")
                new[key] = (old if old and old != "none" else "installed") if b.value else "none"
            FT.save(**new)
            fmsg.content = "✔ Saved. Restart LoRA Studio to apply it."

        def frestart():
            import selfupdate
            restart_soon(fmsg, selfupdate.restart())
        with ui.row().classes("gap-2"):
            ui.button("Save", on_click=fsave).props("unelevated dense no-caps color=primary")
            ui.button("Restart LoRA Studio", on_click=frestart).props("outline dense no-caps")


# --------------------------------------------------------------------------- Folders
def _lines(v) -> str:
    return "\n".join(v or [])


def save_library_settings(**changes) -> str:
    """Write library_settings.txt with the changed values (the rest as they are) and read the folders again."""
    s = LL.load_settings()
    s.update(changes)
    LL.write_settings(s["sd_scripts_dir"], _lines(s["dataset_dirs"]), _lines(s["lora_dirs"]), _lines(s["script_dirs"]),
                      s["sample_dir"], s["models_dir"], "\n".join(f"{b} = {', '.join(k)}" for b, k in s["bases"])
                      if isinstance(s["bases"], list) else s["bases"], s["expected_bases"], s["style"],
                      s["nsfw_blur"], next((k for k, v in LL.NSFW_LEVELS.items() if v == s["nsfw_level"]), "X (8)")
                      if isinstance(s["nsfw_level"], int) else s["nsfw_level"], s["blur_unknown_nsfw"], s["pictures_dir"])
    LIB.settings = LL.load_settings()
    for fn in LL.ON_SETTINGS_SAVED:
        fn(LIB.settings)
    LIB.scan()
    return "✔ Settings saved and folders read again."


def panel_folders(page):
    s = LIB.settings
    note(f"Where your own data is: datasets, LoRAs, model files and ComfyUI pictures. Saved to {LL.SETTINGS_FILE.name} "
         "(you can also edit that file). One folder per line.")
    with card():
        with ui.row().classes("w-full no-wrap gap-3"):
            ds = ui.textarea("Dataset folders (the FIRST one is where new datasets are made)",
                             value=_lines(s["dataset_dirs"])).props("outlined").classes("grow")
            lora = ui.textarea("LoRA folders (with LoRA Manager's files; read by Personal sync)",
                               value=_lines(s["lora_dirs"])).props("outlined").classes("grow")
        scripts = ui.textarea("Extra launch script folders (optional)", value=_lines(s["script_dirs"])) \
            .props("outlined").classes("w-full")
        models = ui.input("Model folder (optional): default folder for model files of a LoRA type without its own",
                          value=s["models_dir"]).props("outlined dense").classes("w-full")
        pics = ui.input("ComfyUI picture folder (optional): its output folder, with a folder per LoRA type",
                        value=s.get("pictures_dir", "")).props("outlined dense").classes("w-full")
        msg = ui.markdown("")

        async def save():
            msg.content = "Saving and reading the folders …"
            msg.content = await run.io_bound(lambda: save_library_settings(
                dataset_dirs=[x.strip() for x in ds.value.splitlines() if x.strip()],
                lora_dirs=[x.strip() for x in lora.value.splitlines() if x.strip()],
                script_dirs=[x.strip() for x in scripts.value.splitlines() if x.strip()],
                models_dir=models.value.strip(), pictures_dir=pics.value.strip()))
        ui.button("Save folders and rescan", on_click=save).props("unelevated no-caps color=primary")


# --------------------------------------------------------------------------- Look and base models
def panel_look(page):
    s = LIB.settings
    note("How the Collection looks and how it tells which base model a LoRA was trained on.")
    with card():
        bases = ui.textarea("Base models: Name = keywords (checked top to bottom)",
                            value="\n".join(f"{b} = {', '.join(k)}" for b, k in s["bases"])).props("outlined") \
            .classes("w-full")
        expected = ui.select(LIB.base_names(), multiple=True, label="Expected bases per character",
                             value=[b for b in s["expected_bases"] if b in LIB.base_names()]) \
            .props("outlined dense use-chips").classes("w-full")
        with ui.row().classes("items-center gap-4"):
            blur = ui.checkbox("Blur NSFW previews (hover to show)", value=s["nsfw_blur"])
            level = ui.select(list(LL.NSFW_LEVELS), label="Blur at level and above",
                              value=next((k for k, v in LL.NSFW_LEVELS.items() if v == s["nsfw_level"]), "X (8)")) \
                .props("outlined dense").classes("w-48")
            unknown = ui.checkbox("Also blur unknown level when the model is marked NSFW", value=s["blur_unknown_nsfw"])
        msg = ui.markdown("")

        async def save():
            b = []
            for line in bases.value.splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, val = line.split("=", 1)
                    b.append((k.strip(), [x.strip() for x in val.split(",") if x.strip()]))
            msg.content = "Saving …"
            msg.content = await run.io_bound(lambda: save_library_settings(
                bases=b, expected_bases=list(expected.value or []), nsfw_blur=blur.value,
                nsfw_level=LL.NSFW_LEVELS.get(level.value, 8), blur_unknown_nsfw=unknown.value))
        ui.button("Save settings and rescan", on_click=save).props("unelevated no-caps color=primary")
    with card():
        pc = LL.purpose_colors(LIB)
        on = ui.checkbox("Colour the Collection cards by purpose (border and label; off = all white)", value=pc["on"])
        picks = {}
        with ui.row().classes("gap-4"):
            for k, lab in LL.PURPOSES:
                with ui.column().classes("gap-0 items-center"):
                    ui.label(lab).classes("ls-note")
                    picks[k] = ui.color_input(value=pc[k]).props("dense outlined").classes("w-36")
        pmsg = ui.markdown("")

        def store():
            LIB.data["purpose_colors"] = {"on": bool(on.value),
                                          **{k: (w.value or LL.PURPOSE_COLORS[k]) for k, w in picks.items()}}
            LIB.save_data()
            pmsg.content = "✔ Saved: the cards are coloured by purpose." if on.value else "✔ Saved: the cards are white."
        with ui.row().classes("gap-2"):
            ui.button("Save colours", on_click=store).props("unelevated dense no-caps color=primary")
            ui.button("Default colours", on_click=lambda: [w.set_value(LL.PURPOSE_COLORS[k]) for k, w in picks.items()]) \
                .props("flat dense no-caps")


# --------------------------------------------------------------------------- Backup
def panel_backup(page):
    import backup as B
    note("A backup is one zip with your settings and LoRA Studio's own databases: folders, model files per LoRA type, "
         "Completed marks, your edits on LoRA pages, old LoRA names, the picture index, file hashes and training logs. "
         "Your datasets, models, LoRAs and pictures are not in it. Restore it here or on the setup page of a new install.")
    with card():
        with ui.row().classes("w-full items-end gap-3"):
            folder = ui.input("Backup folder (remembered)", value=B.tilde(B.backup_dir())).props("outlined dense") \
                .classes("grow")
            tok = ui.checkbox("Include the Hugging Face token (keep the backup private)", value=True)
        msg = ui.markdown("")

        async def go():
            d = os.path.expanduser((folder.value or "").strip()) or str(B.DEFAULT_DIR)
            try:
                out, man = await run.io_bound(B.make_backup, d, bool(tok.value))
            except Exception as e:                       # noqa: BLE001
                msg.content = f"❌ Could not write the backup to {d}: {e}"
                return
            B.remember_dir(Path(d))
            msg.content = (f"✔ Saved {out} ({out.stat().st_size / 1024:,.0f} KB, {len(man['files'])} files"
                           f"{', with your Hugging Face token: keep it private' if man['token'] else ''}).")
            pick.options = B.find_backups([d])
            pick.update()
        ui.button("Back up now", on_click=go).props("unelevated no-caps color=primary")
    with ui.expansion("Restore from a backup").classes("w-full ls-card"):
        pick = ui.select(B.find_backups(), label="Backups found", on_change=lambda e: setattr(
            info, "content", B.describe(e.value) if e.value else "")).props("outlined dense").classes("w-full")
        path = ui.input("Or the path of a backup .zip", on_change=lambda e: setattr(
            info, "content", B.describe(e.value.strip()) if (e.value or "").strip() else "")).props("outlined dense") \
            .classes("w-full")
        info = ui.markdown("")
        rmsg = ui.markdown("")

        async def restore():
            src = (path.value or "").strip() or pick.value
            if not src:
                rmsg.content = "Pick a backup or enter its path."
                return
            if not await confirm("Replace the current settings with this backup (they are kept in restore_backups/ "
                                 "first)? LoRA Studio restarts.", "Restore"):
                return
            try:
                text = B.restore(src)
            except Exception as e:                       # noqa: BLE001
                rmsg.content = f"❌ {e}"
                return
            B.restart_program()
            restart_soon(rmsg, text + "\n\nLoRA Studio restarts now; this page reloads by itself.")
        ui.button("Restore", on_click=restore).props("unelevated dense no-caps color=negative")
