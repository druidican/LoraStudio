# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The Settings panels, one function per group (panel_<key>), built when the group is opened. Each uses the logic
of its own module (launchopts, features, lora_library, backup, selfupdate, updater, sd_settings, ollamactl, …)."""
from __future__ import annotations
__version__ = "1.141126"  # LoRA Studio version, see version.py

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


# --------------------------------------------------------------------------- helpers for long jobs
def log_box(label: str, rows: int = 10):
    return ui.textarea(label).props(f"readonly outlined rows={rows} input-style='font-family:monospace;font-size:12px'") \
        .classes("w-full")


async def thread_lines(work, box):
    """Run work(say) in a thread; every line it says shows in the box (refreshed twice a second)."""
    import asyncio
    lines: list[str] = []
    task = asyncio.ensure_future(run.io_bound(work, lines.append))
    while not task.done():
        box.value = "\n".join(lines)
        await asyncio.sleep(0.5)
    box.value = "\n".join(lines)
    return task.result()


async def saved_upload(e) -> str:
    """An uploaded file, saved to a temporary file; its path."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="lorastudio_up_"))
    p = d / Path(e.file.name).name
    await e.file.save(p)
    return str(p)


# --------------------------------------------------------------------------- Version
def panel_version(page):
    from widgets import stream
    out = ui.html(SU.version_html(), sanitize=False).classes("w-full")
    ui.button("Check again", on_click=lambda: setattr(out, "content", SU.version_html())).props("flat dense no-caps")
    with card():
        note("Check this machine. Looks at your GPU, the trainers and their venv, the model files of every LoRA type, "
             "Ollama, ComfyUI and the picture tools, and can train a tiny LoRA (10 steps at 512 px on 8 made-up "
             "pictures, in a temporary folder that is deleted afterwards) to see that a real run starts, uses the GPU "
             "and saves a file. Nothing of yours is changed. The report (userdata/lorastudio_check_<time>.md; no token, "
             "home folder as ~, no dataset names) is the file to upload when something is not right.")
        log = log_box("Machine check", 14)

        async def check(train: bool):
            await stream(SU.machine_check(train), lambda t: setattr(log, "value", t))
        with ui.row().classes("gap-2"):
            ui.button("Check (no training, about a minute)", on_click=lambda: check(False)).props("outline dense no-caps")
            ui.button("Check + tiny training (a few minutes)", on_click=lambda: check(True)) \
                .props("unelevated dense no-caps color=primary")
    import notice
    with ui.expansion("Licences and credits").classes("w-full ls-card"):
        ui.markdown(f"### {notice.TITLE}\n\n{notice.TEXT}\n\nThe full GPL-3.0 text is in LICENSE, this text in NOTICE, "
                    "both in the program folder.")
    with card():
        note("Report for Claude. One file with the version, the machine and the newest machine check (token and home "
             "folder removed), copied so you can paste it into a chat. For a failed run, use the button on that run in "
             "its dataset's Results.")
        rnote = ui.input("What do you see? (optional)").props("outlined dense").classes("w-full")
        rmsg = ui.markdown("")

        def report():
            import report as R
            path, text = R.make("", rnote.value or "")
            ui.clipboard.write(text)
            rmsg.content = f"✔ Made and copied: userdata/reports/{path.name}."
        ui.button("Report for Claude", on_click=report).props("outline dense no-caps")


# --------------------------------------------------------------------------- Updates
def panel_updates(page):
    import features as FT
    import ghupdate as GH
    import selfupdate as SUP
    with card():
        ui.label("LoRA Studio itself").classes("font-semibold")
        note("Install a newer LoraStudio.zip: only the program files change (your settings, databases, token, logs and "
             "venv are never touched). The files it replaces are kept in update_backups/ first, the self-test runs with "
             "the new files, and when it fails the old files go back by themselves. Then restart LoRA Studio.")
        st = {"zip": ""}
        plan = ui.html("", sanitize=False).classes("w-full")

        async def got_zip(e):
            st["zip"] = await saved_upload(e)
            plan.content = SU.selfupdate_plan_html(st["zip"])
            try:
                ok = any(r["state"] in ("changed", "new") for r in SUP.plan(st["zip"])["rows"])
            except Exception:                            # noqa: BLE001
                ok = False
            inst.set_enabled(ok)
        ui.upload(label="LoraStudio.zip", auto_upload=True, on_upload=got_zip).props("accept=.zip flat bordered") \
            .classes("w-full")
        ulog = log_box("LoRA Studio update")

        async def install():
            ok = await thread_lines(lambda say: SUP.install(st["zip"], say=say), ulog)
            if ok:
                try:
                    import venvcheck as VC
                    r = VC.inspect()
                    ulog.value += "\n" + ("✔ " if r["ok"] else "! ") + "Training venv: " + r["text"]
                except Exception:                        # noqa: BLE001
                    pass

        async def rollback():
            await thread_lines(lambda say: SUP.rollback(say=say), ulog)
        with ui.row().classes("gap-2"):
            inst = ui.button("Install this update", on_click=install).props("unelevated dense no-caps color=primary")
            inst.set_enabled(False)
            ui.button("Restart LoRA Studio", on_click=lambda: restart_soon(rmsg, SUP.restart())).props("outline dense no-caps")
            ui.button("Roll back the last update", on_click=rollback).props("flat dense no-caps")
        rmsg = ui.markdown("")
        with ui.expansion("From GitHub: updates without a zip", value=True).classes("w-full"):
            note(f"Checks github.com/{GH.repo()} (branch {GH.branch()}) for a new release (or, with Betas too, the "
                 "newest upload) and installs it the same safe way as a zip. Only when you press the buttons. While the "
                 "repository is private, a GitHub token with read access to it is needed (fine-grained token: this "
                 "repository, Contents: Read-only); it is saved in userdata/github_token.txt, readable only by you.")
            with ui.row().classes("w-full items-end gap-2"):
                tok = ui.input("GitHub token", password=True,
                               placeholder=(f"saved: {GH.mask(GH.read_token())}" if GH.read_token()
                                            else "github_pat_… (empty: no token)")).props("outlined dense").classes("grow")
                gmd = ui.markdown("")

                def save_tok():
                    gmd.content = GH.save_token(tok.value)
                    tok.value = ""
                ui.button("Save token", on_click=save_tok).props("outline dense no-caps")
                ui.button("Test token", on_click=lambda: setattr(gmd, "content", GH.test_token())).props("flat dense no-caps")
            ui.label("Updates from").classes("font-semibold")
            chan = ui.radio({k: v for k, v in GH.CHANNELS.items()}, value=GH.channel(),
                            on_change=lambda e: (GH.set_channel(GH.CHANNELS[e.value]), gh_inst.set_enabled(False))) \
                .props("inline")
            note("Releases only: the versions published as a release (tested). Betas too: also every newer upload in "
                 "between, which may still have rough edges.")

            async def gh_check():
                c = await run.io_bound(GH.check)
                gmd.content = c["text"]
                gh_inst.set_enabled(bool(c.get("ok")) and not c.get("same"))

            async def gh_install():
                ok = await thread_lines(lambda say: GH.install(say=say), ulog)
                gh_inst.set_enabled(not ok)
            with ui.row().classes("gap-2"):
                ui.button("Check GitHub", on_click=gh_check).props("outline dense no-caps")
                gh_inst = ui.button("Install from GitHub", on_click=gh_install).props("unelevated dense no-caps color=primary")
                gh_inst.set_enabled(False)
            _ = chan
    if FT.have("sd_scripts") or FT.have("musubi"):
        venv_card()
        import updater as UP
        with card():
            ui.label("The trainers").classes("font-semibold")
            note("Updates for sd-scripts and musubi-tuner from GitHub (the branch each one is on). Checked quietly when "
                 "the program starts; nothing is installed until you press Update now. PyTorch / ROCm are never changed, "
                 "and if a test fails after the update, the code and the venv go back to how they were.")
            status = ui.markdown(UP.status_md())
            tlog = log_box("Update log", 14)

            async def tcheck():
                status.content = "Checking for updates …"
                if UP.STATUS["running"] or UP.LOCK.locked():
                    status.content = UP.status_md() + "\n\n(a check or update is already running)"
                    return
                await run.io_bound(UP.check)
                status.content = UP.status_md()

            async def tupdate():
                from widgets import stream
                await stream(UP.update(), lambda r: (setattr(tlog, "value", r[0]), setattr(status, "content", r[1])))
            with ui.row().classes("gap-2"):
                ui.button("Check for updates", on_click=tcheck).props("outline dense no-caps")
                ui.button("Update now", on_click=tupdate).props("unelevated dense no-caps color=primary")
    else:
        note("The trainers (sd-scripts and musubi-tuner missing).")


def venv_card():
    import venvcheck as VC
    from widgets import stream
    with card():
        ui.label("The training venv").classes("font-semibold")
        note("PyTorch decides: every ROCm or CUDA part must be the version PyTorch is built for (a newer part is wrong "
             "too: it crashes trainings with Memory access fault by GPU). Checked at every start and after an update, "
             "without downloading anything. Fix puts the parts that do not fit back and writes the locks again.")
        st = ui.html(SU.venv_html(), sanitize=False).classes("w-full")
        vlog = log_box("Training venv", 8)
        vlog.set_visibility(False)

        def again():
            VC.inspect()
            st.content = SU.venv_html()

        async def fix():
            vlog.set_visibility(True)
            await stream(VC.fix(), lambda t: setattr(vlog, "value", t))
            st.content = SU.venv_html()
        with ui.row().classes("gap-2"):
            ui.button("Check again", on_click=again).props("outline dense no-caps")
            ui.button("Fix the training venv", on_click=fix).props("unelevated dense no-caps color=primary")
        ui.label("A newer PyTorch / ROCm").classes("font-semibold mt-2")
        note("Asks AMD's server (ROCm) or PyPI (CUDA) which PyTorch builds exist; only stable releases count. The "
             "upgrade keeps the old versions in update_backups/, tests the GPU and puts them back when the test fails.")
        nmd = ui.markdown("")

        async def look():
            nmd.content = "Asking the server …"
            n = await run.io_bound(VC.look)
            md = f"**{n['text']}**" if n["newer"] else n["text"]
            if n["newer"] and not n["managed"]:
                md += ("\n\nYour own venv: LoRA Studio shows the command, it does not change it:\n\n```\n"
                       + VC._upgrade_cmd(VC.LAST, n["newest"]) + "\n```")
            nmd.content = md
            up.set_visibility(n["newer"] and n["managed"])
            up.text = f"Upgrade to torch {n['newest']}"

        async def upgrade():
            tgt = VC.NEWER.get("newest")
            if not tgt or not await confirm("Upgrade PyTorch (several GB; no training until it has finished)?",
                                            "Upgrade"):
                return
            vlog.set_visibility(True)
            await stream(VC.upgrade(tgt), lambda t: setattr(vlog, "value", t))
            st.content = SU.venv_html()
        with ui.row().classes("gap-2"):
            ui.button("Look for a newer PyTorch", on_click=look).props("outline dense no-caps")
            up = ui.button("Upgrade", on_click=upgrade).props("unelevated dense no-caps color=primary")
            up.set_visibility(False)


# --------------------------------------------------------------------------- Test pictures (ComfyUI)
def panel_comfy(page):
    import comfytest as CT
    import comfyupscale as U
    s = CT.settings()
    note("After a training run, LoRA Studio can make the same test pictures with each save point in your ComfyUI, "
         "side by side in a dataset's Results, to pick the best epoch (after a queue: Home › Just trained). It reads "
         "your own workflow for the model, sampler, steps, CFG, size and negative prompt, and sends ComfyUI a small "
         "graph of its own nodes with those settings. Saved in comfy_settings.json; your workflows are kept in "
         "comfy_workflows/.")
    with card():
        with ui.row().classes("w-full items-end gap-2"):
            url = ui.input("ComfyUI address", value=s["url"]).props("outlined dense").classes("grow")
            tmsg = ui.markdown("")

            async def test():
                u = (url.value or "").strip() or CT.DEFAULTS["url"]
                tmsg.content = "Asking ComfyUI …"
                ok = await run.io_bound(CT.alive, u)
                tmsg.content = (f"✔ ComfyUI answers at {u} ({len(await run.io_bound(CT.lora_names, u))} LoRAs in its "
                                "list)." if ok else f"❌ ComfyUI does not answer at {u}. Start it, then press Test again.")
            ui.button("Test", on_click=test).props("outline dense no-caps")
        lora_dir = ui.input("The LoRA folder ComfyUI reads (empty = your first LoRA folder)", value=s["lora_dir"]) \
            .props("outlined dense").classes("w-full")
        with ui.row().classes("gap-3"):
            n = ui.number("Test prompts per LoRA", value=s["prompts"], min=1, max=24, precision=0).props("outlined dense")
            around = ui.number("Save points on each side of the one to try first", value=s["around"], min=0, max=6,
                               precision=0).props("outlined dense").classes("w-80")
        auto = ui.checkbox("Make them by themselves after the training queue has finished (one LoRA: the save point "
                           "to try first and the final file; several: the final file of each)", value=s["auto"])
    fams = {}
    for fam, lab in CT.FAMILIES.items():
        with card():
            ui.label(lab).classes("font-semibold")
            info = ui.markdown(_wf_summary(CT, fam))
            fixed = ui.textarea("Fixed tags in front of every test prompt (e.g. quality tags)",
                                value=s["fixed"].get(fam, "")).props("outlined autogrow").classes("w-full")
            wmsg = ui.markdown("")

            async def got(e, fam=fam, info=info, wmsg=wmsg):
                path = await saved_upload(e)
                try:
                    _, note_ = CT.import_workflow(fam, path)
                except Exception as ex:                  # noqa: BLE001
                    wmsg.content = f"❌ Could not read that workflow: {ex}"
                    return
                info.content = _wf_summary(CT, fam)
                wmsg.content = note_
            ui.upload(label="Your workflow (ComfyUI › Workflow › Save / Export, the .json file)", auto_upload=True,
                      on_upload=got).props("accept=.json flat bordered").classes("w-full")
            fams[fam] = fixed
    msg = ui.markdown("")

    def save():
        CT.save_settings(url=(url.value or "").strip() or CT.DEFAULTS["url"], lora_dir=(lora_dir.value or "").strip(),
                         prompts=int(n.value or 8), around=int(around.value or 0), auto=bool(auto.value),
                         fixed={f: w.value for f, w in fams.items()})
        msg.content = "✔ Saved."
    ui.button("Save", on_click=save).props("unelevated no-caps color=primary")
    with card():
        ui.label("Upscale workflow (dataset pictures)").classes("font-semibold")
        note("Used by the Detailed scan (Datasets › Sort, characters) to upscale small pictures, sharpen soft ones and "
             "enlarge detail crops. Give it your own upscale workflow (for example SeedVR2) saved from ComfyUI. It runs "
             "as it is, with your models; LoRA Studio only puts the picture into its Load Image node and, in a SeedVR2 "
             "node, sets the size and the noise. The originals stay in _originals/.")
        us = U.settings()
        uinfo = ui.markdown(f"Workflow: {us['name']} · {us['note']}" if U.ready() else "No upscale workflow yet.")
        umsg = ui.markdown("")

        async def uget(e):
            path = await saved_upload(e)
            try:
                umsg.content = U.import_workflow(path)
            except Exception as ex:                      # noqa: BLE001
                umsg.content = f"❌ Could not use that workflow: {ex}"
                return
            s2 = U.settings()
            uinfo.content = f"Workflow: {s2['name']} · {s2['note']}"
        ui.upload(label="Your upscale workflow (.json)", auto_upload=True, on_upload=uget) \
            .props("accept=.json flat bordered").classes("w-full")
        ui.radio(dict((v, lab) for lab, v in U.NOISE_CHOICES), value=us["noise"],
                 on_change=lambda e: (U.save_settings(noise=e.value), setattr(umsg, "content", "✔ Saved."))).props("inline")


def _wf_summary(CT, fam) -> str:
    wf = CT.settings()["families"].get(fam)
    if not wf:
        return "No workflow yet."
    model = wf["ckpt"] or wf["unet"]
    return (f"Model {model} · {wf['sampler']} / {wf['scheduler']} · {wf['steps']} steps · CFG {wf['cfg']} · "
            f"{wf['width']}×{wf['height']}" + (f" · CLIP skip {abs(wf['clip_skip'])}" if wf.get("clip_skip") else ""))


# --------------------------------------------------------------------------- AI helpers (Ollama, Hugging Face token)
def panel_ai(page):
    import asyncio
    import features as FT
    import sdgui as S
    oc = S.ollama_ctl()
    if not FT.have("ollama"):
        note("Ollama is not installed (Settings › Start and network › What is installed): the detailed sort, writing "
             f"captions and the consistency check by Ollama are hidden. Get it from {FT.ollama_guide()}, tick it there, "
             "restart.")
    url0, use0 = S.oc_settings()
    with card():
        ui.label("Ollama: reads pictures and writes captions").classes("font-semibold")
        with ui.row().classes("w-full items-end gap-2"):
            url = ui.input("Ollama address", value=url0).props("outlined dense").classes("grow")
            ui.button("Test", on_click=lambda: test()).props("outline dense no-caps")
            ui.button("Stop Ollama", on_click=lambda: refresh(oc.stop_all(u()))).props("flat dense no-caps color=negative")
        model = ui.select([], value=None, label="Default model (captions, sorting and picture checks)",
                          on_change=lambda e: set_default(e.value)).props("outlined dense").classes("w-full")
        note("Only models that can read pictures are listed. It is started when a step needs it; Start below only "
             "loads it ahead of time.")
        tmsg = ui.markdown("")
        table = ui.html("", sanitize=False).classes("w-full").props('id="ai_models"')
        msg = ui.markdown("")
        with ui.expansion("Get another model").classes("w-full"):
            with ui.row().classes("w-full items-end gap-2"):
                pname = ui.input("Model name", placeholder="e.g. minicpm-v:8b").props("outlined dense").classes("grow")
                pmd = ui.markdown("")

                async def pull():
                    pmd.content = oc.pull(pname.value, u())
                    while not oc.PULL.get("done"):
                        await asyncio.sleep(2)
                        pmd.content = oc.pull_text()
                    pmd.content = oc.pull_text()
                    refresh()
                ui.button("Download", on_click=pull).props("outline dense no-caps")
            with ui.row().classes("w-full items-end gap-2"):
                rm = ui.select([], label="Model to remove").props("outlined dense").classes("grow")

                async def remove():
                    if rm.value and await confirm(f"Remove {rm.value} from Ollama?", "Remove"):
                        refresh(oc.delete(rm.value, u()))
                ui.button("Remove", on_click=remove).props("flat dense no-caps color=negative")
        note("Only one model runs at a time: Start on another model stops the running one first. Starting a training "
             "run stops Ollama to free video memory; it is not started again by itself.")

    def u():
        return (url.value or "").strip() or oc.DEFAULT_URL

    def refresh(text=""):
        try:
            ms = oc.installed(u())
        except Exception:                                # noqa: BLE001
            ms = []
        _, use = S.oc_settings()
        vision = [m["name"] for m in ms if m.get("vision")]
        table.content = S.ai_models_html(u())
        msg.content = text or ""
        model.options = vision + ([use] if use and use not in vision else [])
        model.set_value(use or None)
        model.update()
        rm.options = [m["name"] for m in ms]
        rm.update()

    def set_default(name):
        _, use = S.oc_settings()
        if name and name != use:
            S.oc_save_settings(u(), name)
            refresh(f"✔ {name} is now the default: used for captions, sorting and picture checks.")

    def test():
        _, m = S.oc_settings()
        S.oc_save_settings(u(), m)
        ok = oc.running(u(), fresh=True) is not None
        tmsg.content = ("✔ Ollama answers at " if ok else "❌ Ollama does not answer at ") + u() + "."
        refresh()

    def action(v):
        act, _, name = (v or "").partition("|")
        if act == "use":
            S.oc_save_settings(u(), name)
            refresh(f"✔ {name} is used for captions, sorting and picture checks.")
        elif act == "start":
            refresh(oc.start(name, u()))
        elif act == "stop":
            refresh(oc.stop(name, u()))
    page.on_ai = action
    refresh()
    with card():
        ui.label("Hugging Face token").classes("font-semibold")
        note(f"Kept in {S.hf_path()} (readable only by you) and written into export HF_TOKEN=\"...\" of launch scripts that "
             "download models. The token is masked in the run log.")
        cur = ui.input("Current token", value=S.read_hf_token(), password=True, password_toggle_button=True) \
            .props("outlined dense readonly").classes("w-full")
        with ui.row().classes("w-full items-end gap-2"):
            new = ui.input("Paste a new token", password=True, placeholder="hf_...").props("outlined dense").classes("grow")
            hmsg = ui.markdown("")

            def save_tok():
                m, _, _ = S.save_hf_token(new.value)
                hmsg.content = m
                new.value = ""
                cur.value = S.read_hf_token()
            ui.button("Save token", on_click=save_tok).props("unelevated dense no-caps color=primary")


# --------------------------------------------------------------------------- Trainers and GPU
def panel_trainers(page):
    import html as _h
    import sd_settings as SSET
    import sdgui as S
    note(f"The two trainers and the GPU they run on. Saved in {Path(S.settings_path()).name} under [general].")
    wd, venv, trace, sd, mu = S.runner_load()
    with card():
        with ui.row().classes("w-full no-wrap gap-3"):
            rn_sd = ui.input("sd-scripts folder (Illustrious, SDXL, Pony, Anima, FLUX.1, SD 1.5 …)",
                             value=sd or str(S.ROOT)).props("outlined dense").classes("grow")
            rn_mu = ui.input("musubi-tuner folder (Z-Image, Qwen-Image, FLUX.2 klein; empty = ~/musubi-tuner)",
                             value=mu).props("outlined dense").classes("grow")
        facts = ui.html(S.trainer_facts_html(), sanitize=False).classes("w-full").props('id="tr_facts"')
    with card():
        ui.label("Picture tools").classes("font-semibold")
        note("Quality rating, look-alikes, crops around the person, watermark removal (LaMa) and gentle upscaling "
             "(Real-ESRGAN). They run in the sd-scripts venv on your GPU, never during a training run, and stop after 5 "
             "idle minutes.")
        pt = ui.html("", sanitize=False)
        pmsg = ui.markdown("")

        def pt_draw(fresh=False):
            try:
                import pictools
                st = pictools.status(fresh)
            except Exception as e:                       # noqa: BLE001
                pt.content = f'<div class="ls-note">✕ {_h.escape(str(e))}</div>'
                return
            have = sorted(p.name for p in pictools.MODELS_DIR.glob("*") if p.suffix in (".pt", ".pth"))
            models = (f" · model files here: {_h.escape(', '.join(have))}" if have else
                      " · the model files (about 2.5 GB) download the first time a tool is used")
            line = ("● ready" + models if st["ok"] else "○ " + _h.escape(st["text"]))
            pt.content = f'<div class="ls-note">{line}<br><span class="rs-mono">{_h.escape(st["python"] or "")}</span></div>'
            inst.set_visibility(not st["ok"])

        async def install():
            import pictools
            pmsg.content = "Installing spandrel into the sd-scripts venv (PyTorch is left as it is) …"
            pmsg.content = await run.io_bound(pictools.install)
            pt_draw(True)
        with ui.row().classes("gap-2"):
            inst = ui.button("Install the picture tools", on_click=install).props("unelevated dense no-caps color=primary")
            ui.button("Check again", on_click=lambda: pt_draw(True)).props("flat dense no-caps")
        pt_draw()
    with ui.expansion("Advanced: low-level trainer and GPU values").classes("w-full ls-card"):
        with ui.row().classes("w-full no-wrap gap-3"):
            rn_wd = ui.input("Working folder (scripts run from here)", value=wd).props("outlined dense").classes("grow")
            rn_venv = ui.input("Venv to activate first (optional)", value=venv).props("outlined dense").classes("grow")
        rn_trace = ui.checkbox("Write a full trace to the log (set -x; for bug reports)", value=bool(trace))
        nv = S.S_NVIDIA()
        note("GPU and CUDA environment (NVIDIA): the AMD ROCm values are not needed and are left out." if nv else
             "GPU and ROCm environment. Change the GFX version when you change GPU. Used by the program, put into every "
             "new launch script, and written into existing scripts with the button below.")
        g = SSET.gfx_load()
        with ui.row().classes("w-full items-end gap-2") as gx_row:
            ui.select({val: f"{lab}: {val}" for lab, val in SSET.KNOWN_GFX}, value=g[1], label="Known GPUs",
                             on_change=lambda e: e.value and gx.set_value(e.value)).props("outlined dense").classes("grow")
            gx = ui.input("GFX version override", value=g[0], placeholder="e.g. 12.0.1").props("outlined dense") \
                .classes("w-40")

            def detect():
                v, m = SSET.detect_gfx()
                if v:
                    gx.value = v
                gmsg.content = m
            ui.button("Detect GPU", on_click=detect).props("outline dense no-caps")
        gx_row.set_visibility(not nv)
        extra = [ui.input(lab, value=g[3 + i], placeholder=f"empty = preset default ({d})").props("outlined dense")
                 .classes("w-full") for i, (_, lab, d) in enumerate(SSET.ENV_EXTRA)]
        gmsg = ui.markdown(g[2])

        def env_save():
            gmsg.content = SSET.env_save(gx.value, *[w.value for w in extra])[0]

        async def env_update():
            if await confirm("Rewrite the export line in all your launch scripts (a .bak is kept)?", "Update"):
                gmsg.content = SSET.env_update_scripts(True)
        with ui.row().classes("gap-2"):
            ui.button("Save environment", on_click=env_save).props("outline dense no-caps")
            ui.button("Update all launch scripts", on_click=env_update).props("flat dense no-caps")
    msg = ui.markdown("")

    def save():
        m, _ = S.runner_save(rn_wd.value, rn_venv.value, rn_trace.value, rn_sd.value, rn_mu.value)
        sdp = (rn_sd.value or "").strip()
        if sdp and m.startswith("Saved to") and Path(os.path.expanduser(sdp)).is_dir():
            for cb in S.ON_SD_DIR:
                try:
                    cb(sdp)
                except Exception as e:                   # noqa: BLE001
                    m += f" (could not update the folder list: {e})"
        msg.content = m
        facts.content = S.trainer_facts_html()
    ui.button("Save", on_click=save).props("unelevated no-caps color=primary")


# --------------------------------------------------------------------------- Model files
def panel_models(page):
    import sd_settings as SSET
    import sdgui as S
    from widgets import stream
    HD = SSET._hd()
    types = dict((v, lab) for lab, v in S.type_choices())
    st = {"tkey": next(iter(types), ""), "flags": [], "slots": []}
    note(f"Model, VAE and text-encoder files for each LoRA type, saved in {Path(S.settings_path()).name}. Pick the type, "
         "then for every file type paste a folder (press Enter) and pick a file. New launch scripts use these files "
         "instead of the preset defaults.")
    with ui.row().classes("w-full items-end gap-3"):
        tsel = ui.select(types, value=st["tkey"], label="LoRA type", on_change=lambda e: load(e.value)) \
            .props("outlined dense").classes("w-72")
        rec = ui.checkbox("Also list files in subfolders (2 levels)", value=True, on_change=lambda: refresh_lists())
        ui.button("Refresh file lists", on_click=lambda: refresh_lists()).props("flat dense no-caps")
    info = ui.markdown("")
    slots_box = ui.column().classes("w-full gap-2")
    dl_md = ui.markdown("")
    msg = ui.markdown("")

    def files_of(d):
        return SSET.list_model_files(d, rec.value)

    def load(tkey):
        st["tkey"] = tkey
        st["slots"].clear()
        st["flags"] = []
        slots_box.clear()
        if not tkey:
            return
        saved = SSET.read_settings().get(tkey, {})
        kn = HD.known(tkey)
        with slots_box:
            for key, default in SSET.type_slots(tkey):
                path = saved.get(key) or default
                d = SSET._slot_dir(path, saved.get(key + ".dir"))
                lab = SSET.slot_label(key)
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    din = ui.input(f"{lab}: folder", value=d).props("outlined dense").classes("w-80")
                    opts = files_of(d)
                    if path and path not in opts:
                        opts = [path] + opts
                    fsel = ui.select(opts, value=path or None, label=f"{lab}: file (--{key})", with_input=True,
                                     new_value_mode="add-unique").props("outlined dense").classes("grow")
                    b = ui.button("⬇ From Hugging Face").props("outline dense no-caps")
                    b.set_visibility(key in kn and HD.is_missing(path or ""))
                slot = {"key": key, "dir": din, "file": fsel, "btn": b}
                din.on("keydown.enter", lambda _, s=slot: relist(s))
                din.on("blur", lambda _, s=slot: relist(s))
                b.on_click(lambda _, s=slot: download_one(s))
                st["slots"].append(slot)
                st["flags"].append(key)
        have = SSET.settings_path().is_file()
        info.content = (f"Settings for this type are read from {SSET.settings_path()}." if saved else
                        "No saved settings for this type yet: showing the defaults from your preset. Save writes them"
                        + (" (the file is created)." if not have else "."))
        dl_all.set_visibility(sum(1 for s in st["slots"] if s["btn"].visible) >= 2)
        link_slot.options = {s["key"]: SSET.slot_label(s["key"]) for s in st["slots"]}
        link_slot.value = st["flags"][0] if st["flags"] else None
        link_slot.update()
        suggest()

    def relist(s):
        cur = s["file"].value
        opts = files_of(s["dir"].value)
        s["file"].options = ([cur] if cur and cur not in opts else []) + opts
        s["file"].update()

    def refresh_lists():
        for s in st["slots"]:
            relist(s)

    def set_slot(key, path):
        s = next((x for x in st["slots"] if x["key"] == key), None)
        if s is None or path is None:
            return
        p = Path(path)
        s["dir"].value = str(p.parent.parent if HD.GENERIC.match(p.name) else p.parent)
        relist(s)
        s["file"].options = list(dict.fromkeys([str(p)] + list(s["file"].options)))
        s["file"].value = str(p)
        s["file"].update()
        s["btn"].set_visibility(False)

    async def download_one(s):
        res = await stream(HD.fetch(st["tkey"], s["key"], slot_dir=s["dir"].value or ""),
                           lambda line: setattr(dl_md, "content", line))
        set_slot(s["key"], res)

    async def download_all():
        todo = HD.missing(st["tkey"])
        if not todo:
            dl_md.content = "✔ Every model file of this LoRA type is on this computer."
            return
        log, n = [], 0
        for key in todo:
            s = next((x for x in st["slots"] if x["key"] == key), None)
            last = {"t": ""}

            def show(line):
                last["t"] = line
                dl_md.content = "\n\n".join(log + [line])
            res = await stream(HD.fetch(st["tkey"], key, slot_dir=(s["dir"].value if s else "") or ""), show)
            if res is not None:
                n += 1
                set_slot(key, res)
            log.append(last["t"])
            if HD.STOP.is_set():
                break
        dl_md.content = "\n\n".join(log + [f"**{n} of {len(todo)} files downloaded.**"])

    def save():
        args = [s["dir"].value for s in st["slots"]] + [s["file"].value for s in st["slots"]]
        n = len(st["slots"])
        S.ST_N = max(S.ST_N, n)
        pad = S.ST_N - n
        dirs, files = args[:n] + [""] * pad, args[n:] + [""] * pad
        msg.content = SSET.st_save(st["tkey"], st["flags"] + [""] * pad, *dirs, *files)[0]
        raw.value = SSET.st_raw()
    with ui.row().classes("gap-2"):
        ui.button(f"Save to {Path(S.settings_path()).name}", on_click=save).props("unelevated no-caps color=primary")
        dl_all = ui.button("⬇ Download all missing files", on_click=download_all).props("outline no-caps")
        ui.button("Stop download", on_click=SSET.st_download_stop).props("flat no-caps color=negative")
    with ui.expansion("Download any file from Hugging Face (paste its link)").classes("w-full ls-card"):
        note("For a file that is not in the built-in list, or another version of it: open the file on huggingface.co and "
             "copy the link (…/blob/main/… or …/resolve/main/…). Your Hugging Face token is used, so gated repos work "
             "once you have accepted their licence. A file in several parts is fetched whole.")
        with ui.row().classes("w-full items-end gap-2"):
            link_slot = ui.select({}, label="For the slot").props("outlined dense").classes("w-56")
            link = ui.input("Hugging Face file link",
                            placeholder="https://huggingface.co/<owner>/<repo>/blob/main/<file>.safetensors") \
                .props("outlined dense").classes("grow")

            async def by_link():
                s = next((x for x in st["slots"] if x["key"] == link_slot.value), None)
                if not link_slot.value or not (link.value or "").strip():
                    dl_md.content = "Pick the slot and paste the link of the file first."
                    return
                res = await stream(HD.fetch(st["tkey"], link_slot.value, slot_dir=(s["dir"].value if s else "") or "",
                                            link=link.value), lambda line: setattr(dl_md, "content", line))
                set_slot(link_slot.value, res)
            ui.button("⬇ Download", on_click=by_link).props("unelevated dense no-caps color=primary")
    with ui.expansion("Suggestions: which files fit this LoRA type and base model", value=True).classes("w-full ls-card"):
        note("Every model file in the slot folders (and the models folder above them) is identified from its header: "
             "model family, text encoder / VAE kind, fp16 / fp8. The best match for each slot is marked ★. Apply "
             "suggestions fills the empty or unsuitable slots; nothing is saved until you press Save.")
        sug_md = ui.markdown("")
        sug = {"picks": {}}
        all_ = ui.checkbox("Also replace files that already fit")
        online_md = ui.markdown("")

        def suggest():
            args = [s["dir"].value for s in st["slots"]] + [s["file"].value for s in st["slots"]]
            n = len(st["slots"])
            S.ST_N = max(S.ST_N, n)
            pad = S.ST_N - n
            md, picks = SSET.st_suggest(st["tkey"], st["flags"] + [""] * pad, *(args[:n] + [""] * pad),
                                        *(args[n:] + [""] * pad))
            sug_md.content, sug["picks"] = md, picks

        def apply():
            req = S.GUIDE.requirements_for(st["tkey"], SSET._type_base(st["tkey"])) if S.GUIDE else {}
            changed = []
            for s in st["slots"]:
                cur = os.path.expanduser((s["file"].value or "").strip())
                new = (sug["picks"] or {}).get(s["key"])
                fits = bool(cur) and "/PATH/TO" not in cur and Path(cur).is_file() and S.GUIDE is not None \
                    and not S.GUIDE.problems(S.GUIDE.identify(cur), s["key"], req.get(s["key"], []))
                if new and (all_.value or not fits) and new != cur:
                    set_slot(s["key"], new)
                    changed.append(f"{SSET.slot_label(s['key'])}: {Path(new).name}")
            msg.content = ("Filled " + "; ".join(changed) + ". Press Save to keep them.") if changed else \
                "Nothing to change: the slots already hold fitting files."

        async def online():
            base = st["slots"][0]["file"].value if st["slots"] else ""
            online_md.content = "Asking Hugging Face and Civitai …"
            online_md.content = await run.io_bound(SSET.st_online, st["tkey"], base, lambda *a, **k: None)
        with ui.row().classes("gap-2"):
            ui.button("Apply suggestions", on_click=apply).props("outline dense no-caps")
            ui.button("Check again", on_click=suggest).props("flat dense no-caps")
            ui.button("Look up online (Hugging Face, Civitai)", on_click=online).props("flat dense no-caps")
    with ui.expansion("LoRA subtypes (for example Pony: its own model files and file tag)").classes("w-full ls-card"):
        note("A subtype trains exactly like its base type (same trainer, flags and recommended settings) but has its own "
             "model files and file tag. Saved as a section such as [pony] with base = sdxl, label and tag.")
        sub_md = ui.markdown(SSET.sub_list_md())
        with ui.row().classes("w-full items-end gap-2"):
            sname = ui.input("Name (e.g. Pony)").props("outlined dense")
            sbase = ui.select({"sdxl": "SDXL", "anima": "Anima"}, value="sdxl", label="Based on").props("outlined dense")
            stag = ui.input("File tag (empty = the name in capitals)").props("outlined dense")

            def add():
                r = SSET.sub_add(sname.value, sbase.value, stag.value)
                smsg.content, sub_md.content = r, SSET.sub_list_md()
                after_subs()
            ui.button("Add subtype", on_click=add).props("unelevated dense no-caps color=primary")
        with ui.row().classes("w-full items-end gap-2"):
            srm = ui.select(dict((v, lab) for lab, v in SSET.sub_choices()), label="Subtype to remove") \
                .props("outlined dense").classes("w-72")

            async def remove():
                if srm.value and await confirm(f"Remove the subtype {srm.value} from settings.txt?", "Remove"):
                    smsg.content = SSET.sub_remove(srm.value, True)
                    sub_md.content = SSET.sub_list_md()
                    after_subs()
            ui.button("Remove subtype", on_click=remove).props("flat dense no-caps color=negative")
        smsg = ui.markdown("")

        def after_subs():
            tsel.options = dict((v, lab) for lab, v in S.type_choices())
            tsel.update()
            srm.options = dict((v, lab) for lab, v in SSET.sub_choices())
            srm.update()
    with ui.expansion(f"{Path(S.settings_path()).name} as it is on disk").classes("w-full ls-card"):
        raw = ui.textarea(value=SSET.st_raw()).props("readonly outlined rows=14 input-style='font-family:monospace'") \
            .classes("w-full")
    load(st["tkey"])


# --------------------------------------------------------------------------- Speed test
def panel_speed(page):
    import speedjobs as SJ
    note("Tick the LoRA types to test. Each one trains a few steps on an invented test dataset (20 plain pictures; "
         "none of your datasets is used) with different settings: fewer swapped blocks, pinned memory, no gradient "
         "checkpointing, other attention, fp8, torch.compile and ROCm speed settings. It measures seconds per step "
         "and video memory. Test some types now and others later; the results stay. When a type has faster "
         "settings, tick Use the faster settings below to make every new launch script of that type use them. "
         "Close ComfyUI before a test; Ollama is stopped by itself.")
    table = ui.html(SJ.table_html(), sanitize=False).classes("w-full")

    def tested():
        return {t["key"]: t["label"] for t in SJ.types() if t["result"]}
    with card():
        ready = {t["key"]: t["label"] for t in SJ.types() if t["ready"]}
        pick = ui.select(ready, multiple=True, label="Test these LoRA types").props("outlined dense use-chips") \
            .classes("w-full")
        depth = ui.radio({v: k for k, v in SJ.DEPTHS}, value="quick").props("dense")
        msg = ui.markdown("")
        with ui.row().classes("gap-2"):
            ui.button("Start the speed test", on_click=lambda: (setattr(msg, "content", SJ.start(
                list(pick.value or []), depth.value or "quick")), follow())).props("unelevated dense no-caps")
            ui.button("Stop", on_click=lambda: setattr(msg, "content", SJ.stop())) \
                .props("outline dense no-caps color=negative")
        live = log_box("Progress", 12)
        live.set_visibility(bool(SJ.JOB["lines"]))
    with card():
        ui.label("Results").classes("font-bold")
        t0 = tested()
        which = ui.select(t0, label="LoRA type", value=next(iter(t0), None)).props("outlined dense").classes("w-80")
        use = ui.checkbox("Use the faster settings for new launch scripts of this type")
        note("Existing launch scripts stay as they are; make a new version in step 9. Untick to go back to the "
             "normal settings.")
        use_msg = ui.markdown("")
        rep = ui.markdown("").classes("w-full")

    def show(k):
        busy["set"] = True
        if not k:
            use.value, rep.content = False, ""
            use.disable()
        else:
            text, can = SJ.report_of(k)
            use.value, rep.content = SJ.use_on(k), text
            use.enable() if can else use.disable()
        busy["set"] = False
    busy = {"set": False}

    def use_changed(e):
        if busy["set"] or not which.value:
            return
        use_msg.content = SJ.set_use(which.value, bool(e.value))
        table.content = SJ.table_html()
    which.on_value_change(lambda e: show(e.value))
    use.on_value_change(use_changed)
    show(which.value)

    def follow():
        if not (SJ.JOB["running"] or SJ.JOB["done"] or SJ.JOB["lines"]):
            return
        live.set_visibility(True)
        live.value = SJ.live_text()
        table.content = SJ.table_html()
        t = tested()
        if t != which.options:
            which.set_options(t)
        cur = which.value if which.value in t else (SJ.JOB["done"][-1] if SJ.JOB["done"] else next(iter(t), None))
        if cur != which.value:
            which.value = cur
    was = {"run": SJ.JOB["running"]}

    def tick():
        if SJ.JOB["running"] or was["run"]:              # while it runs, and once more after it ended
            follow()
        was["run"] = SJ.JOB["running"]
    ui.timer(2.0, tick)
