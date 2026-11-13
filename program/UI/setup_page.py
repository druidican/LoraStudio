# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The setup page (first start, or ./start.sh --setup) and the notice shown once before anything else.
The questions and what is written are in firstrun.py; this file only draws them."""
from __future__ import annotations
__version__ = "1.131126b"  # LoRA Studio version, see version.py

import time

from nicegui import run, ui

import firstrun as FR
from firstrun import STEPS, expand, lines, tilde

RELOAD_JS = ("setTimeout(function wait(){ fetch(location.href, {cache: 'no-store'}).then(function(r){ "
             "if (r.ok) location.reload(); else setTimeout(wait, 2000); }).catch(function(){ setTimeout(wait, 2000); }); "
             "}, 6000);")


def notice_gate():
    """The notice (who owns sd-scripts / musubi-tuner …) over the page, until it is confirmed once."""
    import notice
    if notice.accepted():
        return
    with ui.dialog().props("persistent maximized") as d, ui.card().classes("items-center w-full"):
        with ui.column().classes("max-w-3xl w-full gap-3 ls-page"):
            logo = notice.logo_data_url()
            if logo:
                ui.image(logo).classes("w-full max-w-md self-center rounded-lg")
            ui.markdown(f"## {notice.TITLE}\n\n{notice.TEXT}\n\nThis page is shown once. You can read it again in "
                        "Settings › Version › Licences and credits, and in the NOTICE file in the program folder.")
            ok = ui.checkbox("I have read this and agree")
            go = ui.button("Continue to LoRA Studio", on_click=lambda: (notice.accept("page"), d.close())) \
                .props("unelevated no-caps")
            go.bind_enabled_from(ok, "value")
    d.open()


def build(hint: dict):
    import backup as BK
    ui.add_css(".fr-steps{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 14px}"
               ".fr-step{padding:3px 11px;border-radius:14px;font-size:13px;border:1px solid var(--ls-line);opacity:.65}"
               ".fr-step.cur{opacity:1;font-weight:600;border-color:var(--ls-accent,#C98A1B)}"
               ".fr-step.done{opacity:.9}")
    notice_gate()
    st = {"i": 0}
    v = {}                                               # the answers, by firstrun.write_all's names

    def bar():
        return '<div class="fr-steps">' + "".join(
            f'<span class="fr-step{" cur" if k == st["i"] else " done" if k < st["i"] else ""}">'
            f'{f"{k}. " if k else ""}{n}</span>' for k, n in enumerate(STEPS)) + "</div>"

    with ui.column().classes("ls-page w-full max-w-4xl mx-auto gap-3 p-6"):
        ui.label("LoRA Studio setup").classes("text-2xl font-bold")
        steps = ui.html(bar(), sanitize=False)
        cols = []

        def step_col():
            c = ui.column().classes("w-full gap-2")
            cols.append(c)
            return c

        with step_col():                                 # 0 welcome
            ui.markdown("Welcome! A few questions so LoRA Studio knows where your things are. **Every step can be "
                        "skipped** (press *Skip*) and everything can be changed later in **Settings**.\n\n"
                        "Moving from another computer or install? **Import a backup** (made with Settings › Backup) "
                        "and skip all the steps.")
            with ui.expansion("Import from a backup", value=bool(hint["backups"])).classes("w-full"):
                opts = {p: label for label, p in hint["backups"]}
                bk_pick = ui.select(opts, label="Backups found on this computer",
                                    value=next(iter(opts), None)).props("outlined dense").classes("w-full")
                bk_path = ui.input("Or the path of a backup .zip",
                                   placeholder="~/LoraStudio-backups/LoraStudio-backup-....zip").props("outlined dense") \
                    .classes("w-full")
                bk_info = ui.markdown(BK.describe(bk_pick.value) if bk_pick.value else "")
                bk_pick.on_value_change(lambda e: setattr(bk_info, "content", BK.describe(e.value) if e.value else ""))
                bk_path.on("blur", lambda: setattr(bk_info, "content", BK.describe(bk_path.value.strip())
                                                   if bk_path.value.strip() else ""))
                bk_msg = ui.markdown("")

                def do_import():
                    src = (bk_path.value or "").strip() or bk_pick.value
                    if not src:
                        bk_msg.content = "Pick a backup or enter the path of one."
                        return
                    try:
                        text = BK.restore(src)
                    except Exception as e:               # noqa: BLE001
                        bk_msg.content = f"❌ {e}"
                        return
                    BK.restart_program()
                    bk_msg.content = text + "\n\n**Starting LoRA Studio** with your restored settings; this page " \
                                            "reloads by itself."
                    ui.run_javascript(RELOAD_JS)
                ui.button("Import this backup and start LoRA Studio", on_click=do_import).props("unelevated no-caps")

        with step_col():                                 # 1 trainers
            ui.markdown("### 1. Trainers\nThe training programs. `install.sh` puts both in the LoRA Studio folder; "
                        "only change these if yours are somewhere else.")
            v["sd"] = ui.input("sd-scripts folder (SDXL, Illustrious, Pony, Anima, Flux, ...)",
                               value=tilde(hint["sd"])).props("outlined dense").classes("w-full")
            v["mu"] = ui.input("musubi-tuner folder (Z-Image, Qwen-Image, FLUX.2)",
                               value=tilde(hint["mu"])).props("outlined dense").classes("w-full")
            t_note = ui.markdown("")

        with step_col():                                 # 2 datasets
            ui.markdown("### 2. Datasets\nWhere your datasets are: one folder per LoRA, with its pictures in a repeat "
                        "folder like `10_mychar`. New datasets are made in the first folder. Several folders: one per "
                        "line.")
            v["datasets"] = ui.textarea("Dataset folder(s)", value="\n".join(
                tilde(x) for x in str(hint["datasets"]).splitlines())).props("outlined dense rows=2").classes("w-full")
            v["make_datasets"] = ui.checkbox("Create the folder if it does not exist", value=True)
            d_note = ui.markdown("")

        with step_col():                                 # 3 models
            ui.markdown("### 3. Models\nThe folder with your base models, text encoders and VAEs (for example "
                        "ComfyUI's `models` folder, with `checkpoints/`, `vae/`, `text_encoders/` ...). The file lists "
                        "in Settings › Model files start here; the files themselves are picked per LoRA type later.")
            v["models"] = ui.input("Model folder", value=tilde(hint["models"]) if hint["models"] else "",
                                   placeholder="e.g. ~/ComfyUI/models").props("outlined dense").classes("w-full")
            v["adapter"] = ui.checkbox(f"Download the Z-Image Turbo training adapter ({FR.ADAPTER_FILE}, about 0.3 GB) "
                                       "into <model folder>/loras/zimage (only needed to train Z-Image Turbo)")
            m_note = ui.markdown("")

        with step_col():                                 # 4 comfyui
            found = ", ".join(f"`{tilde(c)}`" for c in hint["comfy"]) or "none found"
            ui.markdown("### 4. ComfyUI\nYour finished LoRAs (with the LoRA Manager files) and the folder ComfyUI "
                        f"saves pictures in, for the **Collection**. ComfyUI installs found: {found}.")
            v["loras"] = ui.textarea("LoRA folder(s), one per line", value="\n".join(tilde(x) for x in hint["loras"])) \
                .props("outlined dense rows=2").classes("w-full")
            v["pictures"] = ui.input("ComfyUI picture folder (its output folder)",
                                     value=tilde(hint["pictures"]) if hint["pictures"] else "") \
                .props("outlined dense").classes("w-full")
            c_note = ui.markdown("")

        with step_col():                                 # 5 token
            ui.markdown("### 5. Hugging Face token\nOnly needed for gated models (for example FLUX.1-dev). Make one at "
                        "https://huggingface.co/settings/tokens (\"Read\" is enough). Saved in `huggingtoken.txt`, "
                        "readable only by you.")
            v["token"] = ui.input("Token (hf_...)", password=True, password_toggle_button=True) \
                .props("outlined dense").classes("w-full")
            k_note = ui.markdown("")

            async def test_token():
                t = (v["token"].value or "").strip()
                if not t:
                    k_note.content = "No token entered."
                    return

                def who():
                    from huggingface_hub import whoami
                    return whoami(token=t).get("name", "?")
                try:
                    k_note.content = f"✔ The token works: Hugging Face user **{await run.io_bound(who)}**"
                except Exception as e:                   # noqa: BLE001
                    k_note.content = f"✖ Could not check it ({type(e).__name__}); it is saved anyway when you continue."
            ui.button("Test the token", on_click=test_token).props("outline dense no-caps")

        with step_col():                                 # 6 ollama
            ui.markdown("### 6. Ollama (optional)\nA local Ollama with a vision model (for example `qwen2.5vl:7b`) "
                        "writes captions and checks dataset pictures. Without it, everything else works.")
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                v["ollama_url"] = ui.input("Ollama address", value=hint["ollama"]).props("outlined dense").classes("grow")
                look = ui.button("Look for models").props("outline dense no-caps")
            v["ollama_model"] = ui.select(hint["ollama_models"], label="Vision model for captions", new_value_mode="add",
                                          value=hint["ollama_models"][0] if hint["ollama_models"] else None) \
                .props("outlined dense").classes("w-full")
            o_note = ui.markdown(hint["ollama_note"])

            async def look_ollama():
                ms, n = await run.io_bound(FR.ollama_models, v["ollama_url"].value.strip())
                v["ollama_model"].set_options(ms, value=ms[0] if ms else None)
                o_note.content = n
            look.on_click(look_ollama)

        with step_col():                                 # 7 backups
            ui.markdown("### 7. Backups\nWhere **Settings › Backup** saves backups of your settings and LoRA Studio's "
                        "databases (not your datasets or models). Pick a folder that is copied off this computer, for "
                        "example a synced folder or a second drive.")
            v["backup"] = ui.input("Backup folder", value=tilde(hint["backup"])).props("outlined dense").classes("w-full")

        with step_col():                                 # 8 finish
            ui.markdown("### Summary")
            summary = ui.markdown("")
            fin_msg = ui.markdown("")
            fin = ui.button("Finish and start LoRA Studio").props("unelevated no-caps")

        with ui.row().classes("gap-2 mt-2"):
            back = ui.button("◀ Back").props("outline no-caps")
            skip = ui.button("Skip this step").props("flat no-caps")
            nxt = ui.button("Start the setup ▶").props("unelevated no-caps")

    def val(k):
        return v[k].value

    # ---- notes per step
    def note_trainers():
        out = []
        for name, p, probe in (("sd-scripts", val("sd"), "sdxl_train_network.py"),
                               ("musubi-tuner", val("mu"), "src/musubi_tuner")):
            d = expand(p) if p else None
            if not d:
                out.append(f"- {name}: skipped (set it later in Settings › Trainers and GPU)")
            elif (d / probe).exists():
                out.append(f"- {name}: ✔ found in `{tilde(d)}`")
            else:
                out.append(f"- {name}: ✖ not found in `{tilde(d)}` (run `./install.sh`, or enter the right folder)")
        return "\n".join(out)

    def note_datasets():
        out = []
        for d in lines(val("datasets")):
            p = expand(d)
            if p.is_dir():
                n = FR.count_datasets(p)
                out.append(f"- `{tilde(p)}`: ✔ {n} dataset{'s' if n != 1 else ''} with repeat folders")
            else:
                out.append(f"- `{tilde(p)}`: does not exist yet" + (" (it will be created)" if val("make_datasets") else ""))
        return "\n".join(out) or "Skipped: `~/LoraStudio-data/datasets` is used."

    def note_models():
        t = val("models")
        if not t:
            return "Skipped: you pick the model files per LoRA type in Settings › Model files."
        p = expand(t)
        if not p.is_dir():
            return f"✖ `{tilde(p)}` does not exist."
        subs = [n for n in ("checkpoints", "diffusion_models", "unet", "vae", "text_encoders", "clip", "loras")
                if (p / n).is_dir()]
        return f"✔ {FR.count_files(p)} model files in `{tilde(p)}`" + (f" (folders: {', '.join(subs)})" if subs else "")

    def note_comfy():
        out = []
        for d in lines(val("loras")):
            p = expand(d)
            out.append(f"- LoRAs `{tilde(p)}`: " + (f"✔ {FR.count_files(p, ('.safetensors',))} LoRA files"
                                                     if p.is_dir() else "✖ does not exist"))
        if val("pictures"):
            p = expand(val("pictures"))
            out.append(f"- Pictures `{tilde(p)}`: " + ("✔ found" if p.is_dir() else "✖ does not exist"))
        return "\n".join(out) or ("Skipped: the Collection works with your datasets only; set these later in "
                                  "Settings › Folders.")

    for k, el, fn in (("sd", t_note, note_trainers), ("mu", t_note, note_trainers), ("datasets", d_note, note_datasets),
                      ("make_datasets", d_note, note_datasets), ("models", m_note, note_models),
                      ("loras", c_note, note_comfy), ("pictures", c_note, note_comfy)):
        v[k].on("blur" if k not in ("make_datasets",) else "update:model-value",
                lambda _, el=el, fn=fn: setattr(el, "content", fn()))

    def make_summary():
        def row(k, x):
            return f"| {k} | {x} |"
        om = val("ollama_model")
        rows = ["| | |", "|---|---|",
                row("sd-scripts", f"`{val('sd')}`" if val("sd") else "*skipped*"),
                row("musubi-tuner", f"`{val('mu')}`" if val("mu") else "*skipped*"),
                row("Datasets", ", ".join(f"`{x}`" for x in lines(val("datasets"))) or "`~/LoraStudio-data/datasets`"),
                row("Models", (f"`{val('models')}`" if val("models") else "*skipped*")
                    + (" + Z-Image adapter download" if val("adapter") else "")),
                row("LoRA folders", ", ".join(f"`{x}`" for x in lines(val("loras"))) or "*skipped*"),
                row("ComfyUI pictures", f"`{val('pictures')}`" if val("pictures") else "*skipped*"),
                row("Hugging Face token", "entered" if (val("token") or "").strip() else "*skipped*"),
                row("Ollama", f"`{om}` at `{val('ollama_url')}`" if om else "*skipped*"),
                row("Backups", f"`{val('backup')}`" if val("backup") else "`~/LoraStudio-backups`")]
        return "\n".join(rows) + "\n\nEverything can be changed later in **Settings**."

    # ---- navigation
    last = len(cols) - 1

    def show(i):
        st["i"] = i = max(0, min(last, i))
        for k, c in enumerate(cols):
            c.set_visibility(k == i)
        steps.content = bar()
        back.set_visibility(i > 0)
        skip.set_visibility(0 < i < last)
        nxt.set_visibility(i < last)
        nxt.text = "Start the setup ▶" if i == 0 else "Next ▶"
        if i == 1:
            t_note.content = note_trainers()
        elif i == 2:
            d_note.content = note_datasets()
        elif i == 3:
            m_note.content = note_models()
        elif i == 4:
            c_note.content = note_comfy()
        elif i == last:
            summary.content = make_summary()

    clear_by_step = {1: ["sd", "mu"], 2: ["datasets"], 3: ["models"], 4: ["loras", "pictures"], 5: ["token"],
                     6: ["ollama_model"], 7: ["backup"]}

    def do_skip():
        """Skip = leave this step's answers empty, then go on."""
        for k in clear_by_step.get(st["i"], []):
            v[k].value = None if k == "ollama_model" else ""
        if st["i"] == 3:
            v["adapter"].value = False
        show(st["i"] + 1)
    nxt.on_click(lambda: show(st["i"] + 1))
    back.on_click(lambda: show(st["i"] - 1))
    skip.on_click(do_skip)

    async def do_finish():
        fin.disable()
        log: list[str] = []
        om = val("ollama_model")
        vals = {"sd": val("sd"), "mu": val("mu"), "datasets": val("datasets"), "make_datasets": val("make_datasets"),
                "models": val("models"), "loras": val("loras"), "pictures": val("pictures"), "token": val("token"),
                "ollama_url": val("ollama_url") if om else "", "ollama_model": om, "backup": val("backup")}
        try:
            await run.io_bound(FR.write_all, vals, log)
            fin_msg.content = "  \n".join(log)
            if val("adapter"):
                fin_msg.content = "  \n".join(log + ["Downloading the Z-Image Turbo training adapter ..."])
                await run.io_bound(FR.download_adapter, val("models"), log)
            FR.DONE.write_text(f"setup page finished {time.strftime('%Y-%m-%d %H:%M')}\n")
        except Exception as e:                           # noqa: BLE001
            log.append(f"❌ {e!r}")
            fin_msg.content = "  \n".join(log)
            fin.enable()
            return
        log.append("\n**Starting LoRA Studio** ... this page reloads by itself in a few seconds.")
        fin_msg.content = "  \n".join(log)
        BK.restart_program()
        ui.run_javascript(RELOAD_JS)
    fin.on_click(do_finish)
    show(0)
