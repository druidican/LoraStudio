# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The login page (--auth user:password, or Settings › Start and network) and the key that signs its cookie."""
from __future__ import annotations
__version__ = "1.191126h"  # LoRA Studio version, see version.py

import hmac
import secrets

import lspaths


def storage_secret() -> str:
    """The key that signs the browser's login cookie (made once, kept in userdata)."""
    f = lspaths.USER / ".session_key"
    try:
        return f.read_text().strip()
    except OSError:
        k = secrets.token_hex(32)
        lspaths.USER.mkdir(exist_ok=True)
        f.write_text(k)
        f.chmod(0o600)
        return k


ON = {"auth": False}


def allowed() -> bool:
    """The browser asking may see the program: no login is set, or it has logged in (also for plain routes)."""
    if not ON["auth"]:
        return True
    from nicegui import app
    try:
        return bool(app.storage.user.get("ok"))
    except Exception:                                    # noqa: BLE001
        return False


def install(user: str, pw: str):
    """A login page at /login; the browser keeps a signed cookie, so a reload does not ask again. /login?next=/m
    goes back to the phone page after the login."""
    from nicegui import app, ui
    ON["auth"] = True

    @ui.page("/login")
    def login(next: str = "/"):                         # noqa: A002 (the query parameter's name)
        back = next if next.startswith("/") and not next.startswith("//") else "/"

        def check():
            if hmac.compare_digest(u.value, user) and hmac.compare_digest(p.value, pw):
                app.storage.user["ok"] = True
                ui.navigate.to(back)
            else:
                msg.text = "Wrong user name or password."
        ui.query("body").style("background:#1C1F24")
        with ui.card().classes("absolute-center w-80 items-stretch"):
            logo = lspaths.PROG / "logo" / "lorastudio_login.jpg"
            if logo.is_file():
                ui.image(logo).classes("rounded")
            ui.label("LoRA Studio").classes("text-xl font-bold text-center")
            u = ui.input("User name").props("outlined dense")
            p = ui.input("Password", password=True).props("outlined dense").on("keydown.enter", check)
            ui.button("Log in", on_click=check).props("unelevated no-caps color=primary")
            msg = ui.label("").classes("text-red-600 text-sm")
