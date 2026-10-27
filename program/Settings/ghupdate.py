# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Updates of LoRA Studio straight from GitHub (Settings › Updates › From GitHub), on a button.

The newest upload to the main branch of github.com/druidican/LoraStudio is downloaded as a zip and installed exactly
like a LoraStudio.zip (selfupdate.install): only the program files that differ change, the old ones are kept in
update_backups/ first, the self-test runs, and when it fails the old files go back by themselves.

Check GitHub asks only for the newest commit and its version.py (small). The commit LoRA Studio was last updated
from is remembered in userdata/github_update.json, so a new upload with the same version number is still seen.

While the repository is private, a GitHub token is needed: a fine-grained token with read access to the repository's
Contents (github.com › Settings › Developer settings › Personal access tokens), saved in userdata/github_token.txt,
readable only by you. A public repository needs none.
"""
from __future__ import annotations
__version__ = "1.261026z"  # LoRA Studio version, see version.py

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import lspaths as _LP

API = os.environ.get("LS_GITHUB_API", "https://api.github.com").rstrip("/")
REPO = "druidican/LoraStudio"
BRANCH = "main"
TOKEN_FILE = _LP.USER / "github_token.txt"
STATE_FILE = _LP.USER / "github_update.json"
DOWNLOADS = _LP.USER / "update_backups" / "downloads"


# --------------------------------------------------------------------------- settings and token
def state() -> dict:
    try:
        d = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(**kw):
    d = state()
    d.update(kw)
    try:
        STATE_FILE.write_text(json.dumps(d, indent=1), encoding="utf-8")
    except OSError:
        pass


def repo() -> str:
    return (state().get("repo") or REPO).strip().strip("/")


def branch() -> str:
    return (state().get("branch") or BRANCH).strip()


def read_token() -> str:
    try:
        for line in TOKEN_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                return line.strip()
    except OSError:
        pass
    return ""


def save_token(tok: str) -> str:
    tok = (tok or "").strip()
    if not tok:
        try:
            TOKEN_FILE.unlink()
        except OSError:
            pass
        return "GitHub token removed."
    if not re.fullmatch(r"[A-Za-z0-9_]{20,255}", tok):
        return "❌ That does not look like a GitHub token (ghp_… or github_pat_…)."
    TOKEN_FILE.write_text(tok + "\n", encoding="utf-8")
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass
    return f"✔ GitHub token saved ({mask(tok)}), readable only by you."


def mask(tok: str) -> str:
    return f"{tok[:6]}…{tok[-4:]}" if len(tok or "") > 14 else ("•" * len(tok or ""))


# --------------------------------------------------------------------------- talking to GitHub
def _get(url: str, accept="application/vnd.github+json", timeout=30):
    h = {"Accept": accept, "User-Agent": "LoRA-Studio", "X-GitHub-Api-Version": "2022-11-28"}
    tok = read_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def _why(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 404:
            return ("the repository was not found: while it is private, a GitHub token with read access to it is "
                    "needed (below)" if not read_token() else
                    "the repository or branch was not found, or the token has no access to it")
        if e.code == 401:
            return "GitHub did not accept the token (expired or mistyped?)"
        if e.code == 403:
            return "GitHub refused (too many requests without a token, or the token lacks read access): try later"
        return f"GitHub answered HTTP {e.code}"
    return f"GitHub cannot be reached ({getattr(e, 'reason', e)})"


def check() -> dict:
    """{ok, sha, date, message, version, have, newer, text}: the newest upload against what is installed."""
    import selfupdate as SU
    import version as V
    r, b = repo(), branch()
    try:
        c = json.loads(_get(f"{API}/repos/{r}/commits/{urllib.parse.quote(b)}").read().decode("utf-8"))
        sha = c["sha"]
        date = ((c.get("commit") or {}).get("committer") or {}).get("date", "")[:16].replace("T", " ")
        msg = ((c.get("commit") or {}).get("message") or "").splitlines()[0][:160]
        src = _get(f"{API}/repos/{r}/contents/version.py?ref={sha}", accept="application/vnd.github.raw").read()
        ver = SU._literal(src.decode("utf-8", "replace"), "VERSION") or "?"
    except Exception as e:                               # noqa: BLE001
        return {"ok": False, "text": f"❌ {_why(e)}."}
    last = state().get("sha", "")
    newer = SU._vkey(ver) > SU._vkey(V.VERSION)
    same = last == sha
    older = SU._vkey(ver) < SU._vkey(V.VERSION)
    if same:
        txt = f"✔ You have the newest upload: {ver} ({sha[:7]}, {date})."
    elif older:
        txt = (f"The newest upload on GitHub is **{ver}** ({sha[:7]}, {date}), older than your **{V.VERSION}**. "
               "Installing it goes back.")
    elif newer:
        txt = f"**A newer version is on GitHub: {ver}** ({sha[:7]}, {date}) · you have {V.VERSION}."
    else:
        txt = (f"**A new upload of {ver}** is on GitHub ({sha[:7]}, {date})"
               + (" · you were last updated from " + last[:7] if last else
                  " · LoRA Studio does not know which upload you have: Install takes only the files that differ") + ".")
    if msg:
        txt += f"\n\nLast change: *{msg}*"
    return {"ok": True, "sha": sha, "date": date, "message": msg, "version": ver, "have": V.VERSION, "newer": newer,
            "same": same, "older": older, "text": txt}


def download(sha: str, say=print) -> Path | None:
    """The repository at this commit as a zip, in update_backups/downloads/ (a few MB)."""
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    out = DOWNLOADS / f"LoraStudio-github-{sha[:7]}.zip"
    if out.is_file() and zipfile.is_zipfile(out):
        return out
    say(f"== Downloading {repo()} at {sha[:7]} from GitHub …")
    tmp = out.with_suffix(".part")
    try:
        with _get(f"{API}/repos/{repo()}/zipball/{sha}", accept="application/vnd.github+json", timeout=120) as r, \
                open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    except Exception as e:                               # noqa: BLE001
        say(f"❌ {_why(e)}.")
        tmp.unlink(missing_ok=True)
        return None
    if not zipfile.is_zipfile(tmp):
        say("❌ GitHub sent something that is not a zip file.")
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(out)
    say(f"   {out.stat().st_size / 1e6:.1f} MB")
    return out


def install(say=print) -> bool:
    """Check, download and install the newest upload (selfupdate.install does the safe part)."""
    import selfupdate as SU
    c = check()
    if not c.get("ok"):
        say(c["text"])
        return False
    say(re.sub(r"[*]", "", c["text"]).split("\n")[0])
    z = download(c["sha"], say)
    if z is None:
        return False
    try:
        pl = SU.plan(z)
    except Exception as e:                               # noqa: BLE001
        say(f"❌ {e}")
        return False
    if not any(r["state"] in ("changed", "new") for r in pl["rows"]):
        save_state(sha=c["sha"], at=time.strftime("%Y-%m-%d %H:%M"), version=c["version"])
        say(f"✔ Nothing to install: your files are the ones of this upload ({c['sha'][:7]}).")
        return True
    ok = SU.install(z, say=say)
    if ok:
        save_state(sha=c["sha"], at=time.strftime("%Y-%m-%d %H:%M"), version=c["version"])
    return ok
