# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Settings profiles (step 9): training settings for one LoRA type, to share with others and to import.

A profile is a small TOML file:

    [profile]
    name = "Flux portrait – sharp faces"
    type = "flux1dev"
    author = "Mike"
    note = "Tested on 40 portrait datasets"

    [character]
    learning_rate = 8e-5
    network_dim = 32

Sections per purpose: character, style, concept (objects, worn items and actions use concept). Only the keys in KEYS
are read, each checked against its kind and range: an imported value never becomes shell text it was not meant to be
(launch scripts are bash). Unknown keys and refused values are named when importing. Model files, memory settings
(block swap, fp8, batch size), paths and tokens are never in a profile.

"Original" is LoRA Studio's own settings (no file, always there). Imported and exported profiles are kept in
userdata/profiles/<type>/<name>.toml; the profile chosen for a dataset and type is in the dataset's meta
("profile": {type: name}). make_plan applies it after the recommendations (apply_cfg); for a saved script,
apply_text writes the values into the script text.
"""
from __future__ import annotations
__version__ = "1.261126b"  # LoRA Studio version, see version.py

import json
import re
import time
import tomllib
from pathlib import Path

import lspaths as LP

DIR = LP.USER / "profiles"
ORIGINAL = ""                                          # the key of LoRA Studio's own settings
PURPOSES = ("character", "style", "concept")
WORD = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,47}$")   # optimizer, scheduler, sampling names: one safe shell word

# key: (where, kind, range or None, label). where: "cfg:<key>" (the plan) or "--<flag>" (the trainer's option)
KEYS = {
    "learning_rate":           ("cfg:lr", float, (1e-7, 0.1), "learning rate"),
    "text_encoder_lr":         ("--text_encoder_lr", float, (0.0, 0.1), "text encoder learning rate"),
    "network_train_unet_only": ("--network_train_unet_only", bool, None, "train only the model (not the text encoders)"),
    "network_dim":             ("--network_dim", int, (1, 1024), "rank (network dim)"),
    "network_alpha":           ("--network_alpha", float, (0.01, 1024), "alpha"),
    "optimizer_type":          ("cfg:optimizer", "word", None, "optimizer"),
    "lr_scheduler":            ("cfg:scheduler", "word", None, "scheduler"),
    "lr_warmup_steps":         ("cfg:warmup", int, (0, 100000), "warm-up steps"),
    "max_grad_norm":           ("--max_grad_norm", float, (0.0, 100.0), "max grad norm"),
    "noise_offset":            ("--noise_offset", float, (0.0, 1.0), "noise offset"),
    "min_snr_gamma":           ("--min_snr_gamma", float, (0.0, 50.0), "min SNR gamma"),
    "discrete_flow_shift":     ("--discrete_flow_shift", float, (0.0, 50.0), "flow shift"),
    "timestep_sampling":       ("--timestep_sampling", "word", None, "timestep sampling"),
    "weighting_scheme":        ("--weighting_scheme", "word", None, "weighting scheme"),
    "caption_dropout_rate":    ("--caption_dropout_rate", float, (0.0, 1.0), "caption dropout"),
    "steps":                   ("cfg:steps", int, (50, 100000), "target steps"),
    "epochs":                  ("cfg:epochs", int, (1, 500), "epochs"),
}
TEXT_FLAG = {"cfg:lr": "--learning_rate", "cfg:optimizer": "--optimizer_type", "cfg:scheduler": "--lr_scheduler",
             "cfg:warmup": "--lr_warmup_steps", "cfg:epochs": "--max_train_epochs"}


def purpose_key(purpose: str) -> str:
    return "concept" if purpose in ("object", "worn", "action", "concept") else (
        purpose if purpose in PURPOSES else "character")


# --------------------------------------------------------------------------- checking
def check(key: str, value) -> tuple[object, str]:
    """(the value as the program uses it, '') or (None, why it is refused)."""
    if key not in KEYS:
        return None, "not a setting LoRA Studio knows"
    _, kind, rng, _ = KEYS[key]
    if kind is bool:
        if isinstance(value, bool):
            return value, ""
        return None, "must be true or false"
    if kind == "word":
        if isinstance(value, str) and WORD.match(value):
            return value, ""
        return None, "must be one plain name (letters, digits, _ . -)"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "must be a number"
    if kind is int and float(value) != int(value):
        return None, "must be a whole number"
    v = kind(value)
    if rng and not (rng[0] <= v <= rng[1]):
        return None, f"must be between {rng[0]:g} and {rng[1]:g}"
    return v, ""


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _toml(v) -> str:
    return json.dumps(v) if isinstance(v, str) else _fmt(v)


def parse(text: str) -> tuple[dict, list[str]]:
    """A profile file → ({"name", "type", "author", "note", "made", "settings": {purpose: {key: value}}}, problems).
    Raises ValueError when it is not a profile at all."""
    try:
        d = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"not a TOML file ({e})") from e
    head = d.get("profile")
    if not isinstance(head, dict) or not str(head.get("name") or "").strip() or not str(head.get("type") or "").strip():
        raise ValueError("no [profile] header with a name and a LoRA type")
    out = {"name": str(head["name"]).strip()[:80], "type": str(head["type"]).strip()[:40],
           "author": str(head.get("author") or "")[:80], "note": str(head.get("note") or "")[:400],
           "made": str(head.get("made") or "")[:20], "settings": {}}
    if not WORD.match(out["type"]):
        raise ValueError(f"the LoRA type {out['type']!r} is not a type name")
    problems = []
    for sec, vals in d.items():
        if sec == "profile":
            continue
        if sec not in PURPOSES or not isinstance(vals, dict):
            problems.append(f"[{sec}] skipped: the sections are {', '.join(PURPOSES)}")
            continue
        for k, v in vals.items():
            good, why = check(k, v)
            if why:
                problems.append(f"[{sec}] {k} = {v!r} skipped: {why}")
            else:
                out["settings"].setdefault(sec, {})[k] = good
    if not out["settings"]:
        raise ValueError("no usable settings in it" + (": " + "; ".join(problems[:4]) if problems else ""))
    return out, problems


def to_toml(p: dict) -> str:
    """The profile as a TOML file (json.dumps writes TOML-compatible strings)."""
    import version
    lines = ["# LoRA Studio settings profile: import it in a dataset's Train part (step 9) > Settings profile.",
             "[profile]"]
    for k in ("name", "type", "type_label", "author", "note", "made"):
        if p.get(k):
            lines.append(f"{k} = {json.dumps(str(p[k]), ensure_ascii=False)}")
    lines.append(f"lorastudio = {json.dumps(version.VERSION)}")
    for sec in PURPOSES:
        vals = (p.get("settings") or {}).get(sec)
        if vals:
            lines += ["", f"[{sec}]"] + [f"{k} = {_toml(v)}" for k, v in vals.items() if k in KEYS]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the store
def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")[:60] or "profile"


def _path(tkey: str, name: str) -> Path:
    return DIR / tkey / f"{_slug(name)}.toml"


def names(tkey: str) -> list[str]:
    """The profiles kept for a type (by their names), Original not included."""
    out = []
    for f in sorted((DIR / tkey).glob("*.toml")) if (DIR / tkey).is_dir() else []:
        try:
            out.append(parse(f.read_text(encoding="utf-8"))[0]["name"])
        except (OSError, ValueError):
            continue
    return out


def load(tkey: str, name: str) -> dict | None:
    if not name:
        return None
    try:
        return parse(_path(tkey, name).read_text(encoding="utf-8"))[0]
    except (OSError, ValueError):
        return None


def save(tkey: str, p: dict) -> Path:
    f = _path(tkey, p["name"])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(to_toml({**p, "type": tkey}), encoding="utf-8")
    return f


def delete(tkey: str, name: str) -> str:
    try:
        _path(tkey, name).unlink()
    except OSError as e:
        return f"❌ {e}"
    return f"✔ Profile {name!r} deleted. Datasets that used it go back to Original."


def import_text(text: str, tkey: str, new_name: str) -> str:
    """Keeps an imported profile under the name you chose (Original and your other profiles stay)."""
    try:
        p, problems = parse(text)
    except ValueError as e:
        return f"❌ Not imported: {e}."
    if p["type"] != tkey:
        return (f"❌ Not imported: this profile is for the LoRA type `{p['type']}`, this launch script is "
                f"`{tkey}`. Import it in a dataset with that type.")
    name = (new_name or "").strip()[:80] or p["name"]
    if name.lower() == "original":
        return "❌ Original is LoRA Studio's own settings: choose another name."
    p = {**p, "name": name, "orig_name": p["name"]}
    if p["name"] != p["orig_name"] and not p["note"]:
        p["note"] = f"Imported as {name!r} (its own name: {p['orig_name']!r})."
    save(tkey, p)
    purp = ", ".join(p["settings"])
    return (f"✔ Imported as **{name}** ({purp}). Pick it under Settings profile."
            + ("\n\nSkipped:\n" + "\n".join(f"- {x}" for x in problems) if problems else ""))


# --------------------------------------------------------------------------- reading a script (export)
def _sg():
    import sys
    return sys.modules.get("sdgui") or __import__("sdgui")


def from_script(text: str, purpose: str) -> dict:
    """The profile settings in a launch script's text, for one purpose: {key: value}. Read with the same helpers as
    step 9's form (sdgui.form_value)."""
    S = _sg()
    out = {}
    for k, (where, kind, _, _) in KEYS.items():
        if where == "cfg:steps":
            m = re.search(r"\(target (\d+)\)", text)
            if m:
                out[k] = int(m.group(1))
            continue
        name = TEXT_FLAG.get(where, where).lstrip("-")
        if kind is bool:
            out[k] = re.search(rf"(?<![\w-])--{name}(?![\w-])", text) is not None
            continue
        v = S.form_value(text, name)
        if not v:
            continue
        try:
            raw = float(v) if kind in (int, float) else v
        except ValueError:
            continue
        good, why = check(k, raw)
        if not why:
            out[k] = good
    return out


def export_text(text: str, tkey: str, type_label: str, purpose: str, name: str, author: str = "",
                note: str = "") -> tuple[str, str]:
    """(TOML text, file name) of this script's settings; also kept as a profile of yours."""
    p = {"name": name.strip()[:80] or "My settings", "type": tkey, "type_label": type_label,
         "author": author.strip()[:80], "note": note.strip()[:400], "made": time.strftime("%Y-%m-%d"),
         "settings": {purpose_key(purpose): from_script(text, purpose)}}
    save(tkey, p)
    return to_toml(p), f"{_slug(p['name'])}.toml"


# --------------------------------------------------------------------------- applying
def chosen(ds, tkey: str) -> str:
    import dsflow as F
    return ((F.meta(ds).get("profile") or {}).get(tkey) or "") if ds else ""


def choose(ds, tkey: str, name: str):
    import dsflow as F
    m = F.meta(ds)
    m.setdefault("profile", {})[tkey] = name or ""
    F.meta_save(ds, m)


def _set_flag(flags: list, flag: str, v) -> list:
    pat = rf"(?:^|\s){re.escape(flag)}(?:[ =]\S+)?(?=\s|$)"
    if isinstance(v, bool):
        flags = [re.sub(pat, " ", f).strip() for f in flags]
        return [f for f in flags if f] + ([flag] if v else [])
    if any(re.search(pat, f) for f in flags):
        return [re.sub(pat, lambda m: (" " if m.group(0)[:1].isspace() else "") + f"{flag} {_fmt(v)}", f) for f in flags]
    return flags + [f"{flag} {_fmt(v)}"]


def apply_cfg(cfg: dict, p: dict | None, purpose: str) -> tuple[dict, list[str]]:
    """The plan with the profile's values for this purpose: (cfg, notes)."""
    vals = ((p or {}).get("settings") or {}).get(purpose_key(purpose)) or {}
    if not vals:
        return cfg, ([f"Settings profile {p['name']!r} has no settings for {purpose_key(purpose)} LoRAs: LoRA Studio's "
                      "own are used."] if p else [])
    cfg = dict(cfg)
    flags = list(cfg.get("flags", []))
    for k, v in vals.items():
        where = KEYS[k][0]
        if where == "cfg:steps":
            cfg["steps"], cfg["targets"] = v, (round(v * 0.7), v, round(v * 1.3))
        elif where.startswith("cfg:"):
            cfg[where[4:]] = _fmt(v) if where == "cfg:lr" else v
        else:
            flags = _set_flag(flags, where, v)
    cfg["flags"] = flags
    return cfg, [f"Settings profile {p['name']!r}: " + ", ".join(f"{KEYS[k][3]} {_fmt(v)}" for k, v in vals.items())]


def apply_text(text: str, p: dict | None, purpose: str) -> str:
    """A saved script's text with the profile's values (target steps only change a new plan); written with the same
    helpers as step 9's form (sdgui.set_arg / set_switch)."""
    S = _sg()
    vals = ((p or {}).get("settings") or {}).get(purpose_key(purpose)) or {}
    for k, v in vals.items():
        where = KEYS[k][0]
        if where == "cfg:steps":
            continue
        name = TEXT_FLAG.get(where, where).lstrip("-")
        text = S.set_switch(text, name, v)[0] if isinstance(v, bool) else S.set_arg(text, name, _fmt(v))[0]
    return mark(text, p)


def mark(text: str, p: dict | None) -> str:
    """The `# lorastudio-profile:` comment (which profile made this script)."""
    text = re.sub(r"^# lorastudio-profile: .*\n", "", text, flags=re.M)
    if p:
        text = re.sub(r"^(#!/bin/bash\n)", lambda m: m.group(1) + f"# lorastudio-profile: {p['name']}\n", text, count=1)
    return text
