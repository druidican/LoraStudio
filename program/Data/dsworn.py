# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""dsworn.py - datasets of a worn item (clothes, equipment, jewelry): the kinds and where they are worn, the item found in
each picture (item_judge) and close-ups where it is small (add_closeups), Make the words the same, the balance check,
and weight per kind (kinds_block: the pictures linked into .lsflow/kinds_view/ in the launch script).
"""
from __future__ import annotations
__version__ = "0"  # LoRA Studio version, see version.py

import re
from pathlib import Path

import dscaptions
import dsflow as F
import dssort


# ---- worn items: the look words, the kinds and where they are worn
WORN_LOOK = {
    "jewelry": {"gold", "golden", "silver", "metal", "gem", "gems", "gemstone", "jewel", "jewels", "diamond", "pearl",
                "pearls", "ruby", "emerald", "sapphire", "amethyst", "crystal", "crystals", "ornate", "floral", "flower",
                "engraved", "filigree", "chain", "beads", "beaded", "enamel", "platinum", "brass", "copper", "rhinestone",
                "rhinestones", "sparkle", "shiny", "leaf", "star", "heart", "moon", "butterfly", "ribbon", "bow"},
    "clothes": {"plaid", "striped", "stripes", "polka", "dot", "print", "floral", "lace", "frills", "frilled", "ruffles",
                "leather", "denim", "silk", "satin", "wool", "knit", "knitted", "embroidered", "embroidery", "buttons",
                "zipper", "pleated", "pattern", "checkered", "sequins", "fur", "velvet", "trim", "cutout", "ribbon", "bow"},
    "equipment": {"metal", "leather", "plate", "plates", "straps", "strap", "buckle", "buckles", "pouch", "pouches",
                  "rivets", "trim", "ornate", "engraved", "scratched", "worn", "brass", "steel", "chainmail", "padded",
                  "gold", "silver", "emblem", "insignia", "glowing"},
}
KIND_WORDS = {
    "jewelry": ["ring", "earrings", "necklace", "bracelet", "hair clip", "hair ornament", "brooch", "piercing",
                "choker", "anklet", "pendant", "tiara", "hairpin", "ear clip", "nose ring", "cufflinks", "watch"],
    "clothes": ["dress", "shirt", "jacket", "coat", "skirt", "pants", "shorts", "sweater", "hoodie", "boots", "shoes",
                "gloves", "hat", "scarf", "socks", "thighhighs", "kimono", "uniform", "vest", "cape"],
    "equipment": ["armor", "helmet", "gauntlets", "backpack", "bag", "belt", "holster", "shield", "goggles",
                  "headphones", "pauldrons", "greaves", "harness", "quiver", "satchel"],
}
PLACE_WORDS = {
    "jewelry": ["on finger", "on index finger", "on middle finger", "on ring finger", "on thumb", "in hair", "on ear",
                "around neck", "on wrist", "on ankle"],
    "clothes": ["upper body", "lower body", "full body"],
    "equipment": ["on back", "on hip", "on head", "on arm", "on chest"],
}


ITEM_SMALL = 0.02                                        # the item covers less than 2 % of the picture: small
ITEM_SURE = 0.12                                         # detector score from which the item counts as seen
CLOSEUP_MIN = 384                                        # a close-up needs at least this many pixels a side


def item_judge(e: dict) -> dict:
    """{"seen", "kind", "box", "frac", "small", "crop"} from the item finder's boxes for one picture."""
    W, H = e.get("W") or 0, e.get("H") or 0
    best = max((m for m in e.get("marks") or [] if m.get("score", 0) >= ITEM_SURE), key=lambda m: m["score"],
               default=None)
    if not best or not W or not H:
        return {"seen": False}
    x0, y0, x1, y1 = best["box"]
    frac = (x1 - x0) * (y1 - y0) / (W * H)
    out = {"seen": True, "kind": best.get("label", ""), "box": best["box"], "frac": round(frac, 4),
           "small": frac < ITEM_SMALL}
    if out["small"]:
        side = min(W, H, max(int(max(x1 - x0, y1 - y0) * 3), 512))
        if side >= CLOSEUP_MIN:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            l = int(min(max(0, cx - side / 2), W - side))
            t = int(min(max(0, cy - side / 2), H - side))
            out["crop"] = [l, t, l + side, t + side]
    return out


def add_closeups(ds, paths) -> list[Path]:
    """Select, a worn item: a close-up around the item as a new picture next to each kept picture where it is small
    (closeup_<name>.png; the picture itself stays). Made once: an existing close-up is left alone."""
    from PIL import Image, ImageOps
    res = dssort.sort_result(ds) or {}
    find = dssort.sort_lookup(res.get("pics") or {})
    made = []
    for p in map(Path, paths):
        if p.name.startswith("closeup_"):
            continue
        it = (find(p) or {}).get("item") or {}
        if not it.get("crop"):
            continue
        dest = p.with_name(f"closeup_{p.stem}.png")
        if dest.exists():
            continue
        try:
            with Image.open(p) as im:
                ImageOps.exif_transpose(im).convert("RGB").crop(tuple(it["crop"])).save(dest)
            made.append(dest)
        except OSError:
            continue
    if made:
        F.ensure_originals(ds, made)
    return made


def _rules(text):
    """'ring = rings, band' lines -> [(name, [other words])]."""
    out = []
    for line in (text or "").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        name, _, rest = line.partition("=")
        name = name.strip().lower()
        if name:
            out.append((name, [w.strip().lower() for w in rest.split(",") if w.strip() and w.strip().lower() != name]))
    return out


def worn_setup(ds) -> dict:
    """{'kinds': 'ring = rings\nhair clip = hairclip', 'places': '...'} as typed in step 7."""
    w = F.meta(ds).get("worn_items") or {}
    return {"kinds": w.get("kinds", ""), "places": w.get("places", "")}


def set_worn_setup(ds, kinds, places):
    m = F.meta(ds)
    m["worn_items"] = {"kinds": (kinds or "").strip(), "places": (places or "").strip()}
    F.meta_save(ds, m)


def suggest_worn(ds, ext=".txt", trigger=""):
    """Kinds and places found in the captions, with the other words the tagger used for them."""
    wk = F.meta(ds).get("worn_kind") or "jewelry"
    rows, _ = dscaptions.tag_counts(ds, ext, trigger)
    tags = [t for t, _ in rows]

    def found(words):
        lines = []
        for w in words:
            alts = [t for t in tags if t != w and (t == w + "s" or t.replace(" ", "") == w.replace(" ", ""))]
            if w in tags or alts:
                lines.append(w + (" = " + ", ".join(alts) if alts else ""))
        return "\n".join(lines)
    return found(KIND_WORDS.get(wk, [])), found(PLACE_WORDS.get(wk, []))


def worn_names(ds):
    w = worn_setup(ds)
    return [n for n, _ in _rules(w["kinds"])], [n for n, _ in _rules(w["places"])]


def same_words(ds, ext=".txt", trigger="") -> str:
    """Make the kind and place words the same in every caption: the other words become the name, the name is added
    when a tag mentions it ('gold ring' also gets 'ring'), and the order is trigger, kinds, places, the rest. Returns
    the balance between the kinds."""
    import sd_prepare
    w = worn_setup(ds)
    kinds, places = _rules(w["kinds"]), _rules(w["places"])
    if not kinds:
        return "Write the kinds first (one per line, for example `ring` or `hair clip = hairclip, hair ornament`)."
    trig = (trigger or ds).strip().lower()
    alias = {a: n for n, al in kinds + places for a in al}
    names_k, names_p = [n for n, _ in kinds], [n for n, _ in places]
    changed, n_pics, count, none = 0, 0, {k: 0 for k in names_k}, []
    sd_prepare.record_change(f"kind words of {ds} {ext}", {dscaptions.set_file(p, ext): (dscaptions.read_set(p, ext) or "") + "\n"
                                                     for p in F.images(ds) if dscaptions.read_set(p, ext)})
    for p in F.images(ds):
        text = dscaptions.read_set(p, ext)
        if not text:
            continue
        n_pics += 1
        parts = [alias.get(x.strip().lower(), x.strip().lower()) for x in re.split(r"[,\n]", text) if x.strip()]
        have_k = [k for k in names_k if any(x == k or re.search(rf"\b{re.escape(k)}\b", x) for x in parts)]
        have_p = [q for q in names_p if q in parts]
        rest = [x for x in parts if x != trig and x not in have_k and x not in have_p]
        new = ([trig] if trig else []) + have_k + have_p + list(dict.fromkeys(rest))
        for k in have_k:
            count[k] += 1
        if not have_k:
            none.append(p.name)
        line = ", ".join(dict.fromkeys(new))
        if line != text.strip():
            dscaptions.set_file(p, ext).write_text(line + "\n", encoding="utf-8")
            changed += 1
    return f"✔ {changed} of {n_pics} captions changed. " + balance_text(count, n_pics, none)


def balance_text(count, n, none=()):
    if not n:
        return ""
    parts = " · ".join(f"**{k}** {c}" for k, c in sorted(count.items(), key=lambda kv: -kv[1]))
    low = [k for k, c in count.items() if c < n / 3] if len(count) > 1 else []
    out = f"Pictures per kind: {parts} (of {n})."
    if low:
        out += (f" ⚠ {', '.join(low)} {'is' if len(low) == 1 else 'are'} in fewer than a third of the pictures: the "
                "trigger word will mostly mean the others. Add pictures of it, or turn on **Weight per kind** (step 7), "
                "which gives its pictures more repeats.")
    if none:
        out += f" {len(none)} caption(s) name no kind ({', '.join(none[:4])}{' …' if len(none) > 4 else ''})."
    return out


def worn_balance(ds, ext=".txt"):
    kinds, _ = worn_names(ds)
    if not kinds:
        return ""
    count, n = {k: 0 for k in kinds}, 0
    for p in F.images(ds):
        t = (dscaptions.read_set(p, ext) or "").lower()
        if not t:
            continue
        n += 1
        for k in kinds:
            if re.search(rf"\b{re.escape(k)}\b", t):
                count[k] += 1
    return balance_text(count, n)


# ---- worn items: weight per kind (each kind its own repeat folder in the training view)
MAX_WEIGHT = 5


def _kind_of(text: str, order) -> str:
    """The kind a caption names, rarest kind first (the same rule as the launch script's grep -iwF)."""
    t = (text or "").lower()
    for k in order:
        if re.search(rf"(?<![\w]){re.escape(k)}(?![\w])", t):
            return k
    return ""


def kind_weights(ds, ext=".txt") -> dict:
    """{"on", "order": [kinds, rarest first], "rows": [(kind, pictures, weight, auto weight)], "none": pictures naming
    no kind, "n", "n_eff"}: each picture counts for one kind (the rarest it names); a kind with fewer pictures gets a
    weight (more repeats) so that every kind counts about the same, at most MAX_WEIGHT; your own numbers win."""
    kinds, _ = worn_names(ds)
    m = F.meta(ds).get("worn_weights") or {}
    caps = [dscaptions.read_set(p, ext) or "" for p in F.images(ds)]
    raw = {k: sum(1 for c in caps if re.search(rf"(?<![\w]){re.escape(k)}(?![\w])", c.lower())) for k in kinds}
    order = sorted(kinds, key=lambda k: (raw[k], kinds.index(k)))
    count = {k: 0 for k in kinds}
    none = 0
    for c in caps:
        k = _kind_of(c, order)
        if k:
            count[k] += 1
        else:
            none += 1
    top = max(count.values(), default=0)
    rows, n_eff = [], none
    for k in order:
        auto = min(MAX_WEIGHT, max(1, round(top / count[k]))) if count[k] else 1
        w = int((m.get("w") or {}).get(k) or auto)
        rows.append((k, count[k], w, auto))
        n_eff += count[k] * w
    return {"on": bool(m.get("on")) and bool(kinds), "order": order, "rows": rows, "none": none, "n": len(caps),
            "n_eff": n_eff}


def set_kind_weights(ds, on: bool, weights: dict) -> str:
    m = F.meta(ds)
    m["worn_weights"] = {"on": bool(on), "w": {k: max(1, min(20, int(v or 1))) for k, v in (weights or {}).items()}}
    F.meta_save(ds, m)
    kw = kind_weights(ds)
    if not kw["on"]:
        return "✔ Saved. New launch scripts train every picture with its repeat folder's repeats, as before."
    return ("✔ Saved. New launch scripts (step 9) give each kind its own repeat folder: "
            + " · ".join(f"**{k}** {n} × {w}" for k, n, w, _ in kw["rows"])
            + (f" · no kind {kw['none']} × 1" if kw["none"] else "")
            + f" (counts as {kw['n_eff']} pictures instead of {kw['n']}). Make the launch script again in step 9.")


def kinds_block(ds, ext: str) -> str:
    """The launch script's part that builds the view: links to the pictures (nothing is moved), one repeat folder per
    kind with the repeats × weight, made fresh at every run. '' when weights are off."""
    kw = kind_weights(ds, ext)
    if not kw["on"]:
        return ""
    def q(x):
        return x.replace('"', "").replace("$", "").replace("`", "").replace("|", "")
    entries = " ".join(f'"{q(k)}|{w}|{q(k)}"' for k, _, w, _ in kw["rows"])
    return "\n".join([
        "# Weight per kind (worn item, LoRA Studio › step 7): every kind gets its own repeat folder in a view made",
        "# fresh at every run, with links to the pictures (nothing is moved). kind|weight|word, rarest kind first.",
        'KINDS_VIEW="${TRAIN_DIR}/.lsflow/kinds_view"',
        f"KINDS=({entries})",
        'rm -rf "${KINDS_VIEW}" && mkdir -p "${KINDS_VIEW}"',
        'for d in "${TRAIN_DIR}"/[0-9]*_*/; do',
        '  rep="$(basename "$d")"; rep="${rep%%_*}"',
        '  for f in "$d"*; do',
        '    case "${f,,}" in *.png|*.jpg|*.jpeg|*.webp|*.bmp) ;; *) continue ;; esac',
        '    base="${f%.*}"; dest="other"; w=1',
        f'    if [ -f "${{base}}{ext}" ]; then',
        '      for k in "${KINDS[@]}"; do',
        '        IFS="|" read -r kn kw kword <<< "$k"',
        f'        if grep -qiwF -- "$kword" "${{base}}{ext}"; then dest="$kn"; w="$kw"; break; fi',
        '      done',
        '    fi',
        '    sub="${KINDS_VIEW}/$((rep * w))_${dest// /-}"',
        '    mkdir -p "$sub"',
        '    ln -s "$f" "$sub/"',
        '    for c in "${base}".*txt "${base}".caption; do',
        '      if [ -f "$c" ]; then ln -s "$c" "$sub/"; fi',
        '    done',
        '  done',
        'done',
        'echo "Weight per kind: $(ls "${KINDS_VIEW}" | tr \'\\n\' \' \')"',
        "",
    ])
