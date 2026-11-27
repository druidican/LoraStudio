# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""dssort.py - steps 3 to 5 of a dataset: the usable scan (pictures that cannot train go to _reserve/), the sort (each
picture judged by the picture tools and Ollama: good, detail, look-alike, unusable, with framing, outfit and place;
sort_run, the result in .lsflow/), and Select (rank: a varied set of the wanted size, your pins first; apply_select
moves the rest to _reserve/). active_sort gives the sort's entries of the pictures that train now.
"""
from __future__ import annotations
__version__ = "0"  # LoRA Studio version, see version.py

import re
import time
from collections import Counter
from pathlib import Path

import datacheck as DC
import dsflow as F
import dsworn
import dsresize


# --------------------------------------------------------------------------- 3 Usable scan
CAT_WORDS = (("broken", "cannot be read"), ("small", "too small"), ("blurry", "blurry"))


def usable_scan(ds, types, upscale: bool = True) -> dict:
    imgs = F.candidates(ds)
    side = F.min_side(types)
    tech = DC.technical(imgs, side)
    tools = F.tools_ok()
    up = tools and upscale
    pics = {}
    counts = Counter()
    for p, r in tech.items():
        r["issues"] = dsresize.soften_small(r["issues"], r.get("w"), r.get("h"), side, up)
        bad = [t for s, t in r["issues"] if s == DC.BAD]
        cat = "usable"
        if bad:
            cat = next((c for c, w in CAT_WORDS if any(w in b for b in bad)), "other")
        counts[cat] += 1
        pics[str(p)] = {"w": r.get("w"), "h": r.get("h"), "cat": cat,
                        "why": "; ".join(t for _, t in r["issues"]), "bad": bool(bad)}
    res = {"at": time.strftime("%Y-%m-%d %H:%M"), "min_side": side, "pics": pics, "counts": dict(counts),
           "sig": F._sig(imgs)}
    F._save(ds, "scan.json", res)
    F.mark_step(ds, "scan", {"not_usable": len(pics) - counts["usable"]})
    return res


def move_to_reserve(ds, paths) -> tuple[int, list]:
    import trim
    return trim.move_to_reserve(F.root(ds), [Path(p) for p in paths])


def back_from_reserve(ds) -> tuple[int, list]:
    import trim
    return trim.move_back(F.root(ds))


def reserve_count(ds) -> int:
    import trim
    return len(trim.reserve_list(F.root(ds))) if ds else 0


# --------------------------------------------------------------------------- 4 Sort
def _content_cached(p: Path, model: str, cache: dict):
    st = p.stat()
    return cache.get(f"{p}|{st.st_size}|{int(st.st_mtime)}|{model}"), f"{p}|{st.st_size}|{int(st.st_mtime)}|{model}"


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def sort_run(ds, types, purpose, wm_removed, url, model, upscale: bool = True):
    """Generator: yields progress lines; the result is saved to sort.json (read it with sort_result)."""
    import sd_prepare
    S = F.sg()
    F.STOP.clear()
    imgs = F.candidates(ds)
    if not imgs:
        yield "No pictures in this dataset's repeat folders."
        return
    side = F.min_side(types)
    t0 = time.time()
    yield f"Pass 1 · quick: size, sharpness and duplicates of {len(imgs)} pictures …"
    tech = DC.technical(imgs, side)
    tools = F.tools_ok()
    pics = {}
    for p, r in tech.items():
        r["issues"] = dsresize.soften_small(r["issues"], r.get("w"), r.get("h"), side, tools and upscale)
        bad = [t for s, t in r["issues"] if s == DC.BAD]
        dup = next((t for s, t in r["issues"] if "near-duplicate" in t), "")
        pics[str(p)] = {"w": r.get("w") or 0, "h": r.get("h") or 0, "sharp": r.get("sharp") or 0,
                        "cat": "out" if bad else ("dup" if dup else "good"), "why": "; ".join(bad) or dup,
                        "warn": [t for s, t in r["issues"] if s == DC.WARN and "near-duplicate" not in t]}
    pass1_out = [p for p, e in pics.items() if e["cat"] == "out"]
    rest = [Path(p) for p, e in pics.items() if e["cat"] != "out"]
    # the picture tools (sd-scripts venv): a quality score, photo or drawing, person boxes, and look-alikes
    tools_note = ""
    det: dict = {}                                       # the detector's boxes per picture
    borders = {}                                         # frames, black or white bars around the picture
    for i, p in enumerate([] if wm_removed else rest):
        if F.STOP.is_set():
            break
        if i % 20 == 0:
            yield f"Pass 1 · looking for frames and bars: {i + 1} of {len(rest)}"
        b = DC.border_box(p)
        if b:
            borders[str(p)] = b
    if tools:
        import pictools
        A = {}
        try:
            for i in range(0, len(rest), 16):
                if F.STOP.is_set():
                    break
                yield f"Pass 1 · rating quality and finding look-alikes: {min(i + 16, len(rest))} of {len(rest)}"
                A.update(pictools.analyze(rest[i:i + 16]))
        except Exception as e:                           # noqa: BLE001
            tools, tools_note = False, f"picture tools: {e}"
        for p, a in A.items():
            if p in pics:
                pics[p].update(aes=a["aes"], kind=a["kind"], persons=a.get("persons") or [])
        if not wm_removed and tools:                     # watermarks and logos: an exact detector (OWLv2)
            try:
                for i in range(0, len(rest), 16):
                    if F.STOP.is_set():
                        break
                    yield f"Pass 1 · looking for watermarks and logos: {min(i + 16, len(rest))} of {len(rest)}"
                    det.update(pictools.find_marks(rest[i:i + 16]))
            except Exception as e:                       # noqa: BLE001
                tools_note = (tools_note + "; " if tools_note else "") + f"watermark finder: {e}"
        if purpose == "worn" and tools:                  # a worn item: where is it, and how big?
            words = dsworn.worn_names(ds)[0] or dsworn.KIND_WORDS.get(F.meta(ds).get("worn_kind") or "jewelry", [])
            try:
                for i in range(0, len(rest), 16):
                    if F.STOP.is_set():
                        break
                    yield f"Pass 1 · looking for the worn item: {min(i + 16, len(rest))} of {len(rest)}"
                    for k, e in pictools.find_items(rest[i:i + 16], words).items():
                        if k in pics:
                            pics[k]["item"] = dsworn.item_judge(e)
            except Exception as e:                       # noqa: BLE001
                tools_note = (tools_note + "; " if tools_note else "") + f"item finder: {e}"
        good = {p: A[p] for p in A if p in pics and pics[p]["cat"] == "good"}
        for a, b, sim in sorted(pictools.lookalikes(good, 0.95) if tools else [], key=lambda t: -t[2]):
            ea, eb = pics[a], pics[b]
            if ea["cat"] != "good" or eb["cat"] != "good":
                continue
            worse, better = (a, b) if (ea.get("aes", 0), ea["sharp"]) < (eb.get("aes", 0), eb["sharp"]) else (b, a)
            pics[worse]["cat"] = "dup"
            pics[worse]["why"] = f"shows the same as {Path(better).name} ({sim:.0%} alike); that one rates higher"
    url = (url or "").strip() or sd_prepare.oc_settings()[0]
    model = model or sd_prepare.oc_settings()[1]
    import features as FT
    ai = FT.have("ollama") and bool(model) and S.CAP is not None and not S.RUN.running()
    note = "" if ai else ("Ollama is not installed: quick pass only" if not FT.have("ollama") else
                          "no Ollama model chosen (Settings › AI helpers)" if not model else
                          "a training run is active" if S.RUN.running() else "Ollama is not available")
    root_ = F.root(ds)
    marks = DC.reconcile(root_, DC.marks_load(root_)) if not wm_removed else {}
    crop_min = side or 512
    done = 0
    seen_marks: set = set()

    def put_marks(p: Path, found: dict | None = None):
        """The marks entry of a picture: Ollama's marks (when it was asked), the detector's and a frame / bars."""
        rp = DC.rel(root_, p)
        ent = marks.get(rp)
        st = p.stat()
        sig = f"{st.st_size}|{int(st.st_mtime)}"
        if ent and ent.get("choice"):                     # you decided already (the file changed by your choice)
            return
        if ent and ent.get("sig") == sig and found is None and ent.get("v") == 2:
            return
        if found is None and ent and ent.get("sig") == sig:      # Ollama's marks from an earlier sort stay
            found = {"W": ent.get("W"), "H": ent.get("H"), "subject": ent.get("subject"),
                     "marks": [m for m in ent.get("marks") or [] if m.get("by") != "detector"]}
        W, H = (found["W"], found["H"]) if found and found.get("W") else DC.shown_size(p)
        subj = dsresize.main_person(pics[str(p)].get("persons"), W, H) or (found or {}).get("subject")
        allm = DC.merge_marks((found or {}).get("marks") or [], DC.detector_marks(det.get(str(p)), subj))
        border = borders.get(str(p))
        if border and subj:                              # a 'bar' that cuts into the person is a plain background
            b, sb = border["box"], subj
            inside = max(0, min(b[2], sb[2]) - max(b[0], sb[0])) * max(0, min(b[3], sb[3]) - max(b[1], sb[1]))
            if inside < 0.97 * max(1, (sb[2] - sb[0]) * (sb[3] - sb[1])):
                border = None
        if not allm and not border:
            if ent and not ent.get("choice"):
                marks.pop(rp, None)
            return
        pics[str(p)]["wm"] = True
        marks[rp] = {"W": W, "H": H, "marks": allm, "subject": subj, "sig": sig, "choice": "", "border": border,
                     "v": 2, "options": DC.options_with_border(W, H, allm, subj, crop_min, border)}
    pool = None
    if ai:
        cache = sd_prepare._dc_cache_load()
        import detailscan as DSC
        with_detail = DSC.enabled(ds)                    # the detailed scan's questions in the same go
        fresh = {}

        def needs(p):
            d0, _ = _content_cached(p, model, cache)
            return d0 is None or "features_visible" not in d0   # none yet, or from before the usefulness check
        todo = [p for p in rest if needs(p)]
        if todo:                                         # several pictures at once: Ollama answers them together
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=DC.PARALLEL)
            futs = {str(p): pool.submit(DC.content, url, model, p, 600.0, with_detail) for p in todo}
        t2 = time.time()
        kinds_found = {}                                 # the kind of each picture (photo, drawing …): lean.py
        for i, p in enumerate(rest, 1):
            if F.STOP.is_set():
                note = "stopped"
                break
            d, key = _content_cached(p, model, cache)
            if str(p) in (futs if pool else {}):
                eta = (f", about {S.fmt((time.time() - t2) / max(done, 1) * (len(todo) - done))} left"
                       if done else "")
                yield f"Pass 2 · detailed (Ollama, {DC.PARALLEL} at once): {done + 1} of {len(todo)}{eta}"
                try:
                    d = futs[str(p)].result()
                except Exception as e:                   # noqa: BLE001
                    sd_prepare._dc_cache_save(cache)
                    note = f"Ollama stopped answering: {e}"
                    break
                cache[key] = d
                done += 1
                if with_detail:
                    fresh[str(p)] = d
                if done % 5 == 0:
                    sd_prepare._dc_cache_save(cache)
            e = pics[str(p)]
            if d.get("medium"):
                kinds_found[p] = d["medium"]
            e["framing"] = _norm(d.get("framing")) or "other"
            e["outfit"], e["place"] = _norm(d.get("outfit")), _norm(d.get("place"))
            e["part"] = _norm(d.get("part_only"))
            e["teaches"] = DC.teaches(d)
            e["udetail"] = DC.unique_detail(d)
            e["face"] = DC.face_size(d)
            if e["part"] in ("none", "no", "n/a", "''"):
                e["part"] = ""
            issues = DC.content_issues(d, purpose)
            if e["part"] and purpose != "style":         # a detail shot is not "no clear subject"
                issues = [x for x in issues if not any(w in x[1] for w in ("no person", "no clear main subject",
                                                                          "face not visible", "face partly", "teaches almost",
                                                                          "shows little"))]
                if e["cat"] == "good" and DC.plain_part(e["part"]) and not e.get("udetail"):
                    e["cat"], e["why"] = "poor", f"only the {e['part']}: shows nothing that identifies the character"
                elif e["cat"] == "good":
                    e["cat"], e["why"] = "detail", f"only the {e['part']}: teaches a detail the full pictures show too small"
            bad = [t for s, t in issues if s == DC.BAD and "watermark" not in t]
            if bad and e["cat"] in ("good", "detail"):
                e["cat"], e["why"] = "poor", "; ".join(bad)
            e["warn"] += [t for s, t in issues if s == DC.WARN and "watermark" not in t]
            if DC._true(d.get("text_or_watermark")) and not wm_removed:
                e["wm"] = True
                rp = DC.rel(root_, p)
                ent = marks.get(rp)
                st = p.stat()
                sig = f"{st.st_size}|{int(st.st_mtime)}"
                # a picture you already decided on is left alone (its file changed because of your choice)
                if not (ent and ent.get("choice")) and (not ent or ent.get("sig") != sig or ent.get("v") != 2):
                    yield f"Pass 2 · where are the watermarks in {p.name}?"
                    try:
                        put_marks(p, DC.find_marks(url, model, p))
                        seen_marks.add(str(p))
                    except Exception:                   # noqa: BLE001
                        pass
                else:
                    seen_marks.add(str(p))
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
        sd_prepare._dc_cache_save(cache)
        if kinds_found:
            try:
                import lean
                lean.remember(ds, kinds_found)
            except Exception:                            # noqa: BLE001
                pass
        if fresh:
            try:
                DSC.remember(ds, fresh, model)
            except Exception:                            # noqa: BLE001
                pass
    if not wm_removed:                                   # the detector and frames also without Ollama's word
        for p in rest:
            if str(p) not in seen_marks and (det.get(str(p)) or str(p) in borders or DC.rel(root_, p) in marks):
                try:
                    put_marks(p)
                except OSError:
                    pass
        DC.marks_save(root_, marks)
    # quality: sharpness rank in this dataset, size against what the type needs, minus warnings
    order = sorted(pics.values(), key=lambda e: e["sharp"])
    for i, e in enumerate(order):
        sharp = i / max(1, len(order) - 1)
        size_ = min(1.0, min(e["w"], e["h"]) / max(side or 1024, 1))
        # sharpness only decides between pictures that show the same much: a slightly soft picture that shows the
        # character well beats a crisp one that shows little
        if "aes" in e:                                   # the quality score of the picture tools leads
            aes = max(0.0, min(1.0, (e["aes"] - 3.0) / 4.0))
            e["q"] = 0.5 * aes + 0.15 * sharp + 0.2 * size_ - 0.05 * len(e["warn"])
        else:
            e["q"] = 0.4 * sharp + 0.4 * size_ - 0.05 * len(e["warn"])
        if e.get("teaches"):                             # how much the picture shows of the subject
            e["q"] += 0.12 * (e["teaches"] - 3) + (0.05 if e.get("udetail") else 0.0)
        if purpose == "character" and e.get("face") in ("medium", "large"):
            e["q"] += 0.05
        e["q"] = round(e["q"], 3)
    # too alike: the same framing, outfit and place more than a few times (the best ones are kept)
    combos: dict = {}
    for p, e in pics.items():
        if e["cat"] in ("good", "detail") and e.get("framing"):
            combos.setdefault((e["framing"], e["outfit"], e["place"]), []).append(p)
    groups = []
    for c, ps in combos.items():
        if len(ps) > F.SAME_LIMIT:
            ps.sort(key=lambda x: -pics[x]["q"])
            for x in ps[2:]:
                if pics[x]["cat"] == "good":
                    pics[x]["cat"] = "alike"
                    pics[x]["why"] = f"one of {len(ps)} alike: {' · '.join(v for v in c if v)}"
            groups.append({"label": " · ".join(v for v in c if v), "n": len(ps)})
    groups.sort(key=lambda g: -g["n"])
    if purpose == "worn":                                # the item not found: set aside (pin it 📌 to keep it anyway)
        for e in pics.values():
            it = e.get("item")
            if it and not it.get("seen") and e["cat"] in ("good", "detail", "alike"):
                e["cat"], e["why"] = "poor", "the worn item was not found in it (item finder)"
            elif it and it.get("small"):
                e["warn"] = (e.get("warn") or []) + [
                    "the item is small in it: Select adds a close-up of it" if it.get("crop") else
                    "the item is small in it, and too small for a close-up"]
    usable = [e for e in pics.values() if e["cat"] in ("good", "detail", "alike")]
    fr = Counter(e.get("framing", "") for e in usable if e.get("framing"))
    outfits = len({e["outfit"] for e in usable if e.get("outfit")})
    places = len({e["place"] for e in usable if e.get("place")})
    good_n = sum(1 for e in usable if e["cat"] in ("good", "detail"))
    need = {"character": 15, "style": 20, "concept": 10}.get(DC.base_purpose(purpose), 15)
    if good_n >= need and (purpose != "character" or fr.get("full body", 0) or not ai):
        baseline = "good"
    elif good_n >= need:
        baseline = "thin"
    else:
        baseline = "weak"
    notes = DC.balance({Path(p): {"content": {"framing": e.get("framing"), "outfit": e.get("outfit"),
                                              "place": e.get("place"), "people": 1}}
                        for p, e in pics.items() if e.get("framing") and e["cat"] != "out"}, purpose) if ai else []
    res = {"at": time.strftime("%Y-%m-%d %H:%M"), "purpose": purpose, "types": list(types or []), "ai": ai,
           "note": note, "pics": pics, "groups": groups, "framing": dict(fr), "outfits": outfits, "places": places,
           "baseline": baseline, "notes": notes, "pass1_out": len(pass1_out), "sig": F._sig(F.candidates(ds)),
           "seconds": round(time.time() - t0, 1), "good": good_n, "tools": tools, "tools_note": tools_note}
    F._save(ds, "sort.json", res)
    F.mark_step(ds, "sort", {"baseline": baseline})
    yield "done"


def sort_result(ds) -> dict | None:
    return F._load(ds, "sort.json")


def sort_lookup(pics: dict):
    """A finder for a picture's sort entry: by its path, else by its file name (and stem). The paths change when
    the repeat folder is renamed, the dataset is reached by another path, or step 6 saves a picture as another
    format; the picture is still the one that was sorted."""
    by_name, by_stem = {}, {}
    for k, v in (pics or {}).items():
        by_name.setdefault(Path(k).name, v)
        by_stem.setdefault(Path(k).stem, v)

    def find(p):
        p = str(p)
        return (pics or {}).get(p) or by_name.get(Path(p).name) or by_stem.get(Path(p).stem)
    return find


# --------------------------------------------------------------------------- 5 Select
QUOTA = {"close-up": 0.35, "upper body": 0.35, "full body": 0.2}


def suggest_size(ds, res=None) -> tuple[int, str, str]:
    """(number, range text, why)."""
    res = res or sort_result(ds) or {}
    m = F.meta(ds)
    purpose = m["purpose"]
    cands = [e for e in (res.get("pics") or {}).values() if e["cat"] in ("good", "detail", "alike")]
    o, pl = res.get("outfits", 0), res.get("places", 0)
    if purpose == "style":
        n, rng = max(40, min(100, 40 + 2 * pl)), "40–100"
    elif purpose in F.CONCEPT_KINDS:
        n, rng = max(20, min(40, 15 + 2 * o + pl)), "20–40"
    else:
        n, rng = max(25, min(50, 20 + 2 * o + pl)), "30–50"
    labels = F.type_labels()
    tname = " + ".join(labels.get(t, t) for t in m["types"]) or "this"
    why = ((f"{tname}, art style, {pl} different places and scenes: a style needs variety (people, objects, "
            f"landscapes, light) more than a big number; about {n} pictures.") if purpose == "style" and (o or pl) else
           (f"{tname}, {F.PURPOSE_LABEL.get(purpose, purpose).lower()}, {o} outfits and {pl} places: more than about "
            f"{n} pictures mostly adds repeats of the same look.") if o or pl else
           f"Without the detailed sort there is no outfit or place information; {n} is the usual size.")
    if cands and len(cands) < n:
        n = len(cands)
        why += f" There are only {len(cands)} usable pictures, so all of them."
    return n, rng, why


def rank(ds, target: int, mode: str | None = None) -> list[dict]:
    """Every picture with keep True/False and a reason. mode "safe": quality first, then what each picture adds;
    "varied": what each picture adds counts as much as quality (more close-ups, angles, outfits, places, details)."""
    mode = mode or F.meta(ds).get("select_mode") or "safe"
    if mode == "varied":
        return _rank_varied(ds, target)
    res = sort_result(ds) or {}
    pics = res.get("pics") or {}
    find = sort_lookup(pics)
    have = {str(p) for p in F.candidates(ds)}
    rows = []
    for p in sorted(have):
        e = find(p)
        if e is None:
            rows.append({"path": p, "cat": "new", "q": 0.5, "framing": "", "outfit": "", "place": ""})
        else:
            rows.append({"path": p, **e})
    cands = [r for r in rows if r["cat"] in ("good", "detail", "alike", "new")]
    cands.sort(key=lambda r: (r["cat"] == "alike", -r["q"]))
    pinned, cands = F._apply_pins(ds, rows, cands)
    purpose = F.meta(ds)["purpose"]
    keep, fr_n, combo_n = list(pinned), Counter(), Counter()
    for r in pinned:
        f = r.get("framing") or "other"
        fr_n[f] += 1
        combo_n[(f, r.get("outfit"), r.get("place"))] += 1
    quota = {k: max(1, round(v * target)) for k, v in QUOTA.items()} if purpose == "character" else {}
    detail_max = max(1, round(0.12 * target))
    for r in cands:
        if len(keep) >= target:
            break
        f = r.get("framing") or "other"
        c = (f, r.get("outfit"), r.get("place"))
        if r["cat"] == "detail" and sum(1 for k in keep if k["cat"] == "detail") >= detail_max:
            continue
        if f in quota and fr_n[f] >= quota[f] and f != "full body":
            continue
        if r.get("framing") and combo_n[c] >= F.SAME_LIMIT - 1:
            continue
        keep.append(r)
        fr_n[f] += 1
        combo_n[c] += 1
    for r in cands:                                      # fill up by quality when the quotas left room
        if len(keep) >= target:
            break
        if r not in keep:
            keep.append(r)
    kept = {id(r) for r in keep}
    for r in rows:
        r["keep"] = id(r) in kept
        f = r.get("framing") or ""
        if r["keep"]:
            if r["cat"] == "detail":
                r["reason"] = r.get("why") or "detail shot"
            elif r["cat"] == "new":
                r["reason"] = "added after the sort"
            else:
                bits = [f] if f else []
                if r.get("outfit"):
                    bits.append(r["outfit"])
                if r.get("udetail"):
                    bits.append(f"shows {r['udetail']}")
                r["reason"] = "kept: " + (" · ".join(bits) if bits else f"quality {r['q']:.2f}")
        else:
            if r["cat"] in ("out", "dup", "poor"):
                r["reason"] = r.get("why") or r["cat"]
            elif r["cat"] == "alike":
                r["reason"] = r.get("why") or "too alike"
            else:
                r["reason"] = "lower quality, or adds the same framing, outfit and place again"
    F._pin_reasons(rows)
    return rows


def mix_text(rows) -> str:
    k = [r for r in rows if r["keep"]]
    fr = Counter((r.get("framing") or "unknown") for r in k if r["cat"] != "detail")
    det = sum(1 for r in k if r["cat"] == "detail")
    parts = [f"{n} {f}" for f, n in fr.most_common()]
    if det:
        parts.append(f"{det} detail shot{'s' if det != 1 else ''} (about {100 * det // max(1, len(k))} %)")
    return " · ".join(parts)


def apply_select(ds, keep_paths) -> str:
    keep = {str(Path(p)) for p in keep_paths}
    stems = {Path(p).stem for p in keep}               # a close-up stays with the picture it was made from
    keep |= {str(p) for p in F.images(ds) if p.name.startswith("closeup_") and p.stem[8:] in stems}
    drop = [p for p in F.images(ds) if str(p) not in keep]
    n, err = move_to_reserve(ds, drop) if drop else (0, [])
    joined = [p for p in F.extra_pictures(ds) if str(p) in keep] if F.include_extra(ds) else []
    j, jerr = F.join_repeat_folder(ds, joined) if joined else (0, [])
    err += jerr
    close = dsworn.add_closeups(ds, F.images(ds)) if F.meta(ds).get("purpose") == "worn" else []
    kept = [str(p) for p in F.images(ds)]
    F.mark_step(ds, "select", {"kept": len(kept), "reserve": reserve_count(ds)})
    F._save(ds, "select.json", {"at": time.strftime("%Y-%m-%d %H:%M"), "kept": sorted(kept)})
    return (f"✔ {len(kept)} pictures in the repeat folder; {n} moved to `{F.RESERVE}/` (not deleted, with their captions)"
            + (f"; {j} from outside it joined" if j else "")
            + (f"; {len(close)} close-up(s) of the worn item added (closeup_*.png)" if close else "") + "."
            + (f" Could not move: {'; '.join(err[:3])}" if err else ""))


# --------------------------------------------------------------------------- what trains now
def active_sort(ds) -> tuple[dict, list[Path]]:
    """The sort's entries of the pictures that are in the repeat folder now ({path: entry}), and the pictures there
    that the sort never saw (added later). The sort result itself describes the pictures it was run on, which after
    Select (the rest went to _reserve/) or Add pictures is not the folder that will be trained."""
    so = sort_result(ds) or {}
    find = sort_lookup(so.get("pics") or {})
    act, unsorted = {}, []
    for p in F.images(ds):
        e = find(p)
        if e is not None:
            act[str(p)] = e
        else:
            unsorted.append(p)
    return act, unsorted


def _rank_varied(ds, target: int) -> list[dict]:
    res = sort_result(ds) or {}
    pics = res.get("pics") or {}
    find = sort_lookup(pics)
    rows = []
    for p in sorted(str(x) for x in F.candidates(ds)):
        e = find(p)
        rows.append({"path": p, "cat": "new", "q": 0.5, "framing": "", "outfit": "", "place": ""} if e is None
                    else {"path": p, **e})
    cands = [r for r in rows if r["cat"] in ("good", "detail", "alike", "new")]
    embs = {}
    try:
        import pictools
        for r in cands:
            a = pictools.cached(r["path"])
            if a and a.get("emb"):
                embs[r["path"]] = pictools.emb(a)
    except Exception:                                    # noqa: BLE001
        embs = {}
    detail_max = max(1, round(0.2 * target))
    pinned, cands = F._apply_pins(ds, rows, cands)
    keep, seen = list(pinned), {"framing": Counter(), "outfit": Counter(), "place": Counter()}
    for r in pinned:
        for k in seen:
            if r.get(k):
                seen[k][r[k]] += 1
    while len(keep) < target and cands:
        best, best_s = None, -9.0
        for r in cands:
            if r["cat"] == "detail" and sum(1 for k in keep if k["cat"] == "detail") >= detail_max:
                continue
            s_ = 0.45 * r["q"] - (0.15 if r["cat"] == "alike" else 0.0)
            for k, w in (("framing", 0.25), ("outfit", 0.15), ("place", 0.15)):
                v = r.get(k) or ""
                if v:
                    s_ += w / (1 + seen[k][v])               # something not in the selection yet counts most
            if r["cat"] == "detail":
                s_ += 0.1
            if embs.get(r["path"]) and keep:
                sims = [sum(a * b for a, b in zip(embs[r["path"]], embs[k["path"]])) for k in keep if embs.get(k["path"])]
                if sims:
                    s_ -= 0.5 * max(0.0, max(sims) - 0.8)     # looks much like one already kept
            if s_ > best_s:
                best, best_s = r, s_
        if best is None:
            break
        keep.append(best)
        cands.remove(best)
        for k in ("framing", "outfit", "place"):
            if best.get(k):
                seen[k][best[k]] += 1
    kept = {id(r) for r in keep}
    for r in rows:
        r["keep"] = id(r) in kept
        if r["keep"]:
            bits = [b for b in (r.get("framing"), r.get("outfit"), r.get("place")) if b]
            r["reason"] = ("detail shot" if r["cat"] == "detail" else "kept for variety: " + " · ".join(bits)
                           if bits else f"quality {r['q']:.2f}")
        else:
            r["reason"] = r.get("why") or ("not needed: others show the same" if r["cat"] in ("good", "alike", "new")
                                           else r["cat"])
    F._pin_reasons(rows)
    return rows
