# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Compare settings, the pages' part (the logic is Training/compare.py): the start card in the dataset's Train part,
the comparisons and the blind rating in its Results part, and "Best settings per LoRA type" in Collection › Insights."""
from __future__ import annotations
__version__ = "1.261126b"  # LoRA Studio version, see version.py

import time

from nicegui import run, ui

import compare as C
from widgets import confirm, note, pic_url


# --------------------------------------------------------------------------- Train: start a comparison
def start_card(get_ds, get_tkey):
    """get_ds(), get_tkey(): the dataset and LoRA type of step 9 right now."""
    with ui.expansion("Compare settings: which settings make the best LoRA of this type?").classes("w-full ls-card"):
        note("Trains this dataset several times from your standard launch script, each run with ONE setting changed, "
             "then makes the same 6 test pictures of every run (3 prompts × 2 seeds, through ComfyUI) for you to "
             "rate blind in Results › Compare settings. Over several datasets this shows which settings win for "
             "this LoRA type. The runs train into _compare/ in the dataset folder (not into the Collection).")
        boxes = {}
        with ui.row().classes("items-center gap-4"):
            for g, lab in C.GROUPS.items():
                boxes[g] = ui.checkbox(lab, value=g in ("lr", "epochs"), on_change=lambda: show())
        what = ui.markdown("")
        msg = ui.markdown("")

        def ticked():
            return [g for g, b in boxes.items() if b.value]

        async def show():
            ds, t = get_ds(), get_tkey()
            if not ds or not t:
                what.content = "Pick the LoRA type above first."
                return
            try:
                pv = await run.io_bound(C.preview, ds, t, ticked())
            except Exception as e:                       # noqa: BLE001
                what.content = f"❌ {e}"
                return
            lines = [f"- {v['label']}" for v in pv["variants"]]
            if "epochs" in ticked():
                lines.append("- the save points of your standard script (about half and three quarters of the way; "
                             "no extra run)")
            hrs = f", about {pv['hours']} hours of training" if pv.get("hours") else (
                ", each as long as one normal training of this type (run Settings › Speed test for an estimate)")
            what.content = (f"**{pv['runs']} run(s)**{hrs}:\n\n" + "\n".join(lines)
                            + "\n\nTip: start it in the evening; the pictures are made after the queue, by themselves.")

        async def go():
            ds, t = get_ds(), get_tkey()
            if not await confirm("Put the runs of this comparison in the queue?", "Start the comparison"):
                return
            msg.content = await run.io_bound(C.start, ds, t, ticked())

        ui.button("Start the comparison", on_click=go).props("unelevated dense no-caps color=primary")
        ui.timer(0.2, show, once=True)
    return show


# --------------------------------------------------------------------------- Results: the comparisons and rating
def _status(e: dict) -> str:
    st = C.run_states(e)
    done = sum(1 for v in st.values() if v == "done")
    if any(v in ("queued", "running") for v in st.values()):
        return f"training: {done} of {len(st)} runs done"
    pics = e.get("pics") or {}
    if not pics:
        return (e.get("msg") or "") + (" · " if e.get("msg") else "") + "runs done, test pictures not made yet"
    sm = C.summary(e)
    if sm["rated"] < sm["rows"]:
        return f"rated {sm['rated']} of {sm['rows']} rows"
    return "rated: " + "; ".join(f"{C.GROUPS.get(g, g)}: {r['text']}" for g, r in sm["groups"].items())


class ResultsCard:
    def __init__(self, get_ds):
        self.get_ds = get_ds
        self.cid = ""
        with ui.expansion("Compare settings: rate the test pictures of your comparisons", value=False) \
                .classes("w-full ls-card") as self.exp:
            note("Each row shows the same prompt and seed from every run of a comparison, shuffled and named only A, B, "
                 "C …: tap the best picture (or two). None are good / All look the same count too. The answers "
                 "(which run was which, and what won) show when every row is rated, or with Show the answers.")
            self.lst = ui.column().classes("w-full gap-1")
            self.msg = ui.markdown("")
            self.rate_col = ui.column().classes("w-full gap-3")
        self.exp.on_value_change(lambda e: e.value and self.draw_list())

    def draw_list(self):
        ds = self.get_ds()
        self.lst.clear()
        exps = C.all_comparisons(ds) if ds else []
        with self.lst:
            if not exps:
                note("No comparisons of this dataset yet: start one in Train › Compare settings.")
            for e in exps:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    when = time.strftime("%d %b %H:%M", time.localtime(e["created"]))
                    ui.label(f"{when} · {e['tkey']} · " + ", ".join(C.GROUPS[g] for g in e["groups"])) \
                        .classes("font-semibold shrink-0")
                    ui.label(_status(e)).classes("ls-note grow truncate")
                    ui.button("Rate", on_click=lambda e=e: self.open(e["id"])).props("unelevated dense no-caps color=primary")
                    if not e.get("rows"):
                        ui.button("Make pictures", on_click=lambda e=e: self.make(e["id"])).props("outline dense no-caps")
                    ui.button("Delete", on_click=lambda e=e: self.delete(e["id"])).props("flat dense no-caps color=negative")

    def make(self, cid):
        self.msg.content = C.make_in_background(cid)

    async def delete(self, cid):
        if not await confirm("Delete this comparison (its scripts, LoRA files and test pictures go to the Trash)?",
                             "Delete"):
            return
        self.msg.content = await run.io_bound(C.delete, cid)
        self.rate_col.clear()
        self.draw_list()

    def open(self, cid: str, reveal: bool = False):
        self.cid = cid
        e = C.load(cid)
        rows = C.rows(cid)
        self.rate_col.clear()
        with self.rate_col:
            if not rows:
                note("No test pictures yet: they are made after the queue (ComfyUI must run), or press Make pictures.")
                return
            sm = C.summary(e)
            show = reveal or sm["rated"] >= sm["rows"]
            lab = C.labels(e)
            with ui.row().classes("items-center gap-2"):
                ui.label(f"Rated {sm['rated']} of {sm['rows']} rows").classes("font-semibold")
                ui.button("Show the answers" if not show else "Hide the answers",
                          on_click=lambda: self.open(cid, not show)).props("flat dense no-caps")
            if show:
                with ui.card().props("flat bordered").classes("ls-card w-full gap-1"):
                    ui.label("What won").classes("font-semibold")
                    for g, r in sm["groups"].items():
                        ui.markdown(f"**{C.GROUPS.get(g, g)}:** {r['text']}" + (" ✔ clear" if r["clear"] else ""))
                    ui.markdown("Picks per run: " + ", ".join(f"{lab.get(k, k)} {n}" for k, n in
                                                             sorted(sm["wins"].items(), key=lambda kv: -kv[1])) or "none")
                    if sm["none"] or sm["same"]:
                        note(f"None good: {sm['none']} row(s) · all the same: {sm['same']} row(s).")
            for r in rows:
                self.row(cid, r, show, lab)

    def row(self, cid, r, show, lab):
        with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
            ui.label(r["prompt"][:220]).classes("ls-note")
            picks = set(r["pick"]) if isinstance(r["pick"], list) else set()
            with ui.row().classes("w-full gap-2"):
                for n, (k, f) in enumerate(zip(r["order"], r["pics"])):
                    chosen = k in picks
                    with ui.column().classes("gap-0 items-center"):
                        ui.image(pic_url(f, 512)).classes("w-56 rounded cursor-pointer" + (
                            " ring-4 ring-amber-500" if chosen else "")).on(
                            "click", lambda k=k: self.toggle(cid, r, k))
                        ui.label(C.LETTERS[n] + (f" · {lab.get(k, k)}" if show else "") + (" ★" if chosen else "")) \
                            .classes("text-sm font-semibold")
            with ui.row().classes("gap-2 items-center"):
                for v, t in (("none", "None are good"), ("same", "All look the same")):
                    ui.button(t, on_click=lambda v=v: self.set(cid, r["i"], v)).props(
                        ("unelevated color=primary" if r["pick"] == v else "outline") + " dense no-caps")
                if r["pick"]:
                    ui.button("Undo", on_click=lambda: self.set(cid, r["i"], None)).props("flat dense no-caps")

    def toggle(self, cid, r, k):
        cur = list(r["pick"]) if isinstance(r["pick"], list) else []
        if k in cur:
            cur.remove(k)
        else:
            cur = (cur + [k])[-2:]                       # the best one, or two
        self.set(cid, r["i"], cur or None)

    def set(self, cid, i, v):
        C.pick(cid, i, v)
        self.open(cid)
        self.draw_list()


# --------------------------------------------------------------------------- Insights: best settings per type
def best_card():
    with ui.card().props("flat bordered").classes("ls-card w-full gap-2"):
        ui.label("Best settings per LoRA type (from your comparisons)").classes("font-semibold")
        note("Adds up every rated comparison (Train › Compare settings). A setting is offered when it won clearly in at "
             "least 2 comparisons and nothing else won as often; ticked, new launch scripts of that type use it (your "
             "standard script stays as the rest). Untick to go back.")
        box = ui.column().classes("w-full gap-1")
        msg = ui.markdown("")

        async def draw():
            data = await run.io_bound(C.best)
            u = C.uses()
            import dsflow as F
            labels = F.type_labels()
            box.clear()
            with box:
                if not data:
                    note("No rated comparisons yet.")
                for t, gs in data.items():
                    ui.label(labels.get(t, t)).classes("font-semibold mt-1")
                    for g, d in gs.items():
                        with ui.row().classes("items-center gap-3 no-wrap"):
                            ui.label(C.GROUPS.get(g, g)).classes("w-44 shrink-0")
                            ui.label(d["text"]).classes("ls-note grow")
                            if d["offer"]:
                                cur = (u.get(t) or {}).get(g) == d["offer"]
                                ui.checkbox(f"Use {C.value_label(g, d['offer'])} for new launch scripts", value=cur,
                                            on_change=lambda e, t=t, g=g, v=d["offer"]: setattr(
                                                msg, "content", C.set_use(t, g, v if e.value else None)))
        ui.button("Read again", on_click=draw).props("flat dense no-caps")
        ui.timer(0.3, draw, once=True)
    return draw
