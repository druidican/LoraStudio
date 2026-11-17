# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Writes LoRA Studio's own test-picture workflows (program/Training/workflows/), one per model family, with the
settings the model cards advise. Normal ComfyUI 'Save' format, plus widgets_values_named for comfytest.read_workflow.
Run: python tools/make_workflows.py"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "program" / "Training" / "workflows"
SUBJECT = "1girl, solo, looking at viewer, upper body, simple background"
FAMILIES = {
    # file: (checkpoint, positive prefix, negative, sampler, scheduler, steps, cfg, clip skip, v-pred)
    "sdxl.json": ("sd_xl_base_1.0.safetensors", "", "worst quality, low quality, blurry",
                  "dpmpp_2m", "karras", 28, 7.0, 0, False),
    "illustrious.json": ("Illustrious-XL-v0.1.safetensors", "masterpiece, best quality",
                         "worst quality, comic, multiple views, bad quality, low quality, lowres, displeasing, "
                         "very displeasing, bad anatomy, bad hands, monochrome, extra digits, fewer digits",
                         "euler_ancestral", "normal", 28, 6.0, 0, False),
    "pony.json": ("ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
                  "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_anime", "",
                  "euler_ancestral", "normal", 25, 7.0, -2, False),
    "noobai_eps.json": ("NoobAI-XL-v1.1.safetensors", "masterpiece, best quality, newest, absurdres, highres, safe",
                        "worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, "
                        "mutated hands", "euler_ancestral", "normal", 28, 5.5, 0, False),
    "noobai_vpred.json": ("NoobAI-XL-Vpred-v1.0.safetensors", "masterpiece, best quality, newest, absurdres, highres, safe",
                          "worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, "
                          "mutated hands", "euler", "normal", 28, 5.0, 0, True),
}
NAMES = {"CheckpointLoaderSimple": ["ckpt_name"], "CLIPTextEncode": ["text"], "EmptyLatentImage": ["width", "height", "batch_size"],
         "ModelSamplingDiscrete": ["sampling", "zsnr"], "CLIPSetLastLayer": ["stop_at_clip_layer"],
         "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
         "SaveImage": ["filename_prefix"], "VAEDecode": []}


def build(ck, pos, neg, sampler, sched, steps, cfg, skip, vpred, title):
    nodes, links = [], []

    def node(i, typ, pos_, widgets, inputs, outputs, ttl=None, size=(320, 120)):
        n = {"id": i, "type": typ, "pos": list(pos_), "size": list(size), "flags": {}, "order": i - 1, "mode": 0,
             "inputs": [{"name": nm, "type": t, "link": None} for nm, t in inputs],
             "outputs": [{"name": nm, "type": t, "links": [], "slot_index": k} for k, (nm, t) in enumerate(outputs)],
             "properties": {"Node name for S&R": typ}, "widgets_values": widgets,
             "widgets_values_named": dict(zip(NAMES[typ], widgets))}
        if ttl:
            n["title"] = ttl
        nodes.append(n)

    def link(a, aslot, b, bslot):
        lid = len(links) + 1
        na = next(n for n in nodes if n["id"] == a)
        nb = next(n for n in nodes if n["id"] == b)
        t = na["outputs"][aslot]["type"]
        na["outputs"][aslot]["links"].append(lid)
        nb["inputs"][bslot]["link"] = lid
        links.append([lid, a, aslot, b, bslot, t])
    node(1, "CheckpointLoaderSimple", (40, 60), [ck], [], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
         f"Model ({title})", (380, 100))
    text = (pos + ",\n\n" if pos else "") + SUBJECT
    node(3, "CLIPTextEncode", (460, 40), [text], [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
         "Positive (the model's own words first)", (420, 160))
    node(4, "CLIPTextEncode", (460, 240), [neg], [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
         "Negative", (420, 140))
    node(5, "EmptyLatentImage", (460, 420), [832, 1216, 1], [], [("LATENT", "LATENT")], None, (300, 110))
    if vpred:
        node(2, "ModelSamplingDiscrete", (40, 220), ["v_prediction", True], [("model", "MODEL")], [("MODEL", "MODEL")],
             "V-prediction, zero terminal SNR", (320, 90))
    if skip:
        node(9, "CLIPSetLastLayer", (40, 340), [skip], [("clip", "CLIP")], [("CLIP", "CLIP")], "CLIP skip 2", (300, 70))
    node(6, "KSampler", (940, 60), [123456, "randomize", steps, cfg, sampler, sched, 1.0],
         [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
         [("LATENT", "LATENT")], None, (320, 270))
    node(7, "VAEDecode", (1300, 60), [], [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], None, (220, 50))
    node(8, "SaveImage", (1560, 60), ["lorastudio/test"], [("images", "IMAGE")], [], None, (420, 600))
    clip_src = (9, 0) if skip else (1, 1)
    if skip:
        link(1, 1, 9, 0)
    link(*clip_src, 3, 0)
    link(*clip_src, 4, 0)
    if vpred:
        link(1, 0, 2, 0)
        link(2, 0, 6, 0)
    else:
        link(1, 0, 6, 0)
    link(3, 0, 6, 1)
    link(4, 0, 6, 2)
    link(5, 0, 6, 3)
    link(6, 0, 7, 0)
    link(1, 2, 7, 1)
    link(7, 0, 8, 0)
    nodes.sort(key=lambda n: n["id"])
    return {"last_node_id": max(n["id"] for n in nodes), "last_link_id": len(links), "nodes": nodes, "links": links,
            "groups": [], "config": {}, "extra": {}, "version": 0.4}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    titles = {"sdxl.json": "SDXL 1.0", "illustrious.json": "Illustrious-XL", "pony.json": "Pony Diffusion V6 XL",
              "noobai_eps.json": "NoobAI-XL EPS", "noobai_vpred.json": "NoobAI-XL V-pred"}
    for f, v in FAMILIES.items():
        (OUT / f).write_text(json.dumps(build(*v, titles[f]), indent=1), encoding="utf-8")
        print("wrote", OUT / f)
