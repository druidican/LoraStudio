#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""An invented world to test LoRA Studio in, with nothing of the owner's in it.

  python3 tools/make_test_world.py <work folder> [--datasets N]

Makes in the work folder:
  datasets/   N invented datasets (robots named pip, bolt, nova, …: coloured shapes with captions)
  loras/      an empty LoRA folder
  sd-scripts/ a pretend sd-scripts: its venv puts a pretend `accelerate` first on PATH, which prints sd-scripts'
              progress lines, writes a save point per epoch and the final LoRA (a few bytes), in a few seconds
  app/userdata/  settings for the above, the notice accepted and the setup page done
Then start it with HOME=<work folder>/home and the app in <work folder>/app (tools/run_test_app.sh copies the app).
"""
import argparse
import json
import os
import random
import stat
from pathlib import Path

NAMES = ["pip", "bolt", "nova", "gizmo", "rusty", "sprocket", "cog", "widget", "dynamo", "flux"]

FAKE_ACCELERATE = r'''#!/usr/bin/env python3
# pretend `accelerate launch …`: sd-scripts' progress lines, save points and the final LoRA, in a few seconds
import re, sys, time, os
a = " ".join(sys.argv)
def opt(n, d=""):
    m = re.search(r"--" + n + r"[= ]+\"?([^\" ]+)", a)
    return m.group(1) if m else d
out_dir, name = os.path.expanduser(opt("output_dir", ".")), opt("output_name", "lora")
epochs = int(opt("max_train_epochs", "3") or 3)
steps = 4 * epochs
os.makedirs(out_dir, exist_ok=True)
print("running training / 学習開始", flush=True)
print(f"  num epochs / epoch数: {epochs}", flush=True)
print(f"  total optimization steps / 学習ステップ数: {steps}", flush=True)
n = 0
for e in range(1, epochs + 1):
    print(f"\nepoch {e}/{epochs}", flush=True)
    for s in range(4):
        n += 1
        time.sleep(0.4)
        print(f"steps: {100 * n // steps}%| {n}/{steps} [00:0{n % 10}<00:01, 1.00it/s, avr_loss=0.{90 - n:02d}]", flush=True)
    if e < epochs:
        open(os.path.join(out_dir, f"{name}-{e:06d}.safetensors"), "wb").write(b"pretend")
        print(f"saving checkpoint: {out_dir}/{name}-{e:06d}.safetensors", flush=True)
open(os.path.join(out_dir, f"{name}.safetensors"), "wb").write(b"pretend")
print(f"model saved.", flush=True)
'''


def picture(path: Path, seed: int, w: int, h: int):
    from PIL import Image, ImageDraw
    r = random.Random(seed)
    im = Image.new("RGB", (w, h), tuple(r.randint(40, 200) for _ in range(3)))
    d = ImageDraw.Draw(im)
    cx, cy = w // 2 + r.randint(-80, 80), h // 2 + r.randint(-60, 60)
    body = tuple(r.randint(120, 255) for _ in range(3))
    d.rectangle([cx - 120, cy - 60, cx + 120, cy + 220], fill=body)                 # the robot's body
    d.ellipse([cx - 90, cy - 220, cx + 90, cy - 70], fill=body)                     # its head
    d.ellipse([cx - 50, cy - 170, cx - 20, cy - 140], fill=(20, 20, 20))
    d.ellipse([cx + 20, cy - 170, cx + 50, cy - 140], fill=(20, 20, 20))
    for k in range(r.randint(2, 6)):                                                 # things around it
        x, y = r.randint(0, w - 80), r.randint(0, h - 80)
        d.rectangle([x, y, x + r.randint(30, 120), y + r.randint(30, 120)], fill=tuple(r.randint(0, 255) for _ in range(3)))
    im.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--datasets", type=int, default=6)
    ap.add_argument("--pictures", type=int, default=12)
    a = ap.parse_args()
    W = Path(a.work).resolve()
    ds_root, loras, sd = W / "datasets", W / "loras", W / "sd-scripts"
    for p in (ds_root, loras, sd / "venv" / "bin", W / "fakebin", W / "home", W / "app" / "userdata"):
        p.mkdir(parents=True, exist_ok=True)
    for f in ("train_network.py", "sdxl_train_network.py", "flux_train_network.py", "sd3_train_network.py"):
        (sd / f).write_text("# pretend sd-scripts\n")
    (sd / "venv" / "bin" / "activate").write_text(f'export PATH="{W / "fakebin"}:$PATH"\n')
    acc = W / "fakebin" / "accelerate"
    acc.write_text(FAKE_ACCELERATE)
    acc.chmod(acc.stat().st_mode | stat.S_IEXEC)
    sizes = [(1024, 1024), (832, 1216), (1216, 832), (640, 640)]
    for i in range(a.datasets):
        name = NAMES[i % len(NAMES)] + ("" if i < len(NAMES) else str(i))
        rep = ds_root / name / f"10_{name}"
        rep.mkdir(parents=True, exist_ok=True)
        for k in range(a.pictures):
            w, h = sizes[(i + k) % len(sizes)]
            picture(rep / f"{name}_{k:02d}.jpg", i * 100 + k, w, h)
            if i % 2 == 0:                                                               # half of them have captions
                (rep / f"{name}_{k:02d}.txt").write_text(f"{name}, a robot, standing, simple background\n")
    ud = W / "app" / "userdata"
    (ud / "library_settings.txt").write_text(
        f"[folders]\nsd_scripts_dir = {sd}\ndataset_dirs =\n    {ds_root}\nlora_dirs =\n    {loras}\n"
        f"script_dirs =\nsample_dir =\nmodels_dir = {W / 'models'}\npictures_dir = {W / 'pictures'}\n"
        f"backup_dir = {W / 'backups'}\n", encoding="utf-8")
    (ud / "notice_accepted.json").write_text(json.dumps({"version": 1, "accepted": "test"}))
    (ud / ".setup_done").write_text("test world\n")
    (ud / "features.json").write_text(json.dumps({"sd_scripts": True, "musubi": False, "ollama": True}))
    (W / "models").mkdir(exist_ok=True)
    (W / "pictures").mkdir(exist_ok=True)
    print(f"test world in {W}: {a.datasets} datasets, pretend sd-scripts in {sd}")


if __name__ == "__main__":
    main()
