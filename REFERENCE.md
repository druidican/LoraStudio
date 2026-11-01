# LoRA Studio: the full reference

Everything LoRA Studio does, page by page and file by file. For a first look, read the [README](README.md) or the **Manual** tab in the program.

Your LoRA collection, your datasets and training in one window, with four top tabs in the order you work:

- **Collection**: Browse (Pictures or Status), the LoRA page (Overview, Pictures, Training, Edit), Maintenance
  (Sync and Clean up)
- **Datasets**: New dataset, Edit existing, Improvements; nine steps from a folder of pictures to the launch script
- **Training**: Run (launch scripts, the queue and the live run) and Results (every finished run)
- **Settings**: a grouped list that opens on a Health check

A **top bar** on every page shows the training run (step, loss, time left, what is waiting), the machine (GPU load
and temperature, VRAM, CPU and RAM, from `rocm-smi` on AMD or `nvidia-smi` on NVIDIA) and, on the right, which
Ollama model is in video memory: **● Ollama: qwen2.5vl:7b**, or **○ Ollama: none**. Click it to open Training › Run.
When a run ends, the bar says so, with the save point to try first.

## Datasets

**What the dataset is:** only its own repeat folder, `<N>_<dataset name>` (for example `10_mychar` in `mychar/`).
Pictures anywhere else in the dataset folder (`_reserve/`, `_extra/`, other repeat folders like `5_closeups`, other
folders, the dataset folder itself) are left out of the counts, the checks, the captions and Improvements, so they
never give false warnings. An older dataset without a folder of that name uses the repeat folders directly in it,
as sd-scripts does. To use those pictures, tick **Also look at the pictures outside the repeat folder** in *2 Usable
scan*: the scan, the sort and the selection then see them too (marked *from _reserve/*, …), and the ones you keep in
*5 Select* move into the repeat folder with their captions. The setting switches itself off when a new launch script
is saved. Saving a launch script also moves other repeat folders directly in the dataset folder to `_extra/`,
because sd-scripts would train every `<N>_name` folder there (nothing is deleted).

**New dataset** starts at *1 Folder*: pick a folder in your dataset folder that holds only pictures (or any folder
somewhere else; its pictures are copied in), name the dataset and set the repeats. LoRA Studio makes
`<name>/<N>_<name>/` with the pictures, `_originals/` with untouched copies, `model/` and `log/`, and goes on with the
steps. **Edit existing** lists every dataset with its next step; open one to see the state of each step and open any
of them (an older dataset gets its LoRA types from its launch scripts). *Add pictures from a folder* copies the
pictures of a folder that are good enough and not already in the dataset. **Improvements** lists what could be better
for each dataset, from what is known already (too many pictures, no full-body pictures, missing captions, a
contradiction in the captions, watermarks not decided, no launch script); nothing is scanned there.

**Fix the ticked datasets** (Improvements) runs steps 2-7 for every ticked dataset, one after the other in the
background, with "less is better" defaults: not usable pictures to `_reserve/` (unless that would be more than half of
them: then nothing moves and the report says to look at step 2), the dataset's own LoRA type and purpose (else Anima +
Character), the sort, the suggested selection at most 50 pictures (the rest to `_reserve/`), watermarks with the
recommended crop, resize, and captions only for pictures without one. The consistency check, Review and Launch script
are left for you. A report per dataset shows what was done; *Stop* ends it after the current picture. It does not
start while a training run uses the GPU.

**Settings › Automatic improvement** changes these defaults (saved in `batch_settings.json`, part of the backup): how
many pictures to keep (the suggestion at most N, or always N), the safety limit for step 2, the LoRA type and purpose
for a dataset without one, what happens to watermarks (remove, crop, set aside, leave), upscaling on or off, captions
(only missing, or all again), the consistency check (left out, or run: every trait answered with "This picture only", or the findings
waiting for you in step 7) and a
launch script at the end. The Improvements page shows the current choices with a link to change them.

The steps (the bar above them shows the dataset, its LoRA types and purpose; the step buttons go anywhere):

1. **Folder**: see above.
2. **Usable scan** (optional): size, sharpness, compression, broken files; no AI. *Move not usable to _reserve*.
3. **Setup**: one or more **LoRA types**, the **purpose**, **Watermarks are already removed**, and the trigger word.
   Purposes: *Character / person*, *Art style*, *Object / item*, *Action / pose* and *Concept / effect*. For the last
   three, step 3 asks for the words of the thing itself (e.g. "katana, sword", "kneeling", "glowing aura"): they go into
   *Never describe*, and the captions name the trigger word instead (an object's captions describe who holds or wears it
   and the scene; an action's describe the person, clothes and place, never the pose). With several types the sort uses the strictest size, step 7 writes one caption set
   per caption style and step 9 one launch script per type.
4. **Sort**: pass 1 is quick (too blurry, too small for your LoRA types, near-duplicates); pass 2 asks the Ollama model
   about every picture (one clear subject, framing, outfit, place, quality, watermarks, and a 1-5 rating of how much
   the picture teaches about the subject, plus a detail it shows well; remembered). A rating of 1 (subject tiny,
   hidden, cut off or lost in the background) leaves the picture out, 2 is a warning, and the rating and a shown
   detail count in the quality score (not for style LoRAs). For a character it also asks the face size and whether
   the identifying features can be seen: hidden by clothes, pose or angle, or a tiny face on a small figure, leaves
   the picture out; plain hands, feet, legs or arms without anything special are left out instead of kept as
   detail shots; sharpness weighs less, so a slightly soft picture that shows the character well beats a crisp one. Answers from before this check are asked again once. It lists groups of
   pictures that are too alike (the best ones are kept), duplicates, **detail shots** (only the hands, tail, horns or
   back: kept, they teach details the full pictures show too small), pictures that are not good enough, and
   watermarks. The **baseline** says whether the good pictures are enough for a LoRA.
5. **Select**: a suggested size for the type, purpose and variety (a character: 30-50), picked by quality first and
   then by what each picture adds (framing, outfit, place), with about 35 % close-ups, 35 % upper body, 20 % full body
   (all full-body pictures stay) and about 10 % detail shots. Tick or untick, then *Confirm selection*: the others go
   with their captions to `_reserve/` (not deleted; *Move all back* undoes it). **Pins** under each picture:
   *📌 Always* keeps it and *⛔ Never* leaves it out, whatever the rules say; saved at once in the dataset (by picture
   name), kept through new sorts and used by the automatic improvement too. **How to pick**: *Safe* (the cleanest
   pictures first) or *Varied* (what a picture adds counts as much as its quality: more close-ups, angles, outfits,
   places and details; look-alikes count less). The choice is kept per dataset; the batch fix has its own default.
6. **Resize**: keep the shape and scale down to a longest side (default 1.5× the type's resolution, changeable). Per
   picture you can keep the original size or **crop** (a box with left, right, top and bottom, free or 3:4, 1:1, 4:3).
   Nothing is padded or squared: the trainer groups pictures by shape. Before a picture is changed, a copy goes to
   `_originals/`; *Put the originals back* undoes it. **Watermarks** found by the sort are offered here: crop, set
   aside or keep.
7. **Tags**: one caption set per caption style: `IMG.txt` for the first LoRA type's style, `IMG.sentences.txt` (or
   `IMG.tags.txt`) for the other. *Write the captions* writes all sets; the caption settings (length, detail, never
   describe, try on 3 pictures) are in the fold-out. **Consistency** is checked right there: the character's own traits
   (hair, eyes, build, skin, horns, a signature object) that are the same in most pictures are taken out of the
   captions so the trigger word learns them, words that mean the same become one word, and odd pictures that say
   something else are shown so you can look (fix, or this picture only). Clothes, places, poses, expressions and light
   stay in each caption. *Apply to the other caption set too* takes the same traits out of the second set.
   **Character tags → trigger words** (character datasets, WD tags): the WD tagger writes every tag, also the
   character's own; the card lists all tags with their picture counts (traits in at least 40 % of the pictures of the
   describable kinds ticked already). *Move the ticked tags* removes them from every caption of `IMG.txt` and adds them
   to the Character description; the captions with all tags are kept in `.lsflow/captions_alltags_txt/` and *Put them
   back* restores them and takes the moved words out of the description again. A new tagging run starts afresh.
   **Character description** (character datasets): the traits the trigger word learns, named in words (*slim build,
   tall, blonde hair, blue eyes*), saved in the dataset. They are added to the LoRA's trigger words in its LoRA Manager
   data (after the trigger word; a changed description replaces the old words), so ComfyUI offers them next to the
   trigger word; the captions do not change. *Suggest from the tags* takes the traits the WD tagger saw in at least
   40 % of the pictures while writing the tags (hair and eye colour, freckles, tattoos, face, makeup; never ethnicity,
   skin, age or body parts); build and height are typed by you. The LoRA page's training card shows it, and the
   versions are read again by themselves when it changes; *Sync now* passes it on to LoRA Manager.
   **For a style LoRA** the captions describe everything *in* the picture (the people with their hair, eyes and
   clothes, objects, the place, the light, the composition) and never the style itself: *Never describe* holds the
   art style, medium, technique and artist words, so the trigger word learns the style; the consistency check is
   hidden. For a concept, write the concept itself in *Never describe*.
8. **Review**: the folder, every picture with its caption(s) to edit, and checks (each picture has a caption, the
   trigger word first, no empty or very long captions, caption files without a picture, pictures outside the repeat
   folder).
9. **Launch script**: one per LoRA type, on one page. **Your plan** shows the step calculation (pictures × repeats ÷
   batch × epochs) against the recommended range, the training time from your last run of the type, and likeness
   against flexibility. **Settings** has the main values (epochs, repeats, batch, learning rate, dim, alpha,
   resolution, save every N epochs), each with the recommendation and what it means; a change is written into the
   script text when you leave the field. Folds below: *More settings* (optimizer, scheduler, saving, model files,
   advanced, the repeat folder), *Why these settings* and *The script text* (edit by hand; the settings follow it).
   The bottom bar: *Fill with recommended settings* (it also gives the repeat folder the recommended count, with its
   copies in `_originals/` and `_reserve/`), *Check flags against my trainer*, *Save*, *Save as new version* and
   *Add to the queue*.

The locked traits of earlier versions (`traits.txt`) are no longer used.

### Picture tools (quality, look-alikes, smart crop, watermark removal, upscaling)

Small AI models that run in the **sd-scripts venv** on your GPU (it already has PyTorch). Settings › Trainers and GPU
shows whether they are ready; *Install the picture tools* adds the one small package the venv lacks (`spandrel`)
and leaves PyTorch alone. The model files (about 2.5 GB) download into `pictools_models/` the first time a tool is used.
The tools never run during a training run (starting a run stops them) and stop by themselves after 5 idle minutes.
Without them everything works as before.

- **Quality rating** (step 4 Sort): a score per picture from the LAION aesthetic predictor on CLIP ViT-L/14, blended
  with sharpness and size. The best pictures are kept first in step 5.
- **Look-alikes** (step 4 Sort): pictures that show the same thing (CLIP, 95 % alike) even when cropped, resized or
  re-coloured; the lower-rated one is marked as a duplicate.
- **Crop around the person** (step 6 Resize): the crop panel starts around the person (a person detector; boxes only,
  no faces or identities), and *Around the person* suggests it for any shape.
- **Watermark removal** (Watermarks, and the batch fix): the recommended choice is *Remove* when the marks cover less
  than 20 % of the picture: LaMa paints over them and the rest of the picture stays untouched. The original goes to
  `_set_aside/originals/`. If removal fails, the recommended crop is used instead. **Each different watermark is kept
  once as a cut-out** in `<dataset>/_originals/_watermarks/` (PNG, the largest copy of each mark), with
  `watermarks.json` listing which pictures it was on. Each picture in the list also has **✎ Paint over it yourself**
  (paint the mark with a brush; exactly that is removed) and **✂ Crop it yourself** (move the edges; a warning shows
  when a found mark is still inside), with a *now* picture of the result. **Watermarks are done** (step 6 bottom bar)
  ends the watermark questions for the dataset: undecided pictures stay as they are. Pictures you decided on are never
  looked at again by a new sort. The automatic crops use the person finder's box as the part that must stay, and keep
  at least 85 % of it (the head always).
- **Gentle upscaling** (step 2 and step 6): a picture a little too small for your LoRA type (down to 40 % of its size)
  is no longer "not usable": step 6 upscales it with Real-ESRGAN (x2plus, which keeps skin, freckles and fine lines) only up to
  the size the type trains at. Pictures smaller than that still go to `_reserve/`.

### Launch script names and `_old_runs/`

A launch script is named after the LoRA it makes: `bobSDXLV2.sh` writes `bobSDXLV2.safetensors`. *Save as new* on an
existing script suggests the next version (`bobSDXLV3.sh`) and moves the output name inside it up too, so a new run
never writes over an older version. Each dataset folder shows **one launch script per LoRA type**, the newest; the
older ones move to `<dataset>/_old_runs/` (they still run from there: their `SCRIPT_DIR` points one folder up, and
Training › Results still finds them). Scripts in the queue or running are never moved or renamed. At start, LoRA Studio
tidies the existing dataset folders the same way; renamed scripts are listed in `<dataset>/.lsflow/script_renames.json`.

## Training

**Run** has one row per launch script (*Add*; a saved state shows as *Resume from epoch N*; completed ones are hidden
unless *show finished*; tick several and *Add the ticked to the queue*, or *Tick the ones not run*; *Mark as run*
on a never-run, failed or stopped one marks that version as finished and hides it), the queue (move up or down, remove; the time of each comes from your earlier runs of the
type) and, while training, the live run: stages, progress, loss, the save points so far and the next item. **Starting
the queue stops Ollama** to free video memory; it is not started again by itself.

**Results** lists every finished run (newest first) with its result, the save point to try first, whether it is copied
to your LoRA folder and the leftovers. Open one for the **save point to try first** (the earliest saved epoch within
1 % of the lowest loss; it also says when the loss rose at the end, went NaN or was still falling), copy buttons for
every save point (an epoch file is copied as `mycharILLV1-e8.safetensors`), **Clean up this run** (epoch files you did
not copy, saved states and caches to the Trash; the final file, the copied ones, the pictures, captions, launch script
and logs stay), *Train a new version* (Datasets › Edit existing › 9 Launch script), *Run it again*, *★ Best version:
mark / unmark*, TensorBoard and the end of the log.

**Compare runs:** tick two to four runs in the Results list and press *Compare the ticked*. You get one card per run
(result, time, the save point to try first, *★ Mark as the best version*), their **loss per epoch in one chart**, the
**settings that differ** (*show all settings* lists the ones that are the same too) and their **test pictures side by
side**: one row per prompt, one column per run, at the save point to try first or the final one. *Make test pictures
with the same prompts* gives every run the prompts and seeds of the first one and makes the missing pictures in
ComfyUI, one run after the other (pictures made earlier with other prompts are kept as `grid_before.json`). New runs
of the same dataset get the same prompts by themselves.

**★ Best version:** the mark is per dataset and LoRA type (kept in `completed.json` next to the Completed marks). The
Collection then shows that version first: its picture on the tile, its card, and the LoRA page opens on it; the
versions list says *★ best*. Mark it again to remove it, and the newest version is shown again.

## Collection

**Browse** shows your own LoRAs in two looks: **Pictures** (one tile per dataset, the newest picture, a chip per LoRA
type) and **Status** (only problems of the LoRA files themselves; what a dataset still needs is in Datasets ›
Improvements). Click a tile for its **LoRA page**: Overview (preview, Completed, trigger words, copy to the LoRA
folder, LoRA Manager's example pictures), Pictures (everything you made with it in ComfyUI, all versions), Training (the training card, the loss,
all versions) and Edit. **Maintenance** is Sync (LoRA Manager files from your LoRA folder into the datasets) and Clean
up (leftovers of all finished LoRAs, orphaned LoRA Manager files).

## Settings

A list on the left in groups: **Health check** (what is set, what needs a look, what is missing, each with a link to
fix it), Program (*Start and network*), Your files (*Folders*, *Model files*), Tools (*Trainers and GPU* with the
low-level values behind *Advanced*; *AI helpers*), Library (*Look and base models*) and Care (*Backup*, *Updates*).

**AI helpers** lists the installed Ollama models: size, whether it reads pictures, and whether it runs. **Default model** (the
switch at the top, or the round button in the table) picks the model for captions, sorting and checks; **Start** loads a model into video memory (only one at a time: starting
another stops the running one) and **Stop** frees it; *Stop Ollama* stops every model. *Get another model* downloads
or removes one.

## Launch options (Settings › Start and network)

How LoRA Studio starts, saved in `launch_settings.json` (readable only by you):

| Option | What it does |
|---|---|
| Reachable on the network | Listens on all addresses so other computers can open it. Needs a user name and password |
| Port | The address ends in `:port` (default 7860) |
| Login: user name and password | Asked by the browser; also on this computer when set |
| Open the browser at start | Off for a computer without a screen |
| Start page | Collection, Datasets, Training or Settings |
| Check for trainer updates at start | Only looks; installing is always your click |
| Resume the queue after a restart | The queue list is always kept; with this on, waiting items start again (a run cut off by the restart is marked stopped) |
| Notice when a run finishes | A browser notification (the browser asks once) |

*Save and restart* starts the program again with the new options. An option typed on the command line
(`./start.sh --port 7861`, `--host`, `--auth`, `--no-browser`) still wins for that start.

## One folder for the program, everything else anywhere

Put the program in its own folder, called `LoraStudio`, wherever you like. Inside it, the program, your own files and
the Python environment each have their place; sd-scripts, your datasets and your LoRAs can live anywhere else, on
other folders or drives: you link them in **Settings > Folders**.

```
LoraStudio/                   (anywhere)
  start.sh, install.sh        start and install (and the Windows installer)
  lora_studio.py              the launcher
  version.py, selftest.py     the version of every part, the self-test
  README.md, REFERENCE.md, LICENSE, NOTICE, requirements.txt
  program/                    the program, by topic
    Gallery/                  the Collection, the picture gallery, Completed marks, model files
    Data/                     the dataset steps, captions, picture check and tools
    Training/                 launch scripts, LoRA types, the queue and runs, results
    Settings/                 settings, setup page, updates, backup, the venv check, the notice
    Other/                    the look (lora_studio.css), the top bar, the Manual tab, reports
    manual/, logo/            the Manual's chapters and pictures, the logo
  userdata/                   everything that is yours
    library_settings.txt      folders, style, library options
    training_settings.txt     training settings (model files, venv, GPU, subtypes)
    launch_settings.json      launch options (port, network, login, start page, ...)
    huggingtoken.txt          your Hugging Face token
    completed.json, *_cache, picture_index.json, features.json, ...
    gui_logs/, reports/, update_backups/, custom.css
  venv/, constraints.txt      the Python environment (install.sh or start.sh)
  sd-scripts/, musubi-tuner/  the trainers, when install.sh installed them
~/datasets, ~/ComfyUI/...     linked in Settings > Folders
```

An install from before 1.011126 had everything in one folder. The first start of the new version (or `./install.sh`)
moves your files into `userdata/` by itself and puts the old program files aside in
`userdata/update_backups/old_layout_<time>/`. An update in Settings › Updates from such an old version stops with a
clear message instead: run `./install.sh` from the new zip once.

Folders are saved as `~/...`, so they follow whoever is logged in.

## Test pictures with ComfyUI (Training › Results)

**Settings › Test pictures (ComfyUI)**: the ComfyUI address (*Test*), the LoRA folder ComfyUI reads, and per model
family (SDXL / Illustrious / Pony, and Anima) **your own workflow** (the normal `.json` from ComfyUI's Workflow › Save
or Export). LoRA Studio reads it for the model (checkpoint, or diffusion model + text encoder + VAE), CLIP skip,
sampler, scheduler, steps, CFG (also when they come from selector or literal nodes), size and negative prompt, and
fills *Fixed tags* with the quality tags at the start of its prompt (change them as you like). It then sends ComfyUI a
small graph of ComfyUI's **own** nodes with those settings, so custom nodes are not needed; upscaling and face
detailers are left out (the pictures compare epochs).

In a run of **Training › Results**, **Make test pictures**: the save point to try first and two on each side, plus the
final file, each with the same 8 prompts (8 captions of the dataset picked once, the fixed tags in front, the same
seed per row). Each save point is copied for a moment into `<LoRA folder>/_lorastudio_test/` so ComfyUI can load it.
The pictures are kept in `<dataset>/model/compare/<name>/`. **Use this one** under a column copies that save point to
your LoRA folder with its LoRA Manager data, and its first picture as the preview (`<name>.preview.png`).

Where a copied LoRA goes (Results, the LoRA page): into the folder for its LoRA type in your LoRA folder (an existing
`Illustrious` folder in any of your LoRA folders, in any case, else a new one in the first), or next to an older
version when that one is in a subfolder; never into a LoRA folder's root.

**After the queue**: with *Make them by themselves after the training queue has finished* (on), the test pictures of
every successful run of the queue are made when the whole queue is done. When ComfyUI is not running then (it shares
the GPU), LoRA Studio tries again every minute until it answers, so you can start ComfyUI when training is over.

## Speed

The whole page has **one** timer (`ticker.py`): every 2 seconds while something runs (training, the batch fix, a
download), every 6 seconds otherwise, and it only sends what changed. Pictures on the Datasets pages are small files
the browser keeps (in `.thumbs/pages/`, cleaned after 30 days), not data inside every page update. Finished run logs
are read once. A click runs only the part it is for. To see which actions are slow: `LS_PROFILE=1 ./start.sh` prints
every action that takes 0.2 s or more.

## Program version and the self-test

The version is `release.ddmmyy`, with a letter at the end for later updates: **1.241026** is release 1 of 24 Oct 2026;
**1.241026b** is the first update of it (an update on another day gets that day's date and the next letter). It is
shown next to the title, in the browser tab and in the terminal at start.

Every program file carries the version of the update that **last changed it** (`__version__` in the Python files, a
`LoRA Studio version` line in `lora_studio.css`, `install.sh` and `start.sh`); a file an update did not touch keeps
its older version. **Settings › Version** lists every part with its version, marks the ones *changed* by this update,
and shows what each update changed. `version.py` knows which version each part must have in this release: a part
with another version, or a missing one, gets a warning there and in the Health check (usually a file left behind
by a half-done copy: put all the files of the zip in again).

**Machine check** (Settings › Version, or `./venv/bin/python selftest.py --machine`): checks this computer with your
own settings, changing nothing: the program files, your folders and free space, sd-scripts and musubi-tuner (with
their git version), the training venv (PyTorch, ROCm/HIP or CUDA, the GPU it sees, bf16, a quick speed test, the
trainer libraries), the model files of every LoRA type, Ollama, ComfyUI and the picture tools. *Check + tiny
training* then trains a 10-step LoRA at 512 px on 8 made-up pictures with the same launch script LoRA Studio would
write for your SDXL-family type (in a temporary folder that is deleted afterwards; Ollama is stopped first, as for a
real run), and reports whether it ran, its speed and its loss. The report is saved as `lorastudio_check_<time>.md` in
`userdata/` (no token, your home folder shown as `~`, no dataset names): upload it when something is wrong.
`--no-train` skips the training.

**Self-test:** `./venv/bin/python selftest.py` (add `-v` for details) checks in a few seconds, without training and
without changing your files: the versions, that every part loads, that the page builds, a launch script for every
LoRA type (form round trip, command, V1 → V2 naming), the dataset steps on a made-up dataset in a temporary folder
(overview, improvements, repeat folder rename), the automatic improvement settings, caption cleaning and the ComfyUI
workflow reading and test graph, the best-version mark (in a temporary file), reading a LoraStudio.zip and the
rule that only `<N>_<dataset name>` is the dataset. Run it after an update; it ends
with `N of N checks passed` (exit code 0).

## When a training fails: what went wrong, in a few words

The end of a trainer log is long and hard to read. LoRA Studio reads it (`failcodes.py`) and says what went wrong, what
it means and what to do: in the queue, in the top bar, in the Training › Run list and as a card on the run's page in
Results. It knows: out of video memory (OOM), out of normal memory (RAM), the loss became NaN, a model file missing or
broken, a model that does not fit the LoRA type, no pictures to train on, a caption or picture that cannot be read,
the GPU not usable, a missing program or package, an option the trainer does not know, a full disk, no permission,
a failed download, stopped, a crash, many errors, an unknown Python error, an exit code without a message.

**Report for Claude** (on a run's page in Results, and in Settings › Version) makes one Markdown file with the version
and parts, the machine, the newest machine check, the run, LoRA Studio's diagnosis, the launch script and the end of
the log (progress bars shortened). The Hugging Face token and anything that looks like a token are removed and your
home folder is shown as `~`. It is saved in `userdata/reports/` and copied: paste it into a chat with Claude or drop the file
there. (A program cannot post into a claude.ai chat by itself; this is the next best thing.)

## The Manual tab

A guide with text and pictures: the workflow, settings, the nine dataset steps with what to do and not to do, training
and failures, results and comparing versions, the Collection, automatic improvement, updates and the machine check,
and a do-and-don't list with the words used. It is read from `manual/` (one Markdown file per chapter, pictures in
`manual/img/`), so it can be edited and extended. Every picture in it is made from invented test data.

## The logo

The painted logo (`program/logo/lorastudio_logo_full.jpg`, by L. G. H. Druidican): old photographs bound into a large
paintbrush on blue and red paint, with *LoRA Studio · for LoRA making* and *powered by sd-scripts and musubi-tuner*, a nod to
the two trainers by kohya-ss. Its middle (the brush and the photos)
is the icon next to the title, the browser tab icon (`favicon.png`) and the Windows starter icon (`lorastudio.ico`).
When LoRA Studio asks for a login (Settings › Start and network), the login fields sit on the painting
(`lorastudio_login.jpg`), as a card in the middle of the screen.

## The look: lora_studio.css and custom.css

All pages share one look: the same buttons (bordered, one height; orange = the main action of a page), button bars
with the main action at the bottom of each Datasets step, cards with the same corners, and tables with light lines.
Datasets › 6 starts with the watermarks and lists only the pictures that change size; the others are folded away.

The look of the LoRA Studio pages is in `program/Other/lora_studio.css` (an update replaces it). For your own
changes make `custom.css` in the same folder: it is loaded after `lora_studio.css` and an update never touches it.
Restart LoRA Studio after a change.

## Files

Where each part lives is in the picture under *One folder for the program*; Settings › Version lists every part
with its path, its version and what it does.

**Giving LoRA Studio to someone else:** give them the zip, never your LoRA Studio folder. The zip holds only the
program; your folder also holds your settings, your Hugging Face token (`userdata/huggingtoken.txt`), your LoRA and dataset
names, the picture index, logs and the 15 GB venv. They run `./install.sh` and get their own setup page.

## Install with install.sh (recommended)

One script sets up everything in one folder. It asks three questions first, one per part:

| Part | The answers |
|---|---|
| **sd-scripts** (SDXL, Illustrious, Pony, Anima, FLUX.1, SD 1.5, ...) | install it for me · I have it already (give its folder and venv) · none |
| **musubi-tuner** (Z-Image Turbo, Qwen-Image, FLUX.2) | install it for me · I have it already · none |
| **Ollama** (the detailed sort, captions, the consistency check) | install it for me (Linux only, with Ollama's own installer, then offers the vision model `qwen2.5vl:7b`) · it runs already, here or on another computer (give its address) · none |

**None** means: LoRA Studio still makes the launch scripts and prepares datasets, but it cannot run a training with
that trainer (or sort and caption with Ollama). The buttons for a missing part are **hidden**, so nothing can be
started by mistake; the steps and tabs keep their names, greyed out with **(Ollama missing)**, **(sd-scripts
missing)** and so on. What is installed is written to `features.json`; **Settings › Start and network** shows it and
lets you tick a part once you have installed it (LoRA Studio restarts, because the page is built once at start).

With one or both trainers installed, PyTorch is installed for your GPU (AMD with ROCm, NVIDIA with CUDA, asked when
the computer has both). With only your own trainers, or none, only a small venv for LoRA Studio itself is made (about
1 GB, any Python 3.10 or newer); your own sd-scripts is checked (`sdxl_train_network.py` must start in its venv) and
the launch scripts activate its venv. The answers are remembered for the next `./install.sh`; `./install.sh --ask`
asks again. Without questions: `--sd install|existing|none`, `--mu install|existing|none`, `--sd-scripts DIR`,
`--sd-venv DIR`, `--musubi DIR`, `--ollama install|remote|none`, `--ollama-url URL` (the older `--trainers
install|existing` still works). The rest of this section is about the full install, with one Python 3.12 environment
shared by LoRA Studio, sd-scripts and musubi-tuner:

```bash
unzip LoraStudio.zip && cd LoraStudio
./install.sh                 # installs into ~/LoraStudio (outside cloud-synced folders), then:
cd ~/LoraStudio && ./start.sh
```

```
~/LoraStudio/
  start.sh, install.sh, *.py   LoRA Studio
  venv/                        Python 3.12 + PyTorch (ROCm for AMD, CUDA for NVIDIA) + both trainers + Gradio
  sd-scripts/                  git clone (its venv/ is a link to ../venv)
  musubi-tuner/                git clone (its venv/ is a link to ../venv)
  constraints.txt              PyTorch and trainer library versions that later installs must keep
  install.log
```

What it does, step by step (every step says what it found; it asks before anything big):

1. **Checks** git, Python 3.12 (offers `sudo dnf install python3.12` / `apt install`), disk space (about
   25 GB), and the GPU: AMD with its architecture from `rocminfo` (gfx1201 = RX 9070 XT), or NVIDIA.
   With both (a laptop), it asks which one to train on. It warns when the folder is inside a synced folder (MEGA, Dropbox, OneDrive, ...).
2. Copies the LoRA Studio files and, the first time, **imports your settings** (folders, training settings,
   Hugging Face token, Completed marks, caches) from an older LoRA Studio folder it finds. The old folder
   is not changed.
3. Creates **venv/** with Python 3.12 (an older GUI-only venv is moved aside, not deleted).
4. Installs **PyTorch** and **bitsandbytes** (8-bit optimizers): AMD = ROCm 10.1.0 and torch 2.13.0+rocm10.1.0 from AMD's repository for your GFX version (an
   older ROCm build is offered the upgrade on the next `./install.sh`; every ROCm part is pinned to the same version);
   NVIDIA = torch from PyPI (CUDA 13; for an older driver set `CUDA_INDEX` at the top of install.sh).
5. Clones (or updates) **sd-scripts** and **musubi-tuner** and installs their packages, locked to the
   PyTorch above. Their shared libraries (transformers, diffusers, accelerate, huggingface-hub, ...) are
   then locked too, so a later `pip install -U` cannot break them (huggingface-hub 2.x did once).
6. Installs LoRA Studio's own packages (Gradio) within those locks.
7. **Tests**: PyTorch sees the GPU and multiplies in bf16; `sdxl_train_network.py`,
   `flux_train_network.py` and musubi's `zimage_train_network.py` start.
8. Points the **settings** at the new trainers (Settings › Folders and Settings › Trainers + GPU).
9. On an existing install: offers to download the **Z-Image Turbo training adapter** into your model folder and
   sets it up (a new install gets this offer on the setup page).
10. Offers to point your **existing launch scripts** at the new trainers (only their `SD_SCRIPTS_DIR=` /
    `MUSUBI_DIR=` line changes; a `.sh.bak` is kept). Your old `~/sd-scripts` and `~/musubi-tuner` stay.

**Update** everything later by running `./install.sh` again in `~/LoraStudio`: new LoRA Studio files (your
settings stay), `git pull` of both trainers and their packages, without touching PyTorch. Options:
`--dir DIR`, `--gpu amd|nvidia`, `--gfx gfxNNNN`, `--import OLD_FOLDER`, `--reinstall-torch`,
`--no-models`, `--yes` (no questions). The PyTorch / ROCm versions are variables at the top of install.sh.

The trainers and every launch script activate `<trainer>/venv`, which is the shared venv, so nothing else
needs to know about it. start.sh uses the same venv for the GUI.

## Windows (through WSL2)

LoRA Studio, sd-scripts and musubi-tuner are Linux programs; on Windows they run inside **WSL2**, Microsoft's Linux
in Windows, which is also how the trainers and PyTorch work best on Windows. Unpack LoraStudio.zip and double-click
**Install LoRA Studio.cmd**. It runs `install-windows.ps1`, which:

1. installs WSL2 with **Ubuntu 24.04** when it is missing (Windows asks for administrator rights; restart Windows,
   choose an Ubuntu user name and password, then double-click the installer again),
2. looks at the graphics card and Ollama on the Windows side,
3. runs `install.sh` inside Ubuntu (the same three questions as on Linux; LoRA Studio goes to `~/LoraStudio` in
   Ubuntu),
4. puts a **LoRA Studio** starter on the desktop: it starts LoRA Studio in Ubuntu and opens
   http://localhost:7860 in your browser when it is ready. Closing its window stops LoRA Studio.

- **NVIDIA**: the normal Windows driver serves WSL too; nothing to install inside Ubuntu.
- **AMD Radeon**: training in WSL needs AMD's *ROCm on WSL* (Radeon RX 7000/9000, a recent Adrenalin driver):
  https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html. If that is not possible,
  answer *none* for both trainers: LoRA Studio prepares datasets and makes launch scripts, and you train on Linux.
  With a Ryzen CPU that has its own graphics, ROCm 10.1 in WSL can hang at the first GPU use (a Windows GPU
  timeout): disable the CPU graphics in the Device Manager, run `wsl --shutdown`, and try again.
- **Ollama** is installed in **Windows**, not inside Ubuntu: https://ollama.com/download/windows (guide:
  https://github.com/ollama/ollama/blob/main/docs/windows.md). The installer finds a running Ollama and, on Windows 11,
  offers *mirrored* networking (`networkingMode=mirrored` in `%USERPROFILE%\.wslconfig`) so LoRA Studio reaches it at
  http://127.0.0.1:11434. On Windows 10 it sets `OLLAMA_HOST=0.0.0.0` (asked) and uses Windows' WSL address
  instead; that address can change after a restart (Settings › AI helpers).
- Your Windows files are under `/mnt/c` in Ubuntu (for example `/mnt/c/Users/you/Pictures`), but datasets and models
  inside Ubuntu (`\\wsl$\Ubuntu-24.04\home\you` in the Explorer) are read much faster when training.

The Windows installer has been written and checked for mistakes, but not yet run on a real Windows computer: if a
step fails, its message says which, and running it again continues where it stopped.

## Updating LoRA Studio itself (Settings › Updates)

Pick the new **LoraStudio.zip** under *LoRA Studio itself*. It lists every part with the version you have and the one
in the zip (*changes*, *new* or *stays*) and what the update says. **Install this update** (refused while a training
run, the queue or the automatic improvement is busy):

1. only the program files are written (the parts in the zip's `version.py`, `requirements.txt`, `README.md`); your
   settings, databases, token, logs and venv are never touched, and nothing in the zip is run before it is in place
2. the files it replaces are kept in `update_backups/lorastudio_<old version>_<time>/` first
3. the self-test runs with the new files; when it fails, the old files go back by themselves
4. **Restart LoRA Studio** starts it again through `start.sh` (which installs new packages when `requirements.txt`
   changed); the open page reloads by itself when it is back

**Roll back the last update** puts the files of before it back (then restart). Settings › Version shows afterwards
which parts the update changed.

**From GitHub** (same card): *Updates from* chooses what counts: **Releases only** (the default: the newest version
published as a GitHub release, pre-releases not counted) or **Betas too** (the newest upload to the `main` branch).
*Check GitHub* asks github.com/druidican/LoraStudio for that commit and its `version.py` (a few kB) and says whether it is newer, a new upload of the same version, or
what you have. *Install from GitHub* downloads that commit as a zip into `userdata/update_backups/downloads/` and
installs it exactly as above (backup, self-test, back by itself), so a new upload with the same version number works
too: only the files that differ are written. The commit you updated from is kept in `userdata/github_update.json`.
Nothing is checked at start: only the buttons go online. While the repository is private, a **GitHub token** is
needed: a fine-grained token for this repository with *Contents: read*, saved with *Save token* in
`userdata/github_token.txt` (readable only by you, not in backups, removed from every report); an empty field removes
it.

## The training venv: PyTorch, ROCm and CUDA (Settings › Updates)

**The training venv** card checks at every start (no download) that every ROCm or CUDA part is the version PyTorch is
built for, and that `constraints.txt` locks what is installed. **Fix the training venv** puts parts back, writes the
locks again and, when PyTorch was upgraded, installs the requirements again within the new locks.

**Look for a newer PyTorch** asks AMD's server (`stable.repo.amd.com/rocm/whl-next`, for ROCm) or PyPI (CUDA) which
PyTorch builds exist; only stable PyTorch releases count, with the torchvision and torchaudio AMD pairs with them
(for PyTorch 2.14 that is an alpha torchvision). **Upgrade** (after *Yes, upgrade*) keeps the old versions in
`update_backups/venv_before_<time>.txt`, installs the new PyTorch, puts the ROCm parts to its version, tests the GPU,
and puts the old versions back when the test fails; then the locks and the requirements are renewed. Only the venv
install.sh made is changed; for your own venv it shows the command. `./install.sh` keeps a PyTorch that is newer
than its own (2.13.0+rocm10.1.0) and offers the upgrade to an older one.

## Updating sd-scripts and musubi-tuner (Settings > Updates)

When the program starts it quietly runs `git fetch` in both trainer folders (nothing is installed). **Settings >
Updates** then shows, per trainer, how many new commits its branch has, the newest commit messages and whether its
requirements changed; **Check for updates** checks again. **Update now** (needs the install.sh layout with the shared
venv; refused while a training runs, and a trainer with local changes to its files is skipped):

1. tests first: PyTorch on the GPU, bitsandbytes, and `--help` of sdxl / flux / zimage training (a test that already
   fails cannot fail the update)
2. a snapshot of both commits and every package version in `update_backups/<time>/` (the last 10 are kept)
3. `git pull --ff-only` on the branch you have
4. a pip **dry run** of both trainers' requirements with PyTorch / ROCm locked: a conflict stops here
5. the real install, then the tests again
6. if anything fails: back to the old commits and the old package versions, and the tests once more. Settings
   remembers that this version failed, so you are not asked to install it again and again
7. on success `constraints.txt` is written again: PyTorch locked as before, the trainer libraries
   (transformers, diffusers, accelerate, ...) at their new versions

## Install and start without install.sh

```bash
cd /path/to/LoraStudio                       # the folder with the files above
chmod +x start.sh
./start.sh                                  # opens http://127.0.0.1:7860
```

The **first start** creates `./venv` with `requirements.txt` when there is none (install.sh already made one), then
opens the **setup page** in the browser.

## The setup page (first start)

A new install starts with a short step-by-step page instead of the program. **Every step can be skipped**
(*Skip this step*), and everything can be changed later in Settings:

0. **Welcome**, with **Import from a backup**: pick a backup it found (your backup folder, `~/LoraStudio-backups`,
   Downloads, your home folder) or enter the path of one, and LoRA Studio starts with those settings. All other
   steps are skipped.
1. **Trainers**: sd-scripts and musubi-tuner (install.sh put them in the LoRA Studio folder; ✔ / ✖ shows whether
   they are there).
2. **Datasets**: your dataset folder(s); the first is where new datasets are made (created if missing).
3. **Models**: the folder with base models, text encoders and VAEs; optionally download the Z-Image Turbo
   training adapter into it.
4. **ComfyUI**: its LoRA folder(s) and the folder it saves pictures in (for the LoRA page › Pictures).
5. **Hugging Face token** (only for gated models), with a *Test the token* button.
6. **Ollama** (optional): the address and the vision model for captions and the picture check; *Look for models*
   lists what your Ollama has.
7. **Backups**: where Settings > Backup saves.
8. **Summary** -> *Finish and start LoRA Studio*: the settings are written, LoRA Studio restarts on the same
   address and the page turns into the program by itself.

Folders are suggested from what is on the computer: ComfyUI installs in your home folder, `/mnt`, `/media` and
`/opt` (with their `models/`, `models/loras/`, `output/` and the folders in `extra_model_paths.yaml`), and a
running Ollama. Show the page again with `./start.sh --setup` (it starts from your current settings).

## Backup and restore (Settings > Backup)

**Back up now** writes one zip, `LoraStudio-backup-<date>-<time>.zip`, into the backup folder (remembered; set on
the setup page or here): your settings (`library_settings.txt`, `training_settings.txt`), LoRA Studio's databases
(`completed.json`, `library_data.json` with your LoRA page edits and old LoRA names, `picture_index.json`,
`hash_cache.json`, `model_ids.json`), the training logs and, when ticked, your Hugging Face token (the zip is then
readable only by you; keep it private). Your datasets, models, LoRAs and pictures are **not** in it: they stay
where they are.

**Restore** (here, or *Import from a backup* on the setup page of a new install) puts those files back. The current
files are kept in `restore_backups/<time>/` first. On another computer or user, paths that started with the old
home folder or the old LoRA Studio folder are changed to the new ones; LoRA Studio then restarts.

Options are passed on: `./start.sh --port 7861`, `./start.sh --no-browser`,
`./start.sh --host 0.0.0.0 --auth user:password` (network access).
Start over with a clean GUI-only environment: `./start.sh --rebuild-venv` (not for the shared venv from
install.sh: use `./install.sh --reinstall-torch`).

## Where settings live

Each setting has one place:

- **Settings › Folders** (`library_settings.txt`): dataset folders, LoRA folders, extra launch script folders, the
  model folder and the ComfyUI picture folder. The first dataset folder is where new datasets are listed and created.
- **Settings › Trainers + GPU** (`training_settings.txt`, `[general]`): the sd-scripts and musubi-tuner folders,
  working folder, venv, trace, and the GPU / ROCm environment. A new sd-scripts folder needs a restart.
- **Settings › Model files** (`training_settings.txt`): the LoRA type first, then its model, VAE and text-encoder
  files, the suggestions, and LoRA subtypes (such as Pony). A missing file with a known source gets a
  **⬇ From Hugging Face** button (also *Download all missing files*, and the same card on Datasets › 9 Launch script):
  the usual file for that slot (from the sd-scripts / musubi-tuner documentation) is downloaded with your Hugging Face
  token into the models folder (`checkpoints`, `diffusion_models`, `text_encoders`, `vae`, `loras`) or the slot's own
  folder, and the slot is saved. *Download any file from Hugging Face* takes a pasted file link (…/blob/main/… or
  …/resolve/main/…) for any slot; files split in parts are fetched whole; a stopped download goes on from where it
  stopped (`<file>.part`). Gated repos (FLUX.1-dev, FLUX.2-dev, SD 3.5): accept the licence on the repo's page first.
- **Settings › AI helpers**: the Ollama address and vision model (captions, the picture check, the automatic
  preparation) and the Hugging Face token (`huggingtoken.txt`, paste a new one and press *Save token*).
- **Settings › Library** (`library_settings.txt`): style (Dark / Normal), base models, expected bases, NSFW blur, and
  *Colour the Collection cards by purpose* (a border and label colour for character, art style, object, action /
  pose and concept LoRAs, each colour your choice; off = white).
- **Settings › Start and network** (`launch_settings.json`): see above.

## Training › Run

- Two lists side by side with the same look and a fixed height (5 rows; a scrollbar appears when there are more):
  **Available scripts** (your launch scripts by file name, with their resumable states and epoch files) and the
  **Queue**. Click a row to select it (the line below the list says what is selected).
- Below the scripts: **Refresh list**, *Show epoch files*, *Show completed* and **Add selected to queue →**.
  Below the queue: Move up / Move down / Remove / Clear finished / Clear waiting, then *Stop on failure*,
  **Start queue**, **Stop after current** and **Stop now**.
- Each queue item ends as **done**, **failed** (non-zero exit code, or a Python traceback in the log, with the reason
  and the log file name) or **stopped**.
- **Between queue items**: every run is its own program, so when it ends the system frees all its video and normal
  memory; anything the trainer left behind in the run's process group (a helper process after a crash) is stopped
  too. When the next item trains with **other model files** (another base model, DiT, VAE or text encoder, for
  example Anima → SDXL), the queue first makes room: ComfyUI is asked to unload its models (its `/free`, when it
  runs at the address in Settings › Test pictures), Ollama's models are stopped, and it waits up to 90 s until the
  video memory is back near what was in use before the queue started (read with rocm-smi or nvidia-smi). The item's
  note says so (*model changed (Anima → SDXL): video memory free: 15.2 of 16.0 GB*). With the same model files the
  next item starts at once.
- The step counter is a **progress bar** that fills while training runs (orange), green when the run finished OK and
  red when it failed.
- A script that runs several stages (Z-Image Turbo) also shows a **stage row** above the bar: one chip per stage
  (done, running, failed, waiting) with its time, the step count on the training stage, and, when something breaks,
  the stage that failed with the last log lines. A **NaN loss** is reported with the step and epoch where it first
  appeared; such a run counts as failed (queue item *failed*, no Completed mark) even though the script itself ends
  with exit code 0. Studio does not stop a NaN run by itself: use **Stop now**, then retrain at half the learning rate.

## Recommended settings: purpose, priority and renewing

**Training › Launch script** has two choices that shape every new launch script (and the comparison for an existing one):

- **Purpose:** *Character / person*, *Art style* or *Concept / object / pose*. Styles get a higher rank, more varied
  images and fewer views per image; concepts a smaller rank; captions advice differs per purpose.
- **Priority:** *Balanced*, *Best quality* (about 25% more training, more checkpoints) or *Fastest* (about 30% fewer
  steps and 768 px instead of 1024 where the model allows it; SDXL and HunyuanImage keep their resolution).

The values come from `Training_Recommendations.py` (model family × purpose × priority): the step target is
*views per image × images*, kept within a range per family and purpose; then the usual plan picks batch,
accumulation, repeats and epochs. It also sets dim/alpha, learning rate, noise offset, resolution and a few
family-specific flags (Lumina's system prompt; HunyuanImage uses `--fp8_scaled`, not `--fp8_base`), checks
image sizes and caption style, and looks at the chosen model files (an SD3.5 Large file gets block swapping).
"Recommendations used" in the explanation lists every change and why, with the file's version and date.

**Renewing:** press **Export report for Claude**. It writes `LoRA_Studio_report_<date>.md` in the LoraStudio
folder: versions, GPU/ROCm/torch, model files per type, the last 12 runs (result, speed, loss, errors) and your
LoRAs with their settings, problems and your notes (Hugging Face tokens are masked). Add a few words at the end
about what worked and what did not, and send it with `Training_Recommendations.py`; put the returned file in place
of the old one. It is read fresh every time, no restart needed. `python Training_Recommendations.py` prints the table.

## Which model files belong together

In **Settings › Model files**, pick a LoRA type and a base model. The **Suggestions** panel below the file slots then:

- recognises every model file in the slot folders (and the `models` folder above them) from its header, without
  loading it: the model family (Chroma, FLUX.1-dev/schnell, SDXL, SD3, Lumina, ...), the kind of text encoder
  (T5-XXL, CLIP-L, CLIP-G, Qwen3, Gemma-2, ByT5) or VAE (FLUX AE, SD/SDXL VAE, Qwen-Image VAE), and its form:
  bf16/fp16, plain fp8, **ComfyUI fp8_scaled** or a **full checkpoint** with several parts bundled
- checks the base model against the LoRA type (for example a FLUX.1 file chosen for Chroma)
- lists the fitting files for every other slot, best first (★), and the files that do not fit, with the reason
- **Apply suggestions** fills empty or unsuitable slots (tick *Also replace files that already fit* to use all ★
  files); the base model is only swapped for a file of the same family. Nothing is saved until you press Save.
- **Look up online** searches Hugging Face (the official model page and pages matching the file name) and
  Civitai, and shows the lines about text encoders, VAE and settings, with links, plus a Reddit search link and
  the sd-scripts guide for that model family. LoRA folders, ControlNet and similar folders are skipped.

Results are cached in `model_ids.json`, so checking again is instant.

## Versions

All LoRAs of the **same LoRA type in the same dataset** count as versions of one LoRA (mychar · Anima:
`mycharAnimaV1`, `mycharAnimaV2`, also when a file is named differently, like `MyChar_Anima_v3`). The newest is the
highest version number in the file name (or in the training metadata's output name), then the latest training date.
Everywhere, **only the newest version is shown**:

- **Browse › Pictures**: one tile per **dataset** (mychar), showing the picture of its newest LoRA (if that one has no
  picture yet, the newest one that has). Below it, a label for every LoRA type the dataset has, in the order of
  the base models in Settings: `ILL` Illustrious, `PO` Pony, `ANI` Anima, `CHR` Chroma, `F1` Flux, `XL` SDXL, `ZI` Z-Image,
  `15` SD 1.5, `SD3`, `LUM` Lumina, `HY` HunyuanImage. A green label with ✓ is Completed. Click a tile to open its
  newest LoRA's page; click a label (`ILL`, `ANI`, ...) to open that type's page.
- **LoRA page › Pictures**: every picture you made with a LoRA, **all its versions together** (newest first, 120 per page,
  a **Version** filter, the version and strength under each picture). Click a picture to see it large with its
  prompt (copy button), the LoRAs and strengths used, model and seed. On the **LoRA page**, *Pictures made with this
  LoRA* opens it for that LoRA and scans right away.
  - Pictures are linked through ComfyUI's metadata, never by looking at them: the `.json` saved next to each
    picture (same file name), else the PNG's own metadata. What ran counts: bypassed / muted LoRA nodes and
    strength 0 are ignored. Understood: core LoRA loaders, rgthree Power Lora Loader, LoRA Manager, stacks,
    `<lora:name:0.8>` tags and "Lora hashes" lines. Pictures without metadata are skipped.
  - Where it looks: **Settings > Folders > ComfyUI picture folder**, in its folder for the LoRA type (`Anima`,
    `Illustrious`, `Pony`, ... also the short names `ANI`, `ILL` and the base model keywords, up to two levels
    deep). Not found: enter the folder on the LoRA page › Pictures and press *Scan for new pictures*; it is remembered
    for that LoRA type (the `[pictures]` section of `library_settings.txt`). *Scan all picture folders* scans the
    whole ComfyUI folder plus the folders you set.
  - **Old names**: pictures made before a LoRA was renamed (no version number, or an older naming) carry the old
    name. Open *Old names of this LoRA* on the LoRA page › Pictures: it lists names in your pictures that no LoRA file has
    now but that contain the dataset / character name, with the number of pictures. Tick them (or type a name) and
    press *Save old names*; those pictures join the gallery as *old name* (Version filter: *Old names*). Saved in
    `library_data.json`. Renamed files with the same content are also found by their hash when the PNG carries
    Image Saver's LoRA hashes and the LoRA's sha256 is known (LoRA Manager data or Personal sync).
  - The index is kept in `picture_index.json`: a new scan only reads pictures that are new or changed
    (file time and size of the picture and its `.json`) and drops deleted ones.
- **Browse › Status**: one card per dataset + type, showing the newest version. Click the card to open its LoRA page.
  Older versions are only shown when you ask: click the `V1–V3 · 3 versions ▾` chip on the card to fold them out
  below it (newest first, each opens its LoRA page); click it again to fold them in.
- **LoRA page**: the type buttons switch between the LoRA types of the dataset; the **Version** dropdown
  switches to older versions.

## LoRA types

New launch scripts are made **without sample images** (the option was removed: it costs a lot of time with the large
models). Existing scripts that have `--sample_prompts` keep it until you remove it from the script.


Every LoRA type lives in `Lora_Datatypes.py`, in its `PRESETS` list. Training › Launch script re-reads the file every time,
so edits apply without a restart; the tab shows which file is used and how many types it has. Types marked
`[template]` (SD 1.5, SD3, FLUX, Lumina, HunyuanImage) come from the sd-scripts documentation and are not yet
tested on your machine: set their model files in Settings › Model files and press **Check flags** first.
Subtypes such as Pony (a type with its own model files) are still added in Settings › Model files.

### Automatic preparation (standalone sdgui.py only)

The one-click preparation of earlier versions (check, repeats, folders, captions in a row) is only in `sdgui.py` run
on its own. In LoRA Studio the Datasets steps do the same, one step at a time, so you can look at each.

### Recaption with Ollama (tags or sentences)

**Datasets › 7 Tags** has a local vision model in [Ollama](https://ollama.com)
look at every image of the dataset's repeat folders and write a new caption that starts with the trigger word.

- **Caption format: Tags or Sentences.** The recommendation follows the LoRA type at the top of the tab and is
  applied when you change the type (or press *Use recommended*):
  - **Tags** for SDXL, Illustrious, Pony and SD 1.5 (CLIP: trained on booru tags, reads only the first 75 tokens)
    and Anima (trained on tags and sentences; tags are the usual choice).
  - **Sentences** for Z-Image, Qwen-Image, FLUX.1, FLUX.2, Chroma, SD3, Lumina and HunyuanImage (T5 or a language
    model reads the captions).
  - Purpose *Art style* raises the detail to *Everything* (and you should empty *Never describe*).
- **Length** (1 very short ... 5 very long): up to 20 / 35 / 55 / 80 / 110 words, or 8 / 14 / 20 / 28 / 40 tags.
- **Detail** (1 basic ... 4 everything): WHAT is described. Basic = clothing, pose, framing; Normal adds expression
  and place; Detailed adds light, camera angle, accessories and colours; Everything adds background objects, other
  people, photo style and composition. Length decides how much is said about those things.
- The **example** under the sliders changes as you move them (an invented photo; no model is called). For CLIP types
  a bar shows how much of the 75-token limit the caption uses: everything after token 75 is cut off in training.
- **Vision model**: press *Find models*; vision models are listed first, text-only ones are marked. None yet:
  `ollama pull qwen2.5vl:7b` (about 6 GB).
- **Never describe**: what always belongs to the person (face, eye and hair colour, freckles, tattoos, body). In
  sentences a caption that still mentions one is asked again once and listed if it still does; in tags those tags are
  simply dropped (the preview lists them).
- **Try on 3 images** writes nothing. **Recaption all images** replaces the captions; **Undo last change** restores
  them (a safety copy is written to `gui_logs/`). The `LoraManager` folder is never captioned. It refuses to run while
  a training run is active, and frees the model's VRAM when done.
- The address and model are saved in settings.txt (`[general]`: `OLLAMA_URL`, `OLLAMA_MODEL`).

### Check dataset pictures (quality, content, balance)

The full picture check behind **Datasets › 4 Sort** finds pictures that are not recommended for training, with
the reason for each:

- **Technical** (no AI, seconds): too small for the LoRA type (below its training size; far below = not
  recommended), blurry / out of focus (sharpness compared with the rest of the dataset), near-duplicates (the smaller
  or softer one is flagged), extreme aspect ratios, heavy JPEG compression, unreadable files.
- **Content** (the Ollama vision model chosen in *Recaption with Ollama*; a few seconds per picture; answers are
  remembered in `datacheck_cache.json`, so a second check is instant): subject clearly visible and in focus, number of
  people, face hidden or cut off, text / watermark / logo, strong filters, collages and screenshots.
- **Balance** (from the content answers): too many close-ups, no full-body pictures, the same outfit or place in
  most pictures (it would stick to the trigger word).

The rules follow the *Purpose*: **character** (one person, face visible, subject in focus), **style** (crowds are
fine; text, watermarks and UI are not), **concept** (one clear main subject). Not-recommended pictures are ticked;
**Move ticked to _set_aside** moves them with their captions to `<dataset>/_set_aside/from_<repeat folder>/`, which
no trainer and no LoRA Studio scan reads, and the step plan updates. **Move all back** restores them. No Ollama
check runs while a training run is active.

**Watermarks, stamps and site names.** For every picture the content check flags with text, the vision model is
asked where marks are that were **added on top** of the picture afterwards (watermarks, website names and handles,
logo stamps, copyright notes, signatures, camera date stamps, captions, stickers) and where the main subject is.
Text that belongs to the scene (signs, shirts, books) is not a mark: such pictures drop off the list. The review
under the check shows each picture with the marks in red and the subject in green, and the options:

- **Crop A / Crop B**: up to two crops that cut every mark off (with a safety margin, because the model's boxes are
  approximate) and keep the subject whole and the picture at least the LoRA type's minimum size. A is the biggest;
  B keeps the original shape best. The caption stays with the picture.
- **Set aside**: the picture and its caption go to `_set_aside/from_<repeat folder>/` (recommended when no crop
  works, e.g. a watermark across the middle).
- **Keep as is**.

Click an option per picture and press **Apply choices**, or **Use the recommended option for the undecided ones**.
The original of a cropped picture is kept in `<dataset>/_set_aside/originals/<repeat folder>/`, and the boxes and
your choices in `_set_aside/marks.json`, so every choice can be changed later (choosing *Keep as is* puts the
original back). *Move all back* leaves the originals alone. Nothing is painted over: a mark is cropped off or the
picture is not used.

### Z-Image Turbo (musubi-tuner)

**Z-Image Turbo** is trained with kohya-ss' *musubi-tuner*, not sd-scripts. Set the musubi-tuner folder in
**Settings › Model files** (`SDGUI_MUSUBI`; it needs a `venv/` inside) and the four model files for type *Z-Image Turbo*:
`--dit` (`z_image_turbo_bf16.safetensors`), `--vae` (FLUX `ae.safetensors`), `--text_encoder` (`qwen_3_4b.safetensors`)
and `--base_weights` (the ostris training adapter `zimage_turbo_training_adapter_v2.safetensors`, required to train on
Turbo). It is created and run like every other type, but the launch script is **all-in-one**. One run does:

1. **Check files**: musubi-tuner, its venv, the scripts it needs and the four model files exist
2. **Prepare dataset**: `dataset.toml` is built from your `<N>_name` folders (resolution, batch size and repeats), and the
   cache folder `<musubi-tuner>/cache/<name>` is emptied (the cache lives outside the dataset, because the Library
   scans every `.safetensors` file under a dataset)
3. **Cache latents**, 4. **Cache text encoder**, 5. **Train LoRA** (AdamW8bit, 1024 px, rank/alpha 32 = 32, about 2200 steps, block swapping 10 instead of fp8)
6. **Convert for ComfyUI**: every saved LoRA is converted in place (once; the model folder keeps `.comfy_converted`),
   so the Library and Personal sync see the files you can use directly. Use them in ComfyUI on the normal Z-Image
   Turbo model **without** the training adapter.

Batch size and resolution are variables at the top of the script (`BATCH_SIZE`, `RESOLUTION`), and the form fields of
the tab edit them. No sample images are made. `--network_alpha` is always set (musubi defaults it to 1). The type is a
`[template]`: press **Check flags** (it reads the real option list of your musubi-tuner) before the first run.

### Qwen-Image and FLUX.2 klein (musubi-tuner)

Both are trained like Z-Image: same musubi-tuner folder setting, same all-in-one script and stage row, no sample images.
They differ in two ways: the script has **five stages** (no conversion, because musubi's documentation names no
ComfyUI converter for them, so the saved LoRAs are used as they are; test the first epoch in ComfyUI), and a
`MODEL_VERSION` line at the top (form field *Model version*).

- **Qwen-Image** (type *Qwen-Image*): model files `qwen_image_bf16`, `qwen_2.5_vl_7b` and `qwen_image_vae` (plain bf16, never
  fp8 files). 20B parameters, so the script uses `--fp8_base --fp8_scaled --blocks_to_swap 45` and `--fp8_vl` while caching
  the text encoder. System RAM is the limit (32 GB is tight); dim 16, LR 5e-5, 1024 px (documented example values).
- **FLUX.2 klein** (type *FLUX.2 klein*): train on a **base** model. Default `klein-base-4b` (fits 16 GB without swapping);
  for `klein-base-9b` change `MODEL_VERSION`, the DiT and the Qwen3-8B text encoder together. Pick the first part of the
  split text-encoder files. Dim 32, LR 1e-4, `flux2_shift`. The AE is the FLUX.2 `ae.safetensors`.

FLUX.1 Kontext is not included. All three musubi types are `[template]`: run **Check flags** first.

### Chroma

**Chroma1-HD** (`chroma_v10HD.safetensors`, 8.9B, based on FLUX.1-schnell) is trained with sd-scripts' FLUX trainer
in Chroma mode. Set its three model files in **Settings › Model files** (type *Chroma1-HD*): the Chroma model, **T5-XXL**
(`t5xxl_fp16`) and the FLUX **AE** (`ae.safetensors`). There is no CLIP-L. The recommended script uses:

- `--model_type chroma`, `--apply_t5_attn_mask` and `--guidance_scale 0.0` (all three required by sd-scripts for Chroma)
- `--timestep_sampling sigmoid --model_prediction_type raw` (recommended for Chroma)
- `--fp8_base`, gradient checkpointing, cached latents and T5 outputs, batch 1 plus accumulation (16 GB);
  add `--blocks_to_swap 8` (or more) if you run out of memory
- LR 1e-4 AdamW8bit, dim 16 / alpha 16, about 1600 steps over ~10 epochs, a checkpoint every epoch, `--save_state`

The analysis also checks two things specific to Chroma: whether your captions are only short tag lists (Chroma reads
captions through T5 and learns more from short sentences that start with the trigger word), and whether images are
smaller than about 896×896 pixels (they would be upscaled to the 1024 training size). It needs an sd-scripts
version that knows `--model_type` (2025 or newer): press **Check flags** before the first run.

T5-XXL must be the **standalone** encoder (`t5xxl_fp16.safetensors` or `t5xxl_fp8_e4m3fn.safetensors`, the same files ComfyUI uses for FLUX), not a full checkpoint that contains it (like `sd3_medium_incl_clips_t5xxlfp16`). The analysis now warns when any text-encoder or VAE/AE slot points to a full checkpoint; sd-scripts would otherwise stop with *Cannot copy out of meta tensor*. Do not use ComfyUI **fp8_scaled** files (`..._fp8_scaled...`, `t5xxl_fp8_e4m3fn_scaled`) for training either: sd-scripts ignores their scale values and trains on wrong weights without an error. Use the bf16/fp16 files; `--fp8_base` converts them to fp8 itself. The analysis warns about these too. In the Library,
Chroma LoRAs show as base model **Chroma** (label `CHR`); existing settings get Chroma added in front of Flux.

## Copy to the LoRA folder, and cleaning up

**Copy a finished LoRA** (Training › Results, the copy buttons of a run; or the LoRA page › Overview, *Copy this LoRA
to your LoRA folder*). The target is found for you and can be changed before copying:

- the folder where an older version already is (mycharAnimaV2 goes next to mycharAnimaV1, in any LoRA folder),
- else, when the same file is already in a LoRA folder, that folder,
- else a folder for its type in your first LoRA folder (`<LoRA folder>/Illustrious/`, `/Anima/`, `/Pony/`, ...).

**Collection › LoRA page › Edit** fills the empty fields from the training when you open it: the trigger word (the
dataset's), the character (the dataset), notes (epochs, steps, learning rate, dim / alpha, resolution, base model,
final loss, from the LoRA file, or its launch script when the file says little), strength and CLIP strength 1.0. They
are not saved until you press **Save**; the line above the fields says which were filled.

Copying from Training › Results also writes the **LoRA Manager data** next to the copy (`<name>.metadata.json`): the
trigger word, the base model, tags (purpose and type) and notes with the training (dataset, pictures × repeats,
epochs, learning rate, dim / alpha, resolution, base model file, which save point). An existing file keeps everything
you filled in: only empty fields are filled. After every finished run the same data goes to the dataset's
`LoraManager/` folder, so Collection › LoRA page › Edit starts filled in.

The final file is picked by default; an epoch file can be picked instead. An identical file already there is left
alone; a different file with the same name is only replaced when you tick that (the old one goes to the Trash).

**Collection › Maintenance › Clean up** (everything removed goes to your desktop **Trash**: restorable from the file manager,
the space is freed when you empty it; `gio trash`, or the freedesktop Trash directly when gio is missing):

- *Training leftovers of finished LoRAs (all datasets)*: epoch checkpoints, `-state` folders, `.tmp.safetensors`
  files, `.npz` latent caches and musubi-tuner's `cache/<dataset>` folders. Only for LoRAs that have a final file
  (an unfinished one keeps its epochs and states so it can be resumed); a dataset with a running training is skipped.
  *Find leftovers* shows a table per dataset with sizes; nothing moves until you tick the confirmation.
- *Orphaned LoRA Manager meta files*: a `.metadata.json` / `.civitai.info` with its preview and example images whose
  LoRA file is gone. In a LoRA folder: no `<name>.safetensors` next to it. In a dataset's `LoraManager/`: the LoRA
  is in neither the dataset's `model/` folder nor any LoRA folder (your LoRA page edits live there). Images without
  such a meta file are never touched. Every group can be unticked before *Move ticked to Trash*.

*Clean up this run* in Training › Results uses the desktop Trash too, and keeps the epoch files you copied.

## Completed

A **Completed** mark means: this **version** of the LoRA of this LoRA type for this dataset is finished (for
example `mychar · Anima V3`). A new version starts unmarked, so a new V4 launch script shows on Training › Run even when
V3 is Completed. Marks made before versions were tracked count for the versions up to the one they were set for
(`mycharAnimaV3` -> V1-V3).

- **Set automatically** when a training run in the queue ends without error *and* wrote its final
  `.safetensors`. The log says so. A failed or stopped run never sets it.
- **Training › Run**: a launch script whose version is completed is hidden from the launch scripts
  (press *Refresh*). Tick **show finished** to see them again, marked ✓.
- **the Collection**: completed LoRAs get a ✓ badge in Browse. On the **LoRA page**, the
  **Completed** checkbox ticks or unticks it by hand, for example to train that type again.
- **Many at once**: in Browse (either look), tick **Select** above the cards. Clicking a card now
  selects it (✓ in its corner) instead of opening it. *Select all shown* picks every card the current
  filters show, and *Clear selection* empties it. Then press **✓ Mark Completed** or **Mark not completed**.
  A Status card marks its newest version. A Pictures tile is a whole dataset, so it marks the newest version of
  **every** LoRA type of that dataset. Untick **Select** to
  click cards normally again.
- Saved in `completed.json` in the LoraStudio folder.

## What is scanned: only your datasets

the Collection only reads your **dataset folders**. Your LoRAs are the trained `.safetensors` in each dataset's
`model/` folder; their LoRA Manager data (`.metadata.json` and preview) comes from the dataset's `LoraManager/`
folder. The LoRA folder (with downloads and everything else) is **not** scanned, and downloads are not shown.

**Personal sync** (tab in the Collection) is the only thing that reads the LoRA folder. Press
**Sync from LoRA folder** to copy the LoRA Manager files of your own LoRAs (`.metadata.json`, preview and example
images) into each dataset's `LoraManager/` folder:

- A LoRA in the LoRA folder belongs to a dataset when it has the same name and size as a model file in the
  dataset, or the same SHA256 hash (a renamed copy, like `MyChar_Illustrious` for `mycharILLV1`).
- The files are stored under the dataset model file's name (`mycharILLV1.metadata.json`,
  `mycharILLV1.preview.webp`), so the cards find them.
- Pictures are copied when they are missing in the dataset or newer in the LoRA folder.
- The `.metadata.json` files are synced **both ways**: a field that only one side has, or that is empty on one side
  (empty text or list, 0), is copied to the other, also into the LoRA folder's file, so ComfyUI's LoRA Manager sees
  your LoRA page edits and LoRA Studio sees what LoRA Manager added. Where both sides have a different value, the
  newer file wins. Location fields (file name, file path, folder, preview) stay as they are on each side. A `.bak` is
  made before either file is changed; files that already agree are not touched.
- Some fields follow their own rule: the **version name** (LoRA Manager shows it as *Version*, `N/A` when empty) comes
  from LoRA Studio (the training card, or what you typed in Edit) and overwrites LoRA Manager's; **trigger words** come
  from LoRA Manager when LoRA Studio has none, and from the dataset's trigger word when neither has any; the **usage
  tips** (strength, CLIP strength, clip skip) are merged one by one, and with no clip skip anywhere both get the
  default, **1**. The training card itself stays in LoRA Studio's copy only.
- The **preview** LoRA Manager's card shows (its `preview_url`, the picture you chose there) becomes the preview on the
  LoRA page, and LoRA Manager's **example pictures** (the ones you added on its card, kept in its example images folder
  per SHA256, and `<lora>.<n>.png` next to the LoRA) are **linked**, not copied: their paths are kept in the dataset's
  LoRA Manager copy, the LoRA page's Overview shows them under the preview, and the Browse card says *N examples*.
  The folder is read from LoRA Manager's `settings.json` (`~/.config/ComfyUI-LoRA-Manager/` or its custom node
  folder); *LoRA Manager's example pictures folder* (Maintenance) sets it by hand. Sample images from training are not
  shown.

**Insights** (Collection › Insights) reads your datasets, LoRAs and run logs on this computer. A version counts as
*good* when you rated it 4-5 (LoRA page › Edit › *Your rating*), or, not rated, when it is the ★ best version or you
made at least 3 pictures with it; *weak* when rated 1-2; other versions are left out of the good/weak split. Per LoRA
type, the medians of the good ones are shown next to the weak ones (or all others when none is rated weak): pictures,
pictures × repeats per epoch, steps, close-ups, full body, outfits, places, detail shots and caption length, with a
note where they differ clearly (only with at least 2 on each side). Framings, outfits and places come from the
dataset's sort as it is now. **Your graphics card** groups every run log by LoRA type and batch size / resolution:
finished, out of video memory, other failures, and the median seconds per step. Step 5's suggested size shows what
your good LoRAs of the type had; step 9 shows *Your graphics card* (a warning when the batch ran out of memory
before, or that it finished before) and takes the training time from your runs at the same batch size.
**Settings that work for you**: per LoRA type the medians (learning rate, dim, alpha, epochs, steps, batch) and the
most used optimizer, scheduler and resolution of the good versions next to the others, with a note where they
differ by a quarter or more; step 9 shows *Your good LoRAs* with them. **Training too long?**: per LoRA type, where in
the run the save point to try first was (the epoch it advised ÷ the epochs); with 3 or more runs and a median of 80 %
or less, step 9 says how many of the planned epochs may be enough. **How you use your LoRAs**: from the picture index
(LoRA page › Pictures; old names count too), the pictures made with each LoRA, the median strength, when last, and
the LoRAs never used with their size. The LoRA page's Overview says *Used in N of your pictures, usually at strength
X*, and Edit offers that strength when none is set (3 pictures or more). Results are kept for 10 minutes; *Read
again* reads everything again.

**Training cards.** The LoRA page's **Training** tab shows what each version was trained with, in separate fields in
four groups: *Dataset* (dataset, LoRA type, version, trigger words, pictures, repeat folders, captions), *Schedule*
(epochs, steps and how they add up: pictures × repeats ÷ batch ÷ accumulation × epochs, batch, gradient accumulation,
resolution), *Network and optimizer* (base model, trainer, dim / alpha, learning rate, optimizer, scheduler, clip skip,
noise offset, min SNR gamma, precision, seed) and *Result* (trained on, training time, final loss, the save point to
try first, GPU). They are read from the LoRA file's training data first, then its launch script (the one whose
`--output_name` is this version, also in `_old_runs/`), the plan line of the script and the repeat folders, the run log
in `userdata/gui_logs/` and the TensorBoard log. The card is saved in the dataset's `LoraManager/<lora>.metadata.json`
under `lorastudio`, with a scanned mark (the LoRA file's size and time): **Collection › Maintenance › Scan all LoRAs**
reads every version once, a version is read again only when its file changes, and a version that was never read is
read when its LoRA page opens. *Read the training again* (Training tab) reads one version again. The facts cannot be
edited; **Edit** has *Version*, *Clip skip*, *Description*, *Trigger words*, *Notes* and the strengths. The long
training text older versions wrote into *Notes* is taken out at the scan (your own notes stay). New runs write the
GPU's name at the top of their run log, so their card shows it.

After updating from an older LoRA Studio, press **Sync from LoRA folder** once so every dataset has its
pictures and LoRA Manager data under the right names.

## Scanning and the saved scan

- After a scan, the result is saved in `library_cache.pkl` in the LoraStudio folder. Later starts, page
  reloads and tab switches use it and do **not** read the folders again. The Library's *scanned ...* line
  ends with **(cached)** when the saved scan is used.
- The folders are read again automatically when files were **added or deleted** since the last scan (checked
  on every page load, which takes a few milliseconds), when you press **Rescan folders**, after a Personal
  sync, or when the dataset folders or base models in Settings change. A deleted `.safetensors` disappears from the lists
  on the next page load; a leftover `.metadata.json` or preview without its `.safetensors` is ignored. Style, NSFW blur and the other settings don't affect the
  saved scan.
- The saved scan is loaded as soon as the program starts, before the browser opens. Only one scan runs at a
  time: a second browser tab or a page reload during a scan waits for it instead of starting another.
- The terminal says what happened: `Loaded the saved scan from ...`, `Scanned N LoRAs ... in X s`, or why the
  saved scan was not used.
- Pictures in Browse and on the LoRA page are small thumbnails in `.thumbs` (made once, in the background after
  start), so pages stay small and the browser keeps them.

## Style

The **Dark / Normal** button at the top switches straight away. The default for each start is set in
Settings › Library.

## Running the parts on their own

Both still work alone: `python sdgui.py --root ~/sd-scripts --images-dir <dataset folder>` (then it uses
`<sd-scripts>/settings.txt` and `huggingtoken.txt` as before) and `python lora_library.py`.
