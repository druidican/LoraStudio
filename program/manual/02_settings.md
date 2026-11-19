# First start and settings

## The first start

The very first page is a short **notice**: sd-scripts and musubi-tuner belong to kohya-ss, and LoRA Studio is a GUI
for them. Tick *I have read this and agree* and press *Continue to LoRA Studio*; it is shown only once.

Then the **setup page** asks seven things, one at a time: the trainers, your dataset folder, your model folder,
ComfyUI (the LoRA folder and the picture folder; installs it finds are offered), a Hugging Face token (*Test the
token* checks it), Ollama (*Look for models* lists the ones you have) and the backup folder. **Every step can be
skipped** and changed later in Settings. Moving from another computer? Open **Import from a backup** at the top and
pick a backup zip: your settings come back and LoRA Studio starts. **Find on this computer** (on the first page)
searches folders you name and fills in the steps with what it finds (see below); you still go through them and press
Finish. *Settings › Health check › Run the setup page again*
shows it again later (or `./start.sh --setup`).

## Settings

**Settings** has a list on the left, grouped like the work: Start, Datasets, Training, Results, Collection and Care. A number next to an item
says how many things need a look there. It opens on the **Health check**.

![Settings › Health check: what is set and working, and what needs a look](img/health.jpg)

Each line that needs a look has a link straight to the place where it is fixed. *Check again* reads everything anew.

## Start and network

How LoRA Studio starts: the port, whether the browser opens, the start page. Tick *Reachable on the network* to open
LoRA Studio from another computer or a tablet; it then needs a **login** (a user name and a password), asked for on
the painted logo:

![The login page](img/login.jpg)

*Save and restart* uses it at once; `./start.sh` and the desktop starter use it from then on (an option typed on the
command line still wins). The same page has **What is installed**: see [Installing](#install).

## Find on this computer

Write the folders to search, one per line (for example `/mnt/Backup` and `~/AI`); only these and their subfolders
(at most 8 deep) are searched, and the list is remembered. **Search** looks for:

- the trainers: sd-scripts (a folder with `sdxl_train_network.py`) and musubi-tuner (`src/musubi_tuner`);
- ComfyUI (`main.py` with `comfy/`): its `output` folder for the pictures, its `models` and `models/loras` folders, and
  LoRA Manager's example pictures folder (from LoRA Manager's own settings);
- your LoRA folder (where LoRA files are, not the `model/` folders of datasets) and the folder that holds your
  datasets (a dataset is a folder with an `<N>_name` repeat folder of pictures);
- every model file: its header is read (once; a second search is quick) to know what it is, and for each LoRA type
  the fitting files are listed per slot, best first. SDXL, Illustrious, Pony and NoobAI files look the same inside,
  so for their base model the file name decides (`Illustrious-XL…` for Illustrious, `NoobAI…` for NoobAI, `…vpred…`
  for V-pred). When the name says nothing, the short label some downloads carry inside (title, prediction type) is
  read instead; a file where neither says it stays unticked (rename it, and search again).

Each suggestion has a tick and a list with the other finds; it is ticked when nothing usable is set now (empty, not
there, or the preset's `/PATH/TO`). **Use the ticked** saves only the ticked ones: a found dataset or LoRA folder is
added in front of the ones you have; model files go to that LoRA type's section, as in Model files. Hidden folders,
venvs, caches, `_reserve`, `_originals` and system folders are skipped.

## Folders

![Settings › Folders](img/set_folders.jpg)

- **sd-scripts folder**: where the trainer is (with its `venv/`).
- **Dataset folders**: one folder per dataset inside them (for example `~/Datasets/pip/`). New datasets are made in
  the first one. Several folders: one per line.
- **LoRA folders**: where ComfyUI reads LoRAs, together with LoRA Manager's files. Finished LoRAs are copied there
  from a dataset's Results, into the folder of their LoRA type.
- **Extra launch script folders** (optional) and the **Model folder** (where the
  model file lists start).
- **ComfyUI picture folder**: where ComfyUI saves pictures, so the LoRA page can show the ones made with a LoRA.

*Save folders and rescan* reads the Collection again with the new folders.

## Model files

![Settings › Model files: the base model, VAE and text encoders per LoRA type](img/set_models.jpg)

Pick the files once per LoRA type (Illustrious, SDXL, Anima, …): the base model, VAE and text encoders. New launch
scripts use them. *Refresh file lists* reads the folders again; *training_settings.txt as it is on disk* shows the file itself.

- **Suggestions: which files fit**: every model file is read from its header (model family, text encoder or VAE
  kind, fp16 / fp8); the best match for each slot is marked ★. *Apply suggestions* fills the empty or unsuitable
  slots; nothing is saved until you press *Save*. *Look up online* opens the file's page on Hugging Face or Civitai.
- **A file you don't have yet**: next to a missing file there is **⬇ From Hugging Face**: it downloads the usual file
  for that slot into your models folder and fills the slot. *⬇ Download all missing files* does it for the whole
  type, *Stop download* stops (a download goes on from where it stopped next time). With your Hugging Face token
  (AI helpers, below) gated models work too, once you have accepted their licence on huggingface.co. For another
  file, open *Download any file from Hugging Face* and paste the file's link.
- **LoRA subtypes** (for example Pony): a subtype trains exactly like its base type but has its own model files and
  file tag (`pipPONYV1`). *Add subtype*, then pick it as LoRA type above and choose its files.

> **Don't** pick a model of another family (an SD 1.5 checkpoint for SDXL, a VAE as the checkpoint): the run fails
> with "The model does not fit this LoRA type".

## Trainers and GPU

![Settings › Trainers and GPU](img/set_trainers.jpg)

The sd-scripts and musubi-tuner folders and their venv, and the GPU (*Detect GPU*; AMD: the gfx version). If you
installed with `install.sh` and chose *use my own trainers*, these point at your folders and venv.

- **Picture tools**: quality rating, look-alikes, crops around the person, watermark removal and gentle upscaling.
  They run in the sd-scripts venv on your GPU, never during a training run, and stop after 5 idle minutes. *Install
  the picture tools* adds what is missing.
- **Advanced: low-level trainer and GPU values**: environment values for the trainers (*Save environment*), and
  *Update all launch scripts* writes changed values into the scripts you already have.

## Speed test

Which settings make each LoRA type train fastest on **your** computer? The table lists every LoRA type: whether it
can be tested (its model files and trainer are there), its last test (how fast it trains now, and the fastest that
was found) and whether the faster settings are in use.

- Tick the types to test, choose how thorough (*Quick*, *Full*, or *Only how fast it trains now*) and press *Start
  the speed test*. Test a few types now and others later: every result is kept.
- Each type trains a few steps on an invented test dataset (20 plain pictures; none of your datasets is used) with
  one setting changed at a time: fewer swapped blocks, pinned memory for block swapping, no gradient checkpointing,
  other attention (sage, flash, xformers), fp8, torch.compile, and the ROCm settings TunableOp, MIOpen fast search and
  expandable memory. Only what your sd-scripts or musubi-tuner support is tried. A setting that runs out of memory or
  gets a NaN loss is reported as such; then the faster ones are tried together.
- **Results**: pick a type to read its report (each setting with seconds per step, the change against now and the
  video memory). Tick **Use the faster settings for new launch scripts of this type** to make every new launch
  script of that type use the fastest result; your existing scripts stay as they are (make a new version in step 9).
  Untick to go back. A later test of the same type updates the settings in use.
- Close ComfyUI before a test; Ollama is stopped by itself. The training queue does not start while a test runs,
  and a test does not start during a training run. *Stop* ends it after the running step.

## AI helpers

![Settings › AI helpers: the Ollama models, which one is the default, start and stop](img/set_ai.jpg)

**Ollama** reads pictures and writes captions. Set its address and press *Test*. The **default model** (the menu at
the top, or the round button in the table) is used for captions, the sort and the picture checks; only models that
can read pictures can be the default. The table shows every installed model with its size and whether it is in
memory: **Start** loads one ahead of time (only one runs at a time, so the running one stops first), **Stop** frees
the memory, *Stop Ollama* stops all. *Get another model* downloads one by name (for example `qwen2.5vl:7b`) or removes
one. Starting a training run stops Ollama to free video memory; it is not started again by itself. The top bar always
shows which model is running.

**Faster sorting**: the sort and the detailed scan send Ollama **three pictures at once**. Ollama answers them together
only when it is allowed to; on Ubuntu, once:

```
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_NUM_PARALLEL=3"\nEnvironment="OLLAMA_FLASH_ATTENTION=1"\nEnvironment="OLLAMA_KV_CACHE_TYPE=q8_0"\n' | sudo tee /etc/systemd/system/ollama.service.d/speed.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Without it Ollama answers them one after the other, as fast as before.

**Hugging Face token**: for gated models and the downloads above (a *Read* token is enough). It is kept readable only
by you, written into launch scripts that download models, and masked in the run log.

## Test pictures (ComfyUI)

![Settings › Test pictures](img/set_comfy.jpg)

The ComfyUI address (*Test* checks it) and your LoRA folder as ComfyUI sees it. Import your own ComfyUI workflow per
model family once (Workflow › Save in ComfyUI): SDXL, Illustrious, Pony, NoobAI-XL EPS, NoobAI-XL V-pred and Anima
each have their own. A family without one uses your SDXL workflow (and its fixed tags), as before. **Use LoRA
Studio's workflow** sets one up with the model card's advice, at 832×1216, with the checkpoint set for that LoRA
type in Settings › Training:

| Family | In front of every test prompt | Negative prompt | Sampler, CFG, steps |
|---|---|---|---|
| SDXL 1.0 | – | worst quality, low quality, blurry | DPM++ 2M Karras, 7, 28 |
| Illustrious-XL | masterpiece, best quality | worst quality, comic, multiple views, bad quality, low quality, lowres, displeasing, very displeasing, bad anatomy, bad hands, monochrome, extra digits, fewer digits | Euler a, 6, 28 |
| Pony Diffusion V6 XL | score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_anime | – (not needed, says the model card) | Euler a, 7, 25, CLIP skip 2 |
| NoobAI-XL EPS | masterpiece, best quality, newest, absurdres, highres, safe | worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands | Euler a, 5.5, 28 |
| NoobAI-XL V-pred | as EPS | as EPS | Euler, 5, 28, v-prediction with zero terminal SNR |

The same files are in `program/Training/workflows/` to open in ComfyUI. A V-pred LoRA's test pictures always use
v-prediction, also with a workflow of your own. LoRA Studio then makes test pictures with every save point of a run,
with the same model, sampler and size as your workflow: how many prompts, and whether they are made by themselves.
When they are, they are made **right after each run, before the next one in the queue starts**: for the save point
to try first and the final file. Then ComfyUI is told to unload its models, so the next training gets the whole GPU,
and the LoRA card gets the first test picture as its preview. ComfyUI must be running during the queue; when it is
not, the pictures are made as soon as it answers.

**Upscale workflow** (dataset pictures): your own ComfyUI workflow for the Detailed scan's upscales, sharpening and
detail crops, for example **SeedVR2** (with a GGUF model if your video memory needs it). Save it in ComfyUI (*Workflow ›
Save*, or *Export (API)*) and drop the file here. It runs as it is; LoRA Studio only puts each picture into its *Load
Image* node and takes the result from its *Save Image* node. In a *SeedVR2 Video Upscaler* node it also sets:

- the **size**: the short side the LoRA type trains at, and a cap for the long side;
- the **noise** (*Automatic*): none for sharpening at the same size; input noise 0.1 up to 0.4 the more a picture grows
  (it hides the blocks and smearing of small sources); a little latent noise above 3×, which can soften detail.
  *The workflow's own values* leaves them as you set them.

Before an upscale, LoRA Studio unloads the Ollama models, so the two don't share the video memory.

**Inpaint workflow** (removing watermarks): your own ComfyUI inpaint workflow for step 6's *Remove* and *Paint over
it yourself*, for example an SDXL / Illustrious one (Load Image → Grow Mask → Inpaint Model Conditioning → KSampler →
VAE Decode → Image Composite Masked → Save Image). It needs one *Load Image* node whose MASK output says what to fill
in. LoRA Studio cuts a window of about 1024 px around each mark, sends it with the mark as the mask (the transparent
part), gives the KSampler a new seed and pastes back only the filled-in part, so the rest of the picture stays as it
was. Marks close together share one window. While ComfyUI is not running, the picture tools fill in instead.
*Stop using it* goes back to the picture tools.

## Look and base models

How the Collection looks and how it tells which base model a LoRA was trained on:

- **Base models**: `Name = keywords`, checked top to bottom against a LoRA's own data (for example
  `Illustrious = illustrious, nova`).
- **NSFW previews**: blur them (hover to show), from which level, and also when the level is unknown.
- **Colour the Collection cards by purpose**: a border and a label in the colour of the purpose (character, art
  style, object, worn item, action / pose, concept). Pick the colours and press *Save colours*; *Default colours* goes back; off
  keeps every card white.
- The moon button at the top right of every page switches between dark and light.

## Backup, Updates and Version

See [Updates, backups and the machine check](#care).
