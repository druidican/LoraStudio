# First start and settings

The first start opens a setup page (folders, model files, ComfyUI, Hugging Face token, Ollama, a backup to import).
Every step can be skipped and changed later in **Settings**, which opens on the **Health check**.

![Settings › Health check: what is set and working, and what needs a look](img/health.jpg)

Each line that needs a look has a link straight to the place where it is fixed. Numbers next to the items on the
left say how many things need a look there.

## Folders

![Settings › Folders](img/set_folders.jpg)

- **Dataset folder**: one folder per dataset inside it (for example `~/Datasets/pip/`).
- **LoRA folder**: where ComfyUI reads LoRAs. Finished LoRAs are copied there from Training › Results.
- **ComfyUI picture folder**: where ComfyUI saves pictures, so the LoRA page can show the ones made with a LoRA.

## Model files

![Settings › Model files: the base model, VAE and text encoders per LoRA type](img/set_models.jpg)

Pick the files once per LoRA type (Illustrious, SDXL, Anima, …). New launch scripts use them. *Suggestions* reads
the header of every model file and marks the ones that fit the type.

A file you don't have yet? Next to a missing file there is **⬇ From Hugging Face**: it downloads the usual file for
that slot into your models folder and fills the slot. With your Hugging Face token (Settings › AI helpers) gated models
work too, once you have accepted their licence on huggingface.co. For another file, open *Download any file from
Hugging Face* and paste the file's link.

> **Don't** pick a model of another family (an SD 1.5 checkpoint for SDXL, a VAE as the checkpoint): the run fails
> with "The model does not fit this LoRA type".

## Trainers and GPU

![Settings › Trainers and GPU](img/set_trainers.jpg)

The sd-scripts and musubi-tuner folders, the GPU value (AMD: the gfx version), and the picture tools (watermark
removal, upscaling, quality rating). If you installed with `install.sh` and chose *use my own trainers*, these point
at your folders and venv.

## Test pictures (ComfyUI)

![Settings › Test pictures](img/set_comfy.jpg)

Import your own ComfyUI workflow per model family once (Workflow › Save in ComfyUI). LoRA Studio then makes test
pictures with every save point of a run, with the same model, sampler and size as your workflow.

## Start and network, and the login

**Settings › Start and network** sets the port, whether the browser opens at start and the start page. Tick
*Reachable on the network* to open LoRA Studio from another computer or a tablet; it then needs a **login** (a user
name and a password), and asks for it with the login fields on the painted logo:

![The login page](img/login.jpg)

The same page has **What is installed**: see [Installing](#install).
