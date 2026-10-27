# Installing LoRA Studio

A step-by-step walkthrough for **Linux** and **Windows**: what you need first, the install itself, the first start,
updates, moving to a new computer, removing it again, and what to do when something goes wrong.

LoRA Studio was made on Fedora Linux with an AMD Radeon (ROCm). NVIDIA and Windows are supported by the installer but
have not been tried on real hardware yet. If you test one of them, please send the result with the
[Test result form](../../issues/new?template=test_result.yml).

**Contents:** [Before you start](#before-you-start) · [Linux](#linux) · [Windows](#windows) ·
[The first start](#the-first-start) · [Starting it later](#starting-it-later) · [Updating](#updating) ·
[A new computer, or a reinstall](#a-new-computer-or-a-reinstall) · [Removing it](#removing-it) ·
[When something goes wrong](#when-something-goes-wrong)

---

## Before you start

**Get LoRA Studio**: download `LoraStudio.zip` from the [Releases](../../releases) page (newest at the top). You can
also clone the repository instead: `git clone https://github.com/druidican/LoraStudio.git`.

**What you need**

| | |
|---|---|
| System | Linux (tested on Fedora 44; Ubuntu, Debian and Mint are supported by the installer), or Windows 10/11 with WSL2 (the installer sets it up) |
| Graphics card, to train | **AMD Radeon** with ROCm (tested: RX 9070 XT, 16 GB) or **NVIDIA** with CUDA. SDXL, Illustrious and Pony fit 12-16 GB; FLUX and Qwen-Image need more |
| Disk | about 25 GB with the trainers (PyTorch, sd-scripts, musubi-tuner), about 2 GB without, plus your models |
| Internet | for the install (downloads PyTorch, the trainers and, if you want them, Ollama and its model) |

**Optional**
- [Ollama](https://ollama.com) with a vision model: the detailed sort of your pictures and sentence captions. The
  installer can install it on Linux.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI): test pictures of your LoRAs with your own workflow.
- A [Hugging Face token](https://huggingface.co/settings/tokens) (a *read* token is enough): for gated models such as
  FLUX.1-dev, and for downloading missing model files from inside LoRA Studio.

Without a usable graphics card LoRA Studio still prepares datasets and writes launch scripts that you can run on
another computer.

---

## Linux

### 1. The things the installer needs

The installer checks for these and offers to install what is missing, so you can usually skip this step.

- **git** and **Python 3.12**
  - Fedora: `sudo dnf install python3.12 git`
  - Ubuntu / Debian / Mint: `sudo apt install python3.12 python3.12-venv git`
- **AMD Radeon**: the normal `amdgpu` driver (part of the Linux kernel). ROCm itself does **not** need to be
  installed on your system: the installer puts the ROCm libraries inside LoRA Studio's own environment. Your user
  must be in the `render` and `video` groups:
  ```bash
  sudo usermod -aG render,video $USER
  ```
  then log out and in again.
- **NVIDIA**: the NVIDIA driver, so that `nvidia-smi` works in a terminal.

### 2. Unpack and run the installer

```bash
cd ~/Downloads
unzip LoraStudio.zip
cd LoraStudio
./install.sh
```

The installer explains each step as it goes. What it asks:

1. **The notice**: sd-scripts and musubi-tuner belong to kohya-ss, LoRA Studio is only a GUI for them. Type `yes`.
2. **sd-scripts** (trains SDXL, Illustrious, Pony, Anima, FLUX.1, SD 1.5, …): *install it for me*, *I have it
   already* (it asks for the folder), or *none*.
3. **musubi-tuner** (trains Z-Image Turbo, Qwen-Image, FLUX.2): the same three choices.
4. **Ollama**: install it, use one that is already running, or none.
5. **Your graphics card**: detected by itself. A computer with both an AMD and an NVIDIA card (most gaming laptops)
   is asked which one to train on.

It then installs everything into **`~/LoraStudio`**: LoRA Studio, one Python environment (`venv/`) with PyTorch for
your card, and the trainers. This takes a while (PyTorch with ROCm or CUDA is several GB). Everything it prints is
also in `~/LoraStudio/install.log`, and at the end it writes an **install report** to
`~/LoraStudio/userdata/reports/` (without tokens, passwords or your home folder) that you can send when something
does not work.

You can delete the unpacked download folder afterwards; LoRA Studio runs from `~/LoraStudio`.

Useful options (`./install.sh --help` lists all of them):

| Option | What it does |
|---|---|
| `--dir DIR` | install somewhere else than `~/LoraStudio` |
| `--gpu amd` / `--gpu nvidia` | choose the card yourself |
| `--gfx gfx1201` | the AMD architecture, when it is not detected |
| `--sd none --mu none` | no trainers (only prepare datasets and write launch scripts) |
| `--import DIR` | take the settings, token and marks from an older LoRA Studio folder |
| `--ask` | ask every question again (the answers of the last run are remembered) |

### 3. Start it

```bash
cd ~/LoraStudio
./start.sh
```

Your browser opens LoRA Studio at <http://127.0.0.1:7860>. Continue with [The first start](#the-first-start).

---

## Windows

LoRA Studio and the trainers are Linux programs. On Windows they run in **WSL2**, Microsoft's own Linux inside
Windows, which is also how sd-scripts and PyTorch work best on Windows. The installer does all of this for you.

### 1. Before you start

- **NVIDIA**: the normal Windows driver from NVIDIA is enough (WSL2 uses it).
- **AMD Radeon**: training inside WSL needs AMD's *ROCm on WSL* (Radeon RX 7000 or 9000 with a recent Adrenalin
  driver). If your card or driver cannot do that, answer **none** for the trainers: you can still prepare datasets
  and make launch scripts on Windows, and train on a Linux computer.
- **Ollama** (optional): install it in Windows from [ollama.com/download/windows](https://ollama.com/download/windows)
  before you run the installer, so it can be found.

### 2. Run the installer

1. Unpack `LoraStudio.zip` (right-click › *Extract All…*).
2. Open the folder and double-click **`Install LoRA Studio.cmd`**.
3. **If WSL2 is not installed yet**, the installer installs it with Ubuntu 24.04. Windows asks for administrator
   rights. Then:
   - restart Windows,
   - Ubuntu opens once and asks you to choose a user name and password (this is a new Linux user, it can be
     anything; remember the password),
   - double-click **`Install LoRA Studio.cmd`** again.
4. The installer looks at your graphics card and at Ollama, and then runs the Linux installer inside Ubuntu. It asks
   the same questions as on Linux (see [above](#2-unpack-and-run-the-installer)). When it asks for your password, it
   is the Ubuntu password from step 3.
5. At the end it puts a **LoRA Studio** starter on your desktop and writes an install report next to the installer
   (`install_report_windows_<time>.md`).

### 3. Start it

Double-click **LoRA Studio** on the desktop. A black window opens, and your browser opens LoRA Studio when it is ready.
**Closing the black window stops LoRA Studio.**

### Your files on Windows

- Inside Ubuntu, your Windows drives are under `/mnt/c`, `/mnt/d`, …: for example your pictures are at
  `/mnt/c/Users/<you>/Pictures`. You can use those paths for your dataset folders.
- Datasets **inside** Ubuntu train noticeably faster. The Windows Explorer shows Ubuntu's files under
  **Linux › Ubuntu-24.04 › home › <your Ubuntu user>**, so you can copy pictures in there.
- Training hangs at the very start on a computer with a Ryzen processor that has built-in graphics? Turn the built-in
  graphics off in the Windows Device Manager, then run `wsl --shutdown` in a Windows terminal and start again.

---

## The first start

1. **The notice**: the first page says that sd-scripts and musubi-tuner belong to kohya-ss and are not part of
   LoRA Studio. Tick *I have read this and agree* and press *Continue to LoRA Studio*. It is shown only once.
2. **The setup page** asks for your folders and options, one step at a time. Every step can be skipped and changed
   later in **Settings**:
   - **Trainers**: where sd-scripts and musubi-tuner are (already filled in when the installer put them there).
   - **Datasets**: the folder your datasets go into, one folder per LoRA.
   - **Models**: the folder with your base models, text encoders and VAEs (ComfyUI's `models` folder works well).
     Missing model files can later be downloaded from Hugging Face in **Settings › Model files**.
   - **ComfyUI**: your LoRA folder and the folder ComfyUI saves its pictures in.
   - **Hugging Face token**: only needed for gated models and downloads.
   - **Ollama**: its address and the vision model (for example `qwen2.5vl:7b`).
   - **Backups**: where *Settings › Backup* saves backups.
   - Instead of all this you can **import a backup** from another LoRA Studio (see below).
3. LoRA Studio opens on the **Collection**. Open **Settings › Health check** first: it lists anything that is not
   set up yet, with a button to the place where it is fixed.
4. Then **Datasets › New dataset** takes you from a folder of pictures to a launch script in nine steps, and
   **Training** runs it. The **Manual** tab (inside the program) explains every page with pictures.

---

## Starting it later

| | |
|---|---|
| Linux | `cd ~/LoraStudio && ./start.sh` |
| Windows | double-click **LoRA Studio** on the desktop |
| Another port | `./start.sh --port 7861` |
| Without opening the browser | `./start.sh --no-browser`, then open <http://127.0.0.1:7860> |
| The setup page again | `./start.sh --setup` |

**Settings › Start and network** sets the port, the start page, whether the browser opens, and whether other
computers on your network may open LoRA Studio (that needs a login, set on the same page).

---

## Updating

- **In the program**: download the new `LoraStudio.zip`, open **Settings › Updates** and choose the zip. LoRA Studio
  shows what changes, keeps a backup of the old files, tests the new version and goes back by itself when the test
  fails. Restart when it asks.
- **From GitHub**: **Settings › Updates › From GitHub › Check GitHub**, then *Install from GitHub*: the newest upload,
  downloaded and installed the same safe way, without a zip. (While the repository is private it needs a GitHub token
  with read access, saved on the same page.)
- **With the installer**: unpack the new zip and run `./install.sh` from it again. It updates LoRA Studio in
  `~/LoraStudio`, pulls the newest trainers and keeps PyTorch as it is.

Your settings and data are never part of an update. **Settings › Updates** also updates sd-scripts and musubi-tuner,
and checks the training environment (*Fix the training venv* puts back ROCm or CUDA parts that do not fit PyTorch).

---

## A new computer, or a reinstall

1. On the old install: **Settings › Backup › Back up now**. It puts your settings, token, marks and databases into
   one zip (your datasets, LoRAs and models are not in it; copy those folders yourself).
2. Install LoRA Studio on the new computer as above.
3. On the setup page of the first start, choose **Import a backup** and pick the zip.

Everything that belongs to you is in `~/LoraStudio/userdata/`; the program is in `~/LoraStudio/program/`. To share
LoRA Studio with someone, give them `LoraStudio.zip`, **never your LoRA Studio folder**: `userdata/` holds your
Hugging Face token.

---

## Removing it

Make a backup first (above) if you might want your settings again.

- **Linux**: delete the folder: `rm -rf ~/LoraStudio`. Your datasets, LoRAs and models are in your own folders and
  stay. Ollama, if the installer put it there, has its own uninstall (see [ollama.com](https://ollama.com)).
- **Windows**: delete the **LoRA Studio** starter from the desktop, then in a Windows terminal:
  `wsl -d Ubuntu-24.04 -- rm -rf ~/LoraStudio`. To remove Ubuntu completely (this deletes **everything** inside it,
  also any datasets you copied in): `wsl --unregister Ubuntu-24.04`.

---

## When something goes wrong

| Problem | What to do |
|---|---|
| *no AMD (ROCm) or NVIDIA GPU found* | AMD: add your user to the `render` and `video` groups (above) and log in again. NVIDIA: check that `nvidia-smi` works. Or choose the card yourself: `./install.sh --gpu amd` / `--gpu nvidia` |
| *your user is not in the 'render' group* | `sudo usermod -aG render,video $USER`, then log out and in |
| The AMD architecture is not found | `./install.sh --gfx gfx1201` (RX 9070 XT; RX 7900: `gfx1100`) |
| The browser does not open | open <http://127.0.0.1:7860> yourself |
| *address already in use* | another program uses port 7860: `./start.sh --port 7861` |
| A training crashes with *Memory access fault by GPU* (AMD) | **Settings › Updates › Fix the training venv** |
| A training fails | **Training › Results** says what went wrong in plain words, and what to change |
| A model file is missing | **Settings › Model files**: *From Hugging Face* next to the slot |
| Windows: the installer stops after installing WSL | restart Windows, finish Ubuntu's user setup, run the installer again |

**Asking for help**: open an [issue](../../issues/new/choose) and attach the **install report**
(`~/LoraStudio/userdata/reports/install_report_*.md` on Linux, `install_report_windows_*.md` next to the installer on
Windows). For a problem inside the program, **Settings › Version › Report for Claude** makes one file with the log
end, the settings and the machine, without tokens or your home folder.

More detail on every option and file: [REFERENCE.md](REFERENCE.md).
