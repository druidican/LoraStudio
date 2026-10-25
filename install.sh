#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# LoRA Studio version 1.251026o
# LoRA Studio installer: LoRA Studio + sd-scripts + musubi-tuner in ONE folder with ONE Python 3.12 venv.
#
#   unzip LoraStudio.zip && cd LoraStudio && ./install.sh
#
# Result (default ~/LoraStudio, outside synced folders so the 15+ GB are not uploaded):
#   ~/LoraStudio/                 LoRA Studio itself, its settings and start.sh
#   ~/LoraStudio/venv/            Python 3.12 + PyTorch (ROCm for AMD, CUDA for NVIDIA) + both trainers + the GUI
#   ~/LoraStudio/sd-scripts/      git clone of kohya-ss/sd-scripts      (its venv/ is a link to ../venv)
#   ~/LoraStudio/musubi-tuner/    git clone of kohya-ss/musubi-tuner    (its venv/ is a link to ../venv)
#   ~/LoraStudio/constraints.txt  the PyTorch and trainer versions that later installs must not change
#   ~/LoraStudio/install.log      everything this installer printed
#
# Run it again at any time to UPDATE: it copies the new LoRA Studio files (your settings stay), pulls both
# trainers and reinstalls their packages without touching PyTorch.
#
# Options:
#   --dir DIR           install folder (default ~/LoraStudio)
#   --gpu auto|amd|nvidia   (default auto: asks when the computer has both)
#   --gfx gfxNNNN       AMD GPU architecture (default: detected, e.g. gfx1201 for the RX 9070 XT)
#   --import DIR        old LoRA Studio folder to take settings, token, Completed marks and caches from
#   --reinstall-torch   install PyTorch (and ROCm) again even when it already works
#   --no-models         do not offer to download the Z-Image Turbo training adapter
#   --sd install|existing|none      sd-scripts: install it, use yours, or none (only make launch scripts)
#   --mu install|existing|none      the same for musubi-tuner
#   --trainers install|existing     both at once (older option, still works)
#   --sd-scripts DIR    your sd-scripts folder      (sets --sd existing)
#   --sd-venv DIR       its venv (default DIR/venv)
#   --musubi DIR        your musubi-tuner folder    (sets --mu existing)
#   --ollama install|remote|none    Ollama: install it (Linux only), use one that runs (give --ollama-url), or none
#   --ollama-url URL    Ollama's address (default http://127.0.0.1:11434)
#   --ask               ask every question again (the answers of the last run are remembered otherwise)
#   --accept-notice     confirm the notice about sd-scripts / musubi-tuner without a question (--yes does too)
#   --yes               answer every question with its default (no questions asked)
#   --help

set -Eeo pipefail

# ------------------------------------------------------------------ versions (edit here to move on)
PYVER="3.12"
# AMD: the wheels from AMD's repository (ROCm 10.1.0; 10.0.0 was tested on an RX 9070 XT, Fedora 44)
# https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html lists the combinations
ROCM_INDEX="https://stable.repo.amd.com/rocm/whl-next/"
ROCM_VER="10.1.0"                                              # must be the ROCm the torch build below is made for
ROCM_SPEC="rocm[libraries,devel,device-@GFX@]==$ROCM_VER"       # @GFX@ = the detected architecture
# every part of ROCm pinned: one newer part (rocm-sdk-devel 10.1 next to 10.0) made SDXL runs crash with a GPU memory fault
ROCM_PARTS="rocm-sdk-core==$ROCM_VER rocm-sdk-libraries==$ROCM_VER rocm-sdk-devel==$ROCM_VER rocm-sdk-device-@GFX@==$ROCM_VER"
TORCH_SPEC_AMD="torch[device-@GFX@]==2.13.0+rocm$ROCM_VER"      # the same PyTorch 2.13 as before, built for ROCm 10.1
TV_SPEC_AMD="torchvision[device-@GFX@]==0.28.0+rocm$ROCM_VER"
TA_SPEC_AMD="torchaudio==2.11.0.2+rocm$ROCM_VER"
# NVIDIA: PyTorch from PyPI (the CUDA 13 build). Older driver? set CUDA_INDEX, e.g. https://download.pytorch.org/whl/cu128
CUDA_INDEX=""
SD_REPO="https://github.com/kohya-ss/sd-scripts.git"
MU_REPO="https://github.com/kohya-ss/musubi-tuner.git"
ADAPTER_REPO="ostris/zimage_turbo_training_adapter"
ADAPTER_FILE="zimage_turbo_training_adapter_v2.safetensors"

APP_FILES="lora_studio.py version.py selftest.py requirements.txt start.sh install.sh README.md REFERENCE.md LICENSE NOTICE"
APP_DIRS="program"                          # the program by topic (Gallery Data Training Settings Other), manual, logo
DATA_FILES="library_settings.txt training_settings.txt huggingtoken.txt completed.json library_data.json
model_ids.json hash_cache.json picture_index.json launch_settings.json"

# ------------------------------------------------------------------ helpers
SRC="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
TARGET="$HOME/LoraStudio"; GPU="auto"; GFX=""; IMPORT=""; REINSTALL_TORCH=0; MODELS=1; YES=0
TRAINERS=""; EX_SD=""; EX_VENV=""; EX_MU=""; ASK=0; ACCEPT_NOTICE=0; A_SD=""; A_MU=""; A_OL=""; A_OLURL=""
STEP=0

RPT=()                                     # the install report: every step, ok and warning, written at the end
say()  { printf '%s\n' "$*"; }
step() { STEP=$((STEP+1)); printf '\n\033[1m== %d. %s\033[0m\n' "$STEP" "$*"; RPT+=("" "**$STEP. $***"); }
ok()   { printf '   \033[32mok\033[0m %s\n' "$*"; RPT+=("- ok $*"); }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; RPT+=("- ! $*"); }
die()  { printf '\n\033[31mStopped:\033[0m %s\n' "$*"; [ -n "${LOG:-}" ] && printf 'Full log: %s\n' "$LOG"
         RPT+=("" "**Stopped:** $*"); write_report "stopped in step $STEP"; exit 1; }
write_report() {   # write_report "result": reports/install_report_<time>.md (token and home folder removed), and print it
    trap - ERR; set +e +o pipefail                # nothing in here may stop the installer
    [ -d "${TARGET:-/nonexistent}" ] || return 0
    local dir="$TARGET/userdata/reports" f ver gpu
    mkdir -p "$dir" 2>/dev/null || return 0
    f="$dir/install_report_$(date +%Y%m%d_%H%M%S).md"
    ver="$(sed -n 's/^VERSION = "\(.*\)"/\1/p' "$SRC/version.py" 2>/dev/null | head -1)"
    gpu="$( (command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -2) ;
            lspci 2>/dev/null | grep -iE 'vga|3d|display' | sed 's/^[0-9a-f:.]* //' | head -3)"
    {
        echo "# LoRA Studio install report · $(date '+%Y-%m-%d %H:%M')"
        echo ""
        echo "Please look at this install of LoRA Studio (a Gradio app that trains LoRAs with sd-scripts / musubi-tuner)."
        echo ""
        echo "- Result: **$1**"
        echo "- LoRA Studio ${ver:-?} · install.sh run as: install.sh ${ARGS_GIVEN:-(no options)}"
        echo "- System: ${PRETTY_NAME:-unknown} · kernel $(uname -r) · $(uname -m)$([ "${IS_WSL:-0}" = 1 ] && echo ' · WSL2 (Windows)')"
        echo "- Choices: sd-scripts ${SD_MODE:-?} · musubi-tuner ${MU_MODE:-?} · Ollama ${OL_MODE:-?}${OL_URL:+ ($OL_URL)} · GPU ${GPU:-?}${GFX:+ $GFX}"
        echo "- Python: ${PY_SYS:-?} · free disk: ${FREE_GB:-?} GB"
        [ -n "$gpu" ] && echo "- Graphics: $(printf '%s\n' "$gpu" | paste -sd ';' | sed 's/;/ · /g')"
        printf '%s\n' "${RPT[@]}"
        echo ""
        echo "Full log: ${LOG:-?}"
    } | sed -E "s#$HOME#~#g; s#(hf_|sk-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}#[token removed]#g" > "$f" 2>/dev/null || return 0
    say ""
    say "   Install report: $f"
    say "   (to ask for help: send this file, or copy the text between the lines into a chat)"
    say "----------------------------------------------------------------------- copy from here"
    cat "$f"
    say "----------------------------------------------------------------------- to here"
}
trap 'die "a command failed in step $STEP (line $LINENO). Nothing that worked before is undone; fix the cause and run ./install.sh again."' ERR

yes_no() {   # yes_no "question" y|n
    local def="$2" ans
    if [ "$YES" = 1 ] || [ ! -t 0 ]; then [ "$def" = y ]; return; fi
    read -r -p "   $1 ($([ "$def" = y ] && echo Y/n || echo y/N)) " ans || ans=""
    ans="${ans:-$def}"; [[ "$ans" =~ ^[YyJj] ]]
}

ARGS_GIVEN="$*"
while [ $# -gt 0 ]; do
    case "$1" in
        --dir) TARGET="$(readlink -m "${2/#\~/$HOME}")"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --gfx) GFX="$2"; shift ;;
        --import) IMPORT="$(readlink -m "${2/#\~/$HOME}")"; shift ;;
        --reinstall-torch) REINSTALL_TORCH=1 ;;
        --no-models) MODELS=0 ;;
        --trainers) TRAINERS="$2"; shift ;;
        --sd-scripts) EX_SD="$(readlink -m "${2/#\~/$HOME}")"; shift ;;
        --sd-venv) EX_VENV="$(readlink -m "${2/#\~/$HOME}")"; shift ;;
        --musubi) EX_MU="$(readlink -m "${2/#\~/$HOME}")"; A_MU="${A_MU:-existing}"; shift ;;
        --sd) A_SD="$2"; shift ;;
        --mu) A_MU="$2"; shift ;;
        --ollama) A_OL="$2"; shift ;;
        --ollama-url) A_OLURL="$2"; A_OL="${A_OL:-remote}"; shift ;;
        --ask) ASK=1 ;;
        --accept-notice) ACCEPT_NOTICE=1 ;;
        --yes|-y) YES=1 ;;
        --help|-h) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option $1 (see ./install.sh --help)" ;;
    esac
    shift
done
case "$GPU" in auto|amd|nvidia) ;; *) die "--gpu must be auto, amd or nvidia" ;; esac
case "$TRAINERS" in ""|install|existing) ;; *) die "--trainers must be install or existing" ;; esac
[ -n "$EX_SD" ] && [ -z "$A_SD" ] && A_SD=existing
for v in "$A_SD" "$A_MU"; do case "$v" in ""|install|existing|none) ;; *) die "--sd and --mu must be install, existing or none" ;; esac; done
case "$A_OL" in ""|install|remote|none) ;; *) die "--ollama must be install, remote or none" ;; esac

mkdir -p "$TARGET"
LOG="$TARGET/install.log"
exec > >(tee -a "$LOG") 2>&1
say ""
say "LoRA Studio installer, $(date '+%Y-%m-%d %H:%M'), from $SRC into $TARGET"

# ------------------------------------------------------------------ the notice (once): who owns sd-scripts / musubi-tuner
NOTICE_FILE="$TARGET/.lorastudio-notice"
if [ ! -f "$NOTICE_FILE" ] && [ -f "$SRC/NOTICE" ]; then
    say ""
    sed 's/^/   /' "$SRC/NOTICE"
    say ""
    if [ "$YES" = 1 ] || [ "$ACCEPT_NOTICE" = 1 ]; then
        ok "notice confirmed ($( [ "$ACCEPT_NOTICE" = 1 ] && echo --accept-notice || echo --yes ))"
    elif [ ! -t 0 ]; then
        die "the notice above has to be confirmed: run ./install.sh in a terminal, or add --accept-notice"
    else
        read -r -p "   Type yes to confirm that you have read this and agree: " ans || ans=""
        [[ "$ans" =~ ^[Yy]([Ee][Ss])?$|^[Jj][Aa]?$ ]] || die "not confirmed: nothing was installed"
        ok "notice confirmed"
    fi
    date '+%Y-%m-%d %H:%M' > "$NOTICE_FILE"
fi

# ------------------------------------------------------------------ 1. checks
step "Checking the computer"
. /etc/os-release 2>/dev/null || true
DISTRO="${ID:-unknown}"; LIKE="${ID_LIKE:-}"
case "$DISTRO $LIKE" in
    *fedora*|*rhel*) PKG_PY="sudo dnf install -y python${PYVER} git"; PKG_GIT="sudo dnf install -y git" ;;
    *ubuntu*|*debian*|*linuxmint*) PKG_PY="sudo apt install -y python${PYVER} python${PYVER}-venv git"; PKG_GIT="sudo apt install -y git" ;;
    *) PKG_PY="(install Python ${PYVER} with your package manager)"; PKG_GIT="(install git with your package manager)" ;;
esac
ok "system: ${PRETTY_NAME:-unknown}"

# What to install: sd-scripts, musubi-tuner and Ollama, each one installed for you, one you have, or none
ask_path() {   # ask_path "question" default -> the answer (default on Enter)
    local ans
    if [ "$YES" = 1 ] || [ ! -t 0 ]; then printf '%s' "$2"; return; fi
    read -r -p "   $1 [${2:-none}] " ans || ans=""
    ans="${ans:-$2}"; ans="${ans/#\~/$HOME}"; printf '%s' "${ans%/}"
}
ask_choice() {   # ask_choice "question" default "1) ..." "2) ..." ... -> the number
    local q="$1" def="$2" n; shift 2
    if [ "$YES" = 1 ] || [ ! -t 0 ]; then printf '%s' "$def"; return; fi
    say "   $q" >&2
    for o in "$@"; do say "     $o" >&2; done
    read -r -p "   Which one? (Enter = $def) " n || n=""
    printf '%s' "${n:-$def}"
}
IS_WSL=0; grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
[ "$IS_WSL" = 1 ] && ok "running in WSL (Windows): Ollama is installed in Windows, not here"
MODE_FILE="$TARGET/.lorastudio-trainers"          # the answers are remembered for the next ./install.sh
SD_MODE=""; MU_MODE=""; OL_MODE=""; OL_URL=""
case "$TRAINERS" in install) SD_MODE=install; MU_MODE=install ;; existing) SD_MODE=existing ;; esac
CLI_EX_SD="$EX_SD"; CLI_EX_VENV="$EX_VENV"; CLI_EX_MU="$EX_MU"
if [ -z "$TRAINERS$A_SD$A_MU$A_OL" ] && [ "$ASK" = 0 ] && [ -f "$MODE_FILE" ]; then
    . "$MODE_FILE"
    EX_SD="${CLI_EX_SD:-$EX_SD}"; EX_VENV="${CLI_EX_VENV:-$EX_VENV}"; EX_MU="${CLI_EX_MU:-$EX_MU}"
    if [ -z "$SD_MODE" ] && [ -n "${TRAINERS:-}" ]; then          # remembered by an older install.sh
        if [ "$TRAINERS" = install ]; then SD_MODE=install; MU_MODE=install; else SD_MODE=existing; MU_MODE=$([ -n "$EX_MU" ] && echo existing || echo none); fi
    fi
    ok "as last time: sd-scripts $SD_MODE, musubi-tuner $MU_MODE, Ollama ${OL_MODE:-not asked} (./install.sh --ask to answer again)"
fi
[ -n "$A_SD" ] && SD_MODE="$A_SD"
[ -n "$A_MU" ] && MU_MODE="$A_MU"
[ -n "$A_OL" ] && OL_MODE="$A_OL"
[ -n "$A_OLURL" ] && OL_URL="$A_OLURL"
if [ "$OL_MODE" = install ] && [ "$IS_WSL" = 1 ]; then
    warn "Ollama is installed in Windows, not inside WSL: https://ollama.com/download/windows (counted as remote)"
    OL_MODE=remote
fi
FOUND_SD=""
for d in "$HOME/sd-scripts" "$HOME/kohya_ss/sd-scripts" "$HOME/sd-script"; do
    [ -f "$d/sdxl_train_network.py" ] && [ "$d" != "$TARGET/sd-scripts" ] && { FOUND_SD="$d"; break; }
done
FOUND_MU=""
for d in "$HOME/musubi-tuner" "$(dirname "${EX_SD:-$HOME/x}")/musubi-tuner"; do
    [ -d "$d/src/musubi_tuner" ] && [ "$d" != "$TARGET/musubi-tuner" ] && { FOUND_MU="$d"; break; }
done
if [ -z "$SD_MODE" ]; then
    say "   sd-scripts trains SDXL, Illustrious, Pony, Anima, FLUX.1, SD 1.5 and more."
    [ -n "$FOUND_SD" ] && say "   Found sd-scripts in $FOUND_SD"
    n="$(ask_choice "sd-scripts:" "$([ -n "$FOUND_SD" ] && echo 2 || echo 1)" \
        "1) install it for me (with PyTorch for your GPU; 15-25 GB together with musubi-tuner)" \
        "2) I have it already: use mine" \
        "3) none: only make the launch scripts here, I train somewhere else")"
    case "$n" in 2) SD_MODE=existing ;; 3) SD_MODE=none ;; *) SD_MODE=install ;; esac
fi
if [ -z "$MU_MODE" ]; then
    say "   musubi-tuner trains Z-Image Turbo, Qwen-Image and FLUX.2."
    [ -n "$FOUND_MU" ] && say "   Found musubi-tuner in $FOUND_MU"
    n="$(ask_choice "musubi-tuner:" "$([ -n "$FOUND_MU" ] && echo 2 || ([ "$SD_MODE" = install ] && echo 1 || echo 3))" \
        "1) install it for me" "2) I have it already: use mine" "3) none (no Z-Image / Qwen-Image / FLUX.2 training here)")"
    case "$n" in 1) MU_MODE=install ;; 2) MU_MODE=existing ;; *) MU_MODE=none ;; esac
fi
if [ "$SD_MODE" = existing ]; then
    [ -n "$EX_SD" ] || EX_SD="$(ask_path "Your sd-scripts folder:" "${FOUND_SD:-$HOME/sd-scripts}")"
    [ -f "$EX_SD/sdxl_train_network.py" ] || die "no sd-scripts in $EX_SD (sdxl_train_network.py not found). Give the right folder with --sd-scripts DIR"
    [ -n "$EX_VENV" ] || EX_VENV="$(ask_path "The venv sd-scripts runs in:" "$EX_SD/venv")"
    [ -x "$EX_VENV/bin/python" ] || die "no Python in $EX_VENV/bin (the venv of sd-scripts). Give it with --sd-venv DIR"
    ok "using your sd-scripts: $EX_SD (venv $EX_VENV)"
    if TINFO="$("$EX_VENV/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>/dev/null)"; then
        ok "its PyTorch: $TINFO (True = it sees the GPU)"
    else
        warn "PyTorch does not load in $EX_VENV: training will not work until it does"
    fi
fi
if [ "$MU_MODE" = existing ]; then
    [ -n "$EX_MU" ] || EX_MU="$(ask_path "Your musubi-tuner folder:" "${FOUND_MU:-$HOME/musubi-tuner}")"
    if [ ! -d "$EX_MU/src/musubi_tuner" ]; then warn "no musubi-tuner in $EX_MU: counted as none"; MU_MODE=none; EX_MU=""
    else ok "using your musubi-tuner: $EX_MU"; fi
fi
[ "$SD_MODE" = install ] && ok "sd-scripts: installed into $TARGET/sd-scripts"
[ "$MU_MODE" = install ] && ok "musubi-tuner: installed into $TARGET/musubi-tuner"
[ "$SD_MODE" = none ] && ok "sd-scripts: none (launch scripts are made, not run here)"
[ "$MU_MODE" = none ] && ok "musubi-tuner: none"
# Ollama (the detailed sort, captions, the consistency check)
if [ -z "$OL_MODE" ]; then
    HAVE_OL=0; command -v ollama >/dev/null && HAVE_OL=1
    curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && HAVE_OL=1
    say "   Ollama looks at the pictures for the detailed sort and writes the captions (a vision model, about 6 GB)."
    if [ "$IS_WSL" = 1 ]; then
        say "   On Windows, install Ollama in Windows itself: https://ollama.com/download/windows"
        say "   (guide: https://github.com/ollama/ollama/blob/main/docs/windows.md). LoRA Studio here reaches it over the network."
        n="$(ask_choice "Ollama:" "$([ "$HAVE_OL" = 1 ] && echo 1 || echo 2)" \
            "1) it is installed in Windows (or on another computer): ask for its address" \
            "2) none for now (the detailed sort and captions are hidden; tick it later in Settings)")"
        case "$n" in 1) OL_MODE=remote ;; *) OL_MODE=none ;; esac
    else
        n="$(ask_choice "Ollama:" "$([ "$HAVE_OL" = 1 ] && echo 2 || echo 1)" \
            "1) install it for me now (Ollama's own installer from ollama.com, it asks for your password)" \
            "2) it is installed already, here or on another computer" \
            "3) none (the detailed sort and captions are hidden; tick it later in Settings)")"
        case "$n" in 1) OL_MODE=install ;; 2) OL_MODE=remote ;; *) OL_MODE=none ;; esac
    fi
fi
if [ "$OL_MODE" = remote ] && [ -z "$OL_URL" ]; then
    OL_URL="$(ask_path "Ollama's address:" "http://127.0.0.1:11434")"
fi
[ "$OL_MODE" = install ] && OL_URL="${OL_URL:-http://127.0.0.1:11434}"
case "$OL_MODE" in
    install) ok "Ollama: installed for you (step 'Ollama')" ;;
    remote)  ok "Ollama: $OL_URL" ;;
    *)       ok "Ollama: none (the detailed sort and captions are hidden until you tick it in Settings › Start and network)" ;;
esac
printf 'SD_MODE=%q\nMU_MODE=%q\nOL_MODE=%q\nOL_URL=%q\nEX_SD=%q\nEX_VENV=%q\nEX_MU=%q\n' \
    "$SD_MODE" "$MU_MODE" "$OL_MODE" "$OL_URL" "$EX_SD" "$EX_VENV" "$EX_MU" > "$MODE_FILE"
# the rest of the installer: "install" when it installs a trainer (Python 3.12 + PyTorch), else only LoRA Studio's venv
if [ "$SD_MODE" = install ] || [ "$MU_MODE" = install ]; then TRAINERS=install; else TRAINERS=existing; fi

if [ "$TRAINERS" = install ] && ! command -v git >/dev/null; then
    if yes_no "git is missing. Install it now with: $PKG_GIT ?" y; then eval "$PKG_GIT"; else die "git is needed: $PKG_GIT"; fi
fi
command -v git >/dev/null && ok "git $(git --version | awk '{print $3}')"

PY_SYS="$(command -v "python${PYVER}" || true)"
if [ "$TRAINERS" = existing ] && [ -z "$PY_SYS" ]; then        # LoRA Studio alone runs on any Python 3.10 or newer
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
        p="$(command -v "$c" || true)"
        if [ -n "$p" ] && "$p" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then PY_SYS="$p"; break; fi
    done
fi
if [ -z "$PY_SYS" ]; then
    say "   Python ${PYVER} is missing (PyTorch for ROCm/CUDA and both trainers are built for it, not for your default python3)."
    if yes_no "Install it now with: $PKG_PY ?" y; then eval "$PKG_PY"; PY_SYS="$(command -v "python${PYVER}" || true)"; fi
    [ -n "$PY_SYS" ] || die "Python ${PYVER} is needed: $PKG_PY"
fi
"$PY_SYS" -m venv --help >/dev/null 2>&1 || die "Python ${PYVER} has no venv module: $PKG_PY"
ok "Python $("$PY_SYS" -c 'import sys; print(sys.version.split()[0])') ($PY_SYS)"

FREE_GB=$(( $(df -Pk "$TARGET" | awk 'NR==2{print $4}') / 1048576 ))
NEED_GB=25; [ "$TRAINERS" = existing ] && NEED_GB=2
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    warn "only ${FREE_GB} GB free on the disk of $TARGET; about ${NEED_GB} GB are needed."
    yes_no "Continue anyway?" n || die "not enough disk space"
else
    ok "${FREE_GB} GB free in $TARGET"
fi
case "$TARGET" in *"/MEGA"*|*"/Dropbox"*|*"/OneDrive"*|*"/Nextcloud"*)
    warn "$TARGET is inside a synced folder: the venv and both trainers (15+ GB) would be uploaded."
    yes_no "Install there anyway?" n || die "choose another folder with --dir (default ~/LoraStudio)" ;;
esac

# GPU: which vendor, and for AMD which architecture (gfx1201 = RX 9070 / 9070 XT)
HAS_NV=0; HAS_AMD=0
if [ "$TRAINERS" = install ]; then
[ "$IS_WSL" = 1 ] && [ -d /usr/lib/wsl/lib ] && export PATH="$PATH:/usr/lib/wsl/lib"   # WSL: nvidia-smi comes from the Windows driver
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then HAS_NV=1; fi
if [ "$IS_WSL" = 1 ] && [ "$HAS_NV" = 0 ] && [ -e /dev/dxg ] && [ "$GPU" != nvidia ]; then
    warn "WSL with an AMD GPU: PyTorch only sees it after AMD's 'ROCm on WSL' setup (Radeon RX 7000/9000, Adrenalin driver):"
    say "     https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html"
    say "     Without it, choose 'none' for the trainers (./install.sh --ask) and train on Linux, or use --gpu amd --gfx gfxNNNN."
    say "     A Ryzen with built-in graphics: ROCm 10.1 can hang at the first GPU use in WSL. Disable the built-in graphics"
    say "     in the Windows Device Manager, then run wsl --shutdown."
fi
if [ -e /dev/kfd ] || lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -qiE "amd|ati|radeon"; then HAS_AMD=1; fi
if [ "$GPU" = auto ]; then
    if [ "$HAS_NV" = 1 ] && [ "$HAS_AMD" = 1 ]; then
        say "   Both an NVIDIA and an AMD GPU were found (a laptop usually trains on the NVIDIA one)."
        if yes_no "Use the NVIDIA GPU?" y; then GPU=nvidia; else GPU=amd; fi
    elif [ "$HAS_NV" = 1 ]; then GPU=nvidia
    elif [ "$HAS_AMD" = 1 ]; then GPU=amd
    else die "no AMD (ROCm) or NVIDIA GPU found. Use --gpu amd|nvidia if you are sure, or install without trainers (they need a GPU): ./install.sh --sd none --mu none"
    fi
fi
if [ "$GPU" = amd ]; then
    if [ -z "$GFX" ] && command -v rocminfo >/dev/null; then
        GFX="$(rocminfo 2>/dev/null | grep -oE '\bgfx[0-9a-f]{3,4}\b' | grep -v '^gfx000$' | head -1 || true)"
    fi
    if [ -z "$GFX" ]; then        # no rocminfo: read the kernel's topology (gfx_target_version 120001 = gfx1201)
        for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
            v="$(awk '/gfx_target_version/{print $2}' "$p" 2>/dev/null)"
            if [ -n "$v" ] && [ "$v" != 0 ]; then
                GFX="$(printf 'gfx%d%x%x' $((v/10000)) $(((v/100)%100)) $((v%100)))"; break
            fi
        done
    fi
    [ -n "$GFX" ] || die "could not detect the AMD GPU architecture. Pass it with --gfx (gfx1201 for the RX 9070 XT)."
    ok "GPU: AMD $GFX (PyTorch for ROCm)"
    id -nG | grep -qwE "render" || warn "your user is not in the 'render' group: sudo usermod -aG render,video $(id -un) (then log out and in)"
else
    ok "GPU: NVIDIA $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1) (PyTorch for CUDA)"
fi
fi   # TRAINERS = install

# ------------------------------------------------------------------ 2. LoRA Studio files
step "LoRA Studio files"
if [ "$SRC" != "$(readlink -f "$TARGET")" ]; then
    for f in $APP_FILES; do
        [ -f "$SRC/$f" ] || die "$f is missing in $SRC (unpack the whole zip)"
        cp -f "$SRC/$f" "$TARGET/$f"
    done
    for d in $APP_DIRS; do                   # replaced as a whole, so no old file is left behind in it
        [ -d "$SRC/$d" ] || die "the folder $d is missing in $SRC (unpack the whole zip)"
        rm -rf "$TARGET/$d.new" && cp -r "$SRC/$d" "$TARGET/$d.new" && rm -rf "$TARGET/$d" && mv "$TARGET/$d.new" "$TARGET/$d"
    done
    chmod +x "$TARGET/start.sh" "$TARGET/install.sh"
    ok "copied the program to $TARGET (your settings in $TARGET/userdata are not touched)"
else
    ok "running inside $TARGET: files are already in place"
fi

# An install from before 1.011126 (everything in one folder): your files move to userdata/, the old program aside
"$PY_SYS" - "$TARGET" <<'EOF' || warn "could not move the old layout (LoRA Studio does it at its next start)"
import sys
sys.path.insert(0, sys.argv[1] + "/program")
import lspaths
moved = lspaths.migrate(say=lambda m: print("   " + m))
EOF
mkdir -p "$TARGET/userdata"

# Settings from an older LoRA Studio folder (only when this one has none yet)
if [ ! -f "$TARGET/userdata/library_settings.txt" ]; then
    CANDS=()
    if [ -n "$IMPORT" ]; then CANDS=("$IMPORT")
    else
        while IFS= read -r f; do
            d="$(dirname "$f")"; app="$d"; [ "$(basename "$d")" = userdata ] && app="$(dirname "$d")"
            [ "$app" != "$(readlink -f "$TARGET")" ] && [ -f "$app/lora_studio.py" ] && CANDS+=("$d")
        done < <(find "$HOME" -maxdepth 6 -name library_settings.txt -not -path "*/venv/*" -not -path "$TARGET/*" \
                      -not -path "*/update_backups/*" -not -path "*/restore_backups/*" 2>/dev/null | head -8)
    fi
    PICK=""
    if [ ${#CANDS[@]} -gt 0 ]; then
        say "   Found an existing LoRA Studio with your settings:"
        i=1; for d in "${CANDS[@]}"; do say "     $i) $d"; i=$((i+1)); done
        if [ "$YES" = 1 ] || [ ! -t 0 ]; then n=1; else read -r -p "   Take settings, token, Completed marks and caches from which one? (number, Enter = 1, 0 = none) " n || n=1; fi
        n="${n:-1}"
        [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le ${#CANDS[@]} ] && PICK="${CANDS[$((n-1))]}"
    fi
    if [ -n "$PICK" ]; then
        for f in $DATA_FILES; do if [ -f "$PICK/$f" ]; then cp -p "$PICK/$f" "$TARGET/userdata/$f"; fi; done
        [ -d "$PICK/gui_logs" ] && cp -rp "$PICK/gui_logs" "$TARGET/userdata/" 2>/dev/null || true
        ok "copied settings from $PICK (the old folder is not changed)"
    else
        ok "no old settings imported: LoRA Studio asks its setup questions at the end"
    fi
else
    ok "keeping the settings already in $TARGET"
fi

# ------------------------------------------------------------------ 3. venv
if [ "$TRAINERS" = existing ]; then step "A small Python environment for LoRA Studio"; else step "Python ${PYVER} environment"; fi
VENV="$TARGET/venv"; PY="$VENV/bin/python"
if [ "$TRAINERS" = install ] && [ -x "$PY" ] && ! "$PY" -c "import sys; raise SystemExit(sys.version_info[:2] != tuple(map(int, '$PYVER'.split('.'))))" 2>/dev/null; then
    OLD="$VENV.old-$(date +%Y%m%d-%H%M)"
    warn "$VENV is a Python $("$PY" -c 'import sys;print(sys.version.split()[0])') environment (the old GUI-only one); moving it to $OLD"
    mv "$VENV" "$OLD"
fi
if [ ! -x "$PY" ]; then
    "$PY_SYS" -m venv "$VENV"
    ok "created $VENV"
fi
"$PY" -m pip install -q --upgrade pip setuptools wheel
ok "pip $("$PY" -m pip --version | awk '{print $2}')"
PIP() { "$PY" -m pip "$@"; }

if [ "$TRAINERS" = install ]; then
# ------------------------------------------------------------------ 4. PyTorch
step "PyTorch for $( [ "$GPU" = amd ] && echo "ROCm ($GFX)" || echo CUDA )"
HAVE_TORCH="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
TORCH_FITS=0
if [ -n "$HAVE_TORCH" ]; then
    if [ "$GPU" = amd ]; then
        WANT_TORCH="${TORCH_SPEC_AMD##*==}"
        NEWEST="$(printf '%s\n%s\n' "$HAVE_TORCH" "$WANT_TORCH" | sort -V | tail -1)"
        if [ "$HAVE_TORCH" = "$WANT_TORCH" ]; then TORCH_FITS=1
        elif [[ "$HAVE_TORCH" == *rocm* ]] && [ "$NEWEST" = "$HAVE_TORCH" ] && [ "$REINSTALL_TORCH" = 0 ]; then
            TORCH_FITS=1                       # newer than this installer's (upgraded in Settings › Updates): kept
            ok "torch $HAVE_TORCH is newer than this installer's $WANT_TORCH: kept"
        elif [[ "$HAVE_TORCH" == *rocm* ]] && [ "$REINSTALL_TORCH" = 0 ]; then
            say "   Installed: torch $HAVE_TORCH. This installer brings torch $WANT_TORCH (ROCm $ROCM_VER)."
            if yes_no "Install torch $WANT_TORCH with ROCm $ROCM_VER now? (several GB; No keeps the current one)" y; then
                REINSTALL_TORCH=1
            else
                TORCH_FITS=1
            fi
        fi
    else [[ "$HAVE_TORCH" != *rocm* ]] && TORCH_FITS=1 || true; fi
fi
if [ "$TORCH_FITS" = 1 ] && [ "$REINSTALL_TORCH" = 0 ]; then
    ok "already installed: torch $HAVE_TORCH (use --reinstall-torch to install it again)"
else
    if [ "$GPU" = amd ]; then
        say "   Downloading ROCm and PyTorch for $GFX from AMD (several GB, takes a while)..."
        PIP install --force-reinstall --index-url "$ROCM_INDEX" "${ROCM_SPEC//@GFX@/$GFX}" ${ROCM_PARTS//@GFX@/$GFX}
        PIP install --force-reinstall --index-url "$ROCM_INDEX" \
            "${TORCH_SPEC_AMD//@GFX@/$GFX}" "${TV_SPEC_AMD//@GFX@/$GFX}" "$TA_SPEC_AMD"
    else
        say "   Downloading PyTorch for CUDA (several GB, takes a while)..."
        if [ -n "$CUDA_INDEX" ]; then PIP install --force-reinstall --index-url "$CUDA_INDEX" torch torchvision torchaudio
        else PIP install --force-reinstall torch torchvision torchaudio; fi
    fi
    HAVE_TORCH="$("$PY" -c 'import torch; print(torch.__version__)')"
    ok "installed torch $HAVE_TORCH"
fi
if [ "$GPU" = amd ]; then          # every ROCm part must be the version PyTorch was built for
    WANT_ROCM="${HAVE_TORCH##*+rocm}"
    ODD="$("$PY" -c '
import sys
from importlib.metadata import distributions
want = sys.argv[1]
odd = sorted({(d.metadata["Name"] or "") + "==" + d.version for d in distributions()
              if (d.metadata["Name"] or "").lower().startswith("rocm")
              and (d.metadata["Name"] or "").lower() != "rocm-bootstrap" and d.version != want})
print(" ".join(odd))' "$WANT_ROCM")"
    if [ -n "$ODD" ]; then
        warn "ROCm parts that do not match PyTorch (built for ROCm $WANT_ROCM): ${ODD//==/ }"
        if yes_no "Put them back to ROCm $WANT_ROCM? (mixed versions crash trainings with 'Memory access fault by GPU')" y; then
            FIX=""
            for o in $ODD; do FIX="$FIX ${o%%==*}==$WANT_ROCM"; done
            PIP install --no-deps --force-reinstall --index-url "$ROCM_INDEX" $FIX
            ok "ROCm parts are $WANT_ROCM again"
        fi
    else
        ok "all ROCm parts are $WANT_ROCM, as PyTorch expects"
    fi
fi
CONS="$TARGET/constraints.txt"
# versions straight from the installed packages ("pip freeze" prints "name @ file://..." for some wheels)
lock_versions() {   # lock_versions REGEX  -> "name==version" for every installed distribution whose name matches
    "$PY" - "$1" <<'EOF'
import re, sys
from importlib.metadata import distributions
pat = re.compile(sys.argv[1], re.I)
seen = set()
for d in distributions():
    n = (d.metadata["Name"] or "").strip()
    if n and pat.fullmatch(n) and n.lower() not in seen:
        seen.add(n.lower())
        print(f"{n}=={d.version}")
EOF
}
{ echo "# Written by install.sh: versions that later 'pip install' runs must keep (pip install ... -c constraints.txt)"
  lock_versions '(torch|torchvision|torchaudio|triton|pytorch-triton-rocm|rocm[a-z0-9_-]*)'
} > "$CONS"
grep -qi '^torch==' "$CONS" || die "could not read the installed PyTorch version"
ok "PyTorch versions locked in constraints.txt"
# 8-bit optimizers (AdamW8bit etc.); the trainers list it too, but it is installed on its own like in a manual setup
PIP install -q bitsandbytes -c "$CONS"
ok "bitsandbytes $("$PY" -c 'from importlib.metadata import version; print(version("bitsandbytes"))')"

# ------------------------------------------------------------------ 5. trainers
step "$( [ "$SD_MODE" = install ] && echo sd-scripts )$( [ "$SD_MODE$MU_MODE" = installinstall ] && echo ' and ' )$( [ "$MU_MODE" = install ] && echo musubi-tuner )"
get_repo() {   # get_repo URL DIR
    local url="$1" dir="$2"
    if [ -d "$dir/.git" ]; then
        if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
            warn "$(basename "$dir") has local changes: not updated (git -C $dir status)"
        else
            git -C "$dir" pull -q --ff-only && ok "$(basename "$dir") updated to $(git -C "$dir" log -1 --format='%h %cd' --date=short)" \
                || warn "$(basename "$dir"): git pull failed, keeping the current version"
        fi
    else
        git clone -q "$url" "$dir"
        ok "$(basename "$dir") cloned ($(git -C "$dir" log -1 --format='%h %cd' --date=short))"
    fi
    # the trainers and every launch script expect <trainer>/venv: point it at the shared one
    if [ -e "$dir/venv" ] && [ ! -L "$dir/venv" ]; then
        warn "$(basename "$dir")/venv is a real folder: moved to venv.own (the shared venv is used instead)"
        mv "$dir/venv" "$dir/venv.own-$(date +%Y%m%d-%H%M)"
    fi
    ln -sfn ../venv "$dir/venv"
}
if [ "$SD_MODE" = install ]; then
    get_repo "$SD_REPO" "$TARGET/sd-scripts"
    say "   Installing the packages of sd-scripts..."
    ( cd "$TARGET/sd-scripts" && PIP install -q -r requirements.txt -c "$CONS" )
    ok "sd-scripts packages"
fi
if [ "$MU_MODE" = install ]; then
    get_repo "$MU_REPO" "$TARGET/musubi-tuner"
    say "   Installing the packages of musubi-tuner..."
    PIP install -q -e "$TARGET/musubi-tuner" -c "$CONS"
    ok "musubi-tuner packages"
fi

NOW_TORCH="$("$PY" -c 'import torch; print(torch.__version__)')"
[ "$NOW_TORCH" = "$HAVE_TORCH" ] || die "a trainer package replaced PyTorch ($HAVE_TORCH -> $NOW_TORCH). Run ./install.sh --reinstall-torch"

# lock the shared libraries too: an upgrade of one of these broke musubi before (huggingface-hub 2.x)
lock_versions '(transformers|diffusers|accelerate|huggingface[-_]hub|safetensors|tokenizers|bitsandbytes)' >> "$CONS"
ok "trainer library versions locked in constraints.txt"
fi   # TRAINERS = install (steps 4 and 5)

# ------------------------------------------------------------------ 6. GUI packages
step "LoRA Studio packages (Gradio)"
if [ "$TRAINERS" = install ]; then
    PIP install -q -r "$TARGET/requirements.txt" -c "$CONS"
    echo "$GPU ${GFX:-cuda}" > "$VENV/.lorastudio-shared"
else
    PIP install -q -r "$TARGET/requirements.txt"
    rm -f "$VENV/.lorastudio-shared"            # start.sh: this venv is LoRA Studio's own, not the training one
fi
sha256sum "$TARGET/requirements.txt" | cut -d' ' -f1 > "$VENV/.requirements.sha256"
ok "gradio $("$PY" -c 'import gradio; print(gradio.__version__)')"

# ------------------------------------------------------------------ 7. test
step "Testing"
if [ "$SD_MODE" = existing ]; then
    if ( cd "$EX_SD" && timeout 300 "$EX_VENV/bin/python" sdxl_train_network.py --help >/dev/null 2>/tmp/lorastudio-help.txt ); then
        ok "your sd-scripts: sdxl_train_network.py starts in its venv"
    else
        warn "your sd-scripts: sdxl_train_network.py --help failed in $EX_VENV:"; tail -3 /tmp/lorastudio-help.txt | sed 's/^/     /'
    fi
    "$EX_VENV/bin/python" -c 'import torch; ok = torch.cuda.is_available(); print("   torch", torch.__version__, "GPU available:", ok, ("· " + torch.cuda.get_device_name(0)) if ok else "")' \
        || warn "PyTorch does not load in $EX_VENV"
fi
[ "$TRAINERS" = existing ] && ok "LoRA Studio itself: $("$PY" -c 'import gradio; print("gradio", gradio.__version__)')"
if [ "$TRAINERS" = install ]; then
if PIP check >/tmp/lorastudio-pipcheck.txt 2>&1; then ok "pip check: no broken requirements"
else warn "pip check reports (usually harmless, see install.log):"; sed 's/^/     /' /tmp/lorastudio-pipcheck.txt | head -8; cat /tmp/lorastudio-pipcheck.txt >> "$LOG"; fi
"$PY" - <<'EOF' || die "PyTorch cannot use the GPU (see the lines above)"
import torch
ok = torch.cuda.is_available()
kind = f"ROCm/HIP {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
print(f"   torch {torch.__version__}, {kind}, GPU available: {ok}")
if not ok:
    raise SystemExit(1)
print(f"   device: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print(f"   bf16 matmul on the GPU: {'ok' if torch.isfinite(x @ x).all() else 'NOT finite'}")
EOF
if BNB_OUT="$("$PY" -c 'import bitsandbytes as b; print(b.__version__)' 2>&1 | tail -1)" && [[ "$BNB_OUT" =~ ^[0-9] ]]; then
    ok "bitsandbytes $BNB_OUT loads"
else
    warn "bitsandbytes does not load (8-bit optimizers such as AdamW8bit will fail; AdamW works): $BNB_OUT"
fi
for t in "sd-scripts:sdxl_train_network.py" "sd-scripts:flux_train_network.py" "musubi-tuner:src/musubi_tuner/zimage_train_network.py"; do
    d="${t%%:*}"; s="${t#*:}"
    [ "$d" = sd-scripts ] && [ "$SD_MODE" != install ] && continue
    [ "$d" = musubi-tuner ] && [ "$MU_MODE" != install ] && continue
    if ( cd "$TARGET/$d" && timeout 300 "$PY" "$s" --help >/dev/null 2>/tmp/lorastudio-help.txt ); then ok "$d: $(basename "$s") starts"
    else warn "$d: $(basename "$s") --help failed:"; tail -3 /tmp/lorastudio-help.txt | sed 's/^/     /'; fi
done
fi   # TRAINERS

# ------------------------------------------------------------------ Ollama (Linux only; on Windows it is installed in Windows)
OL_MODEL="qwen2.5vl:7b"
if [ "$OL_MODE" = install ]; then
    step "Ollama"
    if command -v ollama >/dev/null; then
        ok "already installed: $(ollama --version 2>/dev/null | tail -1)"
    elif yes_no "Install Ollama now with its own installer (curl -fsSL https://ollama.com/install.sh | sh, asks for your password)?" y; then
        if curl -fsSL https://ollama.com/install.sh | sh; then ok "Ollama installed"
        else warn "the Ollama installer failed: see https://ollama.com/download/linux, then tick Ollama in Settings › Start and network"; fi
    else
        warn "not installed: get it from https://ollama.com/download/linux"
    fi
    if command -v ollama >/dev/null; then
        for i in 1 2 3 4 5; do curl -s -m 2 "$OL_URL/api/tags" >/dev/null 2>&1 && break; sleep 2; done
        if ! curl -s -m 2 "$OL_URL/api/tags" >/dev/null 2>&1; then
            warn "Ollama does not answer at $OL_URL yet (LoRA Studio starts it when it needs it)"
        elif ollama list 2>/dev/null | grep -q "^${OL_MODEL%%:*}"; then
            ok "a ${OL_MODEL%%:*} vision model is there"
        elif yes_no "Download the vision model $OL_MODEL for the sort and captions (about 6 GB)?" y; then
            ollama pull "$OL_MODEL" && ok "$OL_MODEL downloaded" || warn "the download failed: ollama pull $OL_MODEL"
        fi
    fi
fi

# ------------------------------------------------------------------ 8. settings
step "LoRA Studio settings"
FRESH=0
if [ ! -f "$TARGET/userdata/library_settings.txt" ]; then
    FRESH=1
    ok "new install: the first ./start.sh opens a setup page in the browser (folders, models, ComfyUI,"
    say "     Hugging Face token, Ollama, backups; every step can be skipped), or imports a backup"
fi
GPU_KIND="$GPU"                                   # which settings LoRA Studio shows: ROCm (amd) or CUDA (nvidia)
if [ "$GPU_KIND" = auto ]; then
    if [ -e /dev/kfd ]; then GPU_KIND=amd; elif command -v nvidia-smi >/dev/null || [ -x /usr/lib/wsl/lib/nvidia-smi ]; then GPU_KIND=nvidia; fi
fi
"$PY" - "$TARGET" "$SD_MODE" "$EX_SD" "$EX_MU" "$EX_VENV" "$MU_MODE" "$OL_MODE" "$OL_URL" "$IS_WSL" "$GPU_KIND" <<'EOF'
import re, shutil, sys
from pathlib import Path
T = Path(sys.argv[1]); home = str(Path.home())
tilde = lambda p: "~" + str(p)[len(home):] if str(p).startswith(home) else str(p)
def set_keys(path: Path, section: str, values: dict, sep=" = "):
    """Set key = value inside [section] (case-insensitive), keeping every other line. A .bak is kept."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if path.is_file():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    start = next((i for i, l in enumerate(lines) if l.strip().lower() == f"[{section}]"), None)
    if start is None:
        lines += ["", f"[{section}]"]; start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    while end > start + 1 and not lines[end - 1].strip():      # new keys go before the blank line(s)
        end -= 1
    for k, v in values.items():
        hit = next((i for i in range(start + 1, end) if re.match(rf"\s*{re.escape(k)}\s*=", lines[i])), None)
        if hit is None:
            lines.insert(end, f"{k}{sep}{v}"); end += 1
        else:
            lines[hit] = f"{k}{sep}{v}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
import json
sd_mode, ex_sd, ex_mu, ex_venv, mu_mode, ol_mode, ol_url, wsl, gpu = (sys.argv[2:11] + [""] * 9)[:9]
if sd_mode == "existing":        # your own sd-scripts: launch scripts activate its venv
    sd, venv_val = Path(ex_sd), str(Path(ex_venv))
else:                            # installed here (the shared venv is linked in as sd-scripts/venv), or none:
    sd, venv_val = T / "sd-scripts", "none"      # scripts are still written for this folder
mu = Path(ex_mu) if mu_mode == "existing" and ex_mu else T / "musubi-tuner"
UD = T / "userdata"
UD.mkdir(exist_ok=True)
lib = UD / "library_settings.txt"
if lib.is_file():
    set_keys(lib, "folders", {"sd_scripts_dir": tilde(sd)})
keys = {"SDGUI_SD_SCRIPTS": tilde(sd), "SDGUI_WORKDIR": str(sd), "SDGUI_VENV": venv_val, "SDGUI_MUSUBI": tilde(mu)}
if ol_mode in ("install", "remote") and ol_url:
    keys["OLLAMA_URL"] = ol_url
set_keys(UD / "training_settings.txt", "general", keys)
word = {"install": "installed", "existing": "existing", "none": "none", "remote": "remote", "": "installed"}
feat = {"ollama": word.get(ol_mode, "installed"), "sd_scripts": word.get(sd_mode, "installed"),
        "musubi": word.get(mu_mode, "installed"), "platform": "wsl" if wsl == "1" else "linux"}
if gpu in ("amd", "nvidia"):
    feat["gpu"] = gpu
(UD / "features.json").write_text(json.dumps(feat, indent=1), encoding="utf-8")
show = lambda m, p: "none (launch scripts are made, not run here)" if m == "none" else tilde(p)
print(f"   sd-scripts:   {show(sd_mode, sd)}\n   musubi-tuner: {show(mu_mode, mu)}\n"
      f"   Ollama:       {ol_url if ol_mode in ('install', 'remote') else 'none'}\n"
      "   (Settings > Start and network shows what is installed; Settings > Folders the trainers)")
EOF
ok "settings point at the trainers; features.json says what is installed"

# ------------------------------------------------------------------ 9. Z-Image training adapter
if [ "$MODELS" = 1 ] && [ "$FRESH" = 1 ]; then
    step "Z-Image Turbo training adapter (optional)"
    ok "offered on the setup page (it goes into the model folder you choose there)"
elif [ "$MODELS" = 1 ]; then
    step "Z-Image Turbo training adapter (optional)"
    MDIR="$("$PY" - "$TARGET" <<'EOF'
import re, sys
from pathlib import Path
p = Path(sys.argv[1]) / "userdata" / "library_settings.txt"
m = re.search(r"^\s*models_dir\s*=\s*(.+)$", p.read_text(encoding="utf-8"), re.M) if p.is_file() else None
print(str(Path(m.group(1).strip()).expanduser()) if m and m.group(1).strip() else str(Path(sys.argv[1]) / "models"))
EOF
)"
    DEST="$MDIR/loras/zimage"
    if [ -f "$DEST/$ADAPTER_FILE" ]; then
        ok "already there: $DEST/$ADAPTER_FILE"
    elif yes_no "Download $ADAPTER_FILE (needed to train Z-Image Turbo) to $DEST ?" y; then
        mkdir -p "$DEST"
        if "$PY" - "$ADAPTER_REPO" "$ADAPTER_FILE" "$DEST" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
try:
    print("   " + hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))
except Exception as e:                       # one line instead of a traceback
    print(f"   {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    raise SystemExit(1)
EOF
        then ok "downloaded"; else warn "download failed; get it later from https://huggingface.co/$ADAPTER_REPO"; fi
    fi
    if [ -f "$DEST/$ADAPTER_FILE" ]; then
        "$PY" - "$TARGET/userdata/training_settings.txt" "$DEST/$ADAPTER_FILE" <<'EOF'
import re, sys
from pathlib import Path
p, f = Path(sys.argv[1]), sys.argv[2]
t = p.read_text(encoding="utf-8") if p.is_file() else ""
m = re.search(r"^\[zimage\]\s*$(.*?)(?=^\[|\Z)", t, re.M | re.S | re.I)
if m and re.search(r"^\s*base_weights\s*=\s*\S", m.group(1), re.M):
    print("   Z-Image base_weights is already set in Settings > Training: not changed")
else:
    block = f"base_weights = {f}\nbase_weights.dir = {Path(f).parent}\n"
    t = t[:m.end(1)].rstrip("\n") + "\n" + block + "\n" + t[m.end(1):] if m else t.rstrip("\n") + f"\n\n[zimage]\n{block}"
    p.write_text(t, encoding="utf-8")
    print("   set as the Z-Image Turbo adapter (--base_weights) in Settings > Training")
EOF
    fi
fi

# ------------------------------------------------------------------ 10. existing launch scripts
if [ "$TRAINERS" = existing ]; then
    OLD_SCRIPTS=()
else
step "Existing launch scripts"
mapfile -t OLD_SCRIPTS < <(cd "$TARGET" && "$PY" - "$TARGET" 2>/dev/null <<'EOF'
import re, sys
from pathlib import Path
T = Path(sys.argv[1]); home = Path.home()
sys.path.insert(0, str(T / "program"))
import lspaths
lspaths.bootstrap()
try:
    import lora_library as ll                      # LoRA Studio's own reader of library_settings.txt
    s = ll.load_settings()
    dirs = [Path(d) for d in s.get("dataset_dirs", []) + s.get("script_dirs", [])]
except Exception:
    dirs = []
new = {"SD_SCRIPTS_DIR": T / "sd-scripts", "MUSUBI_DIR": T / "musubi-tuner"}
seen = set()
for d in dirs:
    if not d.is_dir():
        continue
    for f in d.rglob("*.sh"):
        if f in seen or f.name.startswith(".gui_run_") or "venv" in f.parts:
            continue
        seen.add(f)
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for var, path in new.items():
            m = re.search(rf'^{var}="([^"]*)"', t, re.M)
            if m and Path(m.group(1).replace("${HOME}", str(home)).replace("$HOME", str(home))).expanduser() != path:
                print(f)
                break
EOF
)
if [ ${#OLD_SCRIPTS[@]} -eq 0 ]; then
    ok "no launch scripts point at other trainer folders"
else
    say "   ${#OLD_SCRIPTS[@]} launch script(s) still use the old ~/sd-scripts or ~/musubi-tuner, for example:"
    for f in "${OLD_SCRIPTS[@]:0:5}"; do say "     $f"; done
    if yes_no "Point them at the new trainers in $TARGET? (only the SD_SCRIPTS_DIR= / MUSUBI_DIR= line changes; a .bak is kept)" y; then
        REL="${TARGET#$HOME}"
        if [ "$REL" != "$TARGET" ]; then BASE='${HOME}'"$REL"; else BASE="$TARGET"; fi
        n=0
        for f in "${OLD_SCRIPTS[@]}"; do
            cp -p "$f" "$f.bak"
            sed -i -E "s|^SD_SCRIPTS_DIR=\"[^\"]*\"|SD_SCRIPTS_DIR=\"$BASE/sd-scripts\"|; s|^MUSUBI_DIR=\"[^\"]*\"|MUSUBI_DIR=\"$BASE/musubi-tuner\"|" "$f"
            n=$((n+1))
        done
        ok "$n launch script(s) updated (the old versions are next to them as .sh.bak)"
    else
        ok "left as they are: they keep using the old trainer folders"
    fi
fi
fi   # TRAINERS = install (step 10)

# ------------------------------------------------------------------ done
trap - ERR
step "Done"
say "   Start LoRA Studio:   cd $TARGET && ./start.sh"
[ "$FRESH" = 1 ] && say "                        (the first start opens the setup page in your browser)"
say "   Update everything:   cd $TARGET && ./install.sh   (or Settings › Updates in LoRA Studio)"
say ""
case "$SD_MODE" in
    install)  say "   sd-scripts:   installed in $TARGET/sd-scripts" ;;
    existing) say "   sd-scripts:   yours, in $EX_SD (venv $EX_VENV); ./install.sh never changes it" ;;
    none)     say "   sd-scripts:   none: launch scripts are made for $TARGET/sd-scripts, run them where you train" ;;
esac
case "$MU_MODE" in
    install)  say "   musubi-tuner: installed in $TARGET/musubi-tuner" ;;
    existing) say "   musubi-tuner: yours, in $EX_MU; ./install.sh never changes it" ;;
    none)     say "   musubi-tuner: none" ;;
esac
case "$OL_MODE" in
    install|remote) say "   Ollama:       $OL_URL" ;;
    *) say "   Ollama:       none. To add it later: $([ "$IS_WSL" = 1 ] && echo https://ollama.com/download/windows || echo https://ollama.com/download/linux)," 
       say "                 then tick it in Settings › Start and network" ;;
esac
say "   Change any of these: ./install.sh --ask"
say "   Log of this run: $LOG"
write_report "finished"
