# Testing LoRA Studio

Thank you for testing. This page says what to try and what to send back, so the result is useful without anyone
looking over your shoulder. It takes about an hour, most of it waiting.

## What is most useful to test

LoRA Studio has been tested on Linux with an AMD Radeon (ROCm). Anything else is new:

| Your computer | Why it helps |
|---|---|
| **NVIDIA** on Linux | the CUDA install of PyTorch and both trainers |
| A **laptop with two GPUs** | the question which GPU to train on |
| **Windows** (NVIDIA or AMD) | the whole Windows installer: WSL2, the desktop starter, Ollama in Windows |
| **No GPU**, or no trainers | the greyed-out parts: does it stay clear what is missing and why? |

## The steps

1. **Install.** Unpack LoraStudio.zip. Linux: `./install.sh`. Windows: double-click **Install LoRA Studio.cmd**.
   Answer the questions as fits your computer; *none* is a fine answer.
2. **Keep the install report.** At the end (also when it stops) the installer writes an install report and shows
   where. On Linux it is in `~/LoraStudio/userdata/reports/`, on Windows next to the installer (and copied, so Ctrl+V pastes
   it). It has no passwords, tokens or your home folder in it.
3. **First start.** Start LoRA Studio (`./start.sh`, or the desktop starter on Windows). The setup page asks for your
   folders; every step can be skipped.
4. **Check this machine.** Settings › Version › **Check this machine**, with *a tiny training* ticked when a
   trainer is installed. It writes `lorastudio_check_<time>.md` in `userdata/`.
5. **A small real run** (with a trainer): make a dataset of 10–20 pictures in Datasets › New dataset, go through the
   steps, and train it with the launch script from step 9. Low settings are fine: this is about whether it runs.
6. **Look around** with a part missing (if you have one missing): is it clear from the greyed names what is missing,
   and does nothing start that cannot work?

## What to send back

- the **install report** (step 2),
- the **machine check** file (step 4),
- for anything that went wrong in LoRA Studio: Settings › Version › **Report for Claude** (or the *Report* button of
  a failed run in Training › Results). It also includes the main lines of the install report.
- a few words: what you expected, what happened, what was unclear. A screenshot helps for anything that looks wrong.

Please don't send your LoRA Studio folder itself: it holds your settings and your Hugging Face token.

See also [Installing](#install).
