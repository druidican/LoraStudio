# Making a dataset

**Datasets › New dataset** starts with a folder of pictures. LoRA Studio makes the dataset folder with one repeat
folder `<N>_<name>` (for example `10_pip`): **only that folder is the dataset**. Pictures anywhere else in the
dataset folder (`_reserve/`, `_extra/`, other folders) are left out of every count and check, so they never give
false warnings.

![Datasets › New dataset: pick a folder, name the dataset, set the repeats](img/ds_new.jpg)

**Edit existing** lists your datasets with their next step; open one to see every step and its state.

![Datasets › Edit existing › pip: the state of each step](img/ds_steps.jpg)

## What makes good training pictures

![Good pictures and the ones the checks catch](img/good_bad.jpg)

For a character: **20–40 pictures**, the same character, different framings, outfits, places and light. Fewer good
pictures beat many similar ones. For a style: 30–80 pictures in that style with different subjects.

![Framing: full body, upper body, close-up](img/framing.jpg)

## 2 · Usable scan

![Step 2: size, sharpness and broken files, in seconds, no AI](img/step2.jpg)

Finds pictures that are too small for the LoRA type, blurry, broken or duplicates. *Move not usable to _reserve*
moves them out (never deletes). Tick *Also look at the pictures outside the repeat folder* to let steps 2, 4 and 5
see pictures in `_reserve/` and other folders too: the ones you keep in step 5 join the repeat folder.

## 3 · Setup

![Step 3: LoRA type, purpose, trigger word](img/step3.jpg)

- **LoRA type**: the base model family (several are fine: one caption set and one launch script per type).
- **Purpose**: character, style, object, action or concept. It decides how the captions are written.
- **Trigger word**: the first word of every caption, the word you put in a prompt to call the LoRA (`pip`).

## 4 · Sort

![Step 4: the quick pass and the Ollama pass](img/step4.jpg)

Pass 1 (no AI) removes what is too small, blurry or a duplicate. Pass 2 asks the Ollama vision model about every
picture: one clear subject, framing, outfit, place, quality, watermarks. It takes about a second or two per picture.

## 5 · Select

![Step 5: the suggested size and the mix of framings](img/step5.jpg)

A suggested number of pictures for the type and purpose, and a selection: **Safe** takes the cleanest pictures first,
**Varied** more close-ups, angles, outfits and details. Untick what you don't want; *Confirm selection* moves the rest
to `_reserve/` (you can swap them back any time).

## 6 · Watermarks and size

![Step 6: watermarks first, then the size of each picture](img/step6.jpg)

Watermarks found by the sort: remove them (the picture tools paint over them), crop them off, set the picture aside or
keep it. Then the size: pictures bigger than needed are scaled down, never squared or padded. The originals stay in
`_originals/`.

## 7 · Tags (captions)

![Step 7: captions and the consistency check](img/step7.jpg)

Writes a caption for every picture with Ollama: tags (Illustrious, Pony, SDXL) or sentences (Anima, Z-Image, FLUX).
For a character, captions describe what **changes** (clothes, pose, place) and leave out what is always the
character, so the trigger word learns it. The consistency check finds traits that are named in different ways.

## 8 · Review

![Step 8: the folder, the checks and every picture with its caption](img/step8.jpg)

The last look: pictures without a caption, captions without the trigger word first, very long captions. Click a
picture to see and edit its caption.

## 9 · Launch script

![Step 9: the plan, the settings and their recommendations](img/step9.jpg)

The recommended settings for the type, purpose and number of pictures, with what each one means. Change a field and
the plan (steps, time) follows. **Save as new version** writes `pipILLV1.sh`, the next one `pipILLV2.sh`; older
scripts move to `_old_runs/`. *Add to the queue* sends it to Training › Run.
