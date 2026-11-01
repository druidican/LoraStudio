# Making a dataset

**Datasets** has three pages: **New dataset**, **Edit existing** and **Improvements**, and the nine steps they share.
A dataset is one folder with one repeat folder `<N>_<name>` (for example `10_pip`): **only that folder is the
dataset**. Pictures anywhere else in the dataset folder (`_reserve/`, `_extra/`, other folders) are left out of every
count and check, so they never give false warnings.

## 1 · Folder (New dataset)

![Datasets › New dataset: pick a folder, name the dataset, set the repeats](img/ds_new.jpg)

Pick a folder in your dataset folder. A folder with only pictures becomes a new dataset: LoRA Studio makes the repeat
folder and moves the pictures in (*Create and continue*). Folders that already are datasets open in **Edit existing**.

**Edit existing** lists your datasets with their next step; open one to see every step and its state.

![Datasets › Edit existing › pip: the state of each step](img/ds_steps.jpg)

**Add pictures from a folder** (in Edit existing): pictures that are good enough and not already in the dataset are
copied into its repeat folder (and into `_originals/`). Then sort and select again.

## What makes good training pictures

![Good pictures and the ones the checks catch](img/good_bad.jpg)

For a character: **20–40 pictures**, the same character, different framings, outfits, places and light. Fewer good
pictures beat many similar ones. For a style: 30–80 pictures in that style with different subjects.

![Framing: full body, upper body, close-up](img/framing.jpg)

## 2 · Usable scan

![Step 2: size, sharpness and broken files, in seconds, no AI](img/step2.jpg)

*Scan the pictures* finds pictures that are too small for the LoRA type, blurry, broken or duplicates. *Move not
usable to _reserve* moves them out (never deletes). Tick *Also look at the pictures outside the repeat folder* to let
steps 2, 4 and 5 see pictures in `_reserve/` and other folders too: the ones you keep in step 5 join the repeat
folder, and the setting switches itself off again when a launch script is saved.

## 3 · Setup

![Step 3: LoRA type, purpose, trigger word](img/step3.jpg)

- **LoRA type**: the base model family. Several are fine: sorting uses the strictest size, step 7 writes one caption
  set per caption style, step 9 one launch script per type.
- **Purpose**: character, style, object, action or concept. It decides how the captions are written and how the
  pictures are judged.
- **Trigger word**: the first word of every caption, the word you put in a prompt to call the LoRA (`pip`).

## 4 · Sort

![Step 4: the quick pass and the Ollama pass](img/step4.jpg)

Pass 1 (no AI) removes what is too small, blurry or a duplicate. Pass 2 asks the Ollama vision model about every
picture: one clear subject, framing, outfit, place, quality, watermarks, and how much the picture shows of the
subject. A picture that teaches almost nothing (the subject tiny, hidden or lost in the background) is left out; one
that shows a detail well (a tattoo, jewellery) ranks higher. For a character, a picture is also left out when clothes,
pose or angle hide nearly everything about the character, when the figure is small and the face tiny, or when it shows
only plain hands or feet (a tail, horns or a tattoo stay as detail shots). Sharpness only decides between pictures
that show the character equally well. It takes about a second or two per picture. Without Ollama only pass 1 runs,
and step 5 then picks by quality alone.

## 5 · Select

![Step 5: the suggested size and the mix of framings](img/step5.jpg)

A suggested number of pictures for the type and purpose, and a selection: **Safe** takes the cleanest pictures first,
**Varied** more close-ups, angles, outfits and details. When you have rated LoRAs, a line says what your good LoRAs of
this type had (pictures, close-ups, full body; see [Insights](#collection)). Untick what you don't want; *Confirm
selection* moves the rest to `_reserve/`; *Move all back from _reserve* undoes it.

The sort can still be wrong. Under each picture: **📌 Always** keeps it whatever the rules say (a unique detail the
sort dropped), **⛔ Never** leaves it out (useless content the sort kept). Pins are saved at once, stay after a new
sort and count in the automatic improvement too; click a pin again to remove it.

## 6 · Watermarks and size

![Step 6: watermarks first, then the size of each picture](img/step6.jpg)

**Watermarks** found by the sort: remove them (the picture tools paint over them), crop them off, set the picture
aside or keep it. *Recommended for the undecided* fills in the ones you have not chosen; *Apply the watermark choices*
does it. Click a picture to paint over a watermark yourself (*Remove what I painted*).

**Size, per picture**: what will happen to each picture, in the *Do* column (scale down, crop, keep). Pictures bigger
than needed are scaled down, never squared or padded. Click a picture to crop it yourself: *Around the person*
suggests a crop, *Use this crop* keeps it. *Resize* does the lot. The originals stay in `_originals/`, and *Put the
originals back* undoes the step.

## 7 · Tags (captions)

![Step 7: captions and the consistency check](img/step7.jpg)

**Write the captions** writes a caption for every picture; *Only pictures without a caption* fills the gaps; *Stop*
stops. **Tags** (Illustrious, Pony, SDXL, Anima) come from the **WD tagger** (wd-eva02-large-tagger-v3, run by
sd-scripts; no Ollama needed, about 1.2 GB downloaded the first time); **sentences** (Z-Image, FLUX, Qwen-Image) come
from Ollama. For a character, captions describe what **changes** (clothes, pose, place); what is always the character
belongs to the trigger word.

![Step 7 for a character: every tag with its count; the ticked ones move next to the trigger word](img/step7_tags.jpg)

**Character tags → trigger words** (tags, character datasets): the WD tagger writes **every** tag, also the
character's own (hair, eyes, freckles …). The card lists all tags with the number of pictures that have them; the
character's traits in most pictures are ticked already, and you tick or untick the rest. **Move the ticked tags**
takes them out of every caption and puts them in the Character description; the captions with all tags are kept,
and **Put them back** undoes it. *Read the tags* reads the captions again after you changed them by hand.

**Character description**: the traits the trigger word learns, in words (*slim build, tall, blonde hair, blue eyes*).
They are put next to the trigger word in the LoRA's LoRA Manager data, so ComfyUI offers them with it and you can add
them to a prompt to make the character stronger. *Suggest from the tags* fills in what the WD tagger saw in most
pictures; add build and height yourself, then *Save*.

**Caption settings** (the folded section): the length and detail of the captions, the **Never describe** list (what
sentence captions leave out for a character), and **Try on 3 pictures** to see the result first. *Undo last change*
puts the captions from before the last run back.

**Consistency, checked while tagging**: the traits of the character that the captions mention (hair, eyes, build,
skin, horns, tattoos, a signature object): the same in most pictures is taken out of the captions so the trigger word
learns it; words that mean the same (athletic, fit, toned) become one word; **odd pictures** that say something else
are shown so you can look. Clothes, places, poses, expressions and light stay. With two caption sets, *Apply to the
other caption set too* takes the same traits out of the second one.

## 8 · Review

![Step 8: the folder, the checks and every picture with its caption](img/step8.jpg)

The last look: pictures without a caption, captions without the trigger word first, very long captions. Click a
picture to see and edit its caption, then *Save caption changes*.

## 9 · Launch script

![Step 9: the plan, the settings and their recommendations](img/step9.jpg)

The recommended settings for the type, purpose and number of pictures, each with your value, the recommendation and
what it means. Change a field and the plan (steps, time) follows; *Use recommended* puts a value back, *Set epochs to
hit the target steps* adjusts the epochs. **More settings** has the optimizer, scheduler, saving, model files, the
advanced values and the repeat folder; **Why these settings** has the notes and the checks before training; **The
script text** shows the script itself.

- **Missing model files**: when a model file the script needs is missing, a card offers *⬇ Download from Hugging Face*
  for it (with your token for gated models).
- **From your own runs** (after a few rated LoRAs, see [Insights](#collection)): *Your graphics card* warns when this
  batch size ran out of video memory before and estimates the time from your runs; *Your good LoRAs* shows their
  settings next to the recommendations; *Epochs* suggests fewer when your runs usually peaked early.

**Save as new version** writes `pipILLV1.sh`, the next one `pipILLV2.sh`; older scripts move to `_old_runs/`. *Add to
the queue* sends it to Training › Run.

## Improvements

**Datasets › Improvements** lists what could be better in each dataset; see [Automatic improvement](#automatic).
