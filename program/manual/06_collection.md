# The Collection

The Collection has four pages: **Browse**, **LoRA page**, **Insights** and **Maintenance**.

## Browse

One tile per dataset with the picture of its newest (or ★ best) LoRA, and a label per LoRA type (✓ = Completed). With
*Colour the Collection cards by purpose* on (Settings › Look and base models), each card has a border and a label in
the colour of its purpose (character, art style, object, worn item, action / pose, concept).

![The Collection](img/collection.jpg)

- **Search** by name, character or trigger word; filter by **Base model**; sort *Newest first* or by *Name*;
  *Only with preview picture*, *Favorites only*.
- **Pictures / Status**: the tiles, or a list with the warnings and problems of the LoRA files themselves (what a
  dataset still needs is in Datasets › Improvements).
- **Select**: tick tiles (or *Select all shown*), then *✓ Mark Completed* or *Mark not completed*. **Completed** marks a
  version as finished: its launch script is hidden in Training › Run. A new version starts unmarked.
- *Rescan folders* reads your LoRA folders again (new or removed files).

## The LoRA page

Click a tile for its **LoRA page**; the *Version* menu at the top switches between the versions.

![The LoRA page of pip](img/lorapage.jpg)

- **Overview**: the preview picture (the one you chose in LoRA Manager; NSFW blur follows Settings › Look and base
  models), the versions, the trigger words with the Character description, how often you used the LoRA in your
  pictures and at which strength, and **LoRA Manager's example pictures** (the ones you added on its card, linked by
  *Sync now*; the tile says how many there are). *Accept the warnings* hides a warning you have looked at (for example
  *Run stopped at epoch 8 of 10* when you stopped it on purpose); press again to see it again. *Copy this LoRA to your
  LoRA folder* copies a version there, into the folder of its LoRA type.
- **Pictures**: every picture you made with this LoRA, all its versions together, found through the `.json` ComfyUI
  saves next to each picture (or the PNG's own metadata), with its prompt; *◀ Newer* / *Older ▶* page through them.
  *Scan for new pictures* reads only new and changed pictures, *Scan all picture folders* all of them. **Old names
  of this LoRA**: names found in your pictures that no LoRA file has now but that look like this LoRA (a rename): tick
  them and their pictures join.
- **Training**: what the version was trained with, in separate fields: the dataset and its repeat folders, epochs and
  how the steps add up, the network and optimizer settings, and the result (training time, final loss, the save point
  to try first). They are read from the LoRA file, its launch script and its logs, once per version; *Read the
  training again* after you changed the script or the log. *Train a new version…* opens Datasets › 9 Train.
- **Edit**: display name, character (groups it in Browse), version (shown in ComfyUI's LoRA Manager), clip skip,
  **Your rating** (4-5 good, 1-2 weak, for Insights), trigger words, description, notes, strength and CLIP strength,
  the preview's NSFW level, favorite, and a flag with its reason. Saved in LoRA Manager's format in the dataset's
  `LoraManager/` folder (a `.bak` is kept).

![The training card of pip V2](img/lp_training.jpg)

## Insights

![Collection › Insights: the datasets behind your good and weak LoRAs](img/insights.jpg)

**Insights** compares your own LoRAs, read on this computer. Rate versions on their LoRA page › Edit (*Your rating*:
4-5 good, 1-2 weak; without a rating the ★ best version and versions you made pictures with count as good). Per LoRA
type it shows:

- **Datasets behind good and weak LoRAs**: pictures, repeats, steps, close-ups, full body, outfits, places, detail
  shots, caption length. Step 5 shows what your good LoRAs of the type had.
- **Your graphics card**: the batch sizes that finished, the ones that ran out of video memory, and the seconds per
  step. Step 9 warns when the batch ran out of memory before and estimates the time from your own runs.
- **Settings that work for you**: learning rate, dim / alpha, epochs, steps and optimizer of your good LoRAs next to the
  others; step 9 shows them next to the recommendations.
- **Training too long?**: where in the run the save point to try first was; step 9 suggests fewer epochs when your runs
  usually peaked early.
- **How you use your LoRAs**: pictures made with each, the strength you use, when last, and LoRAs you never used. The
  LoRA page says how often you used a LoRA and offers that strength.

With few LoRAs these are hints, not rules: each says how many LoRAs it comes from. *Read again* after new runs.

## Maintenance

![Collection › Maintenance: sync with LoRA Manager, scan all LoRAs, free disk space](img/maint.jpg)

- **Sync now**: copies the LoRA Manager files of your own LoRAs (`.metadata.json`, preview, example pictures) between
  the LoRA folder and each dataset's `LoraManager/` folder, where the Collection reads them. A LoRA belongs to a
  dataset when it has the same name and size as a model file in the dataset, or the same SHA256 (renamed copies). It
  works both ways: what only one side has (notes, tags, trigger words, Civitai data) is copied to the other, an empty
  field takes the other side's value, and where both differ the newer file wins. The version name always comes from
  LoRA Studio, trigger words come from LoRA Manager when LoRA Studio has none, and with no clip skip anywhere both get
  1. A `.bak` is made before a file is changed.
- **LoRA Manager's example pictures folder**: where LoRA Manager keeps the pictures you add to a LoRA's card; empty =
  read from LoRA Manager's own settings. They are linked, not copied.
- **Scan all LoRAs**: reads every LoRA of your datasets once (launch script, the LoRA file's training data, repeat
  folders, run log) into the Training tab's fields. A scanned version is read again only when its file changes; tick
  *Read the scanned ones again too* to read them all.
- **Free disk space**: *Leftovers of all finished LoRAs* (epoch files, saved states, `.tmp.safetensors`, latent caches,
  musubi-tuner's caches, only for LoRAs with a final file; *Find leftovers*, tick *Yes*, *Move to Trash*) and
  **Orphaned LoRA Manager files** (a `.metadata.json` with its pictures whose LoRA file is gone). Everything goes to the
  desktop Trash.
