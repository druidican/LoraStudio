# The Collection

One tile per dataset with the picture of its newest (or ★ best) LoRA, and a label per LoRA type (✓ = Completed).

![The Collection](img/collection.jpg)

Click a tile for the **LoRA page**: the versions (the *Version* menu), the pictures you made with it in ComfyUI
(read from the pictures' own metadata), its training data, and Edit (name, version, clip skip, trigger words,
description, notes: saved in LoRA Manager's format).

![The LoRA page of pip](img/lorapage.jpg)

The **Training** tab shows what the version was trained with, in separate fields: the dataset and its repeat folders,
epochs and how the steps add up, the network and optimizer settings, and the result (training time, final loss, the
save point to try first). They are read from the LoRA file, its launch script and its logs, once per version:
*Maintenance › Scan all LoRAs* reads them all, and a version that was not read yet is read when you open it.

![The training card of pip V2](img/lp_training.jpg)

- **Completed** marks a version as finished: its launch script is hidden in Training › Run. A new version starts
  unmarked.
- **Maintenance** syncs LoRA Manager data and previews, and cleans up leftovers of finished runs. *Sync now* works
  both ways for the `.metadata.json` files: what only one side has (notes, tags, trigger words, Civitai data) is
  copied to the other, an empty field takes the other side's value, and where both differ the newer file wins. The
  version name always comes from LoRA Studio, trigger words come from LoRA Manager when LoRA Studio has none, and with
  no clip skip anywhere both get 1. The preview you chose in LoRA Manager, and its example pictures, show on the
  LoRA page too.
