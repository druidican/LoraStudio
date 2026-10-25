# Do and don't

## Do

- **Do** keep each dataset to one character (or one style, one object): mixing two teaches neither well.
- **Do** use different framings, outfits, places and light: the LoRA learns what stays the same.
- **Do** read step 8 Review before every training: a missing trigger word or empty caption is fixed in seconds there.
- **Do** train a new version instead of overwriting: `V1`, `V2`, … are kept, and Results compares them.
- **Do** try the save point Results suggests and the final one with a few test pictures before deciding.
- **Do** mark the version you like with ★ and copy it to your LoRA folder from Results (it brings its LoRA Manager data).
- **Do** run *Check this machine* after installing, after a driver or ROCm update, and when training fails strangely.
- **Do** make a backup (Settings › Backup) before big changes.

## Don't

- **Don't** put pictures of other characters, collages or screenshots with text in the repeat folder.
- **Don't** describe the character's fixed look (hair colour, eyes, body) in the captions of a character LoRA: then
  the trigger word does not learn it.
- **Don't** train with ComfyUI generating at the same time on the same GPU: the run runs out of video memory.
- **Don't** delete `_originals/` or `_reserve/` by hand while working on a dataset: they are how steps 5 and 6 can be undone.
- **Don't** edit a launch script while it is in the queue or training.
- **Don't** give your LoRA Studio folder to someone else: it holds your settings and your Hugging Face token. Give
  them LoraStudio.zip.

## Words used in LoRA Studio

| Word | Meaning |
|---|---|
| **Repeats** | How often each picture is shown per epoch (the `10` in `10_pip`) |
| **Epoch** | One pass over all pictures × repeats; a save point is written every N epochs |
| **Steps** | Pictures × repeats × epochs ÷ batch size: the amount of training |
| **Batch size** | Pictures trained together; higher is faster but needs more video memory |
| **Learning rate** | How big each learning step is; too high burns (harsh colours, NaN), too low learns slowly |
| **Network dim / alpha** | The size of the LoRA; higher holds more detail but copies the pictures sooner |
| **Trigger word** | The first word of every caption; the word that calls the LoRA in a prompt |
| **Save point** | The LoRA as it was after an epoch (`pipILLV2-000010.safetensors`) |
