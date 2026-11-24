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
- **Do** train on the **full** model files (bf16 / fp16). The small int8, fp8 and GGUF versions are for making
  pictures in ComfyUI; musubi-tuner (Z-Image, Qwen-Image, FLUX.2) cannot train on them, and stage 1 says so.
- **Do** train with the VAE you make pictures with, and the one the type is made for (Anima: `qwen_image_vae`,
  Z-Image: the FLUX `ae`).
- **Do** use a **Z-Image Turbo** LoRA at about strength **1.3**: it was trained with the training adapter, which is not
  in the finished file, so it comes through weaker at 1.0. Its test pictures are made at 1.3.
- **Do** keep a settings profile (step 9) for settings you found good, and share the `.toml` file instead of a
  launch script.

## Don't

- **Don't** put pictures of other characters, collages or screenshots with text in the repeat folder.
- **Don't** leave the character's fixed look (hair colour, eyes, body) in the captions of a character LoRA: then
  the trigger word does not learn it. For tags, move those tags to the trigger words in step 7.
- **Don't** train with ComfyUI generating at the same time on the same GPU: the run runs out of video memory.
- **Don't** delete `_originals/` or `_reserve/` by hand while working on a dataset: they are how steps 5 and 6 can be undone.
- **Don't** edit a launch script while it is in the queue or training.
- **Don't** give your LoRA Studio folder to someone else: it holds your settings and your Hugging Face token. Give
  them LoraStudio.zip. (Launch scripts never hold the token: LoRA Studio passes it when a script runs.)
- **Don't** use a Qwen3-**VL** text encoder (the one that also reads pictures) for Z-Image: it needs plain Qwen3 4B.

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
| **Text encoder** | The part of the model that reads the prompt; trained lightly, it learns what the trigger word means |
| **Flow shift** | For flow models (Z-Image, FLUX, Anima): where in the noise the training spends its time; higher = more on large shapes (face, build), lower = more on fine detail |
| **Training adapter** | For Z-Image Turbo: a helper LoRA merged in while training, so the sped-up model can be trained; not part of the finished LoRA |
| **Settings profile** | The training settings of one LoRA type in a `.toml` file, to share and import (step 9) |
| **int8 / fp8 / GGUF** | Smaller versions of a model file for making pictures; train on the full (bf16) file |
