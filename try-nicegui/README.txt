NiceGUI test: Collection › Browse

1. Unzip so this folder lies inside your LoRA Studio folder (next to lora_studio.py), e.g. ~/LoraStudio/try-nicegui
2. In a terminal:   cd ~/LoraStudio/try-nicegui && ./start.sh
3. The page opens at http://127.0.0.1:7870 (Ctrl+C in the terminal stops it)

The first start makes its own venv in this folder (a few minutes, about 300 MB); LoRA Studio's venv is not touched.
It only reads the scan LoRA Studio saved (open the Collection in LoRA Studio once first). It writes nothing.
To remove it: delete this folder.
