---
title: Kelp
emoji: 🌿
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 3.9.1
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


Setup and run:
```
# Setup git lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Setup and run demo
git clone https://github.com/eolecvk/kelp.git
cd kelp
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
serve run demo:app
```