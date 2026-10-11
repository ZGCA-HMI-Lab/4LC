---
title: 4LC Interactive Demo
emoji: 🎬
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.14.0
python_version: 3.10.13
app_file: space.py
startup_duration_timeout: 1h
pinned: false
license: mit
fullWidth: true
---

# 4LC interactive demo

Build a 4D Latent Cache from a video or image, explore camera and time controls, preview conditions, and generate a video.

The Python backend reuses the core 4LC code and the [released model weights](https://huggingface.co/jianhuizhang/4LC-Video).

See [setup and usage](https://github.com/ZGCA-HMI-Lab/4LC/blob/main/app/README.md). The Space entrypoint loads the App and core code from the public GitHub repository and downloads model assets at startup.
