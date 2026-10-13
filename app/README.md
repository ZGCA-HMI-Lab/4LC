<h1><img src="https://raw.githubusercontent.com/ZGCA-HMI-Lab/4LC/main/assets/readme/4lc.svg" width="77" height="36" style="display: inline-block; vertical-align: middle; margin: 0;" alt="4LC"> interactive demo</h1>

Build a 4D Latent Cache from a video or image, explore camera and time controls, preview conditions, and generate a video.

The Python backend reuses the core 4LC code and the [released model weights](https://huggingface.co/jianhuizhang/4LC-Video). The usage guide is built into the App. The public Demo is [4LC-Demo](https://huggingface.co/spaces/jianhuizhang/4LC-Demo).

## Run locally

Prepare the core [environment](https://github.com/ZGCA-HMI-Lab/4LC#installation) and [model assets](https://github.com/ZGCA-HMI-Lab/4LC#models-and-configuration), including both released weights. Building the frontend requires Node.js 22.12 or later. From the repository root:

```bash
python -m pip install -r app/requirements.txt
npm --prefix app ci
npm --prefix app run build
PYTHONPATH=app python -m server
```

Open `http://127.0.0.1:7860/`. Both released profiles load by default; `--model configs/models/flux-all.yaml` loads only one. Run `PYTHONPATH=app python -m server --help` for the other options.

## Hugging Face Space

Create a Gradio Space with ZeroGPU and copy the files in [deploy/huggingface](deploy/huggingface/) to its root. At startup the Space checks out `main` of this repository, builds the frontend and downloads the model assets. To add examples, commit case directories with their input and `example.yaml` to `examples/`, then restart the Space.
