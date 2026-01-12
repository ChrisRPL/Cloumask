# Model Weights

This directory contains downloaded CV model weights. Files are gitignored and downloaded on demand.

## Automatic Download

Models are automatically downloaded on first use:

```python
from backend.cv import ModelManager

manager = ModelManager()
model = manager.get("yolo11m")  # Downloads if not present
```

## Manual Download

To pre-download models:

```python
from backend.cv import download_model, list_available_models

# See available models
for model in list_available_models():
    print(f"{model['name']}: {model['size_mb']}MB - {'Downloaded' if model['downloaded'] else 'Not downloaded'}")

# Download specific model
download_model("yolo11m")
download_model("sam2-hiera-large")
```

## Available Models

| Model | Size | Source | Description |
|-------|------|--------|-------------|
| yolo11m | 40 MB | Ultralytics | Object detection |
| yolo11n | 6 MB | Ultralytics | Object detection (nano) |
| sam2-hiera-large | 900 MB | HuggingFace | Segmentation |
| sam2-hiera-base | 320 MB | HuggingFace | Segmentation (base) |
| scrfd-10g | 16 MB | HuggingFace | Face detection |
| yolo-world-l | 170 MB | Ultralytics | Open vocabulary detection |

## HuggingFace Authentication

Some models require HuggingFace authentication. Set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

Get a token at: https://huggingface.co/settings/tokens

## Custom Models Directory

Override the default location with:

```bash
export CLOUMASK_MODELS_DIR="/path/to/models"
```

## Disk Space

Typical installation with common models:

- Minimal (YOLO only): ~50 MB
- Standard (+ SAM2 base): ~400 MB
- Full (all models): ~1.5 GB
