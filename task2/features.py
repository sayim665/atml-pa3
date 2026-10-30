"""Cache frozen SigLIP patch features (196 x 768, FP16) for every selected image.

    python -m task2.features

Preprocessing is deterministic: RGB, bicubic resize to 224 x 224, scale to [0, 1], normalize with mean 0.5 and
std 0.5 (SigLIP's processor settings). The features are the encoder's final hidden states after SigLIP's
post-layernorm, applied once, in the 14 x 14 row-major patch order.
"""

import time

import numpy as np
import torch

from task1.common import environment_info, save_json
from task2.config import IMAGE_SIZE, NUM_PATCHES, SMOKE_TEST, VISION_DIM, VISION_NAME, data_root
from task2.data import image_path, load_manifest, manifest_images

BATCH = 64


def feature_paths():
    root = data_root() / "features"
    return root / "patch_features_fp16.npy", root / "index.json"


def preprocess(path):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5
    return torch.from_numpy(pixels).permute(2, 0, 1)


def load_vision_model(device):
    from transformers import SiglipVisionConfig, SiglipVisionModel

    if SMOKE_TEST:
        config = SiglipVisionConfig(hidden_size=VISION_DIM, intermediate_size=3072, num_hidden_layers=2,
                                    num_attention_heads=12, image_size=IMAGE_SIZE, patch_size=16)
        torch.manual_seed(0)
        model = SiglipVisionModel(config)
    else:
        model = SiglipVisionModel.from_pretrained(VISION_NAME)
    return model.to(device).eval()


@torch.no_grad()
def encode(model, pixels, check=False):
    outputs = model(pixel_values=pixels, output_hidden_states=check)
    features = outputs.last_hidden_state
    if check:
        post_layernorm = next(module for name, module in model.named_modules() if name.endswith("post_layernorm"))
        manual = post_layernorm(outputs.hidden_states[-1])
        difference = (manual - features).abs().max().item()
        assert difference < 1e-4, f"last_hidden_state is not post-layernorm output ({difference})"
        return features, difference
    return features, None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest()
    images = manifest_images(manifest)
    features_path, index_path = feature_paths()
    features_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        print("features already cached", flush=True)
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model = load_vision_model(device)
    load_seconds = time.time() - started
    store = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float16,
                                      shape=(len(images), NUM_PATCHES, VISION_DIM))
    layernorm_check = None
    for start in range(0, len(images), BATCH):
        keys = images[start:start + BATCH]
        pixels = torch.stack([preprocess(image_path(key)) for key in keys]).to(device)
        features, difference = encode(model, pixels, check=start == 0)
        if difference is not None:
            layernorm_check = difference
        assert features.shape[1:] == (NUM_PATCHES, VISION_DIM), features.shape
        store[start:start + len(keys)] = features.float().cpu().numpy().astype(np.float16)
        if start // BATCH % 20 == 0:
            print(f"{start + len(keys)}/{len(images)} images", flush=True)
    store.flush()
    del store

    info = {
        "images": len(images),
        "shape": [len(images), NUM_PATCHES, VISION_DIM],
        "dtype": "float16",
        "vision_model": VISION_NAME if not SMOKE_TEST else "random-init (smoke test)",
        "post_layernorm_check_max_abs_diff": layernorm_check,
        "model_load_seconds": load_seconds,
        "total_seconds": time.time() - started,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
        "bytes": features_path.stat().st_size,
        "environment": environment_info(),
    }
    save_json(data_root() / "features" / "cache_info.json", info)
    save_json(index_path, {key: row for row, key in enumerate(images)})
    print(info, flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FeatureStore:
    def __init__(self):
        features_path, index_path = feature_paths()
        self.features = np.load(features_path, mmap_mode="r")
        from task1.common import load_json

        self.index = load_json(index_path)

    def get(self, keys):
        rows = [self.index[key] for key in keys]
        return torch.from_numpy(np.stack([self.features[row] for row in rows]))


if __name__ == "__main__":
    main()
