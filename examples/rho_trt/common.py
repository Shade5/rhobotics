"""Shared helpers: policy loading, the YAM Box sample, fixed noise and timing."""

import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINT = os.environ.get("RHO_CHECKPOINT", "microsoft/rho-yam-box")
CACHE_DIR = pathlib.Path(os.environ.get("RHO_TRT_CACHE", "~/.cache/rho_trt")).expanduser()
SAMPLE = pathlib.Path(__file__).parent / "sample" / "yam_box_ep0_f120.npz"
IMAGE_SIZE = 256  # RhoConfig.resize_imgs_with_padding


def load_policy(checkpoint: str = CHECKPOINT, *, attention: str = "sdpa", chunk_size: int = 50):
    """Builds a `RhoPolicy` for a portable checkpoint (local path or Hugging Face repo id).

    The architecture comes from `RhoConfig` defaults, which match the public checkpoints; the strict state-dict load
    checks that. The midtrained checkpoints carry a legacy `policy.json`, so the config is built here instead of
    being decoded from it. chunk_size only sets how many action tokens are denoised (50 for rho-yam-box, 32 for the
    YAM Box finetuning recipe).
    """
    import draccus

    from rho.checkpoints import load_bundle_metadata, resolve_checkpoint
    from rho.policies import make_policy
    from rho.policies.base import PolicyConfig

    path = resolve_checkpoint(checkpoint, include_training_state=False)
    features = load_bundle_metadata(path)["features"]
    config = draccus.decode(
        PolicyConfig,
        {
            "type": "rho",
            "feature_dict": features,
            "chunk_size": chunk_size,
            "n_action_steps": chunk_size,
            "attention_implementation": attention,
            "enable_gradient_checkpointing": False,
        },
    )
    policy = make_policy(config)
    policy.load_from_pretrained(path)
    return policy.eval()


def load_sample() -> dict:
    """Episode 0, frame 120 of microsoft/BusyBox `I2RT_YAM_Box` (scene, left wrist, right wrist cameras).

    Returns the raw uint8 frames [3, 480, 640, 3], the model-space images [3, 3, 256, 256] in [0, 1] (center crop to
    480x480, then resize, as in environments/yambox/configs/yambox_dataset.yaml without augmentation), the
    quantile-normalized 20-d end-effector state and the task string.
    """
    data = np.load(SAMPLE)
    frames = torch.from_numpy(data["images"])
    return {
        "frames": data["images"],
        "images": preprocess_frames(frames),
        "state": torch.from_numpy(data["state"]),
        "task": str(data["task"]),
    }


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [n, h, w, 3] camera frames -> float [n, 3, 256, 256] in [0, 1]."""
    images = frames.permute(0, 3, 1, 2).float().div(255.0)
    h, w = images.shape[-2:]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    images = images[..., top : top + side, left : left + side]
    images = F.interpolate(images, size=(448, 448), mode="bilinear", align_corners=False)
    images = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return images.clamp(0.0, 1.0)


def fixed_noise(chunk_size: int, action_dim: int = 32, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).standard_normal((1, chunk_size, action_dim)).astype(np.float32))


def model_inputs(policy, sample: dict, device: str = "cuda") -> dict:
    """The tensors `FlowMatchingModel.sample_actions` receives inside `RhoPolicy.sample_actions` (batch size 1)."""
    num_images = sample["images"].shape[0]
    return {
        "image": sample["images"][None].to(device),
        "prompt": policy.prepare_prompt({"task": [sample["task"]]}),
        "state": F.pad(sample["state"], (0, policy.config.max_state_dim - sample["state"].shape[-1]))[None, None].to(
            device
        ),
        "image_mask": torch.ones(1, num_images, dtype=torch.int32, device=device),
    }


def timeit(fn, *, warmup: int = 10, iters: int = 100) -> tuple[float, float]:
    """Mean and std latency in ms, synchronizing after every call (what a robot client sees)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.std(times))
