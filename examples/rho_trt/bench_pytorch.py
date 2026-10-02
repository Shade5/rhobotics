"""Baseline: the stock PyTorch sampler (`FlowMatchingModel.sample_actions`) and the full `RhoPolicy.sample_actions`.

Usage: python bench_pytorch.py [--attention sdpa|flash_attention_2] [--chunk-size N]
"""

import argparse

import common
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=common.CHECKPOINT)
parser.add_argument("--attention", default="sdpa")
parser.add_argument("--chunk-size", type=int, default=50)
parser.add_argument("--iters", type=int, default=50)
args = parser.parse_args()

policy = common.load_policy(args.checkpoint, attention=args.attention, chunk_size=args.chunk_size)
flow = policy.model.flow_model
sample = common.load_sample()
inputs = common.model_inputs(policy, sample)
noise = common.fixed_noise(args.chunk_size).cuda()

# What a robot server calls per chunk: images (B, C, H, W) per camera in [0, 1], state, task.
batch = {key: sample["images"][i][None].cuda() for i, key in enumerate(policy.config.image_features)}
batch["observation.state"] = inputs["state"][:, 0, : sample["state"].shape[0]]
batch["task"] = [sample["task"]]


@torch.no_grad()
def model_only():
    return flow.sample_actions(inputs["image"], inputs["prompt"], inputs["state"], noise=noise,
                               image_mask=inputs["image_mask"])  # fmt: skip


@torch.no_grad()
def full_policy():
    return policy.sample_actions(batch, noise=noise)


@torch.no_grad()
def prefix_only():
    return flow.get_image_text_hidden_state(inputs["image"], inputs["prompt"], image_mask=inputs["image_mask"])


for name, fn in [("prefix (VLM) only", prefix_only), ("model sample_actions", model_only),
                 ("RhoPolicy.sample_actions", full_policy)]:  # fmt: skip
    mean, std = common.timeit(fn, iters=args.iters)
    print(f"{name:28s} {mean:7.1f} ms ± {std:.1f}  ({args.attention}, chunk {args.chunk_size})")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
