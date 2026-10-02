"""Benchmarks the TensorRT prefix + compiled denoise step sampler and checks it against the reference actions.

Usage: python bench_trt.py [--checkpoint REPO_OR_PATH] [--engine-dir DIR] [--chunk-size N]
"""

import argparse
import pathlib

import common
import torch
import trt_runtime

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=common.CHECKPOINT)
parser.add_argument("--engine-dir", type=pathlib.Path, default=None)
parser.add_argument("--chunk-size", type=int, default=50)
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--step-mode", default="max-autotune-no-cudagraphs", help="torch.compile mode for the step")
parser.add_argument("--gemm-backends", default=None, help="e.g. ATEN to keep cuBLAS GEMMs under max-autotune")
parser.add_argument("--step", default="compiled", choices=["compiled", "trt"])
args = parser.parse_args()
args.engine_dir = args.engine_dir or common.CACHE_DIR / pathlib.Path(args.checkpoint).name

policy = common.load_policy(args.checkpoint, chunk_size=args.chunk_size)
flow = policy.model.flow_model
# The prefix runs in TensorRT: only the token embeddings, state projector and action expert stay in PyTorch.
backbone = flow.vlm_backbone.model
backbone.vision_tower = None
backbone.layers = None
flow.vlm_backbone.lm_head = None
torch.cuda.empty_cache()

io = torch.load(args.engine_dir / "sample_io.pt")
if args.step == "trt":
    step = trt_runtime.TrtStep(args.engine_dir / "denoise_step.engine", args.chunk_size, flow.config.max_action_dim)
else:
    step = trt_runtime.CompiledStep(flow, args.step_mode, args.gemm_backends)
sampler = trt_runtime.Sampler(flow, args.engine_dir, num_images=io["images"].shape[0], step=step)
print(f"step: {args.step}" + (f", torch.compile mode={args.step_mode}, gemm backends={args.gemm_backends or 'default'}"
                               if args.step == "compiled" else ""))
images, ids, state, noise = (io[k].cuda() for k in ("images", "ids", "state", "noise"))
ref = io["ref"].cuda()
action_dim = policy.config.action_feature.shape[0]

out = sampler.run(images, ids, state, noise)
print(f"max|TensorRT sampler - reference| = {(out - ref)[..., :action_dim].abs().max().item():.2e} "
      f"(max|reference| {ref[..., :action_dim].abs().max().item():.2f})")  # fmt: skip

# The drop-in API (same arguments as FlowMatchingModel.sample_actions) gives the same actions.
prompt = policy.prepare_prompt({"task": [common.load_sample()["task"]]})
api = sampler.sample_actions(images[None], prompt, state, noise=noise)
assert torch.equal(api, out), "sample_actions and run disagree"

mean, std = common.timeit(lambda: sampler.run(images, ids, state, noise), iters=args.iters)
print(f"Sampler.run (float images on GPU)      {mean:6.2f} ms ± {std:.2f}")

frames = (images.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().pin_memory()
state_cpu, noise_cpu = state.cpu().pin_memory(), noise.cpu().pin_memory()
mean, std = common.timeit(lambda: sampler.run(frames, ids, state_cpu, noise_cpu).cpu(), iters=args.iters)
print(f"Sampler.run (uint8 host frames -> host) {mean:6.2f} ms ± {std:.2f}")

# Stage breakdown (outside the CUDA graph, so slightly higher than inside it).
g = sampler._graphs[(ids.shape[0], False)]  # noqa: SLF001
mean, _ = common.timeit(lambda: sampler.prefix.run(torch.cuda.current_stream(), images=g.images, text=g.text, **g.kv),
                        iters=args.iters)  # fmt: skip
print(f"  prefix engine                          {mean:6.2f} ms")
c = sampler.consts
x_t = noise.to(torch.bfloat16)
state_emb = flow.state_projector(state.to(torch.bfloat16))
step_args = (x_t, state_emb, c["mods"][0], c["time_sin"][0], c["final_mod"][0], g.key_bias, *(kv[0] for kv in g.kv_list))
mean, _ = common.timeit(lambda: sampler.step(*step_args), iters=args.iters)
print(f"  denoise step (x{sampler.num_steps})                   {mean:6.2f} ms")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB (+ TensorRT engine)")
