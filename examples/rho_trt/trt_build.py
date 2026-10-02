"""Validates the export-friendly rewrite against the reference sampler, exports the prefix to ONNX and builds its
TensorRT engine.

Usage: python trt_build.py [--checkpoint REPO_OR_PATH] [--engine-dir DIR] [--chunk-size N] [--skip-export]
"""

import argparse
import pathlib
import time

import common
import torch
import trt_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=common.CHECKPOINT)
parser.add_argument("--engine-dir", type=pathlib.Path, default=None)
parser.add_argument("--chunk-size", type=int, default=50)
parser.add_argument("--max-text", type=int, default=128, help="max prompt tokens (template included)")
parser.add_argument("--opt-level", type=int, default=3)
parser.add_argument("--skip-export", action="store_true")
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--build-step", action="store_true", help="also build the denoise step as a TensorRT engine")
args = parser.parse_args()
args.engine_dir = args.engine_dir or common.CACHE_DIR / pathlib.Path(args.checkpoint).name
(args.engine_dir / "onnx").mkdir(parents=True, exist_ok=True)

policy = common.load_policy(args.checkpoint, chunk_size=args.chunk_size)
flow = policy.model.flow_model
sample = common.load_sample()
inputs = common.model_inputs(policy, sample)
noise = common.fixed_noise(args.chunk_size).cuda()
num_images = inputs["image"].shape[1]

# The image features are SigLIP2's second-to-last layer (select_layer=-2 in Bunny's transformers 4.x terms). Under
# transformers 5.x the reference reaches it via a remapped select_layer=-3 on a hidden_states tuple that is doubled
# once output capture hooks are installed, which happens during the first backbone call: so the reference's *first*
# call after loading uses layer 25, and every later call layer 26. We match the steady state.
tower = flow.vlm_backbone.model.vision_tower
vision = getattr(tower.vision_tower, "vision_model", tower.vision_tower)
depth = len(vision.encoder.layers)
vision_layers = depth - 1
lm_layers = flow.hidden_state_idx  # the reference runs hidden_state_idx + 1 layers and reads hidden_states[idx]
print(f"vision layers: {vision_layers}/{depth} (select_layer={tower.select_layer}), LM layers: {lm_layers}")

text_ids = trt_model.prompt_token_ids(flow, inputs["prompt"][0], num_images)
head_len = text_ids["head_len"]
ids = text_ids["ids"]
max_len = num_images * 256 + args.max_text
assert ids.shape[0] <= args.max_text, f"prompt has {ids.shape[0]} tokens > --max-text {args.max_text}"
length = num_images * 256 + ids.shape[0]
print(f"prefix: {num_images} images x 256 + {ids.shape[0]} text tokens = {length} (K/V buffers: {max_len} rows)")

consts = trt_model.step_constants(flow, tower, num_steps=flow.config.num_steps, max_len=max_len)
prefix = trt_model.Prefix(flow, consts, head_len=head_len, max_len=max_len, vision_layers=vision_layers,
                          lm_layers=lm_layers).eval()  # fmt: skip
step = trt_model.DenoiseStep(flow).eval()
images = inputs["image"][0].contiguous()
text = flow.vlm_backbone.model.embed_tokens(ids)
state = inputs["state"]

# 1) The rewrite must match the reference sampler.
with torch.no_grad():
    flow.sample_actions(inputs["image"], inputs["prompt"], state, noise=noise, image_mask=inputs["image_mask"])  # warm
    ref = flow.sample_actions(inputs["image"], inputs["prompt"], state, noise=noise, image_mask=inputs["image_mask"])
    ref_hidden, _ = flow.get_image_text_hidden_state(inputs["image"], inputs["prompt"], image_mask=inputs["image_mask"])
    ours_hidden = flow.vlm_projector(prefix.language(torch.cat([text[:head_len], prefix.vision(images), text[head_len:]])))
    ours = trt_model.sample_actions_eager(prefix, step, consts, images, text, state, noise, length)
    ref2 = flow.sample_actions(inputs["image"], inputs["prompt"], state, noise=noise, image_mask=inputs["image_mask"])


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


action_dim = policy.config.action_feature.shape[0]
print(f"VLM hidden state: rel. error {rel(ours_hidden[None], ref_hidden):.2e}")
print(f"actions: max|rewrite - reference| = {(ours - ref)[..., :action_dim].abs().max().item():.2e}, "
      f"reference run-to-run {(ref2 - ref)[..., :action_dim].abs().max().item():.2e}, "
      f"max|reference| {ref[..., :action_dim].abs().max().item():.2f}")  # fmt: skip
torch.save({k: v.cpu() for k, v in consts.items()}, args.engine_dir / "consts.pt")
torch.save(
    {"images": images.cpu(), "ids": ids.cpu(), "head_len": head_len, "state": state.cpu(), "noise": noise.cpu(),
     "ref": ref.cpu(), "length": length, "max_len": max_len},
    args.engine_dir / "sample_io.pt",
)  # fmt: skip
if args.validate_only:
    raise SystemExit

# 2) Export the prefix to ONNX (strongly typed: bf16 weights, fp32 normalizations and softmax).
onnx_path = args.engine_dir / "onnx/prefix.onnx"
kv_names = [f"kv_{b}" for b in range(len(flow.action_expert.transformer_blocks))]
if not args.skip_export:
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            prefix, (images, text), str(onnx_path), opset_version=19, dynamo=False,
            input_names=["images", "text"], output_names=kv_names, dynamic_axes={"text": {0: "num_tokens"}},
        )  # fmt: skip
    print(f"onnx export: {time.perf_counter() - t0:.0f}s")

step_onnx = args.engine_dir / "onnx/denoise_step.onnx"
step_names = ["x_t", "state_emb", "mods", "time_sin", "final_mod", "key_bias", *kv_names]
if args.build_step and not args.skip_export:
    with torch.no_grad():
        kv = prefix(images, text)
        key_bias = torch.zeros(max_len, device="cuda")
        key_bias[length:] = float("-inf")
        step_args = (noise.to(torch.bfloat16), flow.state_projector(state.to(torch.bfloat16)), consts["mods"][0],
                     consts["time_sin"][0], consts["final_mod"][0], key_bias, *(kv_b[0] for kv_b in kv))  # fmt: skip
        torch.onnx.export(step, step_args, str(step_onnx), opset_version=19, dynamo=False, input_names=step_names,
                          output_names=["v_t"])  # fmt: skip

del policy, flow, prefix, step
torch.cuda.empty_cache()

# 3) Build the engines.
import tensorrt as trt  # noqa: E402

import trt_runtime  # noqa: E402


def build(onnx: pathlib.Path, engine_path: pathlib.Path, shapes: dict[str, tuple[tuple, tuple, tuple]]):
    builder = trt.Builder(trt_runtime.LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    onnx_parser = trt.OnnxParser(network, trt_runtime.LOGGER)
    if not onnx_parser.parse_from_file(str(onnx)):
        raise RuntimeError("\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    config = builder.create_builder_config()
    config.builder_optimization_level = args.opt_level
    if shapes:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in shapes.items():
            profile.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"failed to build {engine_path}")
    engine_path.write_bytes(engine)
    print(f"built {engine_path} in {time.perf_counter() - t0:.0f}s ({engine_path.stat().st_size / 1e9:.2f} GB)")


build(onnx_path, args.engine_dir / "prefix.engine", {
    "images": (tuple(images.shape),) * 3,
    "text": ((head_len + 1, text.shape[1]), tuple(text.shape), (args.max_text, text.shape[1])),
})  # fmt: skip
if args.build_step:
    build(step_onnx, args.engine_dir / "denoise_step.engine", {})
