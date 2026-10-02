# Fast Rho inference (TensorRT + torch.compile)

Low-latency inference for `microsoft/rho-yam-box` at batch size 1: **178.4 ms → ~32 ms per action chunk on an RTX
5090** (3 cameras at 256x256, 10 flow-matching steps, 50-action chunk). The actions match the stock PyTorch sampler to
within 4.1–4.5e-2, which is closer than stock PyTorch's own SDPA and eager attention backends are to each other (4.9e-2).

Everything below was run on a Runpod RTX 5090 pod (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Python 3.12).

## 1. Setup

From the repository root, create a [uv](https://docs.astral.sh/uv/) environment with a CUDA 12.8 PyTorch build
(needed for Blackwell GPUs such as the 5090; older GPUs work too):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Install Rho's inference dependencies, Rho itself, and TensorRT:

```bash
uv pip install "accelerate==1.10.0" "draccus==0.10.0" "einops>=0.8.0" "huggingface-hub[hf_xet]>=1.0.0,<2.0.0" \
  "numpy>=1.26.4" packaging "peft==0.18.0" pillow "roma>=1.5.4" "safetensors>=0.8.0rc.0,<1.0.0" tqdm \
  "transformers==5.10.1" onnx onnxscript
uv pip install --no-deps -e . -e rho_client
uv pip install "tensorrt-cu12==10.14.1.48.post1"
```

This skips the dataset/training extras: `lerobot` is only used for datasets and is installed from git, which can be
slow. `uv pip install -e ".[server]"` (the full install) works as well. No CUDA toolkit or `nvcc` is needed: the
scripts use PyTorch's SDPA attention instead of FlashAttention 2.

The checkpoint downloads automatically on first use. To fetch it ahead of time (~10 GB):

```bash
hf download microsoft/rho-yam-box --local-dir /path/to/rho-yam-box
export RHO_CHECKPOINT=/path/to/rho-yam-box   # default: microsoft/rho-yam-box (Hugging Face repo id)
```

The scripts run on a real YAM Box observation shipped in `sample/yam_box_ep0_f120.npz` (`microsoft/BusyBox`
`I2RT_YAM_Box`, episode 0, frame 120: scene, left-wrist and right-wrist frames, the normalized end-effector state and
the task "Flip the bottom switch off with the left gripper.").

## 2. Baseline inference (stock PyTorch)

```bash
cd examples/rho_trt
python bench_pytorch.py                  # --attention sdpa|eager, --chunk-size 50
```

Prints the latency of the stock VLM prefix, `FlowMatchingModel.sample_actions` (model only) and
`RhoPolicy.sample_actions` (with preprocessing). On the 5090: 35.7 ms, 178.4 ms and 182.8 ms.

## 3. Build the engine

```bash
python trt_build.py                      # ~2-3 min, writes ~10 GB (ONNX + engine)
```

This:

1. Checks the export-friendly rewrite (`trt_model.py`) against the stock sampler on the sample. It prints the VLM
   hidden-state error and the max action difference (expect ~1e-2 and ~3e-2).
2. Exports the prefix (SigLIP2 + Phi-3 + the action expert's cross-attention K/V) to ONNX.
3. Builds a bf16, strongly typed TensorRT engine from it.

Output in `$RHO_TRT_CACHE/<checkpoint name>/` (default `~/.cache/rho_trt/rho-yam-box/`):

| File | Contents |
|---|---|
| `prefix.engine` | TensorRT engine for the prefix (4.1 GB) |
| `consts.pt` | precomputed timestep constants (adaLN modulations, time embeddings, RoPE tables) |
| `sample_io.pt` | sample inputs + reference actions, used by `bench_trt.py` |
| `onnx/` | the exported ONNX model (can be deleted after the build) |

Options:
- `--checkpoint REPO_OR_PATH`: build for another checkpoint, such as your finetuned one.
- `--max-text N`: maximum prompt tokens, template included (default 128).
- `--build-step`: also build the denoise step as a TensorRT engine (slower than the default compiled step; for
  comparison).
- `--validate-only`: run only the accuracy check.

Engines are specific to the GPU model and the checkpoint's weights. Rebuild on each GPU type and after finetuning.

## 4. Optimized inference

```bash
python bench_trt.py                      # --step trt to use the TensorRT step from --build-step
```

The first run compiles the denoise step with `torch.compile` (max-autotune, a few minutes) and captures the CUDA graph;
later calls replay it. It prints the max difference from the reference actions (~4e-2), the end-to-end latency
(~32 ms on the 5090) and a breakdown (prefix engine ~20.6 ms).

In your own code, `Sampler` is a drop-in replacement for `FlowMatchingModel.sample_actions` at batch size 1 with all
three cameras:

```python
import common
import torch
import trt_runtime

policy = common.load_policy("microsoft/rho-yam-box")       # or your finetuned checkpoint
sampler = trt_runtime.Sampler(policy.model.flow_model, common.CACHE_DIR / "rho-yam-box")

images = common.preprocess_frames(frames)                    # uint8 [3, H, W, 3] (scene, left wrist, right wrist)
prompt = policy.prepare_prompt({"task": ["Flip the bottom switch off with the left gripper."]})
state = torch.nn.functional.pad(normalized_state, (0, 12))   # [20] -> [32]
actions = sampler.sample_actions(images[None].cuda(), prompt, state[None, None].cuda())  # [1, 50, 32] bf16
actions = actions[0, :, :20]                                 # model space, like the stock sampler
```

The actions are in model space (normalized, state-relative 6D deltas with absolute grippers), exactly like the stock
`sample_actions`. Convert them to robot commands with the checkpoint's output transforms, as `PolicyInterface` does.
For lower host overhead, call `sampler.run(images, token_ids, state, noise)` with uint8 `[3, 256, 256, 3]` frames
(converted inside the CUDA graph) and `sampler.token_ids(prompt[0])`. The first call for each new prompt length
captures a new CUDA graph.

Files:
- `trt_model.py`: export-friendly rewrite and the precomputed constants.
- `trt_build.py`: validation, ONNX export and engine build.
- `trt_runtime.py`: engines, compiled step and the CUDA-graph sampler.
- `common.py`: policy loading, the sample and timing.
- `bench_pytorch.py`, `bench_trt.py`: benchmarks.
- `profile_trt.py`: kernel and TensorRT per-layer profiles.

## Results

RTX 5090 (32 GB, sm_120), torch 2.8.0+cu128, transformers 5.10.1, TensorRT 10.14. Input: the sample above (786 prefix
tokens). Mean over 100-200 calls after warmup (std < 0.1 ms). Actions are compared in model space with the same fixed
noise; the largest action value is about 1.0.

| # | Configuration | Latency | Speedup | max \|Δ actions\| vs stock | Kept? |
|---|---|---:|---:|---:|---|
| 0 | **Baseline:** stock `FlowMatchingModel.sample_actions`, SDPA | **178.4 ms** | 1.0× | — | — |
| 0b | Stock `RhoPolicy.sample_actions` (adds preprocessing) | 182.8 ms | — | — | — |
| 0c | Stock with eager attention instead of SDPA (bf16 noise floor) | — | — | 4.9e-2 | — |
| 1 | TensorRT prefix + TensorRT step, one CUDA graph | 34.3 ms | 5.2× | 3.9e-2 | No |
| 2 | TensorRT prefix + `torch.compile` step (max-autotune), one CUDA graph | 32.8 ms | 5.4× | 3.9e-2 | Yes |
| 3 | Row 2 with the step's GEMMs forced to cuBLAS | 36.8 ms | 4.8× | 4.7e-2 | No |
| 4 | Row 2 with `torch.compile` in default mode | 36.7 ms | 4.9× | 4.7e-2 | No |
| 5 | **Row 2 + one fused Q/K/V GEMM per self-attention** | **31.5–32.4 ms** | **5.5–5.7×** | 4.1–4.5e-2 | **Yes (final)** |
| 5b | Row 5 with uint8 frames uploaded from pinned host memory and actions copied back | 32.7–32.8 ms | 5.4–5.5× | same | — |

| Stage | Stock | Optimized (row 5) |
|---|---:|---:|
| VLM prefix (SigLIP2 + Phi-3) | 35.7 ms | 20.6 ms, including every step's cross-attention K/V |
| 10 denoising steps | ~143 ms (~14 ms each) | ~11 ms (~1.1 ms each) |

- **Accuracy:** every optimized row is within the stock model's own attention-backend difference (4.9e-2).
- **Not timed individually:** the exact rewrites below all went in together in row 1.
- **Variation:** the range in row 5 is across two rebuilds (TensorRT and Inductor autotuning pick slightly different
  kernels).

## How it works

The sampler is split into two graphs, and the prefix plus all 10 steps (including the Euler updates) are captured in
**one CUDA graph** per prompt length:

1. **Prefix** (TensorRT, bf16, strongly typed): images + prompt embeddings → SigLIP2 → projector → Phi-3 → the action
   expert's cross-attention K/V for all 10 steps × 12 blocks.
2. **Denoise step** (torch.compile): the 12-block cross-attention action expert over 51 query tokens (state + 50
   actions), reading that step's K/V.

Exact (math-preserving) rewrites that make it cheaper:

- **Only the layers whose outputs are used run.**
  - The stock code runs `hidden_state_idx + 1` = 15 Phi-3 layers but reads the output of layer 14, so the rewrite
    runs 14.
  - SigLIP2 stops at the selected layer (26 of 27), skipping the post-layernorm and the pooling head.
- **All cross-attention K/V are computed in the prefix.** The expert's K/V depend only on the VLM output and on the
  timestep, through the adaLN modulation of `kv_norm`. The timestep schedule is fixed, so all 120 K/V pairs come out
  of a few large GEMMs instead of 120 small ones inside the steps.
- **Timestep-only quantities are precomputed with the model's own functions** (`step_constants`). These are the
  sinusoidal embeddings, `embed_time_for_cond`, and every shared-adaLN modulation (base + per-block offset) for every
  step. Recomputing the fp32 time loop the same way makes the values bit-identical.
- **Grouped-query attention without `repeat_interleave`.** The 4 query heads that share each K/V head are folded into
  the query axis.
- **One GEMM for the self-attention Q, K and V** (weights concatenated): 1.21 → 1.09 ms per step.
- **The prompt is variable-length, but the step is compiled once.** K/V buffers are padded to 3 × 256 + 128 rows with
  an additive key bias (0 / −inf), which is exact.

## Lessons learned

### The stock model selects a different SigLIP2 layer on its first call

The repo's transformers 5.x compatibility shim (`Phi5Backbone._apply_transformers_5x_siglip2_compat`) remaps Bunny's
`select_layer=-2` to `-3`. It assumes `hidden_states` holds every layer twice (2N+1 entries). Under the pinned
transformers 5.10.1, the doubling only appears after the first full backbone call with `output_hidden_states=True`:

```
call 1 (first after loading): len(hidden_states) = 28 -> hidden_states[-3] = encoder layer 25
call 2 and later:             len(hidden_states) = 55 -> hidden_states[-3] = encoder layer 26
```

That first call installs output-capture hooks on the SigLIP2 encoder layers. So the **first** inference after loading
a policy uses layer-25 features (8% relative difference in the VLM hidden state, up to 0.3 in actions). Every later
inference uses layer 26, the second-to-last layer that the shim intends. The rewrite matches the steady state
(layer 26), and `trt_build.py` warms the reference up before comparing. Consider warming up the stock policy once
after loading, or selecting the layer without relying on the length of `hidden_states`.

### Loading the checkpoint

- The public midtrained checkpoints carry a legacy `policy.json` (`type: phi4mm_tactile`). `common.load_policy`
  builds `RhoConfig` from its defaults plus the checkpoint's `features.json`. The strict state-dict load verifies the
  architecture.

### TensorRT

- **The prefix runs at the GPU's bf16 tensor-core peak.**
  - Phi-3's QKV projection is 0.22 ms per layer for 786 tokens (~200 TFLOPS); attention uses TensorRT's fused MHA.
  - TensorRT 10.14 picks `sm80_xmma` GEMM kernels on sm_120.
  - The remaining 20.6 ms are mostly GEMMs, so only fewer FLOPs or lower precision (FP8) can make it faster.
- **A TensorRT denoise step was slower than the compiled one** (34.3 vs 32.8 ms end-to-end), as with pi0.5.

### torch.compile for the step

- **Keep Inductor's Triton GEMMs:** restricting `max_autotune_gemm_backends` to cuBLAS (`ATEN`) made the sampler
  *slower* (36.8 ms), as did `mode="default"`.
- The step is ~210 kernels and ~1.1 ms, about 2.4× the weight-streaming bound (~0.82 GB of bf16 weights per step at
  ~1.8 TB/s ≈ 0.46 ms).

## Not done / next steps

- **Fewer, larger kernels in the denoise step** (fused adaLN + GEMM, a persistent megakernel): up to ~6 ms.
- **FP8 prefix GEMMs.** The 5090 has twice the FP8 tensor throughput; this needs a calibration and accuracy study.
- **Batch size > 1, masked cameras, RTC guidance**: the runtime asserts batch size 1 with all cameras valid.
- The FlashAttention 2 baseline was not measured (it needs an sm_120 build).
