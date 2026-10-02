"""Runtime for the optimized Rho sampler: the prefix + all denoising steps are replayed as one CUDA graph.

The prefix (SigLIP2 + Phi-3 + every step's cross-attention K/V) runs as a TensorRT engine; the denoising step runs as
a torch.compile'd module (`CompiledStep`) over fixed-size K/V buffers, so it is compiled once for all prompt lengths.
"""

import pathlib

import tensorrt as trt
import torch
import trt_model

LOGGER = trt.Logger(trt.Logger.WARNING)
_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT32: torch.int32,
}


class Engine:
    def __init__(self, path: pathlib.Path):
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def output_shapes(self, **inputs: torch.Tensor) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
        return {n: (tuple(self.context.get_tensor_shape(n)), _DTYPES[self.engine.get_tensor_dtype(n)])
                for n in self.outputs}  # fmt: skip

    def run(self, stream: torch.cuda.Stream, **tensors: torch.Tensor):
        for name, tensor in tensors.items():
            if name not in self.outputs:
                self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        assert self.context.execute_async_v3(stream.cuda_stream)


class CompiledStep:
    def __init__(self, flow, mode: str = "max-autotune-no-cudagraphs", gemm_backends: str | None = None):
        self.module = trt_model.DenoiseStep(flow).eval()
        if gemm_backends is not None:
            import torch._inductor.config as inductor_config

            inductor_config.max_autotune_gemm_backends = gemm_backends
        # Inductor's own CUDA graphs would conflict with capturing the whole sampler.
        self.fn = torch.compile(self.module, mode=mode, fullgraph=True, dynamic=False)

    @torch.no_grad()
    def __call__(self, *args):
        return self.fn(*args)


class TrtStep:
    """The denoise step as a TensorRT engine (same inputs as `CompiledStep`)."""

    def __init__(self, path: pathlib.Path, chunk_size: int, action_dim: int):
        self.engine = Engine(path)
        self.names = [self.engine.engine.get_tensor_name(i) for i in range(self.engine.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if n not in self.engine.outputs]
        self.out = torch.empty(1, chunk_size, action_dim, dtype=torch.bfloat16, device="cuda")

    def __call__(self, *args):
        tensors = {name: t.contiguous() for name, t in zip(self.inputs, args, strict=True)}
        self.engine.run(torch.cuda.current_stream(), v_t=self.out, **tensors)
        return self.out


class _Graph:
    """Static buffers + CUDA graph for one prompt length."""

    def __init__(self, s: "Sampler", num_tokens: int, uint8_images: bool):  # noqa: FBT001
        self.s = s
        n, size = s.num_images, s.image_size
        self.images = torch.zeros(n, 3, size, size, device="cuda")
        self.images_u8 = torch.zeros(n, size, size, 3, dtype=torch.uint8, device="cuda") if uint8_images else None
        self.ids = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
        self.text = torch.zeros(num_tokens, s.embed_tokens.embedding_dim, dtype=torch.bfloat16, device="cuda")
        self.state = torch.zeros(1, 1, s.state_dim, device="cuda")
        self.noise = torch.zeros(1, s.chunk_size, s.action_dim, device="cuda")
        self.x_t = torch.zeros(1, s.chunk_size, s.action_dim, dtype=torch.bfloat16, device="cuda")
        shapes = s.prefix.output_shapes(images=self.images, text=self.text)
        self.kv = {name: torch.empty(shape, dtype=dtype, device="cuda") for name, (shape, dtype) in shapes.items()}
        self.kv_list = [self.kv[f"kv_{b}"] for b in range(len(self.kv))]
        length = n * 256 + num_tokens
        self.key_bias = torch.zeros(s.max_len, device="cuda")
        self.key_bias[length:] = float("-inf")

        with torch.cuda.stream(s.stream):
            self._launch()  # compiles the step (first time) and runs TensorRT once before capture
            s.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=s.stream):
                self._launch()

    @torch.no_grad()
    def _launch(self):
        s = self.s
        if self.images_u8 is not None:
            self.images.copy_(self.images_u8.permute(0, 3, 1, 2)).div_(255.0)
        self.text.copy_(s.embed_tokens(self.ids))
        s.prefix.run(s.stream, images=self.images, text=self.text, **self.kv)
        state_emb = s.state_projector(self.state.to(torch.bfloat16))
        x_t = self.noise.to(torch.bfloat16)
        c = s.consts
        for i in range(s.num_steps):
            v_t = s.step(x_t, state_emb, c["mods"][i], c["time_sin"][i], c["final_mod"][i], self.key_bias,
                         *(kv[i] for kv in self.kv_list))  # fmt: skip
            x_t = (x_t.float() + s.dt32 * v_t.float()).to(torch.bfloat16)
        self.x_t.copy_(x_t)


class Sampler:
    """Drop-in for `FlowMatchingModel.sample_actions` at batch size 1 (prompt given as token ids, see `token_ids`)."""

    def __init__(self, flow, engine_dir: pathlib.Path, *, num_images: int = 3, image_size: int = 256, step=None):
        consts = torch.load(engine_dir / "consts.pt")
        self.consts = {k: v.cuda() for k, v in consts.items()}
        self.prefix = Engine(engine_dir / "prefix.engine")
        self.max_len = self.consts["cos"].shape[0]
        self.num_steps = self.consts["times"].shape[0]
        self.dt32 = torch.tensor(-1.0 / self.num_steps, dtype=torch.float32, device="cuda")
        self.flow = flow
        self.embed_tokens = flow.vlm_backbone.model.embed_tokens
        self.state_projector = flow.state_projector
        self.step = step or CompiledStep(flow)
        self.num_images, self.image_size = num_images, image_size
        self.chunk_size, self.action_dim = flow.config.chunk_size, flow.config.max_action_dim
        self.state_dim = flow.config.max_state_dim
        self.stream = torch.cuda.Stream()
        self._graphs: dict[tuple[int, bool], _Graph] = {}
        self._token_cache: dict[str, dict] = {}

    def token_ids(self, prompt: str) -> torch.Tensor:
        """Formatted prompt (`RhoPolicy.prepare_prompt`) -> text token ids (cached; prompts rarely change)."""
        if prompt not in self._token_cache:
            out = trt_model.prompt_token_ids(self.flow, prompt, self.num_images)
            assert out["head_len"] == self.prefix_head_len, "prompt template differs from the one the engine was built for"
            self._token_cache[prompt] = out["ids"]
        return self._token_cache[prompt]

    @property
    def prefix_head_len(self) -> int:
        return 3  # `<|im_start|>user<|im_sep|>` precedes the image sentinels in Phi5Backbone.prepare_prompt

    def run(self, images: torch.Tensor, ids: torch.Tensor, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """images: float [n, 3, 256, 256] in [0, 1] or uint8 [n, 256, 256, 3]; ids: [t]; state: [1, 1, 32];
        noise: [1, chunk, 32]. Returns bf16 actions [1, chunk, 32] (model space, like the reference)."""
        uint8_images = images.dtype == torch.uint8
        key = (ids.shape[0], uint8_images)
        if key not in self._graphs:
            self._graphs[key] = _Graph(self, *key)
        g = self._graphs[key]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            (g.images_u8 if uint8_images else g.images).copy_(images, non_blocking=True)
            g.ids.copy_(ids, non_blocking=True)
            g.state.copy_(state, non_blocking=True)
            g.noise.copy_(noise, non_blocking=True)
            g.graph.replay()
            out = g.x_t.clone()
        torch.cuda.current_stream().wait_stream(self.stream)
        return out

    @torch.no_grad()
    def sample_actions(self, image, prompt, state, noise=None, image_mask=None) -> torch.Tensor:
        """Same signature as `FlowMatchingModel.sample_actions` (batch size 1, all cameras valid)."""
        assert state.shape[0] == 1, "batch size 1 only"
        if image_mask is not None:
            assert bool(torch.as_tensor(image_mask).all()), "masked cameras are not supported"
        if noise is None:
            noise = torch.randn(1, self.chunk_size, self.action_dim, device="cuda")
        return self.run(image[0], self.token_ids(prompt[0]), state, noise.float())
