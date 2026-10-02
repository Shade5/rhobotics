"""Export-friendly rewrite of Rho's sampler (`FlowMatchingModel.sample_actions`) at batch size 1, on its own weights.

The sampler is split in two graphs:

* `Prefix`: images + prompt embeddings -> SigLIP2 -> mm_projector -> the first `hidden_state_idx` Phi-3 layers ->
  vlm_projector -> the action expert's cross-attention K/V for *every* (denoising step, block). The cross-attention
  keys/values only depend on the prefix and on the timestep (through the adaLN modulation of `kv_norm`), whose
  schedule is fixed, so all 10 x 12 of them are computed here in a few large GEMMs.
* `DenoiseStep`: one Euler step's velocity v(x_t, t) over the 1 + chunk_size query tokens.

Math-preserving rewrites (same ops and dtypes as the reference, modulo kernel accumulation order):

* Padding patches/tokens and unused layers are never computed: the reference runs one LM layer more than it reads
  (`hidden_state_idx + 1`), and the vision tower's last layer(s) after `select_layer`.
* Everything that depends only on the timestep (sinusoidal time embedding, `embed_time_for_cond`, every adaLN
  modulation) is precomputed with the model's own functions (`step_constants`).
* Grouped-query attention folds the 4 query heads that share a K/V head into the query axis (no repeat_interleave).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

BF16 = torch.bfloat16
ROLES = ("q1", "q2", "kv", "q3")  # adaLN sites of CrossAttentionTransformerBlock
NORMS = {"q1": "q_norm_1", "q2": "q_norm_2", "kv": "kv_norm", "q3": "q_norm_3"}


def layer_norm(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    """nn.LayerNorm on bf16 input (fp32 math, one rounding at the end), written out for the ONNX exporter."""
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + ln.eps)
    return (y * ln.weight.float() + ln.bias.float()).to(x.dtype)


def rms_norm(x: torch.Tensor, norm) -> torch.Tensor:
    """Phi3RMSNorm: fp32 normalization, then weight * x in bf16."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * y.to(x.dtype)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, bias: torch.Tensor | None = None):
    """Plain math attention, fp32 softmax (as HF eager). q [..., Lq, d], k/v [..., Lk, d]."""
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale
    if bias is not None:
        scores = scores + bias
    return torch.matmul(scores.softmax(-1).to(q.dtype), v)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Prefix(nn.Module):
    """images [n, 3, 256, 256] float in [0, 1], text embeddings [t, 3072] -> 12 tensors kv_b [steps, 2, 4, max_len, 128].

    The prompt template puts all `<image>` sentinels right after the first `head_len` text tokens
    (`<|im_start|>user<|im_sep|>`), so the spliced sequence is text[:head_len] + images + text[head_len:].
    Rows >= the real sequence length are padding (masked by the denoise step's key bias).
    """

    def __init__(self, flow, consts: dict, *, head_len: int, max_len: int, vision_layers: int, lm_layers: int):
        super().__init__()
        backbone = flow.vlm_backbone
        tower = backbone.model.vision_tower
        vision = getattr(tower.vision_tower, "vision_model", tower.vision_tower)
        self.patch_embedding = vision.embeddings.patch_embedding
        self.vision_layers = vision.encoder.layers[:vision_layers]
        self.vision_heads = vision.config.num_attention_heads
        self.patch = vision.config.patch_size
        self.mm_projector = backbone.model.mm_projector
        self.lm_layers = backbone.model.layers[:lm_layers]
        self.lm_heads = backbone.config.num_attention_heads
        self.vlm_projector = flow.vlm_projector
        self.blocks = flow.action_expert.transformer_blocks
        self.head_len, self.max_len = head_len, max_len
        self.kv_heads = self.blocks[0].cross_attention.num_kv_heads
        # Constants: SigLIP2's resized position embeddings for a 16x16 grid, RoPE tables, the expert's positional
        # encoding, and the (1 + scale, shift) of every block's kv_norm at every step.
        self.register_buffer("vision_pos", consts["vision_pos"])
        self.register_buffer("cos", consts["cos"][:max_len])
        self.register_buffer("sin", consts["sin"][:max_len])
        self.register_buffer("expert_pos", consts["expert_pos"][:max_len])
        self.register_buffer("kv_scale", consts["mods"][:, :, ROLES.index("kv"), 0])  # [steps, blocks, 2048]
        self.register_buffer("kv_shift", consts["mods"][:, :, ROLES.index("kv"), 1])

    def vision(self, images: torch.Tensor) -> torch.Tensor:
        n, c, h, w = images.shape
        p = self.patch
        x = (images - 0.5) / 0.5  # Phi5ImageProcessor (mean = std = 0.5), fp32
        x = x.view(n, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(n, (h // p) * (w // p), p * p * c)
        x = self.patch_embedding(x.to(BF16)) + self.vision_pos
        for layer in self.vision_layers:
            a = layer.self_attn
            y = layer_norm(x, layer.layer_norm1)
            q, k, v = (proj(y).view(n, -1, self.vision_heads, a.head_dim).transpose(1, 2) for proj in (a.q_proj, a.k_proj, a.v_proj))
            y = attention(q, k, v, a.scale).transpose(1, 2).reshape(n, -1, a.embed_dim)
            x = x + a.out_proj(y)
            y = layer_norm(x, layer.layer_norm2)
            x = x + layer.mlp.fc2(layer.mlp.activation_fn(layer.mlp.fc1(y)))
        return self.mm_projector(x).flatten(0, 1)  # [n * 256, 3072]

    def language(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[0]
        cos, sin = self.cos[:length][:, None], self.sin[:length][:, None]  # [L, 1, 96]
        positions = torch.arange(length, device=x.device)
        causal = torch.where(positions[None, :] > positions[:, None], float("-inf"), 0.0)
        for layer in self.lm_layers:
            a = layer.self_attn
            y = rms_norm(x, layer.input_layernorm)
            q, k, v = a.qkv_proj(y).view(length, 3 * self.lm_heads, a.head_dim).split(self.lm_heads, dim=1)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            y = attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), a.scaling, causal)
            x = x + a.o_proj(y.transpose(0, 1).reshape(length, -1))
            y = rms_norm(x, layer.post_attention_layernorm)
            gate, up = layer.mlp.gate_up_proj(y).chunk(2, dim=-1)
            x = x + layer.mlp.down_proj(up * layer.mlp.activation_fn(gate))
        return x

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat([text[: self.head_len], self.vision(images), text[self.head_len :]])
        x = self.vlm_projector(self.language(x))  # [L, 2048]
        x = x + self.expert_pos[: x.shape[0]].to(x.dtype)
        x = F.pad(x, (0, 0, 0, self.max_len - x.shape[0]))
        steps = self.kv_scale.shape[0]
        out = []
        for b, block in enumerate(self.blocks):
            y = layer_norm(x, block.kv_norm.layer_norm)  # [max_len, 2048]
            y = y[None] * self.kv_scale[:, b, None] + self.kv_shift[:, b, None]  # [steps, max_len, 2048]
            attn = block.cross_attention
            k, v = attn.k_proj(y), attn.v_proj(y)  # [steps, max_len, 512]
            kv = torch.stack([k, v], dim=1).view(steps, 2, self.max_len, self.kv_heads, attn.head_dim)
            out.append(kv.transpose(2, 3).contiguous())  # [steps, 2, 4, max_len, 128]
        return out


class DenoiseStep(nn.Module):
    """x_t [1, chunk, 32] bf16 -> v_t [1, chunk, 32] bf16 for one step.

    state_emb: state_projector(state) [1, 1, 2048]; mods: [blocks, 4 roles, 3 (1 + scale, shift, gate), 2048] for this
    step; time_sin: [2048]; final_mod: [2 (1 + scale, shift), 2048]; kv: 12 x [2, 4, max_len, 128]; key_bias: [max_len].
    """

    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.expert = flow.action_expert
        self.chunk = flow.config.chunk_size
        attn = self.expert.transformer_blocks[0].query_self_attention
        self.heads, self.kv_heads, self.head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        self.groups = self.heads // self.kv_heads
        self.register_buffer("pos", flow.action_expert.positional_encoding[: self.chunk + 1].to(BF16).clone())
        # Self-attention Q/K/V projections share their input: one GEMM per block instead of three.
        for b, block in enumerate(self.expert.transformer_blocks):
            sa = block.query_self_attention
            self.register_buffer(f"qkv_{b}", torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]))

    def _heads(self, q):  # [n, 2048] -> [kv_heads, groups * n, d]
        n = q.shape[0]
        return q.view(n, self.kv_heads, self.groups, self.head_dim).permute(1, 2, 0, 3).reshape(
            self.kv_heads, self.groups * n, self.head_dim
        )

    def _merge(self, attn, y, n):  # [kv_heads, groups * n, d] -> [n, 2048]
        y = y.view(self.kv_heads, self.groups, n, self.head_dim).permute(2, 0, 1, 3).reshape(n, -1)
        return attn.out_proj(y)

    def _kv_heads(self, k):  # [n, 512] -> [kv_heads, n, d]
        return k.view(k.shape[0], self.kv_heads, self.head_dim).transpose(0, 1)

    def forward(self, x_t, state_emb, mods, time_sin, final_mod, key_bias, *kv):
        f = self.flow
        a = f.action_in_proj(x_t[0])  # [chunk, 2048]
        h = f.action_time_mlp_in(torch.cat([a, time_sin.expand_as(a)], dim=-1))
        h = f.action_time_mlp_out(F.silu(h))
        x = torch.cat([state_emb[0], h]) + self.pos  # [1 + chunk, 2048]
        n = x.shape[0]
        scale = 1.0 / math.sqrt(self.head_dim)
        for b, block in enumerate(self.expert.transformer_blocks):
            m = mods[b]
            y = layer_norm(x, block.q_norm_1.layer_norm) * m[0, 0] + m[0, 1]
            sa = block.query_self_attention
            q, k, v = F.linear(y, getattr(self, f"qkv_{b}")).split([sa.embed_dim, sa.kv_dim, sa.kv_dim], dim=-1)
            y = attention(self._heads(q), self._kv_heads(k), self._kv_heads(v), scale)
            x = x + m[0, 2] * self._merge(sa, y, n)
            y = layer_norm(x, block.q_norm_2.layer_norm) * m[1, 0] + m[1, 1]
            ca = block.cross_attention
            y = attention(self._heads(ca.q_proj(y)), kv[b][0], kv[b][1], scale, key_bias)
            x = x + m[1, 2] * self._merge(ca, y, n)
            y = layer_norm(x, block.q_norm_3.layer_norm) * m[3, 0] + m[3, 1]
            x = x + m[3, 2] * block.feedforward(y)
        y = layer_norm(x, self.expert.final_norm.layer_norm) * final_mod[0] + final_mod[1]
        return f.action_head(y[-self.chunk :])[None]


def prompt_token_ids(flow, prompt: str, num_images: int, device: str = "cuda") -> dict:
    """Tokenizes a formatted prompt (`RhoPolicy.prepare_prompt`) like the reference and drops the image sentinels.

    Returns the text token ids and `head_len`, the number of ids before the (consecutive) image sentinels.
    """
    from rho.policies.rho.phi5.backbone import _tokenizer_image_token

    ids = _tokenizer_image_token(prompt, flow.vlm_processor.tokenizer, return_tensors="pt")
    sentinels = (ids < 0).nonzero().flatten()
    if len(sentinels) != num_images or not bool((sentinels.diff() == 1).all()):
        raise ValueError(f"expected {num_images} consecutive <image> sentinels in the prompt, got positions {sentinels}")
    head_len = int(sentinels[0])
    return {"ids": torch.cat([ids[:head_len], ids[head_len + num_images :]]).to(device), "head_len": head_len}


@torch.no_grad()
def step_constants(flow, vision_tower, *, num_steps: int, max_len: int, image_grid: int = 16) -> dict:
    """Everything that depends only on the weights and the fixed timestep schedule, computed with the model's own
    functions on the model's device (so the values are bit-identical to what the reference computes per step)."""
    device = flow.action_in_proj.weight.device
    dt32 = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    time32 = torch.tensor(1.0, dtype=torch.float32, device=device)
    times = []
    while time32 >= -dt32 / 2:  # same loop as FlowMatchingModel.sample_actions
        times.append(time32.clone())
        time32 = time32 + dt32
    expert = flow.action_expert
    mods, time_sin, final = [], [], []
    for t in times:
        tb = t.expand(1).to(flow.dtype)
        time_sin.append(
            flow.create_sinusoidal_pos_embedding(tb, flow.embed_dim, min_period=4e-3, max_period=4.0, device=device)[0]
        )
        cond = flow.embed_time_for_cond(tb)  # [1, 2048]
        base = {role: mod(cond) for role, mod in expert.shared_mod.items()}
        per_block = []
        for block in expert.transformer_blocks:
            roles = []
            for role in ROLES:
                scale, shift, gate = (base[role] + getattr(block, NORMS[role]).offset).chunk(3, dim=-1)
                roles.append(torch.stack([1 + scale[0], shift[0], gate[0]]))
            per_block.append(torch.stack(roles))
        mods.append(torch.stack(per_block))
        scale, shift, _ = expert.final_norm.modulation(cond).chunk(3, dim=-1)
        final.append(torch.stack([1 + scale[0], shift[0]]))

    vision = getattr(vision_tower.vision_tower, "vision_model", vision_tower.vision_tower)
    emb = vision.embeddings
    grid = emb.position_embedding.weight.reshape(emb.position_embedding_size, emb.position_embedding_size, -1)
    shapes = torch.tensor([[image_grid, image_grid]], device=device)
    vision_pos = emb.resize_positional_embeddings(grid, shapes, max_length=image_grid * image_grid)[0]

    rotary = flow.vlm_backbone.model.rotary_emb
    dummy = torch.zeros(1, dtype=flow.dtype, device=device)
    cos, sin = rotary(dummy, torch.arange(max_len, device=device)[None])
    return {
        "times": torch.stack(times),
        "mods": torch.stack(mods),  # [steps, blocks, 4, 3, 2048] bf16
        "time_sin": torch.stack(time_sin),  # [steps, 2048]
        "final_mod": torch.stack(final),  # [steps, 2, 2048]
        "vision_pos": vision_pos,  # [256, 1152]
        "cos": cos[0],
        "sin": sin[0],  # [max_len, 96]
        "expert_pos": expert.positional_encoding[:max_len].clone(),  # fp32
    }


@torch.no_grad()
def sample_actions_eager(prefix: Prefix, step: DenoiseStep, consts: dict, images, text, state, noise, length: int):
    """Reference-equivalent sampler built from the rewritten modules (for validating them before export)."""
    flow = step.flow
    kv = prefix(images, text)
    key_bias = torch.zeros(prefix.max_len, device=images.device)
    key_bias[length:] = float("-inf")
    state_emb = flow.state_projector(state.to(BF16))
    x_t = noise.to(BF16)
    dt32 = torch.tensor(-1.0 / consts["times"].shape[0], dtype=torch.float32, device=images.device)
    for i in range(consts["times"].shape[0]):
        v_t = step(x_t, state_emb, consts["mods"][i], consts["time_sin"][i], consts["final_mod"][i], key_bias,
                   *(kv_b[i] for kv_b in kv))  # fmt: skip
        x_t = (x_t.float() + dt32 * v_t.float()).to(BF16)
    return x_t
