"""
GPT model definition and MuonAdamW optimizer, shared by train.py, train_phase2.py,
train_singlegpu.py, and generate.py.

On an Ampere-or-newer GPU (sm_80+) importing this module downloads/loads the
FlashAttention-3 kernel at import time (see the kernels-community note below) and is
therefore not side-effect-free, but every former caller already paid this same cost
via its own copy of this code. On older GPUs (e.g. TITAN X Pascal, sm_61) the kernel
is skipped and attention falls back to torch.scaled_dot_product_attention; see the
capability-detection block below.
"""

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

# ---------------------------------------------------------------------------
# GPU capability detection and precision / kernel selection
# ---------------------------------------------------------------------------
# FlashAttention-3, Triton (torch.compile's CUDA backend) and hardware bf16 all
# require sm_80+ (Ampere). On older cards (e.g. TITAN X Pascal, sm_61) we fall
# back to torch's built-in scaled_dot_product_attention, disable torch.compile,
# and drop to fp16 or fp32 autocast instead of bf16. On sm_80+ nothing below changes.
_CUDA_CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
HAS_FA3 = _CUDA_CAP >= (8, 0)          # FlashAttention-3 kernel is usable
CAN_COMPILE = _CUDA_CAP >= (7, 0)      # Triton requires sm_70+
LEGACY_GPU = torch.cuda.is_available() and not HAS_FA3
# Volta/Turing (sm_70/75) have real 2:1 fp16 throughput; consumer Pascal (sm_6x,
# e.g. TITAN X Pascal / Xp) runs fp16 math at 1/64 the fp32 rate, so fp16 autocast
# there is catastrophically slow -- use plain fp32 instead (the card's fast path).
_FAST_FP16 = _CUDA_CAP >= (7, 0)

# Autocast/compute dtype: bf16 on Ampere+, fp16 on Volta/Turing, fp32 on Pascal.
if not LEGACY_GPU:
    AMP_DTYPE = torch.bfloat16
elif _FAST_FP16:
    AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = torch.float32
# Muon's polar-express orthogonalization runs in bf16 on Ampere+; fp32 on older
# cards, where bf16 matmul is unsupported by cuBLAS.
_MUON_LOWP = torch.bfloat16 if not LEGACY_GPU else torch.float32

if HAS_FA3:
    from kernels import get_kernel
    # Use kernels-community's build unconditionally (varunneal/flash-attention-3 is Hopper-only
    # but requires trust_remote_code=True since its publisher can't be verified; kernels-community
    # is a verified publisher and works fine on Hopper too).
    repo = "kernels-community/flash-attn3"
    fa3 = get_kernel(repo, version=1).flash_attn_interface
else:
    fa3 = None


def maybe_compile(fn=None, **compile_kwargs):
    """torch.compile(fn, **kw) on sm_70+, else return fn unchanged (Pascal has no Triton)."""
    if fn is None:
        return lambda f: maybe_compile(f, **compile_kwargs)
    return torch.compile(fn, **compile_kwargs) if CAN_COMPILE else fn


# ---------------------------------------------------------------------------
# FlashAttention-free attention fallback (Pascal / no-FA3 path)
# ---------------------------------------------------------------------------
# `_sdpa_attention` is used when the FA3 kernel is unavailable. On sm_61 the
# default SDPA dispatch lands on the MATH backend (mem_efficient runs but its
# backward is 2-3x slower on Pascal; flash/cudnn have no sm_61 kernel). PHASE2_ATTN
# selects the strategy:
#   auto           - let torch.sdpa pick (== math on Pascal)
#   math|mem_eff|cudnn|flash - force that single SDPA backend
#   chunked        - manual attention, queries tiled + K/V sliced to the window,
#                    each tile recomputed in backward. ~12% faster than math on
#                    the phase-2 135M shapes; the default on legacy GPUs.
# PHASE2_ATTN_MASK (bool|bias) picks the mask representation for the SDPA paths;
# PHASE2_ATTN_QCHUNK sets the query tile for 'chunked' (default 512).
_ATTN_MODE = os.environ.get("PHASE2_ATTN", "chunked" if LEGACY_GPU else "auto").lower()
_ATTN_MASK_KIND = os.environ.get("PHASE2_ATTN_MASK", "bool").lower()
_ATTN_QCHUNK = int(os.environ.get("PHASE2_ATTN_QCHUNK", "512"))
_ATTN_BACKENDS = {
    "math": [SDPBackend.MATH],
    "mem_eff": [SDPBackend.EFFICIENT_ATTENTION],
    "mem_efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "cudnn": [SDPBackend.CUDNN_ATTENTION],
    "flash": [SDPBackend.FLASH_ATTENTION],
}.get(_ATTN_MODE)

_attn_mask_cache = {}


def _window_mask(T, left, device, kind, dtype):
    """Cached (T, T) causal+window mask. kind='bool' -> bool keep-mask;
    kind='bias' -> additive float mask (0 keep / -inf drop)."""
    key = (T, left, device, kind, dtype)
    m = _attn_mask_cache.get(key)
    if m is None:
        idx = torch.arange(T, device=device)
        keep = (idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - left)
        if kind == "bias":
            m = torch.zeros((T, T), device=device, dtype=dtype)
            m.masked_fill_(~keep, float("-inf"))
        else:
            m = keep
        _attn_mask_cache[key] = m
    return m


def _chunked_attention(q, k, v, left, q_chunk):
    """Manual causal (optionally windowed) attention. q/k/v: (B, H, T, D).
    Queries are processed in tiles; for a windowed layer only the K/V slice the
    tile can see is read. Each tile is recomputed in backward (checkpoint) so the
    peak score tensor is one tile, not the whole (T, T)."""
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    windowed = left is not None and 0 <= left < T
    out = q.new_empty((B, H, T, D))
    qpos_all = torch.arange(T, device=q.device)

    def tile(qc, kc, vc, qpos, kpos, has_left):
        scores = torch.matmul(qc, kc.transpose(-2, -1)) * scale
        allowed = kpos[None, :] <= qpos[:, None]
        if has_left:
            allowed = allowed & (kpos[None, :] >= qpos[:, None] - left)
        scores = scores.masked_fill(~allowed, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), vc)

    for start in range(0, T, q_chunk):
        end = min(start + q_chunk, T)
        k_start = max(0, start - left) if windowed else 0
        qc = q[:, :, start:end]
        kc = k[:, :, k_start:end]
        vc = v[:, :, k_start:end]
        qpos = qpos_all[start:end]
        kpos = qpos_all[k_start:end]
        if q.requires_grad:
            yc = torch.utils.checkpoint.checkpoint(
                tile, qc, kc, vc, qpos, kpos, windowed, use_reentrant=False
            )
        else:
            yc = tile(qc, kc, vc, qpos, kpos, windowed)
        out[:, :, start:end] = yc
    return out


def _sdpa_attention(q, k, v, window_size):
    """FlashAttention-free attention. q/k/v come in as (B, T, H, D)."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # -> (B, H, T, D)
    n_head, n_kv_head, T = q.size(1), k.size(1), q.size(2)
    if n_head != n_kv_head:  # grouped-query attention: expand kv heads
        rep = n_head // n_kv_head
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    left = window_size[0]
    full_causal = left is None or left < 0 or left >= T

    if _ATTN_MODE == "chunked":
        y = _chunked_attention(q, k, v, None if full_causal else left, _ATTN_QCHUNK)
    else:
        ctx = sdpa_kernel(_ATTN_BACKENDS) if _ATTN_BACKENDS is not None else nullcontext()
        with ctx:
            if full_causal:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif _ATTN_MASK_KIND == "bias":
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=_window_mask(T, left, q.device, "bias", q.dtype))
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=_window_mask(T, left, q.device, "bool", q.dtype))
    return y.transpose(1, 2).contiguous()  # -> (B, T, H, D)


# ---------------------------------------------------------------------------
# Memory-lean output head: softcapped linear + cross-entropy, computed in tiles
# ---------------------------------------------------------------------------
_FUSED_CE = os.environ.get("PHASE2_FUSED_CE", "1" if LEGACY_GPU else "0") == "1"
_CE_CHUNK = int(os.environ.get("PHASE2_CE_CHUNK", "2048"))
# Recompute each transformer block in backward (activation checkpointing) -- trades
# ~25-30% extra compute for a large activation-memory cut, letting the batch grow.
_CKPT_BLOCKS = os.environ.get("PHASE2_CKPT_BLOCKS", "0") == "1"


def _fused_softcap_ce(x, weight, targets, softcap, chunk):
    """cross_entropy(softcap * tanh(x @ weight.T / softcap), targets, ignore_index=-1,
    reduction='mean'), but the (rows, vocab) logits are built one row-tile at a
    time and (when training) recomputed in backward, so the full logit tensor is
    never materialized. Mathematically equal to the dense path."""
    n = x.size(0)
    loss_sum = x.new_zeros(())
    count = x.new_zeros(())
    use_ckpt = x.requires_grad

    def tile_loss(xi, ti):
        li = F.linear(xi, weight).float()
        li = softcap * torch.tanh(li / softcap)
        return F.cross_entropy(li, ti, ignore_index=-1, reduction="sum")

    for i in range(0, n, chunk):
        xi, ti = x[i:i + chunk], targets[i:i + chunk]
        if use_ckpt:
            ls = torch.utils.checkpoint.checkpoint(tile_loss, xi, ti, use_reentrant=False)
        else:
            ls = tile_loss(xi, ti)
        loss_sum = loss_sum + ls
        count = count + (ti != -1).sum()
    return loss_sum / count.clamp(min=1)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Eager-mode autocast can promote norm(q)/norm(k) to fp32 while leaving v in the
        # low precision dtype (this doesn't happen under torch.compile, which is how the
        # training scripts run). Force a consistent dtype so the attention kernel doesn't
        # reject mismatched q/k/v dtypes; a no-op when q/k/v already match.
        q, k, v = q.to(AMP_DTYPE), k.to(AMP_DTYPE), v.to(AMP_DTYPE)

        if HAS_FA3:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            y = _sdpa_attention(q, k, v, window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16 (Ampere+). On legacy GPUs keep them fp32 so the
        # AdamW moment buffers stay fp32 and don't underflow in fp16.
        if not LEGACY_GPU:
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(AMP_DTYPE), sin.to(AMP_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        matrix_optimizer='muon', log=print):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        log(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if matrix_optimizer == 'muon':
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        elif matrix_optimizer == 'sgd':
            param_groups.append(dict(
                kind='sgd', params=matrix_params,
                lr=matrix_lr, momentum=0.9, weight_decay=weight_decay,
                normalize_grad=True, nesterov=True,
            ))
        else:
            param_groups.append(dict(
                kind='adamw', params=matrix_params,
                lr=matrix_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        ckpt_blocks = _CKPT_BLOCKS and x.requires_grad
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if ckpt_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, ve, cos_sin, self.window_sizes[i], use_reentrant=False)
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        if targets is not None and reduction == "mean" and _FUSED_CE:
            # Memory-lean equivalent of the block below: the (B*T, vocab) softcapped
            # logits are the peak allocation of the whole model on Pascal (~3x
            # B*T*vocab*4 bytes of fp32 temporaries) and what caps the batch size.
            # Tile it and recompute each tile in backward instead.
            return _fused_softcap_ce(
                x.view(-1, x.size(-1)), self.lm_head.weight,
                targets.view(-1), softcap, _CE_CHUNK,
            )

        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@maybe_compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@maybe_compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization (bf16 on Ampere+, fp32 on cards without bf16 matmul)
    X = g.to(_MUON_LOWP)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def _step_sgd(self, group):
        lr = group['lr']
        momentum = group['momentum']
        wd = group['weight_decay']
        normalize = group.get('normalize_grad', False)
        nesterov = group.get('nesterov', False)
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if normalize:
                g_norm = grad.norm()
                if g_norm > 0:
                    grad = grad / (g_norm + 1e-8)
            state = self.state[p]
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)
            buf = state['momentum_buffer']
            buf.mul_(momentum).add_(grad)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            # Scale LR by sqrt(rows/cols) for non-square matrices (Muon-style)
            if p.dim() == 2:
                effective_lr = lr * max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            else:
                effective_lr = lr
            if nesterov:
                update = grad + momentum * buf
            else:
                update = buf
            p.add_(update, alpha=-effective_lr)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            elif group['kind'] == 'sgd':
                self._step_sgd(group)

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    # (n_layer, n_head, n_embd, device_batch_size)
    "50M":   ( 8,  4,  512, 12),
    "135M":  (12,  6,  768, 32),
    "1B":    (18, 16, 2048,  3),
    "3B":    (34, 20, 2560,  2),  # ~3.07B params; head_dim=128, matching the other presets
}
