"""
Phase-2 (instruction-tuning) variant of train.py: multi-GPU (DistributedDataParallel)
training on pretokenized, loss-masked chat data instead of raw pretraining text. Launch
with torchrun, e.g.:

    torchrun --standalone --nproc_per_node=3 train_phase2.py --model 3B --resume-from model_weights-3B.pt

Data is sharded across ranks (each rank sees a disjoint ~1/world_size slice of documents), so
using more GPUs gives a genuinely larger effective batch / faster wall-clock progress toward
the token budget, not redundant compute on identical data.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import gc
import json
import math
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from model import GPTConfig, GPT, MODEL_PRESETS, AMP_DTYPE, CAN_COMPILE
from prepare import (
    MAX_SEQ_LEN, Tokenizer, evaluate_bpb, select_training_shards, _document_batches,
    DEFAULT_TRAIN_DATA_DIR, DEFAULT_VAL_FILE,
)

TOKEN_BUDGET = 2**28     # total training token budget (default)

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
is_master = rank == 0


def mprint(*a, **kw):
    if is_master:
        print(*a, **kw)


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Gpt phase-two chat SFT trainer (DDP)")
parser.add_argument("--wandb", type=int, default=1, help="Enable Weights & Biases logging (0 = disabled, non-zero = enabled)")
parser.add_argument("--wandb-project", default="gpt_nemotron",
                    help="Weights & Biases project name")
parser.add_argument("--token-budget", type=int, default=None, help="Total training token budget across all GPUs (default: TOKEN_BUDGET)")
parser.add_argument("--logging-steps", type=int, default=5, help="How often (in steps) to log metrics to wandb")
parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adam", "sgd"],
                    help="Optimizer for matrix parameters: 'muon' (default), 'adam' (AdamW), or 'sgd'")
parser.add_argument("--model", choices=list(MODEL_PRESETS), default="135M",
                    help="Model size preset (default: 135M)")
parser.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Gradient accumulation steps per optimizer step (per rank)")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Per-device (per-GPU) batch size (default: preset value for the selected model)")
parser.add_argument("--matrix-lr", type=float, default=None,
                    help="Learning rate for matrix parameters (default: 0.18)")
parser.add_argument("--embedding-lr", type=float, default=None,
                    help="Learning rate for token/value embeddings (default: 0.6)")
parser.add_argument("--unembedding-lr", type=float, default=None,
                    help="Learning rate for the output head (default: 0.004)")
parser.add_argument("--scalar-lr", type=float, default=None,
                    help="Learning rate for residual scalar parameters (default: 0.5)")
parser.add_argument("--warmup-ratio", type=float, default=0.0)
parser.add_argument("--warmdown-ratio", type=float, default=0.75)
parser.add_argument("--n-layer", type=int, default=None, help="Override preset layer count")
parser.add_argument("--n-head", type=int, default=None, help="Override preset attention heads")
parser.add_argument("--n-embd", type=int, default=None, help="Override preset model width")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for parameter initialization")
parser.add_argument("--weight-decay", type=float, default=None,
                    help="Weight decay (default: 0.035)")
parser.add_argument("--grad-clip", type=float, default=0.0,
                    help="Gradient clipping max norm (0 = disabled)")
parser.add_argument("--save-weights", type=int, default=1,
                    help="Save model weights after training (default: 1=save, 0=skip)")
parser.add_argument("--resume-from", type=Path, default=None,
                    help="Load model weights from a .pt checkpoint before training")
parser.add_argument("--checkpoint-tag", default=None,
                    help="Optional tag included in saved checkpoint filenames")
parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_TRAIN_DATA_DIR),
                    help="Directory containing per-stratum Parquet training shards")
parser.add_argument("--val-file", type=Path, default=Path(DEFAULT_VAL_FILE),
                    help="Pinned validation Parquet file")
parser.add_argument("--train-fraction", type=float, default=1.0,
                    help="Fraction selected independently from each stratum (default: 1.0)")
parser.add_argument("--epochs", type=float, default=1.0,
                    help="Nominal passes over the selected shards; may be fractional, "
                         "e.g. 0.1 (default: 1.0)")
parser.add_argument("--eval-assistant-batches", type=int, default=0,
                    help="Evaluate this many deterministic packed training batches and exit")
parser.add_argument("--eval-results-file", type=Path, default=Path("phase2_accuracy_results.jsonl"),
                    help="Append assistant-only accuracy records here")
parser.add_argument("--eval-data-file", type=Path, default=None,
                    help="Optional v2 Parquet to use instead of training data in eval-only mode")
parser.add_argument("--assistant-val-file", type=Path, default=None,
                    help="Optional held-out v2 Parquet used for assistant-only validation")
parser.add_argument("--assistant-val-batches", type=int, default=0,
                    help="Maximum packed held-out batches to evaluate (0 evaluates 32)")
args = parser.parse_args()

if not 0.0 < args.train_fraction <= 1.0:
    parser.error("--train-fraction must be in (0, 1]")
if args.epochs <= 0:
    parser.error("--epochs must be positive")
if args.eval_assistant_batches < 0:
    parser.error("--eval-assistant-batches must be non-negative")
if not 0.0 <= args.warmup_ratio < 1.0:
    parser.error("--warmup-ratio must be in [0, 1)")
if not 0.0 <= args.warmdown_ratio <= 1.0:
    parser.error("--warmdown-ratio must be in [0, 1]")
if args.warmup_ratio + args.warmdown_ratio > 1.0:
    parser.error("warmup and warmdown ratios cannot sum to more than 1")


TRAIN_PARQUET_PATHS, selected_tokens_per_epoch = select_training_shards(
    args.data_dir, args.train_fraction, args.seed, log=mprint
)
VAL_PATH = str(args.val_file)
# The Nemotron-CC BPB validation set is optional: if it's absent we simply skip
# that end-of-run eval (the SmolTalk assistant-accuracy eval is unaffected).
RUN_BPB_EVAL = args.val_file.is_file()
if not RUN_BPB_EVAL:
    mprint(f"Nemotron-CC validation file not found ({args.val_file}); skipping BPB eval.")

if args.token_budget is not None:
    TOKEN_BUDGET = args.token_budget
elif selected_tokens_per_epoch is not None:
    TOKEN_BUDGET = int(selected_tokens_per_epoch * args.epochs)
# else: legacy single-file dataset with unknown token count -- keep the module default above.

wandb_enabled = bool(args.wandb) and is_master
mprint(f"Wandb logging enabled: {wandb_enabled}")

logging_steps = max(1, int(args.logging_steps))

mprint(f"Using GLOBAL token budget: {TOKEN_BUDGET:,} tokens (across {world_size} GPU(s))")
mprint(
    f"Training data: {len(TRAIN_PARQUET_PATHS)} shards from {args.data_dir} "
    f"({args.train_fraction:.1%} per stratum, {args.epochs} epoch(s))"
)

_n_layer, _n_head, _n_embd, _batch_size = MODEL_PRESETS[args.model]
_n_layer = args.n_layer if args.n_layer is not None else _n_layer
_n_head = args.n_head if args.n_head is not None else _n_head
_n_embd = args.n_embd if args.n_embd is not None else _n_embd
if _n_embd % _n_head:
    parser.error("model width (--n-embd) must be divisible by head count (--n-head)")
mprint(f"Using model preset: {args.model}")
mprint(f"Effective architecture: layers={_n_layer}, heads={_n_head}, width={_n_embd}")
batch_size = args.batch_size if args.batch_size is not None else _batch_size
mprint(f"Per-device batch size: {batch_size} (global batch size: {batch_size * world_size})")

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

WINDOW_PATTERN = "SSSL"

EMBEDDING_LR = args.embedding_lr if args.embedding_lr is not None else 0.6
UNEMBEDDING_LR = args.unembedding_lr if args.unembedding_lr is not None else 0.004
MATRIX_LR = args.matrix_lr if args.matrix_lr is not None else 0.18
SCALAR_LR = args.scalar_lr if args.scalar_lr is not None else 0.5
WEIGHT_DECAY = args.weight_decay if args.weight_decay is not None else 0.035
ADAM_BETAS = (0.75, 0.95)
WARMUP_RATIO = args.warmup_ratio
WARMDOWN_RATIO = args.warmdown_ratio
FINAL_LR_FRAC = 0.1

# ---------------------------------------------------------------------------
# Data loading, sharded by rank
# ---------------------------------------------------------------------------

PHASE2_COLUMNS = ("input_ids", "loss_mask")


def _extract_ids_and_mask(record_batch):
    """Turn a pretokenized-chat record batch into (token_ids, loss_mask) pairs."""
    token_lists = record_batch.column(0).to_pylist()
    loss_masks = record_batch.column(1).to_pylist()
    return list(zip(token_lists, loss_masks))


def make_dataloader_sharded(tokenizer, B, T, split, rank, world_size, buffer_size=1000,
                            parquet_paths=None):
    """Pack pretokenized chats while preserving assistant-only target masks.
    Reuses prepare.py's stratum-interleaving/sharding logic (_document_batches),
    only supplying the pretokenized-schema column/extract pair and, for eval-only
    calls, an override for which non-train file(s) to read (parquet_paths)."""
    row_capacity = T + 1
    batches = _document_batches(
        split, TRAIN_PARQUET_PATHS, VAL_PATH, 128, rank, world_size, log=mprint,
        columns=PHASE2_COLUMNS, extract=_extract_ids_and_mask, val_paths=parquet_paths,
    )
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        doc_buffer.extend(doc_batch)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    row_loss_mask = torch.empty((B, row_capacity), dtype=torch.bool)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, (doc_ids, _) in enumerate(doc_buffer):
                    doc_len = len(doc_ids)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                if best_idx >= 0:
                    doc_ids, doc_mask = doc_buffer.pop(best_idx)
                    doc_len = len(doc_ids)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc_ids, dtype=torch.long)
                    row_loss_mask[row_idx, pos:pos + doc_len] = torch.tensor(doc_mask, dtype=torch.bool)
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i][0]))
                    doc_ids, doc_mask = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc_ids[:remaining], dtype=torch.long)
                    row_loss_mask[row_idx, pos:pos + remaining] = torch.tensor(doc_mask[:remaining], dtype=torch.bool)
                    if len(doc_ids) > remaining:
                        # Preserve the continuation so one nominal epoch does
                        # not silently discard document tails while packing.
                        doc_buffer.append((doc_ids[remaining:], doc_mask[remaining:]))
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        cpu_targets.masked_fill_(~row_loss_mask[:, 1:], -1)
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


def evaluate_assistant_accuracy(model, loader, num_batches):
    """Return exact top-1 accuracy/loss over assistant-masked targets."""
    correct = 0
    assistant_tokens = 0
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            eval_x, eval_y, _ = next(loader)
            with autocast_ctx:
                eval_logits = model(eval_x)
            supervised = eval_y.ne(-1)
            correct += int((eval_logits.argmax(dim=-1)[supervised] == eval_y[supervised]).sum())
            loss_sum += float(F.cross_entropy(
                eval_logits.view(-1, eval_logits.size(-1)), eval_y.view(-1),
                ignore_index=-1, reduction="sum"
            ))
            assistant_tokens += int(supervised.sum())
    return correct, assistant_tokens, loss_sum

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
start_memory = torch.cuda.memory_allocated(device)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")
autocast_ctx = torch.amp.autocast(
    device_type="cuda", dtype=AMP_DTYPE, enabled=(AMP_DTYPE != torch.float32)
)
# fp16 (legacy GPUs) needs gradient scaling to avoid underflow; bf16/fp32 do not.
scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
mprint(f"Autocast dtype: {AMP_DTYPE}, grad scaler: {scaler.is_enabled()}, torch.compile: {CAN_COMPILE}")
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
mprint(f"Vocab size: {vocab_size:,}")


def build_model_config():
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=_n_layer, n_head=_n_head, n_kv_head=_n_head, n_embd=_n_embd,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config()
mprint(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

if args.resume_from is not None:
    if not args.resume_from.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
    checkpoint = torch.load(
        args.resume_from, map_location="cpu", weights_only=True, mmap=True
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    del state_dict, checkpoint
    mprint(f"Loaded model weights from {args.resume_from}")

if args.eval_assistant_batches:
    # Use this trainer's loader directly so tokenization, best-fit packing,
    # splitting, shifting, and assistant masking exactly match training.
    eval_paths = [str(args.eval_data_file)] if args.eval_data_file is not None else None
    if args.eval_data_file is not None and not args.eval_data_file.is_file():
        raise FileNotFoundError(f"Evaluation data file not found: {args.eval_data_file}")
    eval_loader = make_dataloader_sharded(
        tokenizer, batch_size, MAX_SEQ_LEN,
        "validation" if eval_paths else "train", rank, world_size,
        parquet_paths=eval_paths,
    )
    raw_correct, raw_total, raw_loss_sum = evaluate_assistant_accuracy(
        model, eval_loader, args.eval_assistant_batches
    )

    totals = torch.tensor(
        [raw_correct, raw_total, raw_loss_sum], dtype=torch.float64, device=device
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if is_master:
        correct, assistant_tokens, loss_sum = totals.tolist()
        result = {
            "checkpoint": str(args.resume_from),
            "accuracy": correct / assistant_tokens,
            "assistant_tokens": int(assistant_tokens),
            "assistant_loss": loss_sum / assistant_tokens,
            "packed_batches_per_rank": args.eval_assistant_batches,
            "batch_size": batch_size,
            "sequence_length": MAX_SEQ_LEN,
            "world_size": world_size,
            "seed": args.seed,
            "data_paths": TRAIN_PARQUET_PATHS,
            "eval_data_file": str(args.eval_data_file) if args.eval_data_file else None,
        }
        args.eval_results_file.parent.mkdir(parents=True, exist_ok=True)
        with args.eval_results_file.open("a") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        print("ASSISTANT_ACCURACY " + json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0)

param_counts = model.num_scaling_params()
mprint("Parameter counts:")
for key, value in param_counts.items():
    mprint(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
mprint(f"Estimated FLOPs per token: {num_flops_per_token:e}")

grad_accum_steps = args.grad_accum_steps
tokens_per_step = grad_accum_steps * batch_size * MAX_SEQ_LEN          # per rank
tokens_per_step_global = tokens_per_step * world_size                  # across all ranks

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    matrix_optimizer=args.optimizer,
    log=mprint,
)
mprint(f"Matrix optimizer: {args.optimizer}")

raw_model = model  # uncompiled, unwrapped reference: used for eval and saving weights
model = DDP(model, device_ids=[local_rank])
if CAN_COMPILE:
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader_sharded(tokenizer, batch_size, MAX_SEQ_LEN, "train", rank, world_size)
x, y, epoch = next(train_loader)  # prefetch first batch

mprint(f"Gradient accumulation steps (per rank): {grad_accum_steps}")

# Initialize Weights & Biases (rank 0 only)
try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
except Exception:
    commit_hash = datetime.now().strftime('%Y%m%d_%H%M%S')

if wandb_enabled:
    _clip_tag = f"-clip{args.grad_clip}" if args.grad_clip > 0.0 else ""
    run_name = f"gpt-{args.model}-multigpu{world_size}-{args.optimizer}-bs{batch_size}-acc{grad_accum_steps}-seq{MAX_SEQ_LEN}-mlr{MATRIX_LR}-seed{args.seed}-budget{TOKEN_BUDGET}{_clip_tag}-{commit_hash}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_preset": args.model,
            "world_size": world_size,
            "depth": _n_layer,
            "window_pattern": WINDOW_PATTERN,
            "vocab_size": vocab_size,
            "grad_accum_steps": grad_accum_steps,
            "tokens_per_step_global": tokens_per_step_global,
            "embedding_lr": EMBEDDING_LR,
            "unembedding_lr": UNEMBEDDING_LR,
            "matrix_lr": MATRIX_LR,
            "scalar_lr": SCALAR_LR,
            "weight_decay": WEIGHT_DECAY,
            "adam_betas": ADAM_BETAS,
            "warmup_ratio": WARMUP_RATIO,
            "warmdown_ratio": WARMDOWN_RATIO,
            "final_lr_frac": FINAL_LR_FRAC,
            "device_batch_size": batch_size,
            "token_budget": TOKEN_BUDGET,
            "data_dir": str(args.data_dir),
            "train_fraction": args.train_fraction,
            "epochs": args.epochs,
            "selected_shards": len(TRAIN_PARQUET_PATHS),
            "selected_tokens_per_epoch": selected_tokens_per_epoch,
        },
    )

# Schedules (all based on progress = total_tokens_trained / TOKEN_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        t = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
        cosine_factor = (1 + math.cos(math.pi * t)) / 2
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine_factor

def get_muon_momentum(step):
    # Constant high momentum - no warmup needed with Nesterov
    return 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

mprint(f"Starting training | model: {args.model} | params: {num_params / 1e6:.1f}M | GPUs: {world_size}")
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
total_tokens_trained = 0
step = 0

MIN_LOG_STEP = 1

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_tokens_trained / TOKEN_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
        elif group['kind'] == 'sgd':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    if scaler.is_enabled() or args.grad_clip > 0.0:
        scaler.unscale_(optimizer)
    if args.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Coordinated fail-fast: every rank checks its own loss, then all agree via all-reduce
    # so every rank exits together (a rank-local exit would desync the collective calls in
    # the next iteration and hang the others).
    bad = torch.tensor(1.0 if (math.isnan(train_loss_f) or train_loss_f > 100) else 0.0, device=device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item() > 0:
        mprint("FAIL")
        dist.destroy_process_group()
        raise SystemExit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > MIN_LOG_STEP:
        total_training_time += dt
        total_tokens_trained += tokens_per_step_global

    if is_master and step > MIN_LOG_STEP and (step % logging_steps == 0):
        print("train_loss_f", train_loss_f)


    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(tokens_per_step_global / dt)
    mfu = 100 * num_flops_per_token * tokens_per_step_global / dt / H100_BF16_PEAK_FLOPS / world_size
    remaining_tokens = max(0, TOKEN_BUDGET - total_tokens_trained)

    if is_master:
        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining_tokens/1e6:.1f}M tok    ", end="", flush=True)

    # Log to Weights & Biases (respecting --logging-steps)
    if wandb_enabled and (step % logging_steps == 0):
        # Determine current Adam betas from optimizer (they may change over time)
        adam_beta1_current, adam_beta2_current = ADAM_BETAS
        for g in optimizer.param_groups:
            if g.get('kind') == 'adamw' and g.get('betas') is not None:
                adam_beta1_current, adam_beta2_current = g['betas']
                adam_beta1_current = float(adam_beta1_current)
                adam_beta2_current = float(adam_beta2_current)
                break
        wandb.log({
            "train/loss": debiased_smooth_loss,
            "train/raw_loss": train_loss_f,
            "train/epochs": epoch,
            "train/lr_multiplier": lrm,
            "train/momentum": muon_momentum,
            "train/weight_decay": muon_weight_decay,
            "train/adam_beta1": adam_beta1_current,
            "train/adam_beta2": adam_beta2_current,
            "train/step": step,
            "train/throughput_tokens_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }, step=step)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Budget exhausted — but only stop after warmup steps so we don't count compilation
    if step > MIN_LOG_STEP and total_tokens_trained >= TOKEN_BUDGET:
        break

mprint()  # newline after \r training log

# All ranks reach here together (total_tokens_trained is tracked identically on every rank),
# so it's safe for only rank 0 to run eval/save from here on.
dist.barrier()

total_tokens = step * tokens_per_step_global

if is_master:
    raw_model.eval()
    val_bpb = None
    if RUN_BPB_EVAL:
        with autocast_ctx:
            val_bpb = evaluate_bpb(raw_model, tokenizer, batch_size)

    assistant_val = None
    if args.assistant_val_file is not None:
        if not args.assistant_val_file.is_file():
            raise FileNotFoundError(f"Assistant validation file not found: {args.assistant_val_file}")
        assistant_val_loader = make_dataloader_sharded(
            tokenizer, min(batch_size, 8), MAX_SEQ_LEN, "validation", 0, 1,
            parquet_paths=[str(args.assistant_val_file)],
        )
        assistant_val_batches = args.assistant_val_batches or 32
        val_correct, val_tokens, val_loss_sum = evaluate_assistant_accuracy(
            raw_model, assistant_val_loader, assistant_val_batches
        )
        assistant_val = {
            "accuracy": val_correct / val_tokens,
            "loss": val_loss_sum / val_tokens,
            "assistant_tokens": val_tokens,
            "packed_batches": assistant_val_batches,
        }
        print("ASSISTANT_VALIDATION " + json.dumps(assistant_val, sort_keys=True))

    if bool(args.save_weights):
        _clip_tag = f"-clip{args.grad_clip}" if args.grad_clip > 0.0 else ""
        _checkpoint_tag = f"-{args.checkpoint_tag}" if args.checkpoint_tag else ""
        weights_path = (
            f"model_weights{_checkpoint_tag}-{args.model}-multigpu{world_size}-{args.optimizer}-bs{batch_size}-acc{grad_accum_steps}-"
            f"seq{MAX_SEQ_LEN}-mlr{MATRIX_LR}-seed{args.seed}-budget{TOKEN_BUDGET}{_clip_tag}.pt"
        )
        torch.save({
            "config": asdict(config),
            "model_state_dict": raw_model.state_dict(),
            "val_bpb": val_bpb,
            "assistant_validation": assistant_val,
        }, weights_path)
        print(f"Saved model weights to {weights_path}")

    t_end = time.time()
    steady_state_mfu = 100 * num_flops_per_token * tokens_per_step_global * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS / world_size if total_training_time > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    if val_bpb is not None:
        print(f"val_bpb:          {val_bpb:.6f}")
    if assistant_val is not None:
        print(f"assistant_val_accuracy: {assistant_val['accuracy']:.6f}")
        print(f"assistant_val_loss:     {assistant_val['loss']:.6f}")
        print(f"assistant_val_tokens:   {assistant_val['assistant_tokens']}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"model_preset:     {args.model}")
    print(f"world_size:       {world_size}")

    if wandb_enabled:
        final_metrics = {}
        if val_bpb is not None:
            final_metrics["final/val_bpb"] = val_bpb
        if assistant_val is not None:
            final_metrics.update({
                "final/assistant_val_accuracy": assistant_val["accuracy"],
                "final/assistant_val_loss": assistant_val["loss"],
                "final/assistant_val_tokens": assistant_val["assistant_tokens"],
            })
        wandb.log(final_metrics)
        wandb.finish()

    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated(device)
    peak_memory = torch.cuda.max_memory_allocated(device)
    print(f"Training took {t_end - t_start:.2f} seconds.")
    print(f"GPU memory before training: {start_memory / (1024 ** 2):.2f} MB")
    print(f"GPU memory after training: {end_memory / (1024 ** 2):.2f} MB")
    print(f"Peak GPU memory usage during training (rank 0): {peak_memory / (1024 ** 2):.2f} MB")

dist.barrier()
dist.destroy_process_group()
