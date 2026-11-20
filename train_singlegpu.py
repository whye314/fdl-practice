"""
GPT pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train_singlegpu.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import gc
import math
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import wandb

from model import GPTConfig, GPT, MODEL_PRESETS
from prepare import (
    MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb, select_training_shards,
    DEFAULT_TRAIN_DATA_DIR, DEFAULT_VAL_FILE,
)

TOKEN_BUDGET = 2**28     # total training token budget (default)

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Gpt pretraining script")
parser.add_argument("--wandb", type=int, default=1, help="Enable Weights & Biases logging (0 = disabled, non-zero = enabled)")
parser.add_argument("--token-budget", type=int, default=None, help="Training token budget (default: TOKEN_BUDGET from prepare.py)")
parser.add_argument("--logging-steps", type=int, default=5, help="How often (in steps) to log metrics to wandb (default: 1)")
parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adam", "sgd"],
                    help="Optimizer for matrix parameters: 'muon' (default), 'adam' (AdamW), or 'sgd' (SGD with momentum)")
parser.add_argument("--model", choices=list(MODEL_PRESETS), default="135M",
                    help="Model size preset (default: 135M)")
parser.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Gradient accumulation steps per optimizer step (default: 1)")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Per-device batch size (default: preset value for the selected model)")
parser.add_argument("--matrix-lr", type=float, default=None,
                    help="Learning rate for matrix parameters (default: 0.04)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for parameter initialization (default: 42)")
parser.add_argument("--weight-decay", type=float, default=None,
                    help="Weight decay (default: 0.035)")
parser.add_argument("--grad-clip", type=float, default=0.0,
                    help="Gradient clipping max norm (0 = disabled)")
parser.add_argument("--save-weights", type=int, default=1,
                    help="Save model weights after training (default: 1=save, 0=skip)")
parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_TRAIN_DATA_DIR),
                    help="Directory containing per-stratum Parquet training shards")
parser.add_argument("--val-file", type=Path, default=Path(DEFAULT_VAL_FILE),
                    help="Pinned validation Parquet file")
parser.add_argument("--train-fraction", type=float, default=1.0,
                    help="Fraction selected independently from each stratum (default: 1.0)")
args = parser.parse_args()

if not 0.0 < args.train_fraction <= 1.0:
    parser.error("--train-fraction must be in (0, 1]")

# Convert integer flag to boolean and disable wandb if requested
wandb_enabled = bool(args.wandb)
print(f"Wandb logging enabled: {wandb_enabled}")
if not wandb_enabled:
    wandb.disabled = True

# How often to record logs to wandb (every N steps)
logging_steps = max(1, int(args.logging_steps))
print(f"Wandb logging frequency: every {logging_steps} step(s)")

# Use provided token budget or default from prepare.py
if args.token_budget is not None:
    TOKEN_BUDGET = args.token_budget
print(f"Using token budget: {TOKEN_BUDGET:,} tokens")

_n_layer, _n_head, _n_embd, _batch_size = MODEL_PRESETS[args.model]
print(f"Using model preset: {args.model}")
batch_size = args.batch_size if args.batch_size is not None else _batch_size
print(f"Per-device batch size: {batch_size}")

TRAIN_PARQUET_PATHS, selected_tokens = select_training_shards(args.data_dir, args.train_fraction, args.seed)
VAL_PATH = str(args.val_file)
if not args.val_file.is_file():
    raise FileNotFoundError(f"Validation file not found: {args.val_file}")
tokens_note = f" ({selected_tokens:,} tokens)" if selected_tokens is not None else ""
print(f"Training data: {len(TRAIN_PARQUET_PATHS)} shards from {args.data_dir}{tokens_note}")
print(f"Validation file: {VAL_PATH}")

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = args.matrix_lr if args.matrix_lr is not None else 0.18
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = args.weight_decay if args.weight_decay is not None else 0.035
ADAM_BETAS = (0.75, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.75  # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.1    # final LR as fraction of initial

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
device = torch.device("cuda")
start_memory = torch.cuda.memory_allocated(device)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config():
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=_n_layer, n_head=_n_head, n_kv_head=_n_head, n_embd=_n_embd,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config()
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

grad_accum_steps = args.grad_accum_steps
tokens_per_step = grad_accum_steps * batch_size * MAX_SEQ_LEN

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    matrix_optimizer=args.optimizer,
)
print(f"Matrix optimizer: {args.optimizer}")

raw_model = model  # uncompiled reference, used for saving weights
model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "train", TRAIN_PARQUET_PATHS, VAL_PATH)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Token budget: {TOKEN_BUDGET:,} tokens")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Initialize Weights & Biases
# Get git commit hash for run name
try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
except:
    print("Using timestamp as commit hash fallback")
    commit_hash = datetime.now().strftime('%Y%m%d_%H%M%S')

if wandb_enabled:
    _clip_tag = f"-clip{args.grad_clip}" if args.grad_clip > 0.0 else ""
    run_name = f"gpt-{args.model}-{args.optimizer}-bs{batch_size}-acc{grad_accum_steps}-seq{MAX_SEQ_LEN}-mlr{MATRIX_LR}-seed{args.seed}-budget{TOKEN_BUDGET}{_clip_tag}-{commit_hash}"
    wandb.init(
        project="gpt_nemotron",
        name=run_name,
        config={
            "model_preset": args.model,
            "depth": _n_layer,
            "window_pattern": WINDOW_PATTERN,
            "vocab_size": vocab_size,
            "grad_accum_steps": grad_accum_steps,
            "tokens_per_step": tokens_per_step,
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

print(f"Starting training | model: {args.model} | params: {num_params / 1e6:.1f}M")
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
        loss.backward()
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

    if args.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > MIN_LOG_STEP:
        total_training_time += dt
        total_tokens_trained += tokens_per_step



    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(tokens_per_step / dt)
    mfu = 100 * num_flops_per_token * tokens_per_step / dt / H100_BF16_PEAK_FLOPS
    remaining_tokens = max(0, TOKEN_BUDGET - total_tokens_trained)

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
        log_dict = {
            "train/loss": debiased_smooth_loss,
            "train/raw_loss": train_loss_f,
            "train/lr_multiplier": lrm,
            "train/momentum": muon_momentum,
            "train/weight_decay": muon_weight_decay,
            "train/adam_beta1": adam_beta1_current,
            "train/adam_beta2": adam_beta2_current,
            "train/step": step,
            "train/throughput_tokens_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }

        # Evaluate `val/bpb` less frequently: once every 10 * logging_steps
        val_log_interval = max(1, logging_steps * 10)
        if step % val_log_interval == 0:
            model.eval()
            with autocast_ctx:
                last_val_bpb = evaluate_bpb(model, tokenizer, batch_size)
            model.train()
            log_dict["val/bpb"] = last_val_bpb
        wandb.log(log_dict, step=step)

    # GC management (Python's GC causes ~500ms stalls)
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

print()  # newline after \r training log

total_tokens = step * tokens_per_step

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, batch_size)

# Save weights (config + state_dict, enough to reconstruct the model for inference)
if bool(args.save_weights):
    _clip_tag = f"-clip{args.grad_clip}" if args.grad_clip > 0.0 else ""
    weights_path = (
        f"model_weights-{args.model}-{args.optimizer}-bs{batch_size}-acc{grad_accum_steps}-"
        f"seq{MAX_SEQ_LEN}-mlr{MATRIX_LR}-seed{args.seed}-budget{TOKEN_BUDGET}{_clip_tag}.pt"
    )
    torch.save({
        "config": asdict(config),
        "model_state_dict": raw_model.state_dict(),
        "val_bpb": val_bpb,
    }, weights_path)
    print(f"Saved model weights to {weights_path}")

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * tokens_per_step * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"model_preset:     {args.model}")

# Finish Weights & Biases run
if wandb_enabled:
    wandb.log({"final/val_bpb": val_bpb})
    wandb.finish()

torch.cuda.synchronize()
end_time = time.time()
end_memory = torch.cuda.memory_allocated(device)
peak_memory = torch.cuda.max_memory_allocated(device)
print(f"Training took {end_time - t_start:.2f} seconds.")
print(f"GPU memory before training: {start_memory / (1024 ** 2):.2f} MB")
print(f"GPU memory after training: {end_memory / (1024 ** 2):.2f} MB")
print(f"Peak GPU memory usage during training: {peak_memory / (1024 ** 2):.2f} MB")
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(i)
    print(f"GPU {i}: Peak memory usage: {peak_memory / (1024**2)} MB")
