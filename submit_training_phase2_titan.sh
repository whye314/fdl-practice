#!/bin/bash
#SBATCH --job-name=phase2_smoltalk
#SBATCH --time=16:00:00
#SBATCH --qos=titan-1day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=titan
#SBATCH --gres=gpu:4

set -euo pipefail
# Under sbatch, ${BASH_SOURCE[0]} points at Slurm's spool copy of this script,
# not the project. SLURM_SUBMIT_DIR is the directory sbatch was invoked from.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")}"

mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=WARN
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

NUM_GPUS=4
EPOCHS="${EPOCHS:-1}"
WANDB="${WANDB:-1}"
MATRIX_LR="${MATRIX_LR:-5e-4}"
EMBEDDING_LR="${EMBEDDING_LR:-0.0288}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.0018}"
SCALAR_LR="${SCALAR_LR:-0.0018}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-.}"
DATA_DIR="${DATA_DIR:-$HF_HOME/gpt/smol-smoltalk-v2}"
# Optional: shrink the phase-2 training set. A value in (0, 1] keeps that
# fraction of the training rows; a value > 1 keeps that many rows. Empty = use
# the full set. TRAIN_SUBSAMPLE_SEED controls which rows are drawn.
TRAIN_SUBSAMPLE="${TRAIN_SUBSAMPLE:-}"
TRAIN_SUBSAMPLE_SEED="${TRAIN_SUBSAMPLE_SEED:-108}"
VAL_FILE="${VAL_FILE:-$HOME/data/nemotron_cc/validation.parquet}"
ASSISTANT_VAL_FILE="${ASSISTANT_VAL_FILE:-$DATA_DIR/validation.parquet}"
DEFAULT_CHECKPOINT="model_weights-135M-phase1.pt"

CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_DIR/$DEFAULT_CHECKPOINT}"
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

TRAIN_FILE="$DATA_DIR/smol-smoltalk/train.parquet"
if [[ ! -f "$TRAIN_FILE" ]]; then
    mkdir -p "$(dirname "$TRAIN_FILE")"
    PHASE2_TRAIN_FILE="$TRAIN_FILE" .venv/bin/python - <<'PY'
import os
from datasets import load_dataset
from prepare import Tokenizer

ds = load_dataset(
    "HuggingFaceTB/smol-smoltalk",
    split="train",
)

tokenizer = Tokenizer.from_directory()
bos_id = tokenizer.get_bos_token_id()
user_id = tokenizer.enc.encode_single_token("<|reserved_1|>")
assistant_id = tokenizer.enc.encode_single_token("<|reserved_2|>")
end_message_id = tokenizer.enc.encode_single_token("<|reserved_3|>")

def render_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)

def prepare_batch(batch):
    batch_input_ids = []
    batch_loss_masks = []
    batch_num_tokens = []
    for messages in batch["messages"]:
        input_ids = [bos_id]
        loss_mask = [False]
        for message in messages:
            role = message.get("role")
            content = render_content(message.get("content", ""))
            if role == "system":
                role_id = user_id
                content = "System: " + content
            elif role == "user":
                role_id = user_id
            elif role == "assistant":
                role_id = assistant_id
            else:
                raise ValueError(f"Unsupported SmolTalk role: {role!r}")

            content_ids = tokenizer.enc.encode_ordinary(content)
            message_ids = [role_id, *content_ids, end_message_id]
            trains_assistant = role == "assistant"
            input_ids.extend(message_ids)
            loss_mask.extend([trains_assistant] * len(message_ids))

        batch_input_ids.append(input_ids)
        batch_loss_masks.append(loss_mask)
        batch_num_tokens.append(len(input_ids))
    return {
        "input_ids": batch_input_ids,
        "loss_mask": batch_loss_masks,
        "num_tokens": batch_num_tokens,
    }

processed = ds.map(
    prepare_batch,
    batched=True,
    batch_size=256,
    remove_columns=ds.column_names,
    desc="Formatting and token-counting SmolTalk",
)
processed.to_parquet(os.environ["PHASE2_TRAIN_FILE"])
PY
fi

# select_training_shards() globs every stratum dir under --data-dir, so the
# subsampled copy lives in its own sibling tree (not under DATA_DIR) to avoid it
# being picked up as an extra stratum on full-set runs.
TRAIN_DATA_DIR="$DATA_DIR"
if [[ -n "$TRAIN_SUBSAMPLE" ]]; then
    TRAIN_DATA_DIR="${DATA_DIR}-subsample-${TRAIN_SUBSAMPLE}-seed${TRAIN_SUBSAMPLE_SEED}"
    SUBSAMPLE_TRAIN_FILE="$TRAIN_DATA_DIR/smol-smoltalk/train.parquet"
    if [[ ! -f "$SUBSAMPLE_TRAIN_FILE" ]]; then
        mkdir -p "$(dirname "$SUBSAMPLE_TRAIN_FILE")"
        SRC="$TRAIN_FILE" DST="$SUBSAMPLE_TRAIN_FILE" \
        AMOUNT="$TRAIN_SUBSAMPLE" SEED="$TRAIN_SUBSAMPLE_SEED" .venv/bin/python - <<'PY'
import os
import numpy as np
import pyarrow.parquet as pq

src, dst = os.environ["SRC"], os.environ["DST"]
amount, seed = float(os.environ["AMOUNT"]), int(os.environ["SEED"])

table = pq.read_table(src)
n = table.num_rows
k = round(n * amount) if amount <= 1.0 else min(int(amount), n)
if k < 1:
    raise SystemExit(f"TRAIN_SUBSAMPLE={amount!r} selects {k} of {n} rows")
idx = np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))
pq.write_table(table.take(idx), dst)
print(f"Subsampled {k}/{n} training rows -> {dst}")
PY
    fi
fi

log_file="logs/train_phase2_smoltalk_$(date +'%Y%m%d-%H%M%S').log"

echo "[$(date)] Starting phase-two training on ${NUM_GPUS} GPUs"
echo "Checkpoint: $CHECKPOINT"
if [[ -n "$TRAIN_SUBSAMPLE" ]]; then
    echo "Training data: $TRAIN_DATA_DIR/smol-smoltalk/train.parquet (subsampled: $TRAIN_SUBSAMPLE, seed $TRAIN_SUBSAMPLE_SEED, $EPOCHS epoch(s))"
else
    echo "Training data: $TRAIN_FILE ($EPOCHS epoch(s))"
fi
echo "Log: $log_file"

train_cmd=(
    .venv/bin/torchrun
    --standalone
    --nproc_per_node="$NUM_GPUS"
    train_phase2.py
    --model 135M
    --optimizer muon
    --wandb "$WANDB"
    # Titan X/Xp (Pascal, 12 GB, fp32) with chunked attn + fused CE (model.py
    # defaults on legacy GPUs): bs4/accum1 = ~20.5k tok/s, peak ~7.8 GB, effective
    # batch 16 (unchanged from the old bs1*accum4*4gpu). Faster options: bs5 ~21k
    # (9.3 GB), bs6 ~21.8k (10.8 GB, tight), or bs4 GRAD_ACCUM=4 ~23k (effective
    # batch 64). PHASE2_CKPT_BLOCKS=1 exists but is a net slowdown here.
    --batch-size "${BATCH_SIZE:-4}"
    --grad-accum-steps "${GRAD_ACCUM:-1}"
    --matrix-lr "$MATRIX_LR"
    --embedding-lr "$EMBEDDING_LR"
    --unembedding-lr "$UNEMBEDDING_LR"
    --scalar-lr "$SCALAR_LR"
    --warmup-ratio 0.10
    --warmdown-ratio 0.28
    --wandb-project gpt_smoltalk_phase2
    --seed 108
    --resume-from "$CHECKPOINT"
    --checkpoint-tag phase2-v2
    --data-dir "$TRAIN_DATA_DIR"
    --val-file "$VAL_FILE"
    --train-fraction 1.0
    --epochs "$EPOCHS"
)
if [[ -n "${TOKEN_BUDGET:-}" ]]; then
    train_cmd+=(--token-budget "$TOKEN_BUDGET")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    train_cmd+=($EXTRA_ARGS)
fi
if [[ -f "$ASSISTANT_VAL_FILE" ]]; then
    train_cmd+=(--assistant-val-file "$ASSISTANT_VAL_FILE")
fi

"${train_cmd[@]}" 2>&1 | tee "$log_file"

echo "[$(date)] Phase-two training finished"
