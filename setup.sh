#!/usr/bin/env bash
# Source in the shell that will launch Python, for example in a %%bash cell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Run source setup.sh in the shell that will launch Python.' >&2
    exit 1
fi

GPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
if [[ ! -x "$GPT_REPO_DIR/.venv/bin/python" ]]; then
    printf '%s\n' 'Run bash install.sh first.' >&2
    return 1
fi

if [[ -z "${GPT_WORK_DIR:-}" ]]; then
    if [[ -d /kaggle/working ]]; then
        GPT_WORK_DIR=/kaggle/working/gpt
    elif [[ -d /content ]]; then
        GPT_WORK_DIR=/content/gpt-work
    else
        GPT_WORK_DIR="$GPT_REPO_DIR/.notebook"
    fi
fi

export GPT_REPO_DIR GPT_WORK_DIR
export HF_HOME="${HF_HOME:-$GPT_WORK_DIR/cache/huggingface}"
export GPT_CACHE_DIR="${GPT_CACHE_DIR:-$GPT_WORK_DIR/cache/gpt}"
export GPT_TOKENIZER_DIR="${GPT_TOKENIZER_DIR:-$GPT_REPO_DIR/tokenizer/tokenizer}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Use these paths as CLI arguments when launching the Python trainers.
export PHASE1_DATA_DIR="${PHASE1_DATA_DIR:-$GPT_WORK_DIR/data/nemotron_cc_5b}"
export DATA_DIR="${DATA_DIR:-$HF_HOME/gpt/smol-smoltalk-v2}"
export VAL_FILE="${VAL_FILE:-$GPT_WORK_DIR/data/nemotron_cc/validation.parquet}"
export ASSISTANT_VAL_FILE="${ASSISTANT_VAL_FILE:-$DATA_DIR/validation.parquet}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GPT_REPO_DIR}"
export CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_DIR/model_weights-135M-phase1.pt}"

mkdir -p -- "$HF_HOME" "$GPT_CACHE_DIR" || return 1

case ":$PATH:" in
    *":$GPT_REPO_DIR/.notebook-tools/bin:"*) ;;
    *) export PATH="$GPT_REPO_DIR/.notebook-tools/bin:$PATH" ;;
esac
case ":$PATH:" in
    *":$GPT_REPO_DIR/.venv/bin:"*) ;;
    *) export PATH="$GPT_REPO_DIR/.venv/bin:$PATH" ;;
esac
