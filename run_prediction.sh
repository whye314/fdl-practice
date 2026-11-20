#!/usr/bin/env bash
# Runs next-token prediction using the model you like
# Wraps generate.py with a sensible default checkpoint/prompt so it can be run with no
# arguments; pass --checkpoint/--prompt/--num_tokens/--temperature/--seed to override.
#
# Defaults to sampling (--temperature 0.8) since greedy decoding (--temperature 0) tends to
# repeat itself (or echo memorized training snippets verbatim) once the model locks onto a
# locally-confident continuation. Pass --temperature 0 for deterministic greedy output instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate.py"

BEST_CHECKPOINT="${SCRIPT_DIR}/model_weights-135M-phase1.pt"
DEFAULT_PROMPT="The history of artificial intelligence began"
DEFAULT_NUM_TOKENS=100
DEFAULT_TEMPERATURE=0.8

checkpoint="${BEST_CHECKPOINT}"
prompt="${DEFAULT_PROMPT}"
num_tokens="${DEFAULT_NUM_TOKENS}"
temperature="${DEFAULT_TEMPERATURE}"
seed=""

usage() {
    echo "Usage: $0 [--checkpoint PATH] [--prompt TEXT] [--num_tokens N] [--temperature T] [--seed N]"
    echo ""
    echo "  --checkpoint   Path to a model_weights-*.pt file (default: best model so far, ${BEST_CHECKPOINT##*/})"
    echo "  --prompt       Prompt text to continue (default: \"${DEFAULT_PROMPT}\")"
    echo "  --num_tokens   Number of tokens to generate (default: ${DEFAULT_NUM_TOKENS})"
    echo "  --temperature  Sampling temperature (default: ${DEFAULT_TEMPERATURE}). Pass 0 for"
    echo "                 greedy/deterministic decoding, which tends to repeat itself."
    echo "  --seed         Random seed for sampling (only used when --temperature > 0)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) checkpoint="$2"; shift 2 ;;
        --prompt) prompt="$2"; shift 2 ;;
        --num_tokens) num_tokens="$2"; shift 2 ;;
        --temperature) temperature="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}"
    echo "(Run train.py with --save-weights 1, the default, to produce one.)"
    exit 1
fi

extra_args=()
if [[ -n "${seed}" ]]; then
    extra_args+=(--seed "${seed}")
fi

echo "=== Checkpoint: ${checkpoint##*/} ==="
cd "${SCRIPT_DIR}"
uv run python "${PYTHON_SCRIPT}" "${checkpoint}" --prompt "${prompt}" --num_tokens "${num_tokens}" \
    --temperature "${temperature}" "${extra_args[@]}"
