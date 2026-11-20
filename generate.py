"""
Generate text from a trained GPT checkpoint (saved by train.py with
--save-weights 1, the default).

Usage:
    uv run python generate.py <checkpoint.pt> --prompt "Once upon a time"
"""

import argparse
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT, AMP_DTYPE
from prepare import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Loading and generation helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint):
    ckpt = torch.load(checkpoint, map_location=device)
    config = GPTConfig(**ckpt["config"])
    print(f"Model config: {ckpt['config']}")
    print(f"Checkpoint val_bpb: {ckpt.get('val_bpb', 'unknown')}")

    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, Tokenizer.from_directory()


def generate_text(model, config, tokenizer, prompt, num_tokens=100,
                  temperature=0.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    bos_id = tokenizer.get_bos_token_id()
    context = tokenizer.encode(prompt, prepend=bos_id)
    generated_ids = []
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE,
                                       enabled=(AMP_DTYPE != torch.float32))
                    if device.type == "cuda" else nullcontext())

    with torch.no_grad(), autocast_ctx:
        for _ in range(num_tokens):
            window = context[-config.sequence_len:]
            idx = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            next_logits = model(idx)[0, -1]

            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(next_logits).item()

            generated_ids.append(next_id)
            context.append(next_id)

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint")
    parser.add_argument("checkpoint", help="Path to a model_weights-*.pt checkpoint file")
    parser.add_argument("--prompt", required=True, help="Prompt text to continue")
    parser.add_argument("--num_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0 uses greedy decoding")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    model, config, tokenizer = load_model(args.checkpoint)
    generated = generate_text(model, config, tokenizer, args.prompt,
                              args.num_tokens, args.temperature, args.seed)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated next {args.num_tokens} tokens: {generated}")


if __name__ == "__main__":
    main()
