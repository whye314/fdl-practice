"""Interactively generate from a phase-2 checkpoint using its chat format."""

import argparse
from pathlib import Path

import torch

from generate import load_model


DEFAULT_CHECKPOINT = Path(__file__).with_name("model_weights-135M-phase2.pt")


def generate_chat(model, config, tokenizer, prompt, num_tokens=100,
                  temperature=0.8, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    user_id = tokenizer.enc.encode_single_token("<|reserved_1|>")
    assistant_id = tokenizer.enc.encode_single_token("<|reserved_2|>")
    end_message_id = tokenizer.enc.encode_single_token("<|reserved_3|>")
    context = [
        tokenizer.get_bos_token_id(),
        user_id,
        *tokenizer.enc.encode_ordinary(prompt),
        end_message_id,
        assistant_id,
    ]
    generated_ids = []
    device = next(model.parameters()).device
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))

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
            if next_id == end_message_id:
                break
            generated_ids.append(next_id)
            context.append(next_id)

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if args.num_tokens < 0:
        parser.error("--num-tokens must be non-negative")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")

    print(f"Loading checkpoint: {checkpoint}")
    model, config, tokenizer = load_model(checkpoint)
    print("\nReady. Enter a user message, or type /quit (Ctrl-D also exits).")

    while True:
        try:
            prompt = input("\nUser> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if prompt.strip().lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            break
        if not prompt.strip():
            continue
        print("prompt:", prompt)
        generated = generate_chat(
            model, config, tokenizer, prompt,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"Assistant> {generated}")


if __name__ == "__main__":
    main()
