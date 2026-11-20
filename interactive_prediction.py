"""Load a gpt checkpoint once and continuously generate from prompts."""

import argparse
from pathlib import Path

from generate import generate_text, load_model


DEFAULT_CHECKPOINT = Path(__file__).with_name("model_weights-135M-phase1.pt")


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
    print("\nReady. Enter a prompt, or type /quit (Ctrl-D also exits).")

    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if prompt.strip().lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            break
        if not prompt.strip():
            continue

        generated = generate_text(
            model, config, tokenizer, prompt,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"Output> {generated}")


if __name__ == "__main__":
    main()
