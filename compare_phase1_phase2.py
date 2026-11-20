"""Ask the same questions to the best phase-1 and phase-2 models."""

import argparse
from pathlib import Path

from generate import generate_text, load_model
from interactive_prediction_chat import generate_chat


PHASE1_CHECKPOINT = Path(__file__).with_name("model_weights-135M-phase1.pt")
PHASE2_CHECKPOINT = Path(__file__).with_name("model_weights-135M-phase2.pt")
NUM_TOKENS = 400

QUESTIONS = [
    "What is the capital of France, and what is it known for?",
    "Explain how photosynthesis works in simple terms.",
    "Write a short Python function that checks whether a number is prime.",
    "What are the main causes of climate change?",
    "Give me three practical tips for learning a new language.",
    "Summarize the difference between supervised and unsupervised learning.",
    "Plan a healthy dinner for four people, including a shopping list.",
    "Why is the sky blue?",
    "Write a polite email asking to reschedule a meeting.",
    "Describe one important lesson from the history of computing.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-checkpoint", type=Path, default=PHASE1_CHECKPOINT)
    parser.add_argument("--phase2-checkpoint", type=Path, default=PHASE2_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("phase1_phase2_answers.txt"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args()

    for checkpoint in (args.phase1_checkpoint, args.phase2_checkpoint):
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")

    print(f"Loading phase-1 checkpoint: {args.phase1_checkpoint}")
    phase1_model, phase1_config, phase1_tokenizer = load_model(args.phase1_checkpoint)
    print(f"Loading phase-2 checkpoint: {args.phase2_checkpoint}")
    phase2_model, phase2_config, phase2_tokenizer = load_model(args.phase2_checkpoint)

    sections = []
    for number, question in enumerate(QUESTIONS, start=1):
        print(f"Answering question {number}/{len(QUESTIONS)}")
        phase1_answer = generate_text(
            phase1_model, phase1_config, phase1_tokenizer, question,
            num_tokens=NUM_TOKENS, temperature=args.temperature, seed=args.seed,
        )
        phase2_answer = generate_chat(
            phase2_model, phase2_config, phase2_tokenizer, question,
            num_tokens=NUM_TOKENS, temperature=args.temperature, seed=args.seed,
        )
        sections.append(
            f"Question {number}: {question}\n\n"
            f"PHASE 1\n{phase1_answer}\n\n"
            f"PHASE 2\n{phase2_answer}\n"
        )

    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote answers to {args.output}")


if __name__ == "__main__":
    main()
