"""
generate.py
===========
Interactive text generation with the trained MiniLM model.

Type a prompt and the model will complete it.
Type 'quit' or press Ctrl+C to exit.

Author  : André Costa
License : MIT

Usage:
    python3 generate.py
    python3 generate.py --max-tokens 100
    python3 generate.py --temperature 0.9 --top-k 50
"""

import argparse
import torch
from transformer import MiniLM, ModelConfig
from bpe_tokenizer import BPETokenizer


def load_model(checkpoint_path: str, tokenizer_path: str):
    """Load the trained model and tokenizer."""

    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(tokenizer_path)

    print("Loading model...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    cfg_dict = ckpt["model_config"]
    cfg_dict.pop("d_head", None)
    config = ModelConfig(**cfg_dict)

    model = MiniLM(config)

    state_dict = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    print(f"Model ready — {config.n_params / 1e6:.1f}M parameters | device: {device}")
    print(f"Vocab: {config.vocab_size} tokens | Context: {config.seq_len} tokens\n")

    return model, tokenizer, device


def generate(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    """Generate text from a prompt."""
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    return tokenizer.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="MiniLM — Interactive text generation")
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints/best_model.pt")
    parser.add_argument("--tokenizer",    type=str,   default="./tokenizer")
    parser.add_argument("--max-tokens",   type=int,   default=80)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top-k",        type=int,   default=50)
    parser.add_argument("--top-p",        type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint, args.tokenizer)

    print("=" * 55)
    print("  MiniLM — Text Generation")
    print("  Type a prompt and press Enter.")
    print("  Type 'quit' to exit.")
    print("=" * 55)
    print()

    while True:
        try:
            prompt = input("Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = generate(
            model, tokenizer, device,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        print(f"\n{result}\n")
        print("-" * 55)


if __name__ == "__main__":
    main()
