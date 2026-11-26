"""Skill Nexis Week 2 - Character-Level Bigram Language Model.

This single Python entry point builds a small CC0 name corpus (if needed),
trains a character-level bigram language model, records train/validation loss,
saves the model, compares untrained/trained samples, and generates names from
starter characters or sequences.

Run from the project folder:
    python mini_project.py --train
    python mini_project.py --generate --seed Al --count 10
    python mini_project.py --demo
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
DATA_PATH = DATA_DIR / "names.txt"
MODEL_PATH = ARTIFACT_DIR / "bigram_char_model.pt"
METRICS_CSV = RESULTS_DIR / "training_metrics.csv"
METRICS_JSON = RESULTS_DIR / "training_metrics.json"
SAMPLES_PATH = RESULTS_DIR / "generated_samples.txt"
LOSS_PLOT = RESULTS_DIR / "loss_curve.png"

SEED = 42

# Original syllable inventory used to create the included synthetic name corpus.
# The resulting names are original educational data and are released under CC0.
PREFIXES = [
    "Al", "Ari", "Aven", "Bel", "Bri", "Cael", "Cali", "Cora", "Dara",
    "Deli", "Eli", "Elora", "Emi", "Fae", "Fira", "Gala", "Gwen", "Hali",
    "Ira", "Jora", "Kael", "Kira", "Lena", "Liora", "Luna", "Mira", "Nara",
    "Neri", "Nia", "Orin", "Oriana", "Pela", "Quin", "Rhea", "Rina", "Sela",
    "Sera", "Tali", "Tara", "Thali", "Una", "Vela", "Wren", "Yara", "Zara",
    "Zelia", "Aela", "Bela", "Ceri", "Demi", "Eira", "Fenna", "Galen",
    "Hera", "Isla", "Jalen", "Kaia", "Maren", "Nolan", "Oriel", "Perrin",
    "Riven", "Sorin", "Tarin", "Valen", "Willa", "Yestin", "Zorin",
]
MIDDLES = [
    "a", "e", "i", "o", "u", "ae", "ia", "ie", "io", "ara", "eli", "ira",
    "ora", "una", "en", "el", "ar", "il", "or", "y", "ari", "essa",
]
SUFFIXES = [
    "a", "ae", "elle", "ia", "ina", "ine", "ion", "is", "ith", "ius", "ora",
    "or", "en", "eth", "iel", "wyn", "ya", "yn", "us", "an", "on", "el", "ar",
    "er", "wen", "ix", "essa", "ette", "ian", "orae",
]


def build_name_corpus(limit: int = 6000) -> List[str]:
    """Create a deterministic, original corpus of pronounceable fantasy names."""
    candidates = {
        (prefix + middle + suffix).capitalize()
        for prefix in PREFIXES
        for middle in MIDDLES
        for suffix in SUFFIXES
    }
    names = sorted(name for name in candidates if 4 <= len(name) <= 12)
    rng = random.Random(SEED)
    rng.shuffle(names)
    names = names[:limit]
    return sorted(names)


def ensure_dataset(force_rebuild: bool = False) -> List[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force_rebuild or not DATA_PATH.exists():
        names = build_name_corpus()
        DATA_PATH.write_text("\n".join(names) + "\n", encoding="utf-8")
    else:
        names = [line.strip() for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) < 100:
        raise ValueError("The dataset must contain at least 100 non-empty names.")
    return names


def split_dataset(names: Sequence[str], validation_fraction: float = 0.10) -> Tuple[List[str], List[str]]:
    """Split at the name level to avoid splitting one name across both sets."""
    items = list(names)
    random.Random(SEED).shuffle(items)
    val_count = max(1, int(len(items) * validation_fraction))
    val_names = items[:val_count]
    train_names = items[val_count:]
    return train_names, val_names


def build_vocabulary(names: Sequence[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    text = "\n".join(names) + "\n"
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return vocab, stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """Convert characters into integer token IDs."""
    unknown = sorted(set(text) - set(stoi))
    if unknown:
        raise ValueError(f"Character(s) not present in vocabulary: {unknown}")
    return [stoi[ch] for ch in text]


def decode(token_ids: Sequence[int], itos: Dict[int, str]) -> str:
    """Convert integer token IDs back into a string."""
    try:
        return "".join(itos[int(token_id)] for token_id in token_ids)
    except KeyError as exc:
        raise ValueError(f"Unknown token ID: {exc.args[0]}") from exc


class BigramLanguageModel(nn.Module):
    """A neural bigram model: current character predicts the next character."""
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        seed_ids: Sequence[int],
        itos: Dict[int, str],
        max_new_tokens: int = 8,
        temperature: float = 0.85,
        min_total_length: int = 4,
        top_k: int = 8,
    ) -> str:
        if not seed_ids:
            raise ValueError("A non-empty starter character or sequence is required.")
        idx = torch.tensor([list(seed_ids)], dtype=torch.long)
        output_ids = list(seed_ids)
        newline_id = next((i for i, ch in itos.items() if ch == "\n"), None)

        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1])
            logits = logits / max(float(temperature), 1e-4)
            if top_k > 0 and top_k < logits.size(-1):
                values, _ = torch.topk(logits, top_k)
                cutoff = values[:, [-1]]
                logits[logits < cutoff] = float("-inf")
            # Do not let a sampled line break truncate a name too early.
            current_text = decode(output_ids, itos)
            if newline_id is not None and len(current_text) < min_total_length:
                logits[:, newline_id] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            if newline_id is not None and next_id == newline_id:
                break
            output_ids.append(next_id)
            idx = torch.cat((idx, torch.tensor([[next_id]], dtype=torch.long)), dim=1)
        return decode(output_ids, itos).strip()


def evaluate_loss(model: BigramLanguageModel, data: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        x = data[:-1]
        y = data[1:]
        _, loss = model(x, y)
    model.train()
    return float(loss.item())


def generate_samples(
    model: BigramLanguageModel,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    starters: Sequence[str],
    count: int,
    seed_offset: int = 0,
) -> List[str]:
    samples: List[str] = []
    for i in range(count):
        starter = starters[i % len(starters)]
        unknown = [ch for ch in starter if ch not in stoi]
        if unknown:
            raise ValueError(f"Starter {starter!r} contains characters not in the vocabulary: {unknown}")
        torch.manual_seed(SEED + seed_offset + i)
        sample = model.generate(encode(starter, stoi), itos)
        samples.append(f"{i + 1:02d}. Starter: {starter:<3} -> {sample}")
    return samples


def train_model(
    max_iters: int = 1800,
    batch_size: int = 256,
    learning_rate: float = 0.025,
    eval_interval: int = 100,
    force_rebuild_dataset: bool = False,
) -> Dict[str, object]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    names = ensure_dataset(force_rebuild=force_rebuild_dataset)
    train_names, val_names = split_dataset(names)
    vocab, stoi, itos = build_vocabulary(names)
    train_ids = torch.tensor(encode("\n".join(train_names) + "\n", stoi), dtype=torch.long)
    val_ids = torch.tensor(encode("\n".join(val_names) + "\n", stoi), dtype=torch.long)

    model = BigramLanguageModel(len(vocab))
    untrained_model = BigramLanguageModel(len(vocab))
    # Same random initialization for a fair before/after comparison.
    untrained_model.load_state_dict(model.state_dict())

    starters = ["Al", "Be", "Ca", "El", "Fa", "Li", "Mi", "Na", "Or", "Se"]
    before_samples = generate_samples(untrained_model, stoi, itos, starters, 10, seed_offset=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: List[Dict[str, float]] = []

    def get_batch() -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, len(train_ids) - 1, (batch_size,))
        return train_ids[ix], train_ids[ix + 1]

    for step in range(max_iters + 1):
        if step % eval_interval == 0 or step == max_iters:
            train_loss = evaluate_loss(model, train_ids)
            val_loss = evaluate_loss(model, val_ids)
            history.append({"step": int(step), "train_loss": train_loss, "validation_loss": val_loss})
        if step == max_iters:
            break
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "stoi": stoi,
        "itos": itos,
        "model_type": "character-level neural bigram",
        "seed": SEED,
        "dataset_name": "Skill Nexis Open Name Corpus (CC0)",
        "train_name_count": len(train_names),
        "validation_name_count": len(val_names),
        "training_steps": max_iters,
    }
    torch.save(checkpoint, MODEL_PATH)

    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "dataset": "Skill Nexis Open Name Corpus (CC0)",
        "total_names": len(names),
        "training_names": len(train_names),
        "validation_names": len(val_names),
        "vocabulary_size": len(vocab),
        "training_steps": max_iters,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_validation_loss": history[0]["validation_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "loss_history": history,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    after_samples = generate_samples(model, stoi, itos, starters, 10, seed_offset=2000)
    sample_text = [
        "SKILL NEXIS WEEK 2 - GENERATED SAMPLE RECORD",
        "Model: Character-level neural bigram language model",
        "Dataset: Original synthetic name corpus released under CC0",
        "",
        "BEFORE TRAINING (randomly initialized model)",
        *before_samples,
        "",
        "AFTER TRAINING (trained bigram model)",
        *after_samples,
        "",
        "Note: These are stochastic model outputs, not curated names.",
    ]
    SAMPLES_PATH.write_text("\n".join(sample_text) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        steps = [row["step"] for row in history]
        train_losses = [row["train_loss"] for row in history]
        val_losses = [row["validation_loss"] for row in history]
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, train_losses, label="Training loss")
        plt.plot(steps, val_losses, label="Validation loss")
        plt.xlabel("Training step")
        plt.ylabel("Cross-entropy loss")
        plt.title("Character-Level Bigram Model: Loss During Training")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(LOSS_PLOT, dpi=160)
        plt.close()
    except Exception as exc:  # Plotting is supplementary; training results remain saved.
        print(f"Warning: loss plot could not be created: {exc}")

    return summary


def load_model() -> Tuple[BigramLanguageModel, Dict[str, int], Dict[int, str], Dict[str, object]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Saved model not found. Run: python mini_project.py --train")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    vocab = checkpoint["vocab"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    model = BigramLanguageModel(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, stoi, itos, checkpoint


def print_summary(summary: Dict[str, object]) -> None:
    print("Training complete.")
    print(f"Dataset names:       {summary['total_names']}")
    print(f"Train / validation:  {summary['training_names']} / {summary['validation_names']}")
    print(f"Vocabulary size:     {summary['vocabulary_size']}")
    print(f"Training steps:      {summary['training_steps']}")
    print(f"Train loss:          {summary['initial_train_loss']:.4f} -> {summary['final_train_loss']:.4f}")
    print(f"Validation loss:     {summary['initial_validation_loss']:.4f} -> {summary['final_validation_loss']:.4f}")
    print(f"Saved model:         {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved results:       {RESULTS_DIR.relative_to(ROOT)}")


def run_self_test() -> None:
    names = ensure_dataset()
    vocab, stoi, itos = build_vocabulary(names[:100])
    probe = "Alora"
    assert decode(encode(probe, stoi), itos) == probe, "Encode/decode round trip failed."
    model = BigramLanguageModel(len(vocab))
    logits, loss = model(torch.tensor([stoi["A"]]), torch.tensor([stoi["l"]]))
    assert logits.shape == (1, len(vocab)), "Unexpected model output shape."
    assert loss is not None and torch.isfinite(loss), "Model loss is not finite."
    print("Self-test passed: dataset, vocabulary, encode/decode, model forward pass.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and use a character-level bigram name generator.")
    parser.add_argument("--train", action="store_true", help="Train the model and save model/results.")
    parser.add_argument("--generate", action="store_true", help="Generate names using the saved model.")
    parser.add_argument("--demo", action="store_true", help="Show the saved before/after training examples.")
    parser.add_argument("--self-test", action="store_true", help="Run a quick code and encode/decode test.")
    parser.add_argument("--seed", default="Al", help="Starter character or sequence for generation (default: Al).")
    parser.add_argument("--count", type=int, default=10, help="Number of names to generate (default: 10).")
    parser.add_argument("--steps", type=int, default=1800, help="Training iterations when --train is used.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Recreate the included deterministic dataset.")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.train:
        summary = train_model(max_iters=args.steps, force_rebuild_dataset=args.rebuild_dataset)
        print_summary(summary)
        return
    if args.generate:
        if args.count < 1 or args.count > 100:
            parser.error("--count must be between 1 and 100.")
        model, stoi, itos, _ = load_model()
        if any(ch not in stoi for ch in args.seed):
            parser.error("The starter sequence contains characters outside the model vocabulary.")
        for i in range(args.count):
            torch.manual_seed(SEED + 3000 + i)
            print(f"{i + 1:02d}. {model.generate(encode(args.seed, stoi), itos)}")
        return
    if args.demo:
        if not SAMPLES_PATH.exists():
            parser.error("Saved sample file not found. Run --train first.")
        print(SAMPLES_PATH.read_text(encoding="utf-8"))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
