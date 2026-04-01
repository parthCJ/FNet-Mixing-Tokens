"""
FNet vs MLP Baseline Benchmark

USAGE:
    python scripts/benchmark.py --epochs 3 --train-subset 5000 --val-subset 1000

This script trains both FNet and a simple MLP baseline on the same data
and compares their performance metrics side-by-side.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm.auto import tqdm

from fnet_mixing.data import build_ag_news_dataloaders
from fnet_mixing.model import FNetConfig, FNetForSequenceClassification


class SimpleMLP(nn.Module):
    """Simple MLP baseline for comparison."""

    def __init__(
        self, vocab_size: int, hidden_size: int, num_labels: int, dropout: float = 0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        return self.mlp(pooled)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1


def benchmark() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FNet vs MLP")
    parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--val-subset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("Loading data...")
    data = build_ag_news_dataloaders(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
    )

    vocab_size = data.tokenizer.vocab_size

    # ===== FNET =====
    print("\n" + "=" * 60)
    print("TRAINING FNET")
    print("=" * 60)

    fnet_config = FNetConfig(
        vocab_size=vocab_size,
        max_position_embeddings=args.max_length,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 2,
        num_layers=4,
        num_labels=data.num_labels,
        dropout=0.1,
        pad_token_id=data.tokenizer.pad_token_id or 0,
    )

    fnet_model = FNetForSequenceClassification(fnet_config).to(device)
    fnet_optimizer = torch.optim.AdamW(
        fnet_model.parameters(), lr=args.lr, weight_decay=0.01
    )
    fnet_criterion = nn.CrossEntropyLoss()

    fnet_start = time.time()
    fnet_best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            fnet_model, data.train_loader, fnet_optimizer, fnet_criterion, device
        )
        acc, f1 = evaluate(fnet_model, data.val_loader, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}"
        )
        fnet_best_acc = max(fnet_best_acc, acc)

    fnet_time = time.time() - fnet_start
    fnet_params = sum(p.numel() for p in fnet_model.parameters())

    # ===== MLP BASELINE =====
    print("\n" + "=" * 60)
    print("TRAINING MLP BASELINE")
    print("=" * 60)

    mlp_model = SimpleMLP(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_labels=data.num_labels,
        dropout=0.1,
    ).to(device)

    mlp_optimizer = torch.optim.AdamW(
        mlp_model.parameters(), lr=args.lr, weight_decay=0.01
    )
    mlp_criterion = nn.CrossEntropyLoss()

    mlp_start = time.time()
    mlp_best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            mlp_model, data.train_loader, mlp_optimizer, mlp_criterion, device
        )
        acc, f1 = evaluate(mlp_model, data.val_loader, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}"
        )
        mlp_best_acc = max(mlp_best_acc, acc)

    mlp_time = time.time() - mlp_start
    mlp_params = sum(p.numel() for p in mlp_model.parameters())

    # ===== RESULTS =====
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'FNet':<20} {'MLP':<20}")
    print("-" * 60)
    print(f"{'Best Val Accuracy':<25} {fnet_best_acc:.4f}{'':<15} {mlp_best_acc:.4f}")
    print(f"{'Training Time (s)':<25} {fnet_time:.2f}{'':<16} {mlp_time:.2f}")
    print(f"{'Model Parameters':<25} {fnet_params:<20} {mlp_params}")
    print(
        f"{'Params per Accuracy':<25} {fnet_params/fnet_best_acc:<20.0f} {mlp_params/mlp_best_acc:.0f}"
    )
    print("=" * 60)

    # Winner
    if fnet_best_acc > mlp_best_acc:
        print(f"\n✓ FNet wins! +{(fnet_best_acc - mlp_best_acc)*100:.2f}% accuracy")
    elif mlp_best_acc > fnet_best_acc:
        print(f"\n✓ MLP wins! +{(mlp_best_acc - fnet_best_acc)*100:.2f}% accuracy")
    else:
        print("\n✓ Tie! Both models achieved the same accuracy")


if __name__ == "__main__":
    benchmark()
