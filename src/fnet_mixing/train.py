from __future__ import annotations
import argparse
from dataclasses import asdict

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm.auto import tqdm

from fnet_mixing.data import build_ag_news_dataloaders
from fnet_mixing.model import FNetConfig, FNetForSequenceClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FNet on AG News")
    parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--val-subset", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save-path", type=str, default="checkpoints/fnet_ag_news.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data = build_ag_news_dataloaders(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
    )

    config = FNetConfig(
        vocab_size=data.tokenizer.vocab_size,
        max_position_embeddings=args.max_length,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_labels=data.num_labels,
        dropout=args.dropout,
        pad_token_id=data.tokenizer.pad_token_id or 0,
    )

    device = torch.device(args.device)
    model = FNetForSequenceClassification(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, data.train_loader, optimizer, criterion, device)
        acc, f1 = evaluate(model, data.val_loader, device)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            checkpoint = {
                "model_state": model.state_dict(),
                "config": asdict(config),
                "tokenizer": args.tokenizer,
                "metrics": {"val_acc": acc, "val_f1": f1},
            }
            __import__("pathlib").Path(args.save_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            torch.save(checkpoint, args.save_path)
            print(f"Saved new best checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
