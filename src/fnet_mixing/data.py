from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    tokenizer: AutoTokenizer
    num_labels: int


def _collate_batch(tokenizer: AutoTokenizer, max_length: int):
    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [example["text"] for example in examples]
        labels = [example["label"] for example in examples]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = __import__("torch").tensor(labels)
        return encoded

    return collate


def build_ag_news_dataloaders(
    tokenizer_name: str,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    train_subset: int | None = None,
    val_subset: int | None = None,
) -> DataBundle:
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = dataset["train"]
    val_ds = dataset["test"]

    if train_subset is not None:
        train_ds = train_ds.select(range(min(train_subset, len(train_ds))))
    if val_subset is not None:
        val_ds = val_ds.select(range(min(val_subset, len(val_ds))))

    collate = _collate_batch(tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        num_labels=4,
    )
