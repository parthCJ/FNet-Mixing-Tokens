from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FNetConfig:
    vocab_size: int
    max_position_embeddings: int = 256
    hidden_size: int = 256
    intermediate_size: int = 512
    num_layers: int = 4
    num_labels: int = 4
    dropout: float = 0.1
    pad_token_id: int = 0


class FNetMixingLayer(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mixed = torch.fft.fft2(hidden_states, dim=(1, 2)).real
        hidden_states = self.norm1(hidden_states + mixed)
        hidden_states = self.norm2(hidden_states + self.ffn(hidden_states))
        return hidden_states


class FNetEncoder(nn.Module):
    def __init__(self, config: FNetConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FNetMixingLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class FNetForSequenceClassification(nn.Module):
    def __init__(self, config: FNetConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        self.encoder = FNetEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        hidden_states = self.word_embeddings(input_ids) + self.position_embeddings(
            position_ids
        )
        hidden_states = self.embed_dropout(hidden_states)

        hidden_states = self.encoder(hidden_states)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        logits = self.classifier(pooled)
        return logits
