"""
FNet Inference Script - Test trained model on custom text

WORKFLOW:
    STEP 1: Train the model FIRST
        C:/Users/CJ/.conda/envs/mlenv/python.exe scripts/train_ag_news.py --epochs 3 --hidden-size 256 --num-layers 4 --max-length 128 --train-subset 20000 --val-subset 5000
        (takes 20-30 minutes, creates checkpoints/fnet_ag_news.pt)

    STEP 2: After training is done, run inference (FAST)
        C:/Users/CJ/.conda/envs/mlenv/python.exe scripts/inference.py --text "Your news text here" --device cuda
        (outputs: predicted class + confidence)
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from fnet_mixing.model import FNetConfig, FNetForSequenceClassification


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load model and tokenizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = FNetConfig(**checkpoint["config"])
    model = FNetForSequenceClassification(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["tokenizer"])

    return model, tokenizer, checkpoint["metrics"]


def predict(
    text: str, model, tokenizer, device: str = "cpu", max_length: int = 128
) -> dict:
    """Predict class for given text."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )
        probs = torch.softmax(logits, dim=-1)
        pred_class = logits.argmax(dim=-1).item()
        pred_conf = probs[0, pred_class].item()

    return {
        "class": pred_class,
        "confidence": pred_conf,
        "logits": logits[0].cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FNet inference on custom text")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fnet_ag_news.pt")
    parser.add_argument("--text", type=str, required=True, help="News text to classify")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    classes = ["World", "Sports", "Business", "Sci/Tech"]

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, tokenizer, metrics = load_checkpoint(args.checkpoint, device=args.device)
    print(f"Model metrics: {metrics}\n")

    print(f"Input text: {args.text}")
    result = predict(
        args.text, model, tokenizer, device=args.device, max_length=args.max_length
    )

    print(f"Predicted class: {classes[result['class']]}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Logits: {[f'{x:.2f}' for x in result['logits']]}")


if __name__ == "__main__":
    main()
