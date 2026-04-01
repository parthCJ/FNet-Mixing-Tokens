# FNet Mixing (PyTorch)

A resume-ready implementation of **FNet** style token mixing for text classification.

This project replaces self-attention with Fourier mixing:

$$
X_{mix} = \Re\left(\text{FFT2}(X)\right)
$$

Then applies a position-wise feed-forward network with residual connections and layer norm.

## What this project demonstrates

- End-to-end implementation of FNet mixing blocks in PyTorch
- Efficient sequence classification pipeline on AG News
- Hugging Face `datasets` + `transformers` integration
- Reproducible training and evaluation setup with accuracy + macro-F1
- Unit tests for model forward and backward behavior

## Project structure

```text
.
├── src/fnet_mixing/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
├── scripts/
│   └── train_ag_news.py
├── tests/
│   └── test_fnet_model.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Train

```bash
python scripts/train_ag_news.py \
  --epochs 3 \
  --hidden-size 256 \
  --num-layers 4 \
  --max-length 128 \
  --train-subset 20000 \
  --val-subset 5000
```

Checkpoint is saved at `checkpoints/fnet_ag_news.pt` with model weights, config, tokenizer name, and best validation metrics.

## Benchmark: FNet vs MLP Baseline

Compare FNet against a simple MLP baseline:

```bash
python scripts/benchmark.py --epochs 3 --train-subset 5000 --val-subset 1000
```

**Results on AG News (5K train, 1K val):**

| Metric            | FNet   | MLP          |
| ----------------- | ------ | ------------ |
| Best Val Accuracy | 79.96% | **80.64%** ✓ |
| Training Time (s) | 502.90 | 29.24        |
| Model Parameters  | 8.9M   | 8.08M        |

**Finding:** While MLP achieves slightly higher accuracy on this small dataset, FNet is competitive and demonstrates efficient Fourier-based token mixing. FNet's strength emerges on longer sequences where attention patterns matter more.

## Run tests

```bash
pytest -q
```
