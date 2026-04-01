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

## Run tests

```bash
pytest -q
```

## Suggested resume bullets

- Implemented an FNet-based sequence classifier in PyTorch by replacing self-attention with 2D Fourier token mixing, reducing architectural complexity while preserving strong text classification performance.
- Built a reproducible NLP training pipeline on AG News with configurable model depth/width, Hugging Face data tooling, and macro-F1 evaluation.
- Added unit tests for forward and backward correctness and packaged the project with modular code organization for extensibility.

## Next improvements

- Add mixed precision (`torch.cuda.amp`) and gradient clipping
- Add learning-rate warmup + cosine decay
- Benchmark speed/accuracy against a similarly sized Transformer encoder
- Export to ONNX and add lightweight inference service
