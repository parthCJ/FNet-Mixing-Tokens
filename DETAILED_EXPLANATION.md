# FNet-Mixing: Detailed Concept and Implementation Explanation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Concepts](#core-concepts)
3. [Architecture Details](#architecture-details)
4. [Implementation Breakdown](#implementation-breakdown)
5. [Training Pipeline](#training-pipeline)
6. [Data Processing](#data-processing)
7. [Key Innovations](#key-innovations)
8. [Performance Analysis](#performance-analysis)

---

## Project Overview

This project implements **FNet**, a neural architecture that replaces the computationally expensive self-attention mechanism in Transformers with a simple and efficient **Fourier Transform** for token mixing. The implementation focuses on text classification using the AG News dataset.

### Research Paper
FNet: Mixing Tokens with Fourier Transforms (https://arxiv.org/pdf/2105.03824)

### Key Idea
Instead of using O(n²) self-attention to mix tokens, FNet uses O(n log n) Fast Fourier Transform (FFT), making it significantly faster while maintaining competitive accuracy.

---

## Core Concepts

### 1. **Token Mixing**

#### Concept
Token mixing refers to how information flows between different positions (tokens) in a sequence. In natural language, understanding context requires tokens to "communicate" with each other.

#### Traditional Approach: Self-Attention
```
For each token, compute attention weights to all other tokens
Time Complexity: O(n²) where n = sequence length
Memory: O(n²)
```

#### FNet Approach: Fourier Transform
```
Apply 2D Fast Fourier Transform to the entire sequence
Time Complexity: O(n log n)
Memory: O(n)
```

**Why it works:**
- Fourier transforms capture global patterns in the sequence
- The transform naturally mixes information across all positions
- Real part of FFT provides sufficient signal for downstream tasks

---

### 2. **Fast Fourier Transform (FFT)**

#### Mathematical Foundation
The Fourier Transform decomposes a signal into its frequency components:

```
X_fourier = FFT(X_input)
```

For sequences, this reveals:
- **Low frequencies**: Capture slow-varying patterns (overall sentence structure)
- **High frequencies**: Capture rapid changes (local token interactions)

#### 2D FFT in FNet
```python
mixed = torch.fft.fft2(hidden_states, dim=(1, 2)).real
```

**Dimensions:**
- `dim=1`: Sequence length dimension (token position)
- `dim=2`: Hidden dimension (feature space)

**Why 2D?**
- Mixes both across token positions AND feature dimensions
- Creates rich interactions between spatial and feature patterns
- Takes only the `.real` part to keep outputs real-valued

---

### 3. **Residual Connections**

#### Concept
Add the input back to the output of a layer:
```
output = input + transformation(input)
```

#### Benefits
- **Gradient flow**: Allows gradients to flow directly backward
- **Information preservation**: Original signal isn't lost
- **Training stability**: Easier to optimize deep networks

#### In FNet
```python
# After Fourier mixing
hidden_states = self.norm1(hidden_states + mixed)

# After feedforward network
hidden_states = self.norm2(hidden_states + self.ffn(hidden_states))
```

Two residual connections per layer, just like in Transformers.

---

### 4. **Layer Normalization**

#### Concept
Normalize features across the hidden dimension for each sample:
```
normalized = (x - mean) / sqrt(variance + epsilon)
```

#### Purpose
- **Stabilizes training**: Keeps activations in a reasonable range
- **Faster convergence**: Reduces internal covariate shift
- **Better gradient flow**: Prevents vanishing/exploding gradients

#### Placement in FNet
Applied **after** adding residual connections (Post-LN):
```python
self.norm1 = nn.LayerNorm(hidden_size)
hidden_states = self.norm1(hidden_states + mixed)
```

---

### 5. **Position-wise Feed-Forward Network (FFN)**

#### Concept
A two-layer MLP applied independently to each position:

```python
FFN(x) = GELU(Linear1(x))
output = Linear2(FFN_intermediate)
```

#### Architecture
```
Input (hidden_size=256)
   ↓
Linear1 → intermediate_size=512
   ↓
GELU activation
   ↓
Dropout
   ↓
Linear2 → hidden_size=256
   ↓
Dropout
```

#### Purpose
- **Non-linear transformations**: Adds expressiveness after linear mixing
- **Feature refinement**: Processes mixed tokens independently
- **Expansion & compression**: Expands to larger dim, then compresses back

---

### 6. **GELU Activation**

#### Concept
Gaussian Error Linear Unit - a smooth, probabilistic activation:

```
GELU(x) = x * Φ(x)
```
where Φ(x) is the cumulative distribution function of standard normal distribution.

#### Advantages over ReLU
- **Smooth gradients**: No hard cutoff at zero
- **Better for Transformers**: Empirically works better in language models
- **Stochastic regularization**: Implicitly provides regularization

---

### 7. **Embeddings**

#### Word Embeddings
Convert token IDs to dense vectors:
```python
self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
# Example: token_id=5 → [0.23, -0.45, 0.12, ..., 0.67] (256-dim vector)
```

#### Position Embeddings
Encode the position of each token:
```python
self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
# position=0 → [0.11, 0.22, ...], position=1 → [-0.05, 0.33, ...]
```

#### Combined Embeddings
```python
hidden_states = word_embeddings(input_ids) + position_embeddings(position_ids)
```

This gives each token:
1. **Semantic information** (from word embedding)
2. **Positional information** (from position embedding)

---

### 8. **Mean Pooling**

#### Concept
Aggregate sequence representations into a fixed-size vector by averaging:

```python
mask = attention_mask.unsqueeze(-1).float()  # Shape: (batch, seq_len, 1)
pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
```

#### Why Masking?
- Ignore padding tokens (don't count them in the average)
- Only average over real tokens using `attention_mask`

#### Output
Single vector per sequence: `(batch_size, hidden_size)` → ready for classification

---

## Architecture Details

### Overall FNet Architecture

```
Input Text: "Stock market rises sharply"
         ↓
    [Tokenization]
         ↓
    Token IDs: [101, 2005, 2003, 7693, 102]
         ↓
┌────────────────────────────────────────┐
│  EMBEDDINGS LAYER                      │
│  • Word Embeddings (vocab → hidden)    │
│  • Position Embeddings (pos → hidden)  │
│  • Sum both + Dropout                  │
└────────────────────────────────────────┘
         ↓
    Hidden States: (batch, seq_len, hidden_size)
         ↓
┌────────────────────────────────────────┐
│  FNET ENCODER (4 layers)               │
│                                        │
│  Layer 1:                              │
│    • 2D FFT Mixing                     │
│    • Residual + LayerNorm              │
│    • Feed-Forward Network              │
│    • Residual + LayerNorm              │
│  Layer 2-4: Same structure             │
└────────────────────────────────────────┘
         ↓
    Encoded: (batch, seq_len, hidden_size)
         ↓
┌────────────────────────────────────────┐
│  POOLING                               │
│  Mean pooling with attention mask      │
└────────────────────────────────────────┘
         ↓
    Pooled: (batch, hidden_size)
         ↓
┌────────────────────────────────────────┐
│  CLASSIFICATION HEAD                   │
│  Linear(hidden_size → num_labels)      │
└────────────────────────────────────────┘
         ↓
    Logits: (batch, num_labels=4)
         ↓
    Predictions: [World=0, Sports=1, Business=2, Sci/Tech=3]
```

---

## Implementation Breakdown

### 1. **FNetConfig** (Configuration Class)

```python
@dataclass
class FNetConfig:
    vocab_size: int                    # Size of vocabulary
    max_position_embeddings: int = 256 # Max sequence length
    hidden_size: int = 256             # Embedding dimension
    intermediate_size: int = 512       # FFN intermediate dimension
    num_layers: int = 4                # Number of FNet layers
    num_labels: int = 4                # Number of output classes
    dropout: float = 0.1               # Dropout probability
    pad_token_id: int = 0              # ID for padding token
```

**Purpose:** Centralized configuration for model hyperparameters.

---

### 2. **FNetMixingLayer** (Core Building Block)

```python
class FNetMixingLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, hidden_states):
        # Fourier mixing
        mixed = torch.fft.fft2(hidden_states, dim=(1, 2)).real
        hidden_states = self.norm1(hidden_states + mixed)
        
        # Feed-forward with residual
        hidden_states = self.norm2(hidden_states + self.ffn(hidden_states))
        return hidden_states
```

#### Step-by-step Forward Pass

**Input:** `hidden_states` with shape `(batch=8, seq_len=128, hidden=256)`

1. **Fourier Mixing:**
   ```python
   mixed = torch.fft.fft2(hidden_states, dim=(1, 2)).real
   ```
   - Apply 2D FFT across sequence and feature dimensions
   - Extract real part: `mixed` shape = `(8, 128, 256)`
   - This mixes information globally across all tokens

2. **First Residual + Norm:**
   ```python
   hidden_states = self.norm1(hidden_states + mixed)
   ```
   - Add original input to mixed (residual)
   - Normalize: stabilizes training

3. **Feed-Forward Network:**
   ```python
   ffn_out = self.ffn(hidden_states)
   ```
   - `Linear1`: `(8, 128, 256) → (8, 128, 512)`
   - `GELU`: Non-linear activation
   - `Dropout`: Regularization
   - `Linear2`: `(8, 128, 512) → (8, 128, 256)`
   - `Dropout`: More regularization

4. **Second Residual + Norm:**
   ```python
   hidden_states = self.norm2(hidden_states + ffn_out)
   ```
   - Add input to FFN output (second residual)
   - Normalize again

**Output:** `(8, 128, 256)` - same shape as input

---

### 3. **FNetEncoder** (Stacking Multiple Layers)

```python
class FNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            FNetMixingLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
```

**Purpose:** 
- Stack multiple FNet layers (typically 4-12)
- Each layer refines token representations
- Deeper layers capture more abstract patterns

---

### 4. **FNetForSequenceClassification** (Complete Model)

```python
class FNetForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Encoder
        self.encoder = FNetEncoder(config)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```

#### Forward Pass Breakdown

```python
def forward(self, input_ids, attention_mask):
    batch_size, seq_len = input_ids.shape
    
    # 1. Create position IDs
    position_ids = torch.arange(seq_len, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
    # Shape: (8, 128)
    
    # 2. Embed tokens
    word_embeds = self.word_embeddings(input_ids)      # (8, 128, 256)
    pos_embeds = self.position_embeddings(position_ids) # (8, 128, 256)
    hidden_states = word_embeds + pos_embeds            # (8, 128, 256)
    hidden_states = self.embed_dropout(hidden_states)
    
    # 3. Encode through FNet layers
    hidden_states = self.encoder(hidden_states)  # (8, 128, 256)
    
    # 4. Pool sequence to single vector
    mask = attention_mask.unsqueeze(-1).float()  # (8, 128, 1)
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
    # Shape: (8, 256)
    
    # 5. Classify
    logits = self.classifier(pooled)  # (8, 4)
    return logits
```

---

## Training Pipeline

### 1. **Data Loading and Preprocessing**

```python
def build_ag_news_dataloaders(tokenizer_name, max_length, ...):
    # Load AG News dataset (4 classes: World, Sports, Business, Sci/Tech)
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Subset for faster training
    train_ds = dataset["train"].select(range(train_subset))
    val_ds = dataset["test"].select(range(val_subset))
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, 
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, 
                            collate_fn=collate)
```

#### Collate Function (Batch Processing)

```python
def _collate_batch(tokenizer, max_length):
    def collate(examples):
        texts = [ex["text"] for ex in examples]
        labels = [ex["label"] for ex in examples]
        
        # Tokenize with padding and truncation
        encoded = tokenizer(
            texts,
            padding=True,        # Pad to longest in batch
            truncation=True,     # Truncate to max_length
            max_length=max_length,
            return_tensors="pt"
        )
        encoded["labels"] = torch.tensor(labels)
        return encoded
    return collate
```

**Output format:**
```python
{
    "input_ids": tensor([[101, 2023, 2003, ..., 0, 0],      # Token IDs
                         [101, 1996, 3023, ..., 102, 0]]),
    "attention_mask": tensor([[1, 1, 1, ..., 0, 0],         # 1=real, 0=padding
                              [1, 1, 1, ..., 1, 0]]),
    "labels": tensor([2, 1])                                 # Class labels
}
```

---

### 2. **Training Loop**

```python
def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="train"):
        # Move to device (GPU/CPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        total_loss += loss.item() * input_ids.size(0)
    
    return total_loss / len(loader.dataset)
```

#### Key Components

**Loss Function:** CrossEntropyLoss
```python
criterion = nn.CrossEntropyLoss()
# Combines LogSoftmax + NLLLoss
# Input: logits (8, 4), labels (8,)
# Output: scalar loss
```

**Optimizer:** AdamW
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,           # Learning rate
    weight_decay=0.01  # L2 regularization
)
```

---

### 3. **Evaluation**

```python
@torch.no_grad()  # Disable gradient computation
def evaluate(model, loader, device):
    model.eval()  # Set to evaluation mode (disables dropout)
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass only
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1)  # Get predicted class
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1
```

#### Metrics

**Accuracy:** Percentage of correct predictions
```
accuracy = (correct_predictions) / (total_predictions)
```

**Macro F1 Score:** Average F1 across all classes
```
F1 = 2 * (precision * recall) / (precision + recall)
Macro F1 = (F1_class0 + F1_class1 + F1_class2 + F1_class3) / 4
```

---

### 4. **Checkpointing**

```python
# Save best model based on F1 score
if f1 > best_f1:
    best_f1 = f1
    checkpoint = {
        "model_state": model.state_dict(),      # Model weights
        "config": asdict(config),               # Hyperparameters
        "tokenizer": args.tokenizer,            # Tokenizer name
        "metrics": {"val_acc": acc, "val_f1": f1}
    }
    torch.save(checkpoint, args.save_path)
```

---

## Data Processing

### AG News Dataset

**Structure:**
- **Classes:** 4 (World, Sports, Business, Science/Technology)
- **Train:** 120,000 samples
- **Test:** 7,600 samples

**Sample:**
```python
{
    "text": "Stock market rises sharply after Fed announcement",
    "label": 2  # Business
}
```

### Tokenization Process

**Input text:**
```
"Stock market rises sharply"
```

**After tokenization:**
```python
{
    "input_ids": [101, 4518, 2003, 7693, 19420, 102],
    "attention_mask": [1, 1, 1, 1, 1, 1]
}
```

Where:
- `101`: [CLS] token (start of sequence)
- `4518, 2003, 7693, 19420`: Word tokens
- `102`: [SEP] token (end of sequence)

---

## Key Innovations

### 1. **Replacing Self-Attention with FFT**

**Traditional Transformer:**
```python
# O(n²) complexity
attention_scores = Q @ K.T / sqrt(d_k)  # (seq_len, seq_len)
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

**FNet:**
```python
# O(n log n) complexity
output = torch.fft.fft2(input, dim=(1, 2)).real
```

**Speedup:** ~7x faster for long sequences (n=512+)

---

### 2. **Simplicity**

FNet has **no learnable parameters** in the mixing operation. The FFT is a fixed mathematical transformation, making the model:
- Easier to understand
- Faster to train
- More memory-efficient
- Less prone to overfitting

---

### 3. **Global Receptive Field**

Unlike convolutions (local) or sparse attention patterns, FFT naturally:
- Processes all tokens simultaneously
- Captures both local and global dependencies
- Requires no attention masks or positional biases

---

## Performance Analysis

### Benchmark Results (5K train, 1K val)

| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| **FNet** | 79.96% | 502.90s | 8.9M |
| **MLP Baseline** | 80.64% | 29.24s | 8.08M |

### Analysis

**Why MLP wins on small data:**
- AG News classification doesn't require complex token interactions
- Bag-of-words averaging is sufficient for topic classification
- Small dataset → simpler model generalizes better

**When FNet shines:**
- Longer sequences (512+ tokens)
- Tasks requiring understanding of word order and dependencies
- Larger datasets where patterns matter more

**FNet's advantages:**
- Much faster than Transformers (O(n log n) vs O(n²))
- Competitive accuracy on many NLP tasks
- Lower memory footprint
- Better scaling to long sequences

---

## Testing

### Unit Tests

**Test 1: Forward Pass Shape**
```python
def test_fnet_forward_shape():
    config = FNetConfig(vocab_size=1000, ...)
    model = FNetForSequenceClassification(config)
    
    input_ids = torch.randint(0, 999, (8, 32))
    attention_mask = torch.ones(8, 32)
    
    logits = model(input_ids, attention_mask)
    assert logits.shape == (8, 4)  # Batch × num_labels
```

**Test 2: Backward Pass**
```python
def test_fnet_backward_pass():
    model = FNetForSequenceClassification(config)
    
    logits = model(input_ids, attention_mask)
    loss = CrossEntropyLoss()(logits, labels)
    loss.backward()
    
    # Check gradients exist
    assert any(p.grad is not None for p in model.parameters())
```

---

## Summary

### What Makes FNet Special?

1. **Efficiency:** Replaces O(n²) attention with O(n log n) FFT
2. **Simplicity:** No learned parameters in mixing layer
3. **Effectiveness:** Competitive accuracy on many NLP tasks
4. **Scalability:** Better for long sequences than Transformers

### Key Takeaways

- **FFT provides a mathematical alternative to self-attention**
- **Residual connections + LayerNorm = stable training**
- **Position embeddings are crucial** (FFT doesn't encode order by itself)
- **Mean pooling effectively aggregates sequence information**
- **Simple baselines can outperform on easy tasks** (AG News)

### When to Use FNet

✅ **Good for:**
- Long sequences (512+ tokens)
- Tasks where speed matters
- Limited computational resources
- Research into alternative architectures

❌ **Not ideal for:**
- Very small datasets
- Tasks requiring fine-grained attention patterns
- When absolute SOTA accuracy is required

---

## Further Reading

- **Original Paper:** FNet: Mixing Tokens with Fourier Transforms (https://arxiv.org/abs/2105.03824)
- **Transformers:** "Attention is All You Need" (Vaswani et al., 2017)
- **BERT:** Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- **Fourier Analysis:** Understanding signal processing fundamentals

---

*This implementation demonstrates core ML engineering skills: architecture design, efficient data pipelines, proper training loops, evaluation metrics, and reproducible experimentation.*
