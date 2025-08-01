# GPT-2 Training from Scratch

An implementation of GPT-2 training from scratch for educational purpose only. This also implements several optimization steps to speed up the training.

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- tiktoken (for tokenization)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/davidnvq/gpt2-training
cd gpt2-training
```

2. Install dependencies:
```bash
pip install torch tiktoken matplotlib
```

## 🏃‍♂️ Quick Start

### Basic Training

Run the baseline training script:

```bash
python train_baseline.py
```

This will:
- Train a GPT-2 model (124M parameters) on the provided text data
- Display real-time training metrics
- Save the trained model as `model.pth`

### Optimized Training

Run the fully optimized training script (includes all optimizations from steps 1-8):

```bash
python train_optimized.py
```

This achieves **48,025 tokens/sec** with all optimizations:
- bfloat16 precision
- FlashAttention
- torch.compile
- Fused AdamW
- Tensor cores
- And more...

### Distributed Training (DDP)

Run distributed training across multiple GPUs:

```bash
# For 2 GPUs
torchrun --nproc_per_node=2 train_optimized_ddp.py

# For 4 GPUs
torchrun --nproc_per_node=4 train_optimized_ddp.py
```

This achieves **78,475 tokens/sec** with 2 GPUs and scales linearly with more GPUs.

### Training Configuration

The training parameters can be modified in `src/configs.py`:

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False

@dataclass
class TrainingArguments:
    num_epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    eval_freq: int = 15
    eval_iter: int = 1
```

## 🏗️ Model Architecture

The implementation follows the GPT-2 architecture:

- **Multi-Head Self-Attention**: 12 attention heads with causal masking
- **Feed-Forward Networks**: 4x expansion ratio with GELU activation
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Residual Connections**: Around both attention and feed-forward blocks
- **Position Embeddings**: Learned positional encodings
- **Token Embeddings**: 50,257 vocabulary size (GPT-2 tokenizer)

Total Parameters are ~124M.


## 📊 Performance

### Training Speed and Memory Usage

| Step Number | # GPUs used | Improvement | Avg tok/sec | Reserved memory | Note |
|-------------|-------------|-------------|-------------|-----------------|------|
| 0 | 1 GPU | Baseline | 12,548 | 22.14 GB | Initial implementation |
| 1 | 1 GPU | Causal mask on-the-fly | 12,671 | 22.12 GB | Dynamic mask creation reduces memory |
| 2 | 1 GPU | Tensor cores | 16,243 | 22.12 GB | Enable Tensor cores (Ampere GPU optimization) |
| 3 | 1 GPU | Fused AdamW | 16,505 | 22.12 GB | Enable Fused AdamW optimizer |
| 4 | 1 GPU | Pinned memory | 16,583 | 22.12 GB | Pre-allocate and re-use GPU memory |
| 5 | 1 GPU | bfloat16 precision | 29,970 | 11.58 GB | 16-bit brain float precision |
| 6 | 1 GPU | FlashAttention v2 | 38,173 | 5.94 GB | Optimized attention implementation ([FlashAttn2 is used](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)) |
| 7 | 1 GPU | torch.compile | 43,879 | 5.95 GB | PyTorch compilation optimization |
| 8 | 1 GPU | Increased batch size | 48,025 | 11.45 GB | Maximum GPU utilization |
| 9 | 2 GPUs | DDP on 2 GPUs | 78,475 | 11.77 GB | Distributed training across 2 GPUs |

## 📁 Project Structure

```
gpt2-training/
├── baseline_training.py      # Main training script
├── ddp_training.py          # Distributed training (WIP)
├── optimized_training.py    # Optimized training (WIP)
├── src/
│   ├── __init__.py
│   ├── configs.py           # Model and training configurations
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # GPT-2 model implementation
│   └── utils.py             # Training utilities and functions
├── pg145.txt               # Training data (Project Gutenberg text)
└── README.md
```

## 🔧 Usage Examples

### Custom Training

```python
from src.configs import GPTConfig, TrainingArguments
from src.model import GPTModel
from src.data import create_dataloader_v1
from src.utils import train_model_simple_with_timing

# Configure model and training
config = GPTConfig()
training_args = TrainingArguments()

# Create model and data loaders
model = GPTModel(config)
train_loader, eval_loader = create_dataloader_v1(
    batch_size=training_args.batch_size,
    max_length=config.context_length
)

# Train the model
train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
    model=model,
    train_loader=train_loader,
    eval_loader=eval_loader,
    num_epochs=training_args.num_epochs
)
```

### Text Generation

```python
from src.model import generate_and_print_sample

# Generate text with the trained model
generate_and_print_sample(
    model=model,
    tokenizer=tokenizer,
    device=device,
    start_context="The future of artificial intelligence"
)
```

## 📈 Training Monitoring

The training script provides real-time monitoring of:

- **Training Loss**: Cross-entropy loss on training data
- **Validation Loss**: Loss on validation set
- **Training Speed**: Tokens processed per second
- **Memory Usage**: GPU memory allocation and reservation
- **Text Samples**: Generated text samples after each epoch


## 🙏 Acknowledgments
This repository re-implement the original work from [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) and builds upon the original GPT-2 architecture.

