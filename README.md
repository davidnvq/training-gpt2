# GPT-2 Training from Scratch

An implementation of GPT-2 training from scratch. This also implements several optimization steps to speed up the training.

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
python baseline_training.py
```

This will:
- Train a GPT-2 model (124M parameters) on the provided text data
- Display real-time training metrics
- Save the trained model as `model.pth`


## 🙏 Acknowledgments
This repository re-implement the original work from [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) and builds upon the original GPT-2 architecture.

