# CIFAR-100 Image Classification

A PyTorch implementation of an image classification pipeline for the **CIFAR-100** dataset. The repository provides a modular project structure with separate components for model definition, configuration, data loading, and training, making it easy to experiment with different architectures and training strategies. Some sections are intentionally left as placeholders for implementation. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Project Structure

```text
.
├── data
│   └── dataloaders.py      # Dataset loading and preprocessing
│
├── model
│   ├── config.py           # Model and training configuration
│   └── model.py            # Model implementation
│
├── train
│   └── train_model.py      # Training pipeline
├── .python-version
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Features

- Modular project organization
- Configuration-based hyperparameter management
- CIFAR-100 data pipeline
- Training and validation workflow
- Easily extensible model architecture
- Clean separation between data, model, and training logic

---

## Dataset

This project uses the **CIFAR-100** dataset.

- **60,000** RGB images
- **32 × 32** image resolution
- **100** object categories
- **50,000** training images
- **10,000** test images

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd CIFAR100-Image-Classification
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

or with **uv**:

```bash
uv sync
```

---