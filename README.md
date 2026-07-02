# CIFAR-100 Image Classification

A PyTorch implementation of an image classification pipeline for the **CIFAR-100** dataset. The repository provides a modular project structure with separate components for model definition, configuration, data loading, and training, making it easy to experiment with different architectures and training strategies. Some sections are intentionally left as placeholders for implementation. 

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

## Contribution Workflow

1. **Fork** this repository.

2. **Clone** your fork.

```bash
git clone https://github.com/<username>/CIFAR100-Image-Classification.git
cd CIFAR100-Image-Classification
```

3. **Open** the project.

```bash
code .
```

4. **Install** the dependencies.

```bash
uv sync
```

or

```bash
pip install -r requirements.txt
```

5. **Create a branch.**

```bash
git checkout -b feature/<feature-name>
```

6. **Implement** one or more missing components.

7. **Commit and push.**

```bash
git add .
git commit -m "Implement <component>"
git push origin feature/<feature-name>
```

8. **Open a Pull Request** to this repository.