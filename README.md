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

1. **Fork** this repository to your GitHub account.

2. **Clone your fork** to your local machine.

```bash
git clone https://github.com/<your-username>/CIFAR100-Image-Classification.git
cd CIFAR100-Image-Classification
```

3. **Open the project** in Visual Studio Code.

```bash
code .
```

4. **Create a virtual environment** (recommended).

Using `uv`:

```bash
uv sync
```

or using `pip`:

```bash
pip install -r requirements.txt
```

5. **Create a new branch** for your implementation.

```bash
git checkout -b feature/<your-feature-name>
```

6. Complete one or more of the placeholder implementations in the repository.

7. Commit your changes.

```bash
git add .
git commit -m "Implement model architecture"
```

8. Push the branch to your fork.

```bash
git push origin feature/<your-feature-name>
```

9. Open a **Pull Request** from your fork to this repository describing your implementation.