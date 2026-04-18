# Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Paper**: Ioffe & Szegedy, _Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_, ICML 2015  
**Course**: ECE 447 — Data Analysis and Machine Learning for Engineers, Winter 2026  
**Authors**: Hetang Mehta, Nathan Krosel, Maanas Saxena, Nathan Wong  
**Institution**: University of Alberta

---

## Overview

This notebook reproduces two central experimental findings from the Batch Normalization paper in simplified settings:

1. **Experiment 1** — BN accelerates convergence and improves accuracy in a sigmoid FC network on MNIST (reproduces Figure 1a of the paper).
2. **Experiment 2** — BN enables stable training at high learning rates in a CNN on CIFAR-10 (reproduces the qualitative claim from Section 3.3 of the paper).

Each experiment includes a validation/ablation study to confirm the findings hold beyond the paper's exact setup.

---

## Requirements

All dependencies are installed automatically in the first notebook cell via pip. No manual setup is required.

```
torch
torchvision
matplotlib
numpy
```

> The notebook is self-contained and downloads MNIST and CIFAR-10 automatically via `torchvision.datasets`. Datasets are saved to a local `data/` directory on first run.

---

## Hardware

All experiments were developed and tested on **Apple Silicon (MPS backend)**. The notebook auto-detects the available device:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

| Backend             | Supported                                 |
| ------------------- | ----------------------------------------- |
| Apple Silicon (MPS) | ✅ Primary target                         |
| NVIDIA GPU (CUDA)   | ✅ Will use automatically if available    |
| CPU                 | ✅ Fallback — slower but fully functional |

**Estimated runtimes (Apple Silicon / Colab GPU):**

- Experiment 1 (two 50K-step runs): ~15–20 minutes per run
- Experiment 2 (6 LR × BN combinations, 15 epochs each): ~30–45 minutes total

The notebook is fully runnable on **Google Colab** with no modification.

---

## Reproducibility

A fixed random seed (`42`) is applied to Python, NumPy, and PyTorch at the start of each training run via `setSeed()`. Results are deterministic under this configuration.

> Note: Minor variation may occur across different hardware backends (e.g., MPS vs CUDA) due to non-deterministic floating-point operations. The qualitative conclusions are robust to this.

---

## Notebook Structure

```
ECE447_BatchNormalization_Reproductions.ipynb
│
├── Dependencies          — pip install + imports
├── CONFIG                — All hyperparameters in one place (EXP1_PARAMS, EXP2_PARAMS)
├── Data Loading          — get_dataset() for MNIST and CIFAR-10
├── Model Definitions     — FC_Model (Exp 1), CNN_Model (Exp 2)
│
├── Experiment 1: MNIST FC + Sigmoid
│   ├── Part A — Reproduce Figure 1(a): Test accuracy vs. training steps
│   └── Part B — Ablation: hidden layer width ∈ {50, 100, 200}
│
├── Experiment 2: CIFAR-10 CNN, LR Sensitivity
│   ├── Part A — LR sweep ∈ {0.001, 0.01, 0.1} with SGD + momentum
│   └── Part B — Optimizer ablation: SGD vs Adam at LR = 0.1
│
└── Summary of Findings   — Results tables and comparison to paper
```

---

## Experiment Details

### Experiment 1 — MNIST Fully Connected Network

Reproduces **Section 4.1 / Figure 1(a)** of the paper.

| Hyperparameter    | Value                                                 |
| ----------------- | ----------------------------------------------------- |
| Dataset           | MNIST (50,000 train / 10,000 test)                    |
| Architecture      | 3-layer FC, 100 units each, sigmoid activations       |
| BN placement      | `FC → BN → Sigmoid` (pre-activation, per Section 3.2) |
| Bias on FC layers | `False` when BN is used (BN's β subsumes the bias)    |
| Optimizer         | SGD (no momentum)                                     |
| Learning rate     | 0.01                                                  |
| Batch size        | 60                                                    |
| Training steps    | 50,000                                                |
| Random seed       | 42                                                    |
| Checkpoints       | ~20 (every 2,500 steps)                               |

**Part B ablation** — varies hidden units over `[50, 100, 200]`, all else fixed.

**Output figures:**

- `exp1_figure1a_reproduction.png` — Test accuracy vs. training steps
- `exp1_partb_width_ablation.png` — Bar chart of final accuracy across widths

---

### Experiment 2 — CIFAR-10 CNN, Learning Rate Sensitivity

Tests the qualitative claim from **Section 3.3** of the paper in a simplified setting.

| Hyperparameter      | Value                                                       |
| ------------------- | ----------------------------------------------------------- |
| Dataset             | CIFAR-10 (40,000 train / 10,000 test)                       |
| Architecture        | 6-conv CNN (3→32→32→64→64→128→128 channels), GAP, FC output |
| BN placement        | `Conv → BN → ReLU` (pre-activation, per Section 3.2)        |
| Bias on Conv layers | `False` when BN is used                                     |
| Optimizer (Part A)  | SGD + momentum (0.9)                                        |
| Learning rates      | {0.001, 0.01, 0.1}                                          |
| Epochs              | 15                                                          |
| Batch size          | 64                                                          |
| Random seed         | 42                                                          |

> SGD is deliberately used over Adam in Part A because its non-adaptive updates make divergence at high LR directly observable — Adam's adaptive step sizes would mask the instability.

**Part B ablation** — fixes LR = 0.1, compares SGD + momentum vs. Adam, both with and without BN.

**Output figures:**

- `exp2a_lr_sensitivity.png` — 3×2 grid: training loss / train accuracy / test accuracy over epochs, for each LR
- `exp2b_optimizer_ablation.png` — 3×2 grid: same metrics for SGD vs. Adam at LR = 0.1

---

## Key Results

### Experiment 1

| Hidden Units | Without BN | With BN | Improvement |
| ------------ | ---------- | ------- | ----------- |
| 50           | 73.2%      | 97.3%   | +24.1 pp    |
| 100          | 79.6%      | 97.5%   | +17.9 pp    |
| 200          | 83.7%      | 97.8%   | +14.1 pp    |

BN reaches ~97–98% accuracy within ~5K steps. Non-BN remains near chance for most of training.

### Experiment 2

| Learning Rate | Without BN      | With BN | Outcome                        |
| ------------- | --------------- | ------- | ------------------------------ |
| 0.001         | ~30%            | ~75–80% | BN converges faster and higher |
| 0.01          | ~70–75%         | ~78–80% | Both train; BN more stable     |
| 0.1           | ~10% (diverges) | ~80%    | Non-BN fails; BN trains stably |

At LR = 0.1, non-BN loss becomes NaN within the first few epochs. Both SGD and Adam without BN diverge at this rate; both converge with BN.

---

## Correspondence to the Paper

| Experiment                          | Paper Section                                          | Reproduced?                                    |
| ----------------------------------- | ------------------------------------------------------ | ---------------------------------------------- |
| MNIST FC + sigmoid accuracy curve   | Section 4.1, Figure 1(a)                               | ✅ Qualitatively matched                       |
| Hidden width robustness             | Not in paper (original ablation)                       | ✅ New validation                              |
| LR sensitivity (high LR divergence) | Section 3.3 (claim); Section 4.2.1 (ImageNet evidence) | ✅ Qualitatively matched in simplified setting |
| Optimizer ablation                  | Not in paper (original ablation)                       | ✅ New validation                              |

> The paper's primary LR sensitivity evidence is demonstrated on large-scale Inception/ImageNet networks. This notebook reproduces the qualitative phenomenon in a simplified CIFAR-10 setting, which is sufficient per the project scope.

---

## File Structure After Running

```
.
├── ECE447_BatchNormalization_Reproductions.ipynb
├── README.md
├── data/                                      # Auto-created by torchvision
│   ├── MNIST/
│   └── cifar-10-batches-py/
├── exp1_figure1a_reproduction.png
├── exp1_partb_width_ablation.png
├── exp2a_lr_sensitivity.png
└── exp2b_optimizer_ablation.png
```

---

## Citation

```
Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift. Proceedings of the 32nd
International Conference on Machine Learning (ICML), JMLR: W&CP volume 37.
```
