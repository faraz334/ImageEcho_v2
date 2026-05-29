<div align="center">

# ImageEcho 🌊
### Adversarial Machine Learning — Invisible Pixel Attacks

*A production-grade toolkit that exposes critical structural vulnerabilities in modern AI vision models using targeted gradient perturbations.*

[![CI](https://github.com/faraz334/ImageEcho_v2/actions/workflows/ci.yml/badge.svg)](https://github.com/faraz334/ImageEcho_v2/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange?logo=pytorch)
![Engines](https://img.shields.io/badge/Attack%20Engines-10-purple)
![Tests](https://img.shields.io/badge/Tests-48%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 👁️ Core Idea: Fooling the AI Classifier

Can you spot the difference? The human eye cannot tell them apart, but the target deep vision network is completely fooled.

| Original Image | Adversarial Output (ImageEcho) |
| :---: | :---: |
| <img src="assets/cat.png" width="340" alt="Original Image"> | <img src="assets/adversarial_cat.png" width="340" alt="Adversarial Image"> |
| **Human Eye:** Cat (100%) <br> **ResNet50 AI:** Cat (True Positive) | **Human Eye:** Cat (100%) <br> **ResNet50 AI:** ❌ **Toaster / Misclassified** |

> **Mathematical Integrity Profile:** When optimized surgically through boundary attacks like DeepFool, the image matrix maintains a pristine **SSIM of 1.0000** and a stellar **PSNR of 51.1 dB**, keeping changes imperceptible to a human reviewer while fully breaking model confidence.

---

## 🧠 System Architecture & Core Science

Using real neural network gradients via backpropagation through ResNet50, ImageEcho computes exactly which pixels to change to maximize classification error while managing empirical distortion budgets.

[ Input Image (x) ] ──> [ ResNet50 Forward Pass ] ──> [ Target Loss (CrossEntropy) ]
        ▲                                                          │
        │                                                          ▼
[ Perturbed Output ] ◄── [ Attack Engine ] ◄── [ Exact Pixels Gradients (dL/dx) ]
                            (FGSM/PGD/C&W)          (Backpropagation)

---

## 📊 Comprehensive Cross-Engine Benchmarks

*Below is the empirical baseline data matrix matching your terminal generation outputs:*

| Engine | SSIM | PSNR (dB) | Mean $\Delta$ | Fooled Status |
| :--- | :---: | :---: | :---: | :---: |
| **deepfool** | 1.0000 | 51.1 dB | 0.50 | 🟢 **YES (Optimal)** |
| **pgd** | 0.9977 | 34.2 dB | 4.27 | 🟢 **YES** |
| **fgsm** | 0.9943 | 30.1 dB | 7.90 | 🟢 **YES** |
| **auto_pgd** | 0.9943 | 30.1 dB | 7.89 | 🟢 **YES** |
| **lsb** | 0.9995 | 41.1 dB | 2.00 | 🔴 NO |
| **dct** | 0.9985 | 36.1 dB | 3.16 | 🔴 NO |
| **patch** | 0.9999 | 48.3 dB | 0.13 | 🔴 NO |
| **gaussian** | 0.9997 | 42.5 dB | 1.49 | 🔴 NO |
| **jsma** | 0.9999 | 50.1 dB | 0.08 | 🔴 NO |
| **cw** | 1.0000 | $\infty$ | 0.00 | 🔴 NO |

---

## 🛠️ Operational Interfaces

### 1. Interactive Desktop GUI Sandbox
<p align="center">
  <img src="assets/gui_attack.png" width="800" alt="ImageEcho Attack Workspace">
</p>

Review holistic system diagnostics instantly on the analytical benchmark panel:
<p align="center">
  <img src="assets/gui_benchmark.png" width="800" alt="ImageEcho Analytical Dashboard">
</p>

### 2. High-Throughput CLI Engine
```bash
# Run holistic cross-engine benchmark diagnostics on a target sample image
python -m imageecho.cli benchmark cat.png --epsilon 8 --save-report


📐 Object-Oriented Software Architecture
Strategy Pattern: EchoContext dynamically swaps backend attack engines seamlessly at runtime via a unified BaseEngine abstraction.

Template Method Pattern: The orchestration pipeline handles image loading globally, while child classes override only the specific mathematical _perturb() tensor method.