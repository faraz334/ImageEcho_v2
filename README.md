````md
<div align="center">

# ImageEcho

### Adversarial Machine Learning — Invisible Pixel Attacks

*I built a tool that can fool image classifiers using changes the human eye cannot see.*

[![CI](https://github.com/faraz334/ImageEcho_v2/actions/workflows/ci.yml/badge.svg)](https://github.com/faraz334/ImageEcho_v2/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange?logo=pytorch)
![Engines](https://img.shields.io/badge/Attack%20Engines-10-purple)
![Tests](https://img.shields.io/badge/Tests-48%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# What Is This?

Modern AI image classifiers — the same technology used in self-driving cars,
medical imaging, and security systems — can be fooled by tiny pixel-level
changes that are mathematically invisible to the human eye.

ImageEcho demonstrates this by implementing **10 adversarial attack engines**
using real neural network gradients through **ResNet50**.

The system computes:

- Which pixels to change
- How much to change them
- While keeping the image visually identical

The result:

- Humans still see the same image
- The AI predicts something completely different

---

# 👁️ The Core Phenomenon — See It In One Image

Can you spot the difference? The AI can — and it is completely fooled.

<div align="center">

| Original Photo | Adversarial Photo |
| :---: | :---: |
| <img src="assets/cat.png" width="300"> | <img src="assets/adversarial_cat.png" width="300"> |
| **Human:** Cat | **Human:** Cat |
| **AI:** Cat | **AI:** ❌ Wrong Class |
| **SSIM: 1.000** | **SSIM: 0.994 (Invisible)** |

</div>

> SSIM > 0.95 means the perturbation is nearly impossible for humans to notice.

---

# How It Works — Real Gradient-Based Attacks

Most beginner adversarial tools fake gradients using edge detection.

**ImageEcho uses real backpropagation through ResNet50.**

```text
Your image
    |
    v
ResNet50 forward pass
    |
    v
CrossEntropyLoss
    |
    v
dL/dx  (true gradient)
    |
    v
Attack engine crafts perturbation
    |
    v
Adversarial image
```

Because the perturbation is generated using real gradients,
attacks often transfer to other architectures such as:

- VGG16
- EfficientNet
- Cloud Vision APIs

---

# 10 Attack Engines

| Engine | Algorithm | Description |
|--------|-----------|-------------|
| **FGSM** | Fast Gradient Sign Method | One-step gradient attack |
| **PGD** | Projected Gradient Descent | Iterative white-box attack |
| **LSB** | Least Significant Bit | Bit-level perturbation |
| **DCT** | Discrete Cosine Transform | Frequency-domain attack |
| **C&W** | Carlini-Wagner L2 | Optimization-based attack |
| **DeepFool** | Boundary Crossing | Minimal-distance perturbation |
| **AutoPGD** | Adaptive PGD | Automatic step-size optimization |
| **Patch** | Adversarial Patch | Localized square perturbation |
| **Gaussian** | Frequency-Weighted Noise | Human-vision-aware noise |
| **JSMA** | Jacobian Saliency Map | Sparse pixel attack |

---

# Benchmark Results

Tested on a 224×1200 RGB image with epsilon = 8/255.

| Engine | SSIM | PSNR | Pixels Changed | Result |
|--------|------|------|----------------|--------|
| FGSM | 0.9943 | 30.1 dB | 50,176 | ✅ FOOLED |
| PGD | 0.9978 | 34.3 dB | 50,165 | ✅ FOOLED |
| LSB | 0.9995 | 41.1 dB | 50,176 | ❌ same |
| DCT | 0.9985 | 36.1 dB | 50,127 | ❌ same |
| C&W | 1.0000 | inf | 0 | ❌ same |
| DeepFool | 1.0000 | 51.1 dB | 37,681 | ✅ FOOLED |
| AutoPGD | 0.9943 | 30.1 dB | 50,176 | ✅ FOOLED |
| Patch | 0.9999 | 48.3 dB | 1,088 | ❌ same |
| Gaussian | 0.9997 | 42.3 dB | 49,754 | ❌ same |
| JSMA | 0.9999 | 50.1 dB | 501 | ❌ same |

> SSIM > 0.95 means the perturbation remains visually invisible.

---

# Installation

```bash
git clone https://github.com/faraz334/ImageEcho_v2.git
cd ImageEcho_v2

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

# Three Ways To Use ImageEcho

## 1. GUI — Point and Click

```bash
python main.py
```

Features:

- Drag-and-drop image loading
- Engine selection
- Epsilon slider
- Live metrics
- Benchmark tab
- Heatmap visualization
- Real-time adversarial generation

---

## 2. CLI — Terminal Interface

### Attack a single image

```bash
python -m imageecho.cli run photo.png --engine fgsm --epsilon 8
```

### Benchmark all engines

```bash
python -m imageecho.cli benchmark photo.png --epsilon 8 --save-report
```

### Batch process a folder

```bash
python -m imageecho.cli batch ./images/ --engine pgd --output ./results/
```

### Run all engines

```bash
python -m imageecho.cli batch ./images/ --engine all --output ./results/
```

---

## 3. Python API

```python
from imageecho.engines import FgsmEngine
from imageecho.context import EchoContext

ctx = EchoContext(FgsmEngine(epsilon=8/255))

adversarial_image, report = ctx.run(
    "photo.png",
    save_to="result.png"
)

print(report.ssim)
print(report.fooled)
print(report.original_class)
print(report.perturbed_class)
```

---

# GUI Screenshots

## Attack Interface

<p align="center">
  <img src="assets/gui_attack.png" width="900">
</p>

---

## Benchmark Dashboard

<p align="center">
  <img src="assets/gui_benchmark.png" width="900">
</p>

---

# Project Structure

```text
ImageEcho_v2/
├── imageecho/
│   ├── engines/
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   ├── cw.py
│   │   ├── deepfool.py
│   │   ├── autopgd.py
│   │   ├── patch.py
│   │   ├── jsma.py
│   │   ├── gaussian.py
│   │   ├── lsb.py
│   │   └── dct.py
│   │
│   ├── surrogate.py
│   ├── base_engine.py
│   ├── context.py
│   └── cli.py
│
├── gui/
│   ├── main_window.py
│   ├── benchmark_panel.py
│   ├── heatmap_panel.py
│   └── settings_panel.py
│
├── tests/
│   ├── test_engines.py
│   ├── test_context.py
│   └── test_surrogate.py
│
├── docs/
│   ├── ENGINES.md
│   ├── ARCHITECTURE.md
│   └── CONTRIBUTING.md
│
├── assets/
│   ├── cat.png
│   ├── adversarial_cat.png
│   ├── gui_attack.png
│   └── gui_benchmark.png
│
├── main.py
└── CHANGELOG.md
```

---

# Design Patterns Used

| Pattern | Purpose |
|---------|---------|
| **Strategy Pattern** | Swap attack engines dynamically |
| **Template Method** | Shared attack pipeline |
| **Factory Pattern** | Clean engine construction |
| **Value Object** | Immutable perturbation reports |

---

# Technologies

| Technology | Purpose |
|------------|---------|
| PyTorch 2.12 | Neural network gradients |
| ResNet50 | Surrogate classifier |
| PyQt6 | GUI framework |
| Matplotlib | Charts and visualizations |
| scikit-image | SSIM / PSNR metrics |
| Rich | CLI formatting |
| pytest | Automated testing |
| GitHub Actions | Continuous integration |

---

# Documentation

- `docs/ENGINES.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTRIBUTING.md`
- `CHANGELOG.md`

---

# Academic Context

This project implements techniques from major adversarial ML papers:

- FGSM — Goodfellow et al. (2015)
- PGD — Madry et al. (2018)
- Carlini-Wagner — Carlini & Wagner (2017)
- DeepFool — Moosavi-Dezfooli et al. (2016)
- AutoPGD — Croce & Hein (2020)
- JSMA — Papernot et al. (2016)

---

# Future Improvements

- Black-box attacks
- Universal perturbations
- ONNX/TensorRT acceleration
- Defense benchmarking
- Transferability analysis
- Real-time webcam attacks

---

<div align="center">

## Built By Faraz

Python 3.11 · PyTorch · PyQt6

*Exploring the robustness limits of modern computer vision systems.*

</div>
````
