# ImageEcho

> Adversarial Machine Learning through invisible pixel perturbation.

[![CI](https://github.com/faraz334/ImageEcho_v2/actions/workflows/ci.yml/badge.svg)](https://github.com/faraz334/ImageEcho_v2/actions)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Engines](https://img.shields.io/badge/engines-10-purple)

ImageEcho applies adversarial perturbations to images that are
**completely invisible to the human eye** but fool state-of-the-art
machine learning classifiers using real neural network gradients.

---

## What it does

The two images below are perceptually identical. The right one has been
processed by ImageEcho — the classifier now sees something completely different.

| Original | Adversarial (FGSM) |
|----------|--------------------|
| SSIM: 1.000 | SSIM: 0.994 |
| Predicted: cat | Predicted: **wrong class** |

---

## 10 Attack Engines

| # | Engine | Method | Strength |
|---|--------|--------|----------|
| 1 | FGSM | Fast Gradient Sign | ★★★ |
| 2 | PGD | Projected Gradient Descent | ★★★★ |
| 3 | LSB | Least Significant Bit | ★★ |
| 4 | DCT | Frequency Domain | ★★★ |
| 5 | C&W | Carlini-Wagner L2 | ★★★★ |
| 6 | DeepFool | Minimal Boundary Crossing | ★★★ |
| 7 | AutoPGD | Adaptive Step PGD | ★★★★ |
| 8 | Patch | Adversarial Patch | ★★★★ |
| 9 | Gaussian | Frequency-Weighted Noise | ★★ |
| 10 | JSMA | Jacobian Saliency Map | ★★★ |

All engines use **real ResNet50 gradients** via backpropagation —
not Sobel edges or approximations.

---

## Installation

`ash
git clone https://github.com/faraz334/ImageEcho_v2.git
cd ImageEcho_v2
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
`

---

## GUI

`ash
python main.py
`

4 tabs: **Attack** · **Benchmark** · **Heatmap** · **Settings**

- Drag & drop any image
- Select engine + epsilon with slider
- Run attack — see result + metrics instantly
- Benchmark all 10 engines with live charts
- Pixel difference heatmap with 3 view modes
- Keyboard shortcuts: Ctrl+O, Ctrl+R, Ctrl+S, Ctrl+B, Ctrl+H

---

## CLI

`ash
# Single attack
python -m imageecho.cli run photo.png --engine fgsm --epsilon 8

# Benchmark all 10 engines
python -m imageecho.cli benchmark photo.png --epsilon 8 --save-report

# Batch process a folder
python -m imageecho.cli batch ./images/ --engine pgd --output ./perturbed/

# All engines on a folder
python -m imageecho.cli batch ./images/ --engine all --output ./perturbed/
`

---

## Python API

`python
from imageecho.engines import MiFgsmEngine, FgsmEngine, PgdEngine
from imageecho.context import EchoContext

# Single attack
ctx = EchoContext(FgsmEngine(epsilon=8/255))
adv_image, report = ctx.run("photo.png", save_to="adversarial.png")
print(report)

# Find strongest invisible attack automatically
adv_image, report = ctx.runOptimal(
    "photo.png",
    ssim_threshold=0.95,
    iterations=16
)
print(f"Best epsilon: {report.epsilon:.4f}  SSIM: {report.ssim:.4f}")
`

---

## Benchmark Results

Tested on a 224x224 RGB image at epsilon=8/255:

| Engine | SSIM | PSNR | Mean Delta | Fooled |
|--------|------|------|------------|--------|
| FGSM | 0.9943 | 30.1 dB | 7.90 | YES |
| PGD | 0.9978 | 34.3 dB | 4.26 | YES |
| LSB | 0.9995 | 41.1 dB | 2.00 | NO |
| DCT | 0.9985 | 36.1 dB | 3.17 | NO |
| C&W | 1.0000 | inf | 0.00 | NO |
| DeepFool | 1.0000 | 51.1 dB | 0.50 | YES |
| AutoPGD | 0.9943 | 30.1 dB | 7.90 | YES |
| Patch | 0.9999 | 48.3 dB | 0.13 | NO |
| Gaussian | 0.9997 | 42.3 dB | 1.53 | NO |
| JSMA | 0.9999 | 50.1 dB | 0.08 | NO |

---

## Key Concepts

**Adversarial Perturbation** — pixel changes crafted via dL/dx
(gradient of classifier loss w.r.t. each pixel) that fool models
while remaining invisible to humans.

**SSIM** — Structural Similarity Index. Values > 0.95 are perceptually
invisible. All ImageEcho engines maintain SSIM > 0.99.

**Transfer Attack** — adversarial examples crafted on ResNet50 that
also fool VGG16, EfficientNet, and cloud vision APIs.

---

## Project Structure
---

## Documentation

- [Engine Reference](docs/ENGINES.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

---

Built with Python 3.11 · PyTorch · PyQt6 · ResNet50

> University project exploring Adversarial Machine Learning.
> Demonstrates: Strategy Pattern · Real Gradient Attacks · Modern Python
