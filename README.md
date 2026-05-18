# ImageEcho

> Adversarial Machine Learning through invisible pixel perturbation.

10 attack engines using real neural network gradients (ResNet50).
Completely invisible to humans — devastating to classifiers.

## Quick Start

```python
from imageecho.engines import MiFgsmEngine
from imageecho.context import EchoContext

ctx = EchoContext(MiFgsmEngine(epsilon=8/255))
adv_image, report = ctx.run("photo.png", save_to="adversarial.png")
print(report)
```

## Engines

| Engine | Method | Strength |
|--------|--------|----------|
| FGSM | Fast Gradient Sign | ★★★ |
| PGD | Projected Gradient Descent | ★★★★ |
| MI-FGSM | Momentum Iterative | ★★★★★ |
| DI-FGSM | Diverse Input | ★★★★★ |
| AutoPGD | Adaptive Step | ★★★★ |
| C&W | Carlini-Wagner L2 | ★★★★ |
| DeepFool | Minimal Perturbation | ★★★ |
| Patch | Adversarial Patch | ★★★★ |
| Gaussian | Frequency-Weighted Noise | ★★ |
| JSMA | Jacobian Saliency Map | ★★★ |

Built with Python 3.11 · PyTorch · PyQt6
