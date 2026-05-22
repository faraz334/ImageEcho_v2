import torch
from typing import Optional
from ..base_engine import BaseEngine


class JsmaEngine(BaseEngine):
    """
    JSMA - Jacobian Saliency Map Attack
    Selects TOP-K most salient pixels (highest dL/dx magnitude)
    and perturbs only those, maximally, in gradient direction.
    Produces sparse targeted pixel changes - very few pixels altered
    but each changed by the maximum amount.
    """
    name = "jsma"

    def __init__(self, epsilon: float = 8/255,
                 max_pixels: float = 0.01, surrogate=None):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.max_pixels = max_pixels  # fraction of total pixels to perturb

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        device  = self.surrogate.device
        _, h, w = x.shape

        # Get predicted class for untargeted attack
        pred = self.surrogate.predict(x)
        t    = target_class if target_class is not None else pred.class_id

        print(f"  [JSMA] computing saliency map for class {t}...")

        # Get per-pixel saliency HxW
        saliency = self.surrogate.get_jacobian(x.to(device), target_class=t)

        # Number of pixels to perturb
        k = max(1, int(h * w * self.max_pixels))
        print(f"  [JSMA] perturbing top {k} of {h*w} pixels "
              f"({self.max_pixels*100:.1f}%)")

        # Select top-k most salient pixel locations
        flat_saliency = saliency.flatten()
        top_k_indices = torch.topk(flat_saliency, k).indices

        # Get gradient direction for those pixels
        grad  = self.surrogate.get_gradients(
            x.to(device),
            target_class=t,
            targeted=target_class is not None
        )

        x_adv = x.clone().to(device)

        # Perturb only selected pixels by full epsilon in gradient direction
        for flat_idx in top_k_indices:
            row = int(flat_idx) // w
            col = int(flat_idx) % w
            x_adv[:, row, col] += self.epsilon * grad[:, row, col].sign()

        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv.cpu()
