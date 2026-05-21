import torch
from typing import Optional
from ..base_engine import BaseEngine


class PatchEngine(BaseEngine):
    """
    Adversarial Patch Attack
    All perturbation concentrated in one square region.
    Outside the patch: image is completely clean.
    High local perturbation inside patch only.
    Very effective in physical world attacks.
    """
    name = "patch"

    def __init__(self, epsilon: float = 8/255, patch_size: float = 0.15,
                 location: str = "center", steps: int = 40, surrogate=None):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.patch_size = patch_size   # fraction of image min-dimension
        self.location   = location     # "center" or "random"
        self.steps      = steps

    def _get_patch_coords(self, h: int, w: int):
        size = int(min(h, w) * self.patch_size)
        size = max(size, 8)  # minimum 8x8 patch

        if self.location == "random":
            r0 = int(torch.randint(0, max(1, h - size), (1,)))
            c0 = int(torch.randint(0, max(1, w - size), (1,)))
        else:
            r0 = (h - size) // 2
            c0 = (w - size) // 2

        return r0, c0, r0 + size, c0 + size

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        device    = self.surrogate.device
        c, h, w   = x.shape
        r0, c0, r1, c1 = self._get_patch_coords(h, w)

        print(f"  [Patch] patch region: rows {r0}-{r1}, cols {c0}-{c1} "
              f"({r1-r0}x{c1-c0} pixels)")

        x_adv     = x.clone().to(device)
        step_size = 2 * self.epsilon / self.steps

        for _ in range(self.steps):
            grad = self.surrogate.get_gradients(
                x_adv,
                target_class=target_class,
                targeted=target_class is not None
            )

            # Apply gradient step ONLY inside the patch
            patch_grad               = grad[:, r0:r1, c0:c1]
            x_adv[:, r0:r1, c0:c1] += step_size * patch_grad.sign()

            # Project patch pixels back into epsilon-ball
            orig_patch               = x[:, r0:r1, c0:c1].to(device)
            x_adv[:, r0:r1, c0:c1]  = torch.clamp(
                x_adv[:, r0:r1, c0:c1],
                orig_patch - self.epsilon,
                orig_patch + self.epsilon
            )
            x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.cpu()
