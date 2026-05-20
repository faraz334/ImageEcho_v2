import torch
from typing import Optional
from ..base_engine import BaseEngine


class FgsmEngine(BaseEngine):
    """
    FGSM - Fast Gradient Sign Method
    One step: x_adv = x + epsilon * sign(grad)
    Fast but weaker than iterative methods.
    """
    name = "fgsm"

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        grad = self.surrogate.get_gradients(
            x,
            target_class=target_class,
            targeted=target_class is not None
        )
        return x + self.epsilon * grad.sign()
