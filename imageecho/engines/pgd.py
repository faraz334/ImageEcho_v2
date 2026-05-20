import torch
from typing import Optional
from ..base_engine import BaseEngine


class PgdEngine(BaseEngine):
    """
    PGD - Projected Gradient Descent
    Multi-step FGSM with random start inside epsilon-ball.
    Much stronger than one-step FGSM.
    """
    name = "pgd"

    def __init__(self, epsilon: float = 8/255, steps: int = 40,
                 step_size: Optional[float] = None, surrogate=None):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.steps = steps
        self.step_size = step_size or (2.5 * epsilon / steps)

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)

        for _ in range(self.steps):
            grad = self.surrogate.get_gradients(
                x_adv,
                target_class=target_class,
                targeted=target_class is not None
            )

            x_adv = x_adv + self.step_size * grad.sign()
            x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv
