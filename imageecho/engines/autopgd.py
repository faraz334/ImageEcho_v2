import torch
import torch.nn as nn
from typing import Optional
from ..base_engine import BaseEngine


class AutoPgdEngine(BaseEngine):
    """
    AutoPGD - PGD with automatic adaptive step size
    Halves step when loss plateaus. Restarts from best found point.
    Better convergence than fixed-step PGD on most images.
    """

    name = "auto_pgd"

    def __init__(
        self,
        epsilon: float = 8 / 255,
        steps: int = 100,
        n_restarts: int = 1,
        surrogate=None,
    ):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.steps = steps
        self.n_restarts = n_restarts

    def _run_once(self, x: torch.Tensor, target_class: Optional[int]):
        device = self.surrogate.device
        step_size = 2 * self.epsilon
        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_best = x_adv.clone()
        best_loss = -float("inf")
        prev_loss = -float("inf")
        plateau = 0

        for i in range(self.steps):
            grad = self.surrogate.get_gradients(
                x_adv, target_class=target_class, targeted=target_class is not None
            )

            # Track current loss
            with torch.no_grad():
                x_in = x_adv.unsqueeze(0).to(device)
                logits = self.surrogate.model(self.surrogate._normalize(x_in))
                cls = torch.tensor(
                    [
                        (
                            target_class
                            if target_class is not None
                            else int(logits.argmax())
                        )
                    ],
                    device=device,
                )
                loss = float(nn.CrossEntropyLoss()(logits, cls))

            if loss > best_loss:
                best_loss = loss
                x_best = x_adv.clone()

            # Detect plateau and halve step size
            if loss <= prev_loss:
                plateau += 1
            else:
                plateau = 0

            if plateau >= 10:
                step_size = max(step_size / 2, self.epsilon / 100)
                plateau = 0
                if i % 20 == 0:
                    print(
                        f"  [AutoPGD] step {i}  step_size={step_size:.5f}  loss={loss:.4f}"
                    )

            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            prev_loss = loss

        return x_best, best_loss

    def _perturb(
        self, x: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        best_x, best_loss = None, -float("inf")

        for r in range(self.n_restarts):
            print(f"  [AutoPGD] restart {r+1}/{self.n_restarts}")
            x_cand, loss = self._run_once(x, target_class)
            if loss > best_loss:
                best_loss = loss
                best_x = x_cand

        return best_x
