import torch
import torch.optim as optim
import numpy as np
from typing import Optional
from ..base_engine import BaseEngine


class CwEngine(BaseEngine):
    """
    C&W - Carlini-Wagner L2 Attack
    Optimization-based. Minimises L2 distance while maximising misclassification.
    Produces the tightest, most invisible perturbations of any method.
    Uses Adam optimizer - slower but highest quality results.
    """
    name = "cw"

    def __init__(self, epsilon: float = 8/255, steps: int = 200,
                 lr: float = 0.01, c: float = 1.0, kappa: float = 0.0,
                 surrogate=None):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.steps = steps
        self.lr    = lr
        self.c     = c        # trade-off: L2 distance vs misclassification
        self.kappa = kappa    # confidence margin

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        device = self.surrogate.device

        # Work in tanh-space to enforce [0,1] box constraint naturally
        # x = (tanh(w) + 1) / 2  =>  w = arctanh(2x - 1)
        x_np   = x.cpu().numpy()
        w_init = np.arctanh(np.clip(2 * x_np - 1, -0.9999, 0.9999))
        w      = torch.tensor(w_init, dtype=torch.float32,
                              device=device, requires_grad=True)

        optimizer = optim.Adam([w], lr=self.lr)
        best_adv  = x.clone()
        best_l2   = float("inf")

        for step in range(self.steps):
            x_adv  = (torch.tanh(w) + 1) / 2
            delta  = x_adv - x.to(device)
            l2     = (delta ** 2).sum().sqrt()

            logits = self.surrogate.model(
                self.surrogate._normalize(x_adv.unsqueeze(0))
            )

            if target_class is None:
                # Untargeted: push away from predicted class
                true_cls   = logits.argmax().item()
                true_logit = logits[0, true_cls]
                others     = torch.cat([logits[0, :true_cls],
                                        logits[0, true_cls + 1:]])
                f_loss     = torch.clamp(true_logit - others.max() + self.kappa, min=0)
            else:
                # Targeted: pull toward target class
                tgt_logit  = logits[0, target_class]
                others     = torch.cat([logits[0, :target_class],
                                        logits[0, target_class + 1:]])
                f_loss     = torch.clamp(others.max() - tgt_logit + self.kappa, min=0)

            loss = l2 + self.c * f_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep best solution: lowest L2 that fools the model
            if float(l2) < best_l2 and float(f_loss) < 1e-4:
                best_l2  = float(l2)
                best_adv = x_adv.detach().clone()

            if step % 50 == 0:
                print(f"  [C&W] step {step}/{self.steps}  "
                      f"l2={float(l2):.4f}  f={float(f_loss):.4f}")

        # Clamp to epsilon-ball as safety measure
        best_adv = torch.clamp(
            best_adv,
            x.to(device) - self.epsilon,
            x.to(device) + self.epsilon
        )
        return best_adv.cpu()
