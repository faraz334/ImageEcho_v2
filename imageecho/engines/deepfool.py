import torch
from typing import Optional
from ..base_engine import BaseEngine


class DeepFoolEngine(BaseEngine):
    """
    DeepFool Attack
    Finds the MINIMAL perturbation needed to cross the decision boundary.
    Uses linearised boundary approximation - very efficient.
    Produces the smallest possible perturbations that still fool the model.
    """
    name = "deepfool"

    def __init__(self, epsilon: float = 8/255, steps: int = 50,
                 overshoot: float = 0.02, num_classes: int = 10,
                 surrogate=None):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.steps       = steps
        self.overshoot   = overshoot
        self.num_classes = num_classes  # only consider top-k classes for speed

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        device = self.surrogate.device
        x_adv  = x.clone().to(device)

        # Get original predicted class
        with torch.no_grad():
            logits0    = self.surrogate.model(
                self.surrogate._normalize(x_adv.unsqueeze(0))
            )
            orig_class = int(logits0.argmax())

        print(f"  [DeepFool] original class: {orig_class}, "
              f"running up to {self.steps} steps...")

        for step in range(self.steps):
            x_leaf = x_adv.clone().unsqueeze(0).requires_grad_(True)
            logits = self.surrogate.model(
                self.surrogate._normalize(x_leaf)
            )

            # Stop if we already crossed the boundary
            if int(logits.argmax()) != orig_class:
                print(f"  [DeepFool] boundary crossed at step {step}")
                break

            # Gradient of original class logit
            self.surrogate.model.zero_grad()
            logits[0, orig_class].backward(retain_graph=True)
            grad_orig = x_leaf.grad.squeeze(0).detach().clone()

            # Find nearest decision boundary across top-k classes
            pert_min = float("inf")
            r_opt    = None

            top_k = torch.topk(logits.detach(), self.num_classes, dim=1).indices[0]

            for k in top_k:
                k = int(k)
                if k == orig_class:
                    continue

                x_leaf2 = x_adv.clone().unsqueeze(0).requires_grad_(True)
                logits2  = self.surrogate.model(
                    self.surrogate._normalize(x_leaf2)
                )
                self.surrogate.model.zero_grad()
                logits2[0, k].backward()
                grad_k = x_leaf2.grad.squeeze(0).detach()

                w_k    = grad_k - grad_orig
                f_k    = float((logits2[0, k] - logits2[0, orig_class]).detach())
                pert_k = abs(f_k) / (w_k.norm() + 1e-8)

                if float(pert_k) < pert_min:
                    pert_min = float(pert_k)
                    r_opt    = pert_k * w_k / (w_k.norm() + 1e-8)

            if r_opt is None:
                break

            # Step toward nearest boundary with overshoot
            x_adv = x_adv + (1 + self.overshoot) * r_opt
            x_adv = torch.clamp(x_adv, x.to(device) - self.epsilon,
                                 x.to(device) + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.cpu()
