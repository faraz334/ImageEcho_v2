import torch
import numpy as np
from typing import Optional
from ..base_engine import BaseEngine


class LsbEngine(BaseEngine):
    """
    LSB - Least Significant Bit flipping
    Flips the least significant bits of pixel values.
    No gradient needed - purely bit-level manipulation.
    Invisible but weaker against real classifiers.
    """

    name = "lsb"

    def __init__(self, epsilon: float = 8 / 255, bits: int = 2, surrogate=None):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.bits = bits

    def _perturb(
        self, x: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        x_uint8 = (x.cpu().numpy() * 255).astype(np.uint8)
        mask = (1 << self.bits) - 1
        flipped = x_uint8 ^ mask
        x_adv = torch.from_numpy(flipped.astype(np.float32) / 255.0)
        return x_adv.to(self.surrogate.device)
