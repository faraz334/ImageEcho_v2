import torch
import numpy as np
from scipy.fft import dct, idct
from typing import Optional
from ..base_engine import BaseEngine


class DctEngine(BaseEngine):
    """
    DCT - Discrete Cosine Transform attack
    Injects noise into high-frequency DCT coefficients of 8x8 blocks.
    High-frequency changes are invisible to humans but disrupt CNNs.
    """
    name = "dct"

    def _perturb_channel(self, channel: np.ndarray) -> np.ndarray:
        h, w    = channel.shape
        result  = channel.copy()
        eps_val = self.epsilon * 255.0

        for i in range(0, h - 7, 8):
            for j in range(0, w - 7, 8):
                block        = channel[i:i+8, j:j+8].astype(np.float32)
                coeffs       = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                noise        = np.random.uniform(-eps_val, eps_val, (8, 8))
                # Only perturb high-frequency coefficients (bottom-right quadrant)
                noise[:4, :4] = 0
                coeffs      += noise
                block_back   = idct(idct(coeffs, axis=1, norm='ortho'), axis=0, norm='ortho')
                result[i:i+8, j:j+8] = np.clip(block_back, 0, 255)

        return result

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        img_np  = (x.cpu().numpy() * 255.0)   # CxHxW float
        out     = np.zeros_like(img_np)

        for c in range(img_np.shape[0]):
            out[c] = self._perturb_channel(img_np[c])

        return torch.from_numpy(out / 255.0).float().to(self.surrogate.device)
