import torch
import numpy as np
from typing import Optional
from ..base_engine import BaseEngine


class GaussianEngine(BaseEngine):
    """
    Gaussian - Frequency-Weighted Structured Noise
    High-frequency Gaussian noise weighted by DCT importance map.
    Humans are less sensitive to high-frequency perturbations
    so we concentrate noise there for best invisibility.
    No gradient needed - purely noise based.
    """
    name = "gaussian"

    def __init__(self, epsilon: float = 8/255,
                 high_freq_boost: float = 2.0, surrogate=None):
        super().__init__(epsilon=epsilon, surrogate=surrogate)
        self.high_freq_boost = high_freq_boost

    def _frequency_mask(self, h: int, w: int) -> np.ndarray:
        """
        Returns HxW mask where high-frequency areas have higher values.
        Based on distance from DC component of 2D DFT.
        """
        fy       = np.fft.fftfreq(h)[:, None]
        fx       = np.fft.fftfreq(w)[None, :]
        freq_dist = np.sqrt(fy ** 2 + fx ** 2)
        mask     = 1 + self.high_freq_boost * (freq_dist / (freq_dist.max() + 1e-8))
        return mask.astype(np.float32)

    def _perturb(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        _, h, w   = x.shape
        freq_mask = self._frequency_mask(h, w)
        x_np      = x.cpu().numpy()
        noise     = np.random.randn(*x_np.shape).astype(np.float32)

        # Apply frequency mask to each channel
        for c in range(x_np.shape[0]):
            noise_fft  = np.fft.fft2(noise[c])
            noise_fft *= freq_mask
            noise[c]   = np.fft.ifft2(noise_fft).real

        # Normalise noise to epsilon-ball
        max_abs = np.abs(noise).max() + 1e-8
        noise  *= (self.epsilon / max_abs)

        x_adv = np.clip(x_np + noise, 0, 1)
        return torch.from_numpy(x_adv).to(self.surrogate.device)
