from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from .surrogate import SurrogateModel


@dataclass
class PerturbationReport:
    engine_name: str
    epsilon: float
    ssim: float
    psnr: float
    mean_delta: float
    max_delta: float
    pixels_altered: int
    original_class: int
    original_confidence: float
    perturbed_class: int
    perturbed_confidence: float
    fooled: bool

    def __str__(self):
        status = "FOOLED" if self.fooled else "same class"
        return (
            f"[{self.engine_name}] eps={self.epsilon:.4f} "
            f"SSIM={self.ssim:.4f} PSNR={self.psnr:.1f}dB "
            f"meanDelta={self.mean_delta:.2f} -> {status} "
            f"({self.original_class}->{self.perturbed_class})"
        )

    def as_dict(self):
        return self.__dict__


class BaseEngine(ABC):
    name: str = "base"

    def __init__(
        self, epsilon: float = 8 / 255, surrogate: Optional[SurrogateModel] = None
    ):
        self.epsilon = epsilon
        self.surrogate = surrogate or SurrogateModel()

    @abstractmethod
    def _perturb(self, x: torch.Tensor, target_class=None) -> torch.Tensor:
        pass

    def apply(self, image: np.ndarray, target_class=None):
        x = self.surrogate.image_to_tensor(image).to(self.surrogate.device)
        orig = self.surrogate.predict(x)
        x_adv = torch.clamp(self._perturb(x.clone(), target_class), 0.0, 1.0)
        adv_image = self.surrogate.tensor_to_image(x_adv)
        adv_pred = self.surrogate.predict(x_adv)
        delta = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
        return adv_image, PerturbationReport(
            engine_name=self.name,
            epsilon=self.epsilon,
            ssim=compute_ssim(image, adv_image,
                              channel_axis=2, data_range=255),
            psnr=compute_psnr(image, adv_image, data_range=255),
            mean_delta=float(delta.mean()),
            max_delta=float(delta.max()),
            pixels_altered=int((delta.sum(axis=2) > 0).sum()),
            original_class=orig.class_id,
            original_confidence=orig.confidence,
            perturbed_class=adv_pred.class_id,
            perturbed_confidence=adv_pred.confidence,
            fooled=(orig.class_id != adv_pred.class_id),
        )
