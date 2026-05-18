import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from dataclasses import dataclass

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


@dataclass
class Prediction:
    class_id: int
    confidence: float


class SurrogateModel:
    SUPPORTED = {
        "resnet50":        lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "vgg16":           lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    }

    def __init__(self, model_name: str = "resnet50"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SurrogateModel] Loading {model_name} on {self.device}...")
        self.model = self.SUPPORTED[model_name]().to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._loss_fn   = nn.CrossEntropyLoss()

    def get_gradients(self, x: torch.Tensor, target_class=None, targeted: bool = False) -> torch.Tensor:
        x_leaf = x.clone().detach().unsqueeze(0).requires_grad_(True).to(self.device)
        logits  = self.model(self._normalize(x_leaf))
        if target_class is None:
            cls = torch.tensor([logits.argmax().item()], device=self.device)
        else:
            cls = torch.tensor([target_class], device=self.device)
        loss = self._loss_fn(logits, cls)
        if targeted:
            loss = -loss
        self.model.zero_grad()
        loss.backward()
        return x_leaf.grad.squeeze(0).detach()

    def get_jacobian(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        x_leaf = x.clone().detach().unsqueeze(0).requires_grad_(True).to(self.device)
        logits  = self.model(self._normalize(x_leaf))
        logits[0, target_class].backward()
        grad     = x_leaf.grad.squeeze(0).detach()
        return grad.norm(dim=0)

    def predict(self, x: torch.Tensor) -> Prediction:
        with torch.no_grad():
            logits = self.model(self._normalize(x.unsqueeze(0).to(self.device)))
            probs  = torch.softmax(logits, dim=1)
        return Prediction(class_id=int(probs.argmax()), confidence=float(probs.max()))

    @staticmethod
    def image_to_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

    @staticmethod
    def tensor_to_image(t: torch.Tensor) -> np.ndarray:
        return np.clip(t.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
