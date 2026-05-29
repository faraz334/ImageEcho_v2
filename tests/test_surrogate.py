import torch
import numpy as np
import pytest
from imageecho.surrogate import SurrogateModel


@pytest.fixture(scope="module")
def surrogate():
    return SurrogateModel("resnet50")


def test_surrogate_loads(surrogate):
    assert surrogate.model is not None


def test_image_to_tensor(surrogate):
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    t = surrogate.image_to_tensor(img)
    assert t.shape == (3, 64, 64)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0
    assert t.max() <= 1.0


def test_tensor_to_image(surrogate):
    t = torch.rand(3, 64, 64)
    img = surrogate.tensor_to_image(t)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() <= 255


def test_predict_returns_prediction(surrogate):
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    t = surrogate.image_to_tensor(img).to(surrogate.device)
    p = surrogate.predict(t)
    assert 0 <= p.class_id <= 999
    assert 0.0 <= p.confidence <= 1.0


def test_gradients_shape(surrogate):
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    t = surrogate.image_to_tensor(img).to(surrogate.device)
    grad = surrogate.get_gradients(t)
    assert grad.shape == t.shape


def test_gradients_not_zero(surrogate):
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    t = surrogate.image_to_tensor(img).to(surrogate.device)
    grad = surrogate.get_gradients(t)
    assert grad.abs().sum() > 0
