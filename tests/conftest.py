import numpy as np
import pytest


@pytest.fixture(scope="session")
def small_image():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def medium_image():
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
