import numpy as np
import pytest
from imageecho.engines import (
    FgsmEngine,
    PgdEngine,
    LsbEngine,
    DctEngine,
    CwEngine,
    DeepFoolEngine,
    AutoPgdEngine,
    PatchEngine,
    GaussianEngine,
    JsmaEngine,
)

# Small test image — fast to run
TEST_IMAGE = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
EPSILON = 8 / 255
SSIM_MIN = 0.85  # minimum acceptable SSIM


def _run_engine(engine_cls, **kwargs):
    engine = engine_cls(epsilon=EPSILON, **kwargs)
    adv, report = engine.apply(TEST_IMAGE)
    return adv, report


# ──────────────────────────────────────────────────────────────────────────
# Shape tests — output must match input shape
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_cls",
    [
        FgsmEngine,
        PgdEngine,
        LsbEngine,
        DctEngine,
        GaussianEngine,
    ],
)
def test_output_shape(engine_cls):
    adv, _ = _run_engine(engine_cls)
    assert adv.shape == TEST_IMAGE.shape, f"{engine_cls.__name__} output shape mismatch"


# ──────────────────────────────────────────────────────────────────────────
# Dtype tests — output must be uint8
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_cls",
    [
        FgsmEngine,
        PgdEngine,
        LsbEngine,
        DctEngine,
        GaussianEngine,
    ],
)
def test_output_dtype(engine_cls):
    adv, _ = _run_engine(engine_cls)
    assert adv.dtype == np.uint8, f"{engine_cls.__name__} output dtype should be uint8"


# ──────────────────────────────────────────────────────────────────────────
# Pixel range tests — all pixels must be in [0, 255]
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_cls",
    [
        FgsmEngine,
        PgdEngine,
        LsbEngine,
        DctEngine,
        GaussianEngine,
        PatchEngine,
    ],
)
def test_pixel_range(engine_cls):
    adv, _ = _run_engine(engine_cls)
    assert adv.min() >= 0, f"{engine_cls.__name__} has pixels < 0"
    assert adv.max() <= 255, f"{engine_cls.__name__} has pixels > 255"


# ──────────────────────────────────────────────────────────────────────────
# No NaN/Inf tests
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_cls",
    [
        FgsmEngine,
        PgdEngine,
        LsbEngine,
        DctEngine,
        GaussianEngine,
    ],
)
def test_no_nan_inf(engine_cls):
    adv, _ = _run_engine(engine_cls)
    arr = adv.astype(np.float32)
    assert not np.isnan(arr).any(), f"{engine_cls.__name__} has NaN values"
    assert not np.isinf(arr).any(), f"{engine_cls.__name__} has Inf values"


# ──────────────────────────────────────────────────────────────────────────
# SSIM tests — perturbation must stay above minimum threshold
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_cls",
    [
        FgsmEngine,
        PgdEngine,
        LsbEngine,
        DctEngine,
        GaussianEngine,
        PatchEngine,
    ],
)
def test_ssim_threshold(engine_cls):
    _, report = _run_engine(engine_cls)
    assert (
        report.ssim >= SSIM_MIN
    ), f"{engine_cls.__name__} SSIM={report.ssim:.4f} below {SSIM_MIN}"


# ──────────────────────────────────────────────────────────────────────────
# Report completeness tests
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_cls",
    [
        FgsmEngine,
        PgdEngine,
        LsbEngine,
        DctEngine,
        GaussianEngine,
    ],
)
def test_report_fields(engine_cls):
    _, report = _run_engine(engine_cls)
    assert report.engine_name != ""
    assert 0.0 <= report.ssim <= 1.0
    assert report.psnr > 0
    assert report.mean_delta >= 0
    assert report.max_delta >= 0
    assert report.pixels_altered >= 0
    assert isinstance(report.fooled, bool)


# ──────────────────────────────────────────────────────────────────────────
# Engine name tests
# ──────────────────────────────────────────────────────────────────────────


def test_engine_names():
    assert FgsmEngine.name == "fgsm"
    assert PgdEngine.name == "pgd"
    assert LsbEngine.name == "lsb"
    assert DctEngine.name == "dct"
    assert CwEngine.name == "cw"
    assert DeepFoolEngine.name == "deepfool"
    assert AutoPgdEngine.name == "auto_pgd"
    assert PatchEngine.name == "patch"
    assert GaussianEngine.name == "gaussian"
    assert JsmaEngine.name == "jsma"


# ──────────────────────────────────────────────────────────────────────────
# Epsilon respected — perturbation must not wildly exceed epsilon
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_cls",
    [
        FgsmEngine,
        PgdEngine,
    ],
)
def test_epsilon_respected(engine_cls):
    adv, report = _run_engine(engine_cls)
    # max delta in [0,1] space should be close to epsilon
    max_delta_norm = report.max_delta / 255.0
    assert max_delta_norm <= EPSILON + (
        1.0 / 255.0
    ), f"{engine_cls.__name__} max delta {max_delta_norm:.4f} exceeds epsilon {EPSILON:.4f}"
