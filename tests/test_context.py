import numpy as np
from imageecho.context import EchoContext
from imageecho.engines import FgsmEngine, GaussianEngine

TEST_IMAGE = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


def test_context_run_returns_tuple():
    ctx = EchoContext(FgsmEngine(epsilon=8 / 255))
    result = ctx.run(TEST_IMAGE)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_context_run_output_shape():
    ctx = EchoContext(FgsmEngine(epsilon=8 / 255))
    adv, _ = ctx.run(TEST_IMAGE)
    assert adv.shape == TEST_IMAGE.shape


def test_context_set_engine():
    ctx = EchoContext(FgsmEngine(epsilon=8 / 255))
    ctx.set_engine(GaussianEngine(epsilon=8 / 255))
    adv, report = ctx.run(TEST_IMAGE)
    assert report.engine_name == "gaussian"


def test_context_load_numpy():
    ctx = EchoContext(FgsmEngine(epsilon=8 / 255))
    adv, report = ctx.run(TEST_IMAGE)
    assert adv is not None
    assert report is not None


def test_context_run_optimal():
    ctx = EchoContext(FgsmEngine(epsilon=8 / 255))
    adv, report = ctx.runOptimal(
        TEST_IMAGE,
        ssim_threshold=0.85,
        eps_min=1 / 255,
        eps_max=16 / 255,
        iterations=4,
    )
    assert adv.shape == TEST_IMAGE.shape
    assert report.ssim >= 0.0


def test_report_as_dict():
    ctx = EchoContext(FgsmEngine(epsilon=8 / 255))
    _, report = ctx.run(TEST_IMAGE)
    d = report.as_dict()
    assert "ssim" in d
    assert "psnr" in d
    assert "fooled" in d
    assert "engine_name" in d
    assert "pixels_altered" in d


def test_report_str():
    ctx = EchoContext(FgsmEngine(epsilon=8 / 255))
    _, report = ctx.run(TEST_IMAGE)
    s = str(report)
    assert "fgsm" in s
    assert "SSIM" in s
