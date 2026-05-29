import copy
import numpy as np
from PIL import Image
from .base_engine import BaseEngine


class EchoContext:
    def __init__(self, engine: BaseEngine):
        self._engine = engine

    def set_engine(self, engine: BaseEngine):
        self._engine = engine

    def run(self, image_source, target_class=None, save_to=None):
        image = self._load(image_source)
        adv, report = self._engine.apply(image, target_class=target_class)
        if save_to:
            Image.fromarray(adv).save(save_to)
        return adv, report

    def runOptimal(
        self,
        image_source,
        ssim_threshold=0.95,
        eps_min=1 / 255,
        eps_max=32 / 255,
        iterations=16,
        target_class=None,
    ):
        image = self._load(image_source)
        best_adv = None
        best_rpt = None
        lo, hi = eps_min, eps_max

        for i in range(iterations):
            mid = (lo + hi) / 2.0
            print(f"  [runOptimal] iter {i+1}/{iterations}  eps={mid:.5f}")
            engine_copy = copy.deepcopy(self._engine)
            engine_copy.epsilon = mid
            adv, rpt = engine_copy.apply(image, target_class=target_class)
            if rpt.ssim >= ssim_threshold:
                lo = mid
                best_adv = adv
                best_rpt = rpt
            else:
                hi = mid

        if best_rpt is None:
            self._engine.epsilon = eps_min
            best_adv, best_rpt = self._engine.apply(
                image, target_class=target_class)

        return best_adv, best_rpt

    @staticmethod
    def _load(source):
        if isinstance(source, np.ndarray):
            return source
        return np.array(Image.open(source).convert("RGB"))
