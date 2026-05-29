import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSizePolicy,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("QtAgg")


class HeatmapPanel(QWidget):
    """
    Shows pixel difference heatmap between original and adversarial image.
    Visualises exactly which pixels changed and by how much.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig = None
        self._adv = None
        self._setup_ui()

    def set_images(self, original: np.ndarray, adversarial: np.ndarray):
        self._orig = original
        self._adv = adversarial
        self.btn_refresh.setEnabled(True)
        self._draw()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # Top bar
        top = QHBoxLayout()

        lbl = QLabel("Pixel Difference Visualiser")
        lbl.setStyleSheet("color:#ccccee; font-weight:bold; font-size:14px;")

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(
            [
                "Heatmap (magnitude)",
                "Per-channel diff",
                "Side by side",
            ]
        )
        self.combo_mode.setFixedHeight(32)
        self.combo_mode.currentIndexChanged.connect(self._draw)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setFixedHeight(32)
        self.btn_refresh.setEnabled(False)
        self.btn_refresh.clicked.connect(self._draw)

        self.lbl_info = QLabel("Run an attack first")
        self.lbl_info.setStyleSheet("color:#888899; font-size:12px;")

        top.addWidget(lbl)
        top.addStretch()
        top.addWidget(QLabel("View:"))
        top.addWidget(self.combo_mode)
        top.addWidget(self.btn_refresh)
        root.addLayout(top)
        root.addWidget(self.lbl_info)

        # Canvas
        self.fig = Figure(figsize=(10, 4), facecolor="#13131f")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self.canvas)

    def _draw(self):
        if self._orig is None or self._adv is None:
            return

        mode = self.combo_mode.currentIndex()
        self.fig.clear()

        orig = self._orig.astype(np.float32)
        adv = self._adv.astype(np.float32)
        diff = np.abs(orig - adv)

        if mode == 0:
            self._draw_heatmap(diff)
        elif mode == 1:
            self._draw_per_channel(diff)
        else:
            self._draw_side_by_side(orig, adv, diff)

        self.canvas.draw()

        # Update info label
        total_px = diff.shape[0] * diff.shape[1]
        altered = int((diff.sum(axis=2) > 0).sum())
        mean_delta = diff.mean()
        max_delta = diff.max()
        self.lbl_info.setText(
            f"Pixels altered: {altered:,} / {total_px:,}  |  "
            f"Mean Δ: {mean_delta:.3f}  |  Max Δ: {max_delta:.1f}"
        )

    def _draw_heatmap(self, diff: np.ndarray):
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        self.fig.patch.set_facecolor("#13131f")

        magnitude = diff.sum(axis=2)
        im = ax.imshow(magnitude, cmap="hot", interpolation="nearest")
        self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title(
            "Pixel Change Magnitude  (brighter = more change)",
            color="#ccccee",
            fontsize=11,
        )
        ax.axis("off")
        self.fig.tight_layout()

    def _draw_per_channel(self, diff: np.ndarray):
        titles = ["Red channel Δ", "Green channel Δ", "Blue channel Δ"]
        cmaps = ["Reds", "Greens", "Blues"]

        for i in range(3):
            ax = self.fig.add_subplot(1, 3, i + 1)
            ax.set_facecolor("#1a1a2e")
            im = ax.imshow(diff[:, :, i], cmap=cmaps[i],
                           interpolation="nearest")
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(titles[i], color="#ccccee", fontsize=10)
            ax.axis("off")

        self.fig.patch.set_facecolor("#13131f")
        self.fig.tight_layout()

    def _draw_side_by_side(self, orig, adv, diff):
        titles = ["Original", "Adversarial", "Difference (×10)"]
        imgs = [
            orig.astype(np.uint8),
            adv.astype(np.uint8),
            np.clip(diff * 10, 0, 255).astype(np.uint8),
        ]

        for i, (img, title) in enumerate(zip(imgs, titles)):
            ax = self.fig.add_subplot(1, 3, i + 1)
            ax.set_facecolor("#1a1a2e")
            ax.imshow(img, interpolation="nearest")
            ax.set_title(title, color="#ccccee", fontsize=10)
            ax.axis("off")

        self.fig.patch.set_facecolor("#13131f")
        self.fig.tight_layout()
