import sys
import numpy as np
from pathlib import Path
from PIL import Image

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QFileDialog,
    QStatusBar,
    QFrame,
    QTabWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QShortcut,
)

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
from imageecho.context import EchoContext
from gui.benchmark_panel import BenchmarkPanel
from gui.heatmap_panel import HeatmapPanel
from gui.settings_panel import SettingsPanel

ENGINE_MAP = {
    "FGSM": FgsmEngine,
    "PGD": PgdEngine,
    "LSB": LsbEngine,
    "DCT": DctEngine,
    "C&W": CwEngine,
    "DeepFool": DeepFoolEngine,
    "AutoPGD": AutoPgdEngine,
    "Patch": PatchEngine,
    "Gaussian": GaussianEngine,
    "JSMA": JsmaEngine,
}

ENGINE_DESC = {
    "FGSM": "Fast Gradient Sign — one step, fast",
    "PGD": "Projected Gradient Descent — iterative, strong",
    "LSB": "Least Significant Bit flipping",
    "DCT": "Frequency domain — DCT high-freq injection",
    "C&W": "Carlini-Wagner L2 — tightest perturbation",
    "DeepFool": "Minimal boundary crossing perturbation",
    "AutoPGD": "Adaptive step PGD with restarts",
    "Patch": "Concentrated adversarial patch region",
    "Gaussian": "Frequency-weighted structured noise",
    "JSMA": "Jacobian Saliency Map — sparse pixel attack",
}


class AttackWorker(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, image, engine_name, epsilon):
        super().__init__()
        self.image = image
        self.engine_name = engine_name
        self.epsilon = epsilon

    def run(self):
        try:
            engine = ENGINE_MAP[self.engine_name](epsilon=self.epsilon)
            ctx = EchoContext(engine)
            adv, report = ctx.run(self.image)
            self.finished.emit(adv, report)
        except Exception as e:
            self.error.emit(str(e))


class ImagePanel(QLabel):
    image_dropped = pyqtSignal(str)

    def __init__(self, title: str):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            QLabel {
                background: #1e1e2e;
                border: 2px dashed #444466;
                border-radius: 12px;
                color: #888899;
                font-size: 14px;
            }
        """)
        self.setText(f"{title}\n\nDrag & drop an image\nor click Open")
        self.setAcceptDrops(True)

    def set_image(self, img: np.ndarray):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(
            pixmap.scaled(
                self.width(),
                self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        path = e.mimeData().urls()[0].toLocalFile()
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            self.image_dropped.emit(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageEcho — Adversarial ML")
        self.setMinimumSize(1100, 750)
        self._image_np = None
        self._adv_np = None
        self._worker = None

        # Create panels FIRST before any tab building
        self.settings_panel = SettingsPanel()
        self.benchmark_panel = BenchmarkPanel()
        self.heatmap_panel = HeatmapPanel()

        self._setup_ui()
        self._setup_shortcuts()
        self._apply_dark_theme()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border:none; background:#13131f; }
            QTabBar::tab {
                background:#1a1a2e; color:#888899;
                padding:8px 20px; border:none; font-size:13px;
            }
            QTabBar::tab:selected {
                background:#2d2d4e; color:#ddddff;
                border-bottom:2px solid #7777cc;
            }
            QTabBar::tab:hover { background:#222240; }
        """)

        self.tab_attack = QWidget()
        self._build_attack_tab()
        self.tabs.addTab(self.tab_attack, "⚡  Attack")
        self.tabs.addTab(self.benchmark_panel, "📊  Benchmark")
        self.tabs.addTab(self.heatmap_panel, "🔥  Heatmap")

        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        self.tabs.addTab(self.settings_panel, "⚙️  Settings")

        root.addWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — open an image to begin")

    def _build_attack_tab(self):
        layout = QVBoxLayout(self.tab_attack)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        toolbar = QHBoxLayout()

        self.btn_open = QPushButton("Open Image")
        self.btn_open.setFixedHeight(36)
        self.btn_open.clicked.connect(self._open_image)

        self.combo_engine = QComboBox()
        self.combo_engine.addItems(ENGINE_MAP.keys())
        self.combo_engine.setFixedHeight(36)
        self.combo_engine.setMinimumWidth(140)
        self.combo_engine.currentTextChanged.connect(self._engine_changed)

        self.lbl_desc = QLabel(ENGINE_DESC["FGSM"])
        self.lbl_desc.setStyleSheet("color:#888899; font-size:12px;")

        self.lbl_eps = QLabel("ε = 8/255")
        self.lbl_eps.setFixedWidth(80)
        self.lbl_eps.setStyleSheet("color:#aaaacc; font-size:12px;")

        self.slider_eps = QSlider(Qt.Orientation.Horizontal)
        self.slider_eps.setRange(1, 32)
        self.slider_eps.setValue(self.settings_panel.get("default_epsilon"))
        self.slider_eps.setFixedWidth(160)
        self.slider_eps.valueChanged.connect(self._eps_changed)

        self.btn_run = QPushButton("▶  Run Attack")
        self.btn_run.setFixedHeight(36)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_attack)

        self.btn_save = QPushButton("Save Output")
        self.btn_save.setFixedHeight(36)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_output)

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(QLabel("Engine:"))
        toolbar.addWidget(self.combo_engine)
        toolbar.addWidget(self.lbl_desc)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Epsilon:"))
        toolbar.addWidget(self.slider_eps)
        toolbar.addWidget(self.lbl_eps)
        toolbar.addWidget(self.btn_run)
        toolbar.addWidget(self.btn_save)
        layout.addLayout(toolbar)

        panels = QHBoxLayout()
        panels.setSpacing(12)
        left = QVBoxLayout()
        right = QVBoxLayout()

        lbl_l = QLabel("Original")
        lbl_l.setStyleSheet("color:#ccccee;font-weight:bold;font-size:13px;")
        lbl_r = QLabel("Adversarial Output")
        lbl_r.setStyleSheet("color:#ccccee;font-weight:bold;font-size:13px;")

        self.panel_orig = ImagePanel("Original")
        self.panel_adv = ImagePanel("Adversarial")
        self.panel_orig.image_dropped.connect(self._load_image)
        self.panel_adv.image_dropped.connect(self._load_image)

        left.addWidget(lbl_l)
        left.addWidget(self.panel_orig)
        right.addWidget(lbl_r)
        right.addWidget(self.panel_adv)
        panels.addLayout(left)
        panels.addLayout(right)
        layout.addLayout(panels)

        mframe = QFrame()
        mframe.setStyleSheet(
            "QFrame{background:#1a1a2e;border-radius:8px;padding:4px;}"
        )
        ml = QHBoxLayout(mframe)

        self.lbl_ssim = self._metric_label("SSIM", "—")
        self.lbl_psnr = self._metric_label("PSNR", "—")
        self.lbl_delta = self._metric_label("Mean Δ", "—")
        self.lbl_pixels = self._metric_label("Pixels Altered", "—")
        self.lbl_fooled = self._metric_label("Result", "—")

        for lbl in [
            self.lbl_ssim,
            self.lbl_psnr,
            self.lbl_delta,
            self.lbl_pixels,
            self.lbl_fooled,
        ]:
            ml.addWidget(lbl)
            d = QFrame()
            d.setFrameShape(QFrame.Shape.VLine)
            d.setStyleSheet("color:#333355;")
            ml.addWidget(d)

        layout.addWidget(mframe)

    def _metric_label(self, title, value):
        lbl = QLabel(
            f"<b style='color:#888899'>{title}</b><br>"
            f"<span style='color:#eeeeff;font-size:15px'>{value}</span>"
        )
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumWidth(120)
        return lbl

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"),
                  self).activated.connect(self._open_image)
        QShortcut(QKeySequence("Ctrl+R"),
                  self).activated.connect(self._run_attack)
        QShortcut(QKeySequence("Ctrl+S"),
                  self).activated.connect(self._save_output)
        QShortcut(QKeySequence("Ctrl+B"), self).activated.connect(
            lambda: self.tabs.setCurrentIndex(1)
        )
        QShortcut(QKeySequence("Ctrl+H"), self).activated.connect(
            lambda: self.tabs.setCurrentIndex(2)
        )

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background:#13131f; color:#ccccee;
                font-family:'Segoe UI',sans-serif; font-size:13px;
            }
            QPushButton {
                background:#2d2d4e; color:#ddddff;
                border:1px solid #444466; border-radius:6px;
                padding:4px 14px;
            }
            QPushButton:hover   { background:#3d3d6e; }
            QPushButton:pressed { background:#5555aa; }
            QPushButton:disabled { background:#1a1a2e; color:#555566; }
            QComboBox {
                background:#2d2d4e; color:#ddddff;
                border:1px solid #444466; border-radius:6px;
                padding:4px 8px;
            }
            QComboBox QAbstractItemView {
                background:#2d2d4e; color:#ddddff;
                selection-background-color:#5555aa;
            }
            QSlider::groove:horizontal {
                background:#2d2d4e; height:6px; border-radius:3px;
            }
            QSlider::handle:horizontal {
                background:#7777cc; width:14px; height:14px;
                margin:-4px 0; border-radius:7px;
            }
            QLineEdit {
                background:#1a1a2e; color:#ccccee;
                border:1px solid #333355; border-radius:4px;
                padding:4px 8px;
            }
            QSpinBox, QDoubleSpinBox {
                background:#1a1a2e; color:#ccccee;
                border:1px solid #333355; border-radius:4px;
                padding:2px 6px;
            }
            QCheckBox { color:#ccccee; }
            QCheckBox::indicator {
                width:14px; height:14px;
                border:1px solid #444466; border-radius:3px;
                background:#1a1a2e;
            }
            QCheckBox::indicator:checked { background:#7777cc; }
            QStatusBar { background:#0d0d1a; color:#888899; }
        """)

    # ------------------------------------------------------------------ #

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        self._image_np = np.array(Image.open(path).convert("RGB"))
        self.panel_orig.set_image(self._image_np)
        self.panel_adv.setText(
            "Adversarial Output\n\nRun an attack to see result")
        self.btn_run.setEnabled(True)
        epsilon = self.slider_eps.value() / 255.0
        self.benchmark_panel.set_image(self._image_np, epsilon)
        self.status.showMessage(
            f"Loaded: {Path(path).name}  "
            f"({self._image_np.shape[1]}x{self._image_np.shape[0]})"
        )
        self._reset_metrics()

    def _engine_changed(self, name):
        self.lbl_desc.setText(ENGINE_DESC.get(name, ""))

    def _eps_changed(self, val):
        self.lbl_eps.setText(f"ε = {val}/255")
        if self._image_np is not None:
            self.benchmark_panel.set_image(self._image_np, val / 255.0)

    def _run_attack(self):
        if self._image_np is None:
            return
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        engine_name = self.combo_engine.currentText()
        epsilon = self.slider_eps.value() / 255.0
        self.status.showMessage(f"Running {engine_name}  ε={epsilon:.4f} ...")
        self._worker = AttackWorker(self._image_np, engine_name, epsilon)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, adv_np, report):
        self._adv_np = adv_np
        self.panel_adv.set_image(adv_np)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self._update_metrics(report)
        self.heatmap_panel.set_images(self._image_np, adv_np)

        if self.settings_panel.get("auto_save"):
            folder = Path(self.settings_panel.get("output_folder"))
            folder.mkdir(exist_ok=True)
            Image.fromarray(self._adv_np).save(
                folder / f"adv_{report.engine_name}.png")

        status = "FOOLED" if report.fooled else "Same class"
        self.status.showMessage(
            f"Done — {report.engine_name}  "
            f"SSIM={report.ssim:.4f}  {status}  "
            f"(Ctrl+H for heatmap)"
        )

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.status.showMessage(f"Error: {msg}")

    def _save_output(self):
        if self._adv_np is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "adversarial.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            Image.fromarray(self._adv_np).save(path)
            self.status.showMessage(f"Saved to {path}")

    def _on_settings_changed(self, settings: dict):
        self.slider_eps.setValue(settings["default_epsilon"])
        self.status.showMessage("Settings saved")

    def _update_metrics(self, report):
        fc = "#55ff88" if report.fooled else "#ff5555"
        ft = "FOOLED ✓" if report.fooled else "Same Class"
        self.lbl_ssim.setText(
            f"<b style='color:#888899'>SSIM</b><br>"
            f"<span style='color:#eeeeff;font-size:15px'>{report.ssim:.4f}</span>"
        )
        self.lbl_psnr.setText(
            f"<b style='color:#888899'>PSNR</b><br>"
            f"<span style='color:#eeeeff;font-size:15px'>{report.psnr:.1f} dB</span>"
        )
        self.lbl_delta.setText(
            f"<b style='color:#888899'>Mean Δ</b><br>"
            f"<span style='color:#eeeeff;font-size:15px'>{report.mean_delta:.2f}</span>"
        )
        self.lbl_pixels.setText(
            f"<b style='color:#888899'>Pixels Altered</b><br>"
            f"<span style='color:#eeeeff;font-size:15px'>{report.pixels_altered:,}</span>"
        )
        self.lbl_fooled.setText(
            f"<b style='color:#888899'>Result</b><br>"
            f"<span style='color:{fc};font-size:15px'>{ft}</span>"
        )

    def _reset_metrics(self):
        for lbl, title in [
            (self.lbl_ssim, "SSIM"),
            (self.lbl_psnr, "PSNR"),
            (self.lbl_delta, "Mean Δ"),
            (self.lbl_pixels, "Pixels Altered"),
            (self.lbl_fooled, "Result"),
        ]:
            lbl.setText(
                f"<b style='color:#888899'>{title}</b><br>"
                f"<span style='color:#eeeeff;font-size:15px'>—</span>"
            )
