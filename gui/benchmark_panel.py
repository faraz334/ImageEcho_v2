from imageecho.context import EchoContext
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
import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QLabel,
    QHeaderView,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("QtAgg")


ALL_ENGINES = {
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


class BenchmarkWorker(QThread):
    engine_done = pyqtSignal(str, object)  # engine_name, report
    all_done = pyqtSignal()
    error = pyqtSignal(str, str)  # engine_name, error_msg

    def __init__(self, image, epsilon):
        super().__init__()
        self.image = image
        self.epsilon = epsilon

    def run(self):
        for name, cls in ALL_ENGINES.items():
            try:
                engine = cls(epsilon=self.epsilon)
                ctx = EchoContext(engine)
                _, report = ctx.run(self.image)
                self.engine_done.emit(name, report)
            except Exception as e:
                self.error.emit(name, str(e))
        self.all_done.emit()


class BenchmarkPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self._epsilon = 8 / 255
        self._results = {}
        self._worker = None
        self._setup_ui()

    def set_image(self, image: np.ndarray, epsilon: float):
        self._image = image
        self._epsilon = epsilon
        self.btn_run.setEnabled(True)

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # --- Top bar ---
        top = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Full Benchmark (all 10 engines)")
        self.btn_run.setFixedHeight(36)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_benchmark)

        self.lbl_status = QLabel("Load an image on the Attack tab first")
        self.lbl_status.setStyleSheet("color: #888899;")

        top.addWidget(self.btn_run)
        top.addWidget(self.lbl_status)
        top.addStretch()
        root.addLayout(top)

        # --- Progress bar ---
        self.progress = QProgressBar()
        self.progress.setRange(0, len(ALL_ENGINES))
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        self.progress.setStyleSheet("""
            QProgressBar {
                background: #1a1a2e;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background: #7777cc;
                border-radius: 4px;
            }
        """)
        root.addWidget(self.progress)

        # --- Results table ---
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Engine",
                "SSIM",
                "PSNR (dB)",
                "Mean Δ",
                "Max Δ",
                "Pixels Altered",
                "Fooled",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setStyleSheet("""
            QTableWidget {
                background: #1a1a2e;
                color: #ccccee;
                border: 1px solid #333355;
                gridline-color: #222244;
            }
            QHeaderView::section {
                background: #2d2d4e;
                color: #aaaacc;
                border: 1px solid #333355;
                padding: 4px;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background: #3d3d6e;
            }
        """)
        self.table.setMaximumHeight(280)
        root.addWidget(self.table)

        # --- Charts ---
        charts = QHBoxLayout()

        self.fig_bar = Figure(figsize=(5, 3), facecolor="#13131f")
        self.canvas_bar = FigureCanvas(self.fig_bar)
        self.canvas_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.fig_radar = Figure(figsize=(4, 3), facecolor="#13131f")
        self.canvas_radar = FigureCanvas(self.fig_radar)
        self.canvas_radar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        charts.addWidget(self.canvas_bar, stretch=3)
        charts.addWidget(self.canvas_radar, stretch=2)
        root.addLayout(charts)

    # ------------------------------------------------------------------ #

    def _run_benchmark(self):
        if self._image is None:
            return
        self._results = {}
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Running benchmarks...")

        self._worker = BenchmarkWorker(self._image, self._epsilon)
        self._worker.engine_done.connect(self._on_engine_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_engine_error)
        self._worker.start()

    def _on_engine_done(self, name: str, report):
        self._results[name] = report
        self.progress.setValue(len(self._results))
        self.lbl_status.setText(
            f"Completed {len(self._results)}/{len(ALL_ENGINES)}: {name}"
        )

        # Add row to table
        row = self.table.rowCount()
        self.table.insertRow(row)

        fooled_color = QColor(
            "#55ff88") if report.fooled else QColor("#ff5555")
        fooled_text = "YES" if report.fooled else "NO"

        items = [
            name,
            f"{report.ssim:.4f}",
            f"{report.psnr:.1f}",
            f"{report.mean_delta:.2f}",
            f"{report.max_delta:.2f}",
            f"{report.pixels_altered:,}",
            fooled_text,
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 6:
                item.setForeground(fooled_color)
            self.table.setItem(row, col, item)

        # Update bar chart live
        self._draw_bar_chart()

    def _on_engine_error(self, name: str, msg: str):
        self.progress.setValue(self.progress.value() + 1)
        self.lbl_status.setText(f"Error on {name}: {msg}")

    def _on_all_done(self):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(
            f"Benchmark complete — {len(self._results)} engines")
        self._draw_bar_chart()
        self._draw_radar_chart()

    # ------------------------------------------------------------------ #
    #  Charts                                                              #
    # ------------------------------------------------------------------ #

    def _draw_bar_chart(self):
        if not self._results:
            return

        names = list(self._results.keys())
        ssims = [self._results[n].ssim for n in names]
        colors = [
            "#55ff88" if self._results[n].fooled else "#ff5555" for n in names]

        self.fig_bar.clear()
        ax = self.fig_bar.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        self.fig_bar.patch.set_facecolor("#13131f")

        bars = ax.bar(names, ssims, color=colors,
                      edgecolor="#333355", linewidth=0.8)

        # Threshold line
        ax.axhline(
            0.95,
            color="#ffaa44",
            linewidth=1.2,
            linestyle="--",
            label="SSIM 0.95 threshold",
        )

        ax.set_ylim(0.85, 1.01)
        ax.set_ylabel("SSIM", color="#aaaacc", fontsize=10)
        ax.set_title(
            "SSIM per Engine  (green = fooled)", color="#ccccee", fontsize=11, pad=8
        )
        ax.tick_params(colors="#888899", labelsize=8)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.legend(
            fontsize=8, facecolor="#1a1a2e", labelcolor="#aaaacc", edgecolor="#333355"
        )
        self.fig_bar.tight_layout()
        self.canvas_bar.draw()

    def _draw_radar_chart(self):
        if len(self._results) < 3:
            return

        names = list(self._results.keys())
        categories = ["SSIM", "PSNR\n(norm)", "Fooled", "Invisibility"]
        N = len(categories)
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]

        self.fig_radar.clear()
        ax = self.fig_radar.add_subplot(111, polar=True)
        ax.set_facecolor("#1a1a2e")
        self.fig_radar.patch.set_facecolor("#13131f")

        colors = [
            "#7777cc",
            "#55ff88",
            "#ffaa44",
            "#ff5555",
            "#44ccff",
            "#ff44cc",
            "#aaff44",
            "#ff8844",
            "#44ffcc",
            "#cc44ff",
        ]

        for i, name in enumerate(names[:5]):
            r = self._results[name]
            ssim = r.ssim
            psnr = min(r.psnr / 50.0, 1.0)
            fooled = 1.0 if r.fooled else 0.2
            invis = 1.0 - (r.mean_delta / 30.0)
            vals = [ssim, psnr, fooled, invis]
            vals += vals[:1]
            ax.plot(
                angles,
                vals,
                "o-",
                linewidth=1.5,
                color=colors[i],
                label=name,
                markersize=3,
            )
            ax.fill(angles, vals, alpha=0.08, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color="#aaaacc", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(colors="#555577")
        ax.set_title("Engine Profile (top 5)",
                     color="#ccccee", fontsize=10, pad=15)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.1),
            fontsize=7,
            facecolor="#1a1a2e",
            labelcolor="#ccccee",
            edgecolor="#333355",
        )
        self.fig_radar.tight_layout()
        self.canvas_radar.draw()
