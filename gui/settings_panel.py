from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QGroupBox,
    QLineEdit,
    QFileDialog,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
import json
from pathlib import Path

SETTINGS_FILE = Path("imageecho_settings.json")

DEFAULT_SETTINGS = {
    "default_epsilon": 8,
    "ssim_threshold": 0.95,
    "output_folder": "outputs",
    "thread_count": 1,
    "auto_save": False,
}


class SettingsPanel(QWidget):
    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = self._load()
        self._setup_ui()

    def get(self, key):
        return self._settings.get(key, DEFAULT_SETTINGS.get(key))

    def _load(self) -> dict:
        if SETTINGS_FILE.exists():
            try:
                return json.loads(SETTINGS_FILE.read_text())
            except Exception:
                pass
        return DEFAULT_SETTINGS.copy()

    def _save(self):
        SETTINGS_FILE.write_text(json.dumps(self._settings, indent=2))

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(24, 24, 24, 24)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Settings")
        title.setStyleSheet("color:#ccccee; font-weight:bold; font-size:16px;")
        root.addWidget(title)

        # --- Attack defaults ---
        grp_attack = self._group("Attack Defaults")
        gl = QVBoxLayout(grp_attack)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Default epsilon (x/255):"))
        self.spin_eps = QSpinBox()
        self.spin_eps.setRange(1, 32)
        self.spin_eps.setValue(self._settings["default_epsilon"])
        self.spin_eps.setFixedWidth(80)
        row1.addWidget(self.spin_eps)
        row1.addStretch()
        gl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("SSIM invisibility threshold:"))
        self.spin_ssim = QDoubleSpinBox()
        self.spin_ssim.setRange(0.80, 1.00)
        self.spin_ssim.setSingleStep(0.01)
        self.spin_ssim.setDecimals(2)
        self.spin_ssim.setValue(self._settings["ssim_threshold"])
        self.spin_ssim.setFixedWidth(80)
        row2.addWidget(self.spin_ssim)
        row2.addWidget(QLabel("(attacks below this are considered invisible)"))
        row2.addStretch()
        gl.addLayout(row2)

        root.addWidget(grp_attack)

        # --- Output ---
        grp_output = self._group("Output")
        ol = QVBoxLayout(grp_output)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Output folder:"))
        self.edit_folder = QLineEdit(self._settings["output_folder"])
        self.edit_folder.setFixedHeight(30)
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedHeight(30)
        btn_browse.clicked.connect(self._browse_folder)
        row3.addWidget(self.edit_folder)
        row3.addWidget(btn_browse)
        ol.addLayout(row3)

        row4 = QHBoxLayout()
        self.chk_autosave = QCheckBox(
            "Auto-save adversarial images after each attack")
        self.chk_autosave.setChecked(self._settings["auto_save"])
        row4.addWidget(self.chk_autosave)
        row4.addStretch()
        ol.addLayout(row4)

        root.addWidget(grp_output)

        # --- Keyboard shortcuts info ---
        grp_keys = self._group("Keyboard Shortcuts")
        kl = QVBoxLayout(grp_keys)

        shortcuts = [
            ("Ctrl + O", "Open image"),
            ("Ctrl + R", "Run attack"),
            ("Ctrl + S", "Save output"),
            ("Ctrl + B", "Go to Benchmark tab"),
            ("Ctrl + H", "Go to Heatmap tab"),
        ]
        for key, desc in shortcuts:
            row = QHBoxLayout()
            lbl_key = QLabel(key)
            lbl_key.setStyleSheet(
                "background:#2d2d4e; color:#aaaaff; "
                "border-radius:4px; padding:2px 8px; "
                "font-family:monospace; font-size:12px;"
            )
            lbl_key.setFixedWidth(90)
            row.addWidget(lbl_key)
            row.addWidget(QLabel(desc))
            row.addStretch()
            kl.addLayout(row)

        root.addWidget(grp_keys)

        # --- Save / Reset buttons ---
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save Settings")
        btn_save.setFixedHeight(36)
        btn_save.clicked.connect(self._apply)

        btn_reset = QPushButton("Reset Defaults")
        btn_reset.setFixedHeight(36)
        btn_reset.clicked.connect(self._reset)

        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_reset)
        root.addLayout(btn_row)

        root.addStretch()

    def _group(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet("""
            QGroupBox {
                color: #aaaacc;
                border: 1px solid #333355;
                border-radius: 8px;
                margin-top: 8px;
                padding: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
        """)
        return g

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.edit_folder.setText(folder)

    def _apply(self):
        self._settings = {
            "default_epsilon": self.spin_eps.value(),
            "ssim_threshold": self.spin_ssim.value(),
            "output_folder": self.edit_folder.text(),
            "auto_save": self.chk_autosave.isChecked(),
            "thread_count": 1,
        }
        self._save()
        self.settings_changed.emit(self._settings)

    def _reset(self):
        self._settings = DEFAULT_SETTINGS.copy()
        self.spin_eps.setValue(self._settings["default_epsilon"])
        self.spin_ssim.setValue(self._settings["ssim_threshold"])
        self.edit_folder.setText(self._settings["output_folder"])
        self.chk_autosave.setChecked(self._settings["auto_save"])
        self._save()
        self.settings_changed.emit(self._settings)
