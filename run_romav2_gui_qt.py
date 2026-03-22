from __future__ import annotations

import json
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, QStandardPaths, QTimer, Qt, QUrl, Signal
    from PySide6.QtGui import QDesktopServices, QGuiApplication, QImage, QPixmap
    from PySide6.QtTest import QTest
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
        QSpinBox,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 + QtWebEngine is required for this GUI.\n"
        "Install with: uv sync --extra gui\n"
    ) from exc


ROMA_SETTINGS = ("turbo", "fast", "base", "precise", "mega1500", "scannet1500", "wxbs", "satast")
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Primary result first, then diagnostics
PREVIEW_FILES = [
    ("Final Colorized", "final_colorized.png"),
    ("Warped (Smooth)", "warped_src_smooth.png"),
    ("Warped (Raw)", "warped_src_to_ref.png"),
    ("Warped (Regularized)", "warped_src_to_ref_regularized.png"),
    ("Overlap Confidence", "overlap_AB_overlay_on_ref.png"),
    ("Correspondences", "sampled_correspondences.png"),
    ("Difference Map", "difference_before_after.png"),
    ("Before / After", "before_after_triptych.png"),
]


DEFAULT_PREP_PROMPT = """Restore this old photograph in one continuous workflow and produce a single final image.

Follow these steps in this exact order:

1. Detect whether the photo includes a visible physical frame, mount, border card, CDV card edge, daguerreotype case edge, decorative mat, or any non-image surround.
2. If a frame or mount is present, crop/trim it away first so that only the actual photographic image area remains. Do not preserve, duplicate, or expand the frame, card, mount, or surround.
3. Neutralize the aged sepia/brown/yellow cast before colorization. Do not leave the final image with an overall brown antique tint unless a specific object is truly brown.
4. Restore and enhance the photograph carefully:
   - improve sharpness and fine detail
   - recover facial features, hair, clothing texture, and background detail
   - reduce blur, haze, dust, scratches, stains, cracks, and age damage
   - preserve realism and original identity
   - do not over-smooth skin
   - do not invent modern-looking features
5. Colorize the image in a vivid but historically plausible way:
   - use full natural color, not partial sepia tinting
   - produce a genuinely colorful result where the scene calls for it
   - use realistic skin tones and believable fabric, object, and background colors
   - allow distinct greens, blues, reds, sky tones, wood tones, and textile colors when historically appropriate
   - keep saturation realistic, but do not leave the image mostly beige, tan, or brown
6. Expand the image content itself equally in ALL FOUR DIRECTIONS with no exceptions:
   - expand left, right, top, and bottom evenly
   - add 25% of the original cropped image width to the LEFT
   - add 25% of the original cropped image width to the RIGHT
   - add 25% of the original cropped image height to the TOP
   - add 25% of the original cropped image height to the BOTTOM
   - this means the final canvas must be 150% of the original cropped width and 150% of the original cropped height
   - do not place most of the expansion on only one or two sides
   - do not skip any side
   - do not crop after expansion
   - do not add a frame, border, blank margin, or decorative edge
   - extend the actual photographic scene naturally on all four sides
7. Return one final image that already includes frame removal if needed, restoration, full historical colorization, and the four-sided 25% expansion.

Critical requirements:
- Complete all steps in one pass.
- The border expansion must happen in this same final output.
- The expansion must be symmetrical and applied to all four sides.
- Do not expand only horizontally.
- Do not expand only vertically.
- Do not expand only where composition seems convenient.
- Do not require a second prompt."""

class AutoUploadWebPage(QWebEnginePage):
    """QWebEnginePage that can auto-answer file dialogs with a pending file path."""

    auto_file_used = Signal(str)

    def __init__(self, profile: QWebEngineProfile, parent: QWidget | None = None) -> None:
        super().__init__(profile, parent)
        self.pending_upload_path: Path | None = None

    def set_pending_upload_path(self, path: Path | None) -> None:
        self.pending_upload_path = path

    # Qt virtual override
    def chooseFiles(self, mode, old_files, accepted_mime_types):  # noqa: N802
        path = self.pending_upload_path
        if path is not None and path.exists() and path.is_file():
            path_str = str(path.resolve())
            if mode == QWebEnginePage.FileSelectionMode.FileSelectOpenMultiple:
                self.pending_upload_path = None
                self.auto_file_used.emit(path_str)
                return [path_str]
            if mode == QWebEnginePage.FileSelectionMode.FileSelectOpen:
                self.pending_upload_path = None
                self.auto_file_used.emit(path_str)
                return [path_str]
        return super().chooseFiles(mode, old_files, accepted_mime_types)


class PhotoColorizerQt(QMainWindow):
    """PySide6 GUI with an embedded browser for ChatGPT prep + RoMa colorization."""

    BROWSER_PANEL_MIN_WIDTH = 430
    BROWSER_PANEL_MAX_WIDTH = 620

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Photo Colorizer (Qt + In-App Browser)")
        self.resize(1900, 1050)
        self.setMinimumSize(1300, 760)

        self.repo_root = Path(__file__).resolve().parent
        self.runner_script = self.repo_root / "run_romav2_pair.py"
        self.default_outdir = self.repo_root / "outputs" / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.browser_profile_dir = self.repo_root / ".qt_browser_profile"
        self.prewarm_cache_dir = self.repo_root / ".roma_prewarm_cache"
        self.browser_profile_dir.mkdir(parents=True, exist_ok=True)
        self.prewarm_cache_dir.mkdir(parents=True, exist_ok=True)

        self.process: QProcess | None = None
        self.prewarm_process: QProcess | None = None
        self.browser_page: AutoUploadWebPage | None = None
        self.pending_bw_upload_path: Path | None = None
        self._prewarm_setting: str | None = None
        self._prewarm_partial = ""
        self._prewarm_stop_requested = False
        self._awaiting_c1_download = False
        self._c1_download_request_seen = False
        self._download_terminal_logged: set[int] = set()
        self._run_started_monotonic: float | None = None
        self._last_process_output_monotonic: float | None = None
        self._last_run_watchdog_log_monotonic: float | None = None
        self._process_partial = ""
        self._result_pixmaps: dict[str, QPixmap] = {}
        self._current_preview_paths: dict[str, Path] = {}
        self._result_image_labels: dict[str, QLabel] = {}
        self._run_watchdog = QTimer(self)
        self._run_watchdog.setInterval(5000)
        self._run_watchdog.timeout.connect(self._on_run_watchdog_tick)

        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
            }
            QPushButton#primaryAction {
                font-weight: 700;
                min-height: 38px;
            }
            """
        )

        self._init_ui()
        self._load_outputs_placeholder()
        self._navigate_chatgpt()
        self._refresh_workflow_status()
        QTimer.singleShot(700, self._start_background_prewarm)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.setChildrenCollapsible(False)
        root_splitter.setHandleWidth(8)
        self.setCentralWidget(root_splitter)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)

        workflow_tab = self._build_workflow_tab()
        self.left_tabs.addTab(workflow_tab, "Workflow")

        self.output_tabs = QTabWidget()
        self.output_tabs.setDocumentMode(True)
        self.output_tabs.setMinimumHeight(260)
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(6, 6, 6, 6)
        results_layout.setSpacing(6)
        results_layout.addWidget(self.output_tabs, 1)
        self.left_tabs.addTab(results_tab, "Results")
        self._results_tab_index = self.left_tabs.count() - 1

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(6)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Pipeline logs will appear here.")
        log_layout.addWidget(self.log_box, 1)
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_box.clear)
        log_layout.addWidget(clear_log_btn)
        self.left_tabs.addTab(log_tab, "Log")
        self._log_tab_index = self.left_tabs.count() - 1

        left_layout.addWidget(self.left_tabs, 1)

        # ChatGPT panel (left side)
        right_panel = QWidget()
        right_panel.setMinimumWidth(self.BROWSER_PANEL_MIN_WIDTH)
        right_panel.setMaximumWidth(self.BROWSER_PANEL_MAX_WIDTH)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        right_layout.addWidget(self._build_browser_upload_group())
        right_layout.addWidget(self._build_browser_toolbar())
        self.browser = self._build_browser()
        self.browser.setMinimumWidth(self.BROWSER_PANEL_MIN_WIDTH - 40)
        self.browser.setMinimumHeight(250)
        self.browser.setMaximumHeight(360)
        right_layout.addWidget(self.browser, 1)

        # Put browser first so it appears on the LEFT, workflow on the RIGHT.
        root_splitter.addWidget(right_panel)
        root_splitter.addWidget(left_panel)
        root_splitter.setStretchFactor(0, 2)
        root_splitter.setStretchFactor(1, 9)
        root_splitter.setSizes([500, 1400])

    def _build_workflow_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab_layout.addWidget(controls_scroll, 1)

        controls_host = QWidget()
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(8)
        controls_scroll.setWidget(controls_host)

        controls_layout.addWidget(self._build_quickstart_group())
        controls_layout.addWidget(self._build_progress_group())
        controls_layout.addWidget(self._build_controls_group())
        controls_layout.addWidget(self._build_advanced_group())
        controls_layout.addStretch(1)
        return tab

    def _build_quickstart_group(self) -> QGroupBox:
        group = QGroupBox("Quick Workflow")
        layout = QVBoxLayout(group)
        quick_label = QLabel(
            "Use the C1 Workflow panel on the left:\n"
            "Choose B&W -> Add Photo -> Color Step 1 -> Down C1 -> Up C1 -> Colorize."
        )
        quick_label.setWordWrap(True)
        layout.addWidget(quick_label)
        return group

    def _build_progress_group(self) -> QGroupBox:
        group = QGroupBox("Workflow Status")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("B&W"), 0, 0)
        self.step1_state = QLabel("Waiting for B&W image")
        layout.addWidget(self.step1_state, 0, 1)

        layout.addWidget(QLabel("C1"), 1, 0)
        self.step2_state = QLabel("Waiting for colorized download")
        layout.addWidget(self.step2_state, 1, 1)

        layout.addWidget(QLabel("Run"), 2, 0)
        self.step4_state = QLabel("Cannot run yet")
        layout.addWidget(self.step4_state, 2, 1)

        layout.addWidget(QLabel("Warmup"), 3, 0)
        self.warmup_state = QLabel("Checking model cache")
        layout.addWidget(self.warmup_state, 3, 1)
        return group


    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Step 3: Output + Quality")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self.setting_combo = QComboBox()
        self.setting_combo.addItems(list(ROMA_SETTINGS))
        self.setting_combo.setCurrentText("fast")
        self.setting_combo.currentTextChanged.connect(lambda _: self._on_setting_changed_for_prewarm())
        form.addRow("Quality preset", self.setting_combo)

        self.color_opacity_spin = QDoubleSpinBox()
        self.color_opacity_spin.setDecimals(2)
        self.color_opacity_spin.setRange(0.0, 1.0)
        self.color_opacity_spin.setSingleStep(0.05)
        self.color_opacity_spin.setValue(1.0)
        form.addRow("Color opacity", self.color_opacity_spin)

        outdir_row = QHBoxLayout()
        self.outdir_edit = QLineEdit(str(self.default_outdir))
        outdir_row.addWidget(self.outdir_edit, 1)
        outdir_pick_btn = QPushButton("Browse")
        outdir_pick_btn.clicked.connect(self._pick_outdir)
        outdir_row.addWidget(outdir_pick_btn)
        outdir_widget = QWidget()
        outdir_widget.setLayout(outdir_row)
        form.addRow("Output folder", outdir_widget)

        helper = QLabel("Use defaults unless you need precise tuning. Fine-grain controls are in Advanced.")
        helper.setWordWrap(True)
        form.addRow("", helper)

        warm_row = QHBoxLayout()
        warm_hint = QLabel("Tip: model warmup runs in background so first Colorize is faster.")
        warm_hint.setWordWrap(True)
        warm_row.addWidget(warm_hint, 1)
        warm_btn = QPushButton("Warm Up Now")
        warm_btn.clicked.connect(lambda: self._start_background_prewarm(force=True))
        warm_row.addWidget(warm_btn)
        warm_widget = QWidget()
        warm_widget.setLayout(warm_row)
        form.addRow("", warm_widget)
        return group

    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Advanced (Optional)")
        group.setCheckable(True)
        group.setChecked(False)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self.gf_radius_spin = QSpinBox()
        self.gf_radius_spin.setRange(0, 48)
        self.gf_radius_spin.setValue(8)
        form.addRow("Edge smoothing radius", self.gf_radius_spin)

        self.gf_eps_spin = QDoubleSpinBox()
        self.gf_eps_spin.setDecimals(4)
        self.gf_eps_spin.setRange(0.0001, 1.0)
        self.gf_eps_spin.setSingleStep(0.0005)
        self.gf_eps_spin.setValue(0.001)
        form.addRow("Edge sensitivity", self.gf_eps_spin)

        self.chroma_radius_spin = QSpinBox()
        self.chroma_radius_spin.setRange(0, 128)
        self.chroma_radius_spin.setValue(24)
        form.addRow("Chroma filter radius", self.chroma_radius_spin)

        self.reg_thresh_spin = QDoubleSpinBox()
        self.reg_thresh_spin.setDecimals(2)
        self.reg_thresh_spin.setRange(0.0, 0.99)
        self.reg_thresh_spin.setSingleStep(0.01)
        self.reg_thresh_spin.setValue(0.35)
        form.addRow("Regularization threshold", self.reg_thresh_spin)

        self.reg_fallback_combo = QComboBox()
        self.reg_fallback_combo.addItems(["identity", "reference", "none"])
        self.reg_fallback_combo.setCurrentText("identity")
        form.addRow("Regularization fallback", self.reg_fallback_combo)

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(1, 200000)
        self.num_samples_spin.setValue(2000)
        form.addRow("Sample count", self.num_samples_spin)

        self.max_draw_spin = QSpinBox()
        self.max_draw_spin.setRange(1, 200000)
        self.max_draw_spin.setValue(800)
        form.addRow("Max points drawn", self.max_draw_spin)

        self.compile_check = QCheckBox("Enable torch.compile")
        form.addRow(self.compile_check)
        return group


    def _build_browser_upload_group(self) -> QGroupBox:
        group = QGroupBox("C1 Workflow")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        note = QLabel("Step 1: Choose a B&W photo. Then run: Add Photo -> Color Step 1 -> Down C1 -> Up C1 -> Colorize.")
        note.setWordWrap(True)
        layout.addWidget(note)

        # Step 1: pick B&W image
        bw_row = QHBoxLayout()
        self.bw_edit = QLineEdit()
        self.bw_edit.setPlaceholderText("Step 1: black-and-white photo path")
        self.bw_edit.textChanged.connect(lambda _: self._refresh_workflow_status())
        bw_row.addWidget(self.bw_edit, 1)
        pick_btn = QPushButton("Choose B&W")
        pick_btn.clicked.connect(self._on_pick_bw)
        bw_row.addWidget(pick_btn)
        layout.addLayout(bw_row)

        # Hidden/managed fields used by existing pipeline methods.
        self.color_edit = QLineEdit()
        self.color_edit.setPlaceholderText("C1 image path (filled by Up C1)")
        self.color_edit.textChanged.connect(lambda _: self._refresh_workflow_status())
        layout.addWidget(self.color_edit)

        self.prompt_box = QPlainTextEdit()
        self.prompt_box.setPlainText(DEFAULT_PREP_PROMPT)
        self.prompt_box.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.prompt_box.setMaximumHeight(90)
        self.prompt_box.setPlaceholderText("Prompt that will be sent in Color Step 1.")
        self.prompt_box.textChanged.connect(self._refresh_workflow_status)
        self.prompt_box.textChanged.connect(self._maybe_prefill_prompt_now)
        layout.addWidget(self.prompt_box)

        downloads_row = QHBoxLayout()
        self.downloads_edit = QLineEdit(self._default_downloads_dir())
        downloads_row.addWidget(self.downloads_edit, 1)
        dl_btn = QPushButton("Downloads")
        dl_btn.clicked.connect(self._pick_downloads_dir)
        downloads_row.addWidget(dl_btn)
        layout.addLayout(downloads_row)

        # Required button sequence.
        seq = QGridLayout()
        seq.setHorizontalSpacing(6)
        seq.setVerticalSpacing(6)

        add_photo_btn = QPushButton("Add Photo")
        add_photo_btn.clicked.connect(self._attach_selected_bw_to_chatgpt)
        seq.addWidget(add_photo_btn, 0, 0)

        color_step1_btn = QPushButton("Color Step 1")
        color_step1_btn.clicked.connect(self._color_step_1)
        seq.addWidget(color_step1_btn, 0, 1)

        down_c1_btn = QPushButton("Down C1")
        down_c1_btn.clicked.connect(self._download_c1_from_chatgpt)
        seq.addWidget(down_c1_btn, 0, 2)

        up_c1_btn = QPushButton("Up C1")
        up_c1_btn.clicked.connect(self._import_latest_download)
        seq.addWidget(up_c1_btn, 1, 0)

        self.run_btn = QPushButton("Colorize")
        self.run_btn.setObjectName("primaryAction")
        self.run_btn.clicked.connect(self._start_run)
        seq.addWidget(self.run_btn, 1, 1)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        seq.addWidget(self.stop_btn, 1, 2)

        layout.addLayout(seq)

        self.attach_status_label = QLabel("Attachment status: waiting")
        self.attach_status_label.setWordWrap(True)
        layout.addWidget(self.attach_status_label)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        return group

    def _build_browser_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        reload_btn = QPushButton("Reload")
        reload_btn.setMinimumWidth(64)
        reload_btn.clicked.connect(lambda: self.browser.reload())
        layout.addWidget(reload_btn)

        home_btn = QPushButton("ChatGPT")
        home_btn.setMinimumWidth(74)
        home_btn.clicked.connect(self._navigate_chatgpt)
        layout.addWidget(home_btn)

        external_btn = QPushButton("Open In Browser")
        external_btn.setMinimumWidth(120)
        external_btn.clicked.connect(self._open_chatgpt_external)
        layout.addWidget(external_btn)

        self.browser_hint = QLabel("Use ChatGPT here to generate the restored color image.")
        self.browser_hint.setWordWrap(True)
        layout.addWidget(self.browser_hint, 1)
        return bar

    def _build_browser(self) -> QWebEngineView:
        profile = QWebEngineProfile("RoMaQtBrowser", self)
        profile.setCachePath(str(self.browser_profile_dir / "cache"))
        profile.setPersistentStoragePath(str(self.browser_profile_dir / "storage"))
        profile.setPersistentCookiesPolicy(QWebEngineProfile.ForcePersistentCookies)
        profile.downloadRequested.connect(self._on_download_requested)

        page = AutoUploadWebPage(profile, self)
        self.browser_page = page
        page.auto_file_used.connect(self._on_auto_file_used_for_chatgpt)
        browser = QWebEngineView()
        browser.setPage(page)
        browser.loadFinished.connect(self._on_browser_load_finished)
        return browser

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self._refresh_workflow_status()

    def _set_attach_status(self, text: str, *, ok: bool | None = None) -> None:
        if not hasattr(self, "attach_status_label"):
            return
        self.attach_status_label.setText(f"Attachment status: {text}")
        if ok is True:
            self.attach_status_label.setStyleSheet("color: #3ea76a;")
        elif ok is False:
            self.attach_status_label.setStyleSheet("color: #d08b39;")
        else:
            self.attach_status_label.setStyleSheet("")

    @staticmethod
    def _set_step_label(label: QLabel, text: str, ready: bool) -> None:
        if ready:
            label.setText(f"Ready - {text}")
            label.setStyleSheet("color: #3ea76a;")
        else:
            label.setText(f"Pending - {text}")
            label.setStyleSheet("color: #c7a44b;")

    def _python_executable_for_child_process(self) -> Path:
        python_exec = Path(sys.executable)
        if python_exec.name.lower() == "pythonw.exe":
            py_exe = python_exec.with_name("python.exe")
            if py_exe.exists():
                python_exec = py_exe
        return python_exec

    def _prewarm_marker_path(self, setting: str) -> Path:
        safe_setting = "".join(ch for ch in setting if ch.isalnum() or ch in ("-", "_")).strip() or "default"
        return self.prewarm_cache_dir / f"prewarm_{safe_setting}_v1.json"

    def _is_setting_prewarmed(self, setting: str) -> bool:
        return self._prewarm_marker_path(setting).exists()

    def _mark_setting_prewarmed(self, setting: str) -> None:
        marker = self._prewarm_marker_path(setting)
        payload = {
            "setting": setting,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "python": str(self._python_executable_for_child_process()),
        }
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _on_setting_changed_for_prewarm(self) -> None:
        self._refresh_workflow_status()
        QTimer.singleShot(200, self._start_background_prewarm)

    def _start_background_prewarm(self, force: bool = False) -> None:
        setting = self.setting_combo.currentText() if hasattr(self, "setting_combo") else "fast"
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return
        if self.prewarm_process is not None and self.prewarm_process.state() != QProcess.NotRunning:
            return
        if not force and self._is_setting_prewarmed(setting):
            self._refresh_workflow_status()
            return

        ref_sample = self.repo_root / "assets" / "toronto_A.jpg"
        src_sample = self.repo_root / "assets" / "toronto_B.jpg"
        if not ref_sample.exists() or not src_sample.exists():
            self._append_log("[WARN] Warmup skipped: sample assets are missing.")
            return

        warmup_outdir = self.repo_root / "outputs" / "_warmup_cache"
        warmup_outdir.mkdir(parents=True, exist_ok=True)

        python_exec = self._python_executable_for_child_process()
        cmd = [
            str(python_exec),
            "-u",
            str(self.runner_script),
            "--ref",
            str(ref_sample),
            "--src",
            str(src_sample),
            "--outdir",
            str(warmup_outdir),
            "--setting",
            setting,
            "--num-samples",
            "32",
            "--max-draw",
            "16",
            "--guided-filter-radius",
            "0",
            "--guided-filter-eps",
            "0.001",
            "--chroma-filter-radius",
            "0",
            "--regularize-fallback",
            "none",
            "--diag-max-side",
            "512",
            "--filter-max-megapixels",
            "4",
            "--warmup-only",
        ]

        proc = QProcess(self)
        proc.setWorkingDirectory(str(self.repo_root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.started.connect(self._on_prewarm_started)
        proc.readyReadStandardOutput.connect(self._on_prewarm_output)
        proc.errorOccurred.connect(self._on_prewarm_error)
        proc.finished.connect(self._on_prewarm_finished)

        self.prewarm_process = proc
        self._prewarm_setting = setting
        self._prewarm_partial = ""
        self._prewarm_stop_requested = False
        self._append_log(f"[INFO] Starting background warmup for RoMa setting '{setting}'...")
        self._refresh_workflow_status()
        proc.start(cmd[0], cmd[1:])

    def _on_prewarm_started(self) -> None:
        setting = self._prewarm_setting or "unknown"
        self._append_log(f"[INFO] Warmup process started ({setting}).")
        self._refresh_workflow_status()

    def _on_prewarm_output(self) -> None:
        if self.prewarm_process is None:
            return
        raw = bytes(self.prewarm_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not raw:
            return
        self._prewarm_partial += raw
        while "\n" in self._prewarm_partial:
            line, self._prewarm_partial = self._prewarm_partial.split("\n", 1)
            line = line.rstrip()
            if not line:
                continue
            if "[ERROR]" in line or "[WARN]" in line or "[TIMING]" in line or "Warmup-only" in line:
                self._append_log(f"[WARMUP] {line}")

    def _on_prewarm_error(self, error) -> None:
        if self._prewarm_stop_requested:
            return
        name = str(error).split(".")[-1]
        self._append_log(f"[WARN] Warmup process error: {name}")
        self._refresh_workflow_status()

    def _on_prewarm_finished(self, exit_code: int, _exit_status) -> None:
        if self._prewarm_partial:
            trailing = self._prewarm_partial.strip()
            if trailing:
                self._append_log(f"[WARMUP] {trailing}")
            self._prewarm_partial = ""

        setting = self._prewarm_setting or "unknown"
        if self._prewarm_stop_requested:
            self._append_log(f"[INFO] Warmup stopped before completion for setting '{setting}'.")
        elif exit_code == 0:
            try:
                self._mark_setting_prewarmed(setting)
            except Exception as exc:
                self._append_log(f"[WARN] Could not write warmup marker: {exc}")
            self._append_log(f"[INFO] Warmup ready for setting '{setting}'.")
        else:
            self._append_log(f"[WARN] Warmup exited with code {exit_code} for setting '{setting}'.")

        self.prewarm_process = None
        self._prewarm_setting = None
        self._prewarm_stop_requested = False
        self._refresh_workflow_status()

    def _stop_prewarm_if_running(self) -> None:
        if self.prewarm_process is None or self.prewarm_process.state() == QProcess.NotRunning:
            return
        self._prewarm_stop_requested = True
        self.prewarm_process.terminate()
        if not self.prewarm_process.waitForFinished(1200):
            self.prewarm_process.kill()
        self.prewarm_process = None
        self._prewarm_setting = None
        self._prewarm_partial = ""
        self._append_log("[INFO] Background warmup stopped.")

    def _refresh_workflow_status(self) -> None:
        bw_path = Path(self.bw_edit.text().strip()) if hasattr(self, "bw_edit") else Path()
        color_path = Path(self.color_edit.text().strip()) if hasattr(self, "color_edit") else Path()
        bw_ok = bw_path.exists() and bw_path.is_file()
        color_ok = color_path.exists() and color_path.is_file()
        setting = self.setting_combo.currentText() if hasattr(self, "setting_combo") else "fast"
        prewarm_ready = self._is_setting_prewarmed(setting)
        prewarm_running = (
            self.prewarm_process is not None
            and self.prewarm_process.state() != QProcess.NotRunning
            and self._prewarm_setting == setting
        )

        prompt_ok = bool(self.prompt_box.toPlainText().strip()) if hasattr(self, "prompt_box") else False
        runner_ok = self.runner_script.exists()
        run_ready = bw_ok and color_ok and runner_ok

        self._set_step_label(self.step1_state, "B&W selected" if bw_ok else "Select B&W image", bw_ok)
        self._set_step_label(
            self.step2_state,
            "Colorized image imported" if color_ok else "Import/download colorized image",
            color_ok and prompt_ok,
        )
        self._set_step_label(
            self.step4_state,
            "Ready to run" if run_ready else "Need B&W + AI Colorized image",
            run_ready,
        )
        warmup_text = f"Warm model cache for '{setting}'"
        warmup_ok = prewarm_ready
        if prewarm_running:
            warmup_text = f"Warming model cache for '{setting}'"
            warmup_ok = False
        self._set_step_label(self.warmup_state, warmup_text, warmup_ok)

        running = self.process is not None and self.process.state() != QProcess.NotRunning
        if hasattr(self, "run_btn") and not running:
            self.run_btn.setEnabled(run_ready)

    def _show_results_tab(self) -> None:
        if hasattr(self, "left_tabs"):
            self.left_tabs.setCurrentIndex(self._results_tab_index)

    def _show_log_tab(self) -> None:
        if hasattr(self, "left_tabs"):
            self.left_tabs.setCurrentIndex(self._log_tab_index)

    # ------------------------------------------------------------------
    # Browser actions
    # ------------------------------------------------------------------

    @staticmethod
    def _default_downloads_dir() -> str:
        locations = QStandardPaths.standardLocations(QStandardPaths.DownloadLocation)
        if locations:
            return locations[0]
        return str(Path.home() / "Downloads")

    def _navigate_chatgpt(self) -> None:
        self.browser.setUrl(QUrl("https://chatgpt.com/"))

    def _open_chatgpt_external(self) -> None:
        QDesktopServices.openUrl(QUrl("https://chatgpt.com/"))

    def _on_browser_load_finished(self, ok: bool) -> None:
        if not ok:
            return
        current = self.browser.url().toString().lower()
        if "chatgpt.com" not in current:
            return
        # Try a few times because ChatGPT's composer appears after initial page load.
        self._auto_prefill_prompt(retries=5, delay_ms=700)
        if self.pending_bw_upload_path is not None and self.pending_bw_upload_path.exists():
            QTimer.singleShot(900, lambda: self._auto_upload_bw_to_chatgpt(self.pending_bw_upload_path, retries=5, delay_ms=800))

    def _on_auto_file_used_for_chatgpt(self, path_str: str) -> None:
        self._set_attach_status("file selected for ChatGPT", ok=True)
        self._append_log(f"[INFO] ChatGPT file chooser auto-filled: {path_str}")

    def _auto_prefill_prompt(self, retries: int, delay_ms: int) -> None:
        prompt = self.prompt_box.toPlainText().strip() if hasattr(self, "prompt_box") else ""
        if not prompt:
            return

        def _after(success: bool) -> None:
            if success:
                self._append_log("[INFO] Prompt pre-filled into ChatGPT composer.")
                return
            if retries - 1 <= 0:
                QGuiApplication.clipboard().setText(prompt)
                self._append_log(
                    "[WARN] Auto pre-fill failed. Prompt copied to clipboard; click chat box and paste."
                )
                return
            QTimer.singleShot(delay_ms, lambda: self._auto_prefill_prompt(retries - 1, delay_ms))

        self._inject_prompt_into_chatgpt(prompt, silent=True, done=_after)

    def _maybe_prefill_prompt_now(self) -> None:
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            return
        QTimer.singleShot(120, lambda: self._auto_prefill_prompt(retries=2, delay_ms=500))

    @staticmethod
    def _chatgpt_trigger_upload_js() -> str:
        return """
(() => {
  const directInput = document.querySelector('input[type="file"]');
  if (directInput) {
    directInput.click();
    return { ok: true, via: 'direct_file_input' };
  }

  const attachSelectors = [
    'button[aria-label*="attach" i]',
    'button[aria-label*="upload" i]',
    'button[title*="attach" i]',
    'button[title*="upload" i]',
    '[data-testid*="attach" i]',
    '[data-testid*="upload" i]',
    'label[for*="file" i]',
    'button'
  ];

  for (const sel of attachSelectors) {
    const nodes = Array.from(document.querySelectorAll(sel));
    for (const node of nodes) {
      const text = (
        (node.innerText || '') + ' ' +
        (node.getAttribute?.('aria-label') || '') + ' ' +
        (node.getAttribute?.('title') || '')
      ).toLowerCase();
      const looksAttach =
        sel !== 'button' ||
        text.includes('attach') ||
        text.includes('upload') ||
        text.includes('file') ||
        text.trim() === '+' ||
        text.includes('plus');
      if (!looksAttach) continue;
      node.click();
      setTimeout(() => {
        const inputAfterClick = document.querySelector('input[type="file"]');
        if (inputAfterClick) {
          inputAfterClick.click();
        }
      }, 120);
      return { ok: true, via: 'attach_control' };
    }
  }

  const menuItems = Array.from(document.querySelectorAll('[role="menuitem"], button, div[role="button"]'));
  const uploadItem = menuItems.find((m) => {
    const text = (
      (m.innerText || '') + ' ' +
      (m.getAttribute?.('aria-label') || '') + ' ' +
      (m.getAttribute?.('title') || '')
    ).toLowerCase();
    return text.includes('upload') && (text.includes('computer') || text.includes('file') || text.includes('photo'));
  });

  if (uploadItem) {
    uploadItem.click();
    setTimeout(() => {
      const inputAfterMenu = document.querySelector('input[type="file"]');
      if (inputAfterMenu) {
        inputAfterMenu.click();
      }
    }, 120);
    return { ok: true, via: 'upload_menu_item' };
  }

  return { ok: false, reason: 'attach_control_not_found' };
})();
"""

    @staticmethod
    def _chatgpt_send_js() -> str:
        return """
(() => {
  const visible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const tryClick = (el) => {
    if (!el) return false;
    try { el.focus?.(); } catch {}
    try { el.click(); return true; } catch {}
    return false;
  };

  const selectors = [
    'button[data-testid*="composer-send" i]',
    'button[data-testid*="send-button" i]',
    'button[data-testid*="send" i]',
    'button[aria-label*="send" i]',
    'button[title*="send" i]',
    'button[type="submit"]',
    '[role="button"][aria-label*="send" i]'
  ];

  let disabledVia = null;
  for (const sel of selectors) {
    const nodes = Array.from(document.querySelectorAll(sel)).filter(visible);
    if (nodes.length > 0 && disabledVia === null) {
      disabledVia = sel;
    }
    const usable = nodes.find((n) => !n.disabled && n.getAttribute('aria-disabled') !== 'true');
    if (usable && tryClick(usable)) {
      return { ok: true, via: sel };
    }
  }

  const buttons = Array.from(document.querySelectorAll('button,[role="button"]')).filter(visible);
  const fallback = buttons.find((b) => {
    const text = ((b.innerText || '') + ' ' + (b.getAttribute?.('aria-label') || '')).toLowerCase();
    return text.includes('send');
  });
  if (fallback && tryClick(fallback)) {
    return { ok: true, via: 'send_text_fallback' };
  }

  if (disabledVia !== null) {
    return { ok: false, reason: 'send_disabled', via: disabledVia };
  }

  return { ok: false, reason: 'send_button_not_found' };
})();
"""

    @staticmethod
    def _chatgpt_download_js() -> str:
        return """
(() => {
  const visible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const tryClick = (el) => {
    if (!el) return false;
    try { el.focus?.(); } catch {}
    try { el.click(); return true; } catch {}
    return false;
  };

  const imgs = Array.from(document.querySelectorAll('img')).filter(visible);
  const latestImg = imgs.length ? imgs[imgs.length - 1] : null;
  if (latestImg) {
    try {
      latestImg.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
      latestImg.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
    } catch {}
  }

  const directDownloads = Array.from(document.querySelectorAll('a[download],button[download]')).filter(visible);
  if (directDownloads.length > 0) {
    const direct = directDownloads[directDownloads.length - 1];
    if (tryClick(direct)) {
      return { ok: true, via: 'direct_download_attr' };
    }
  }

  const nodes = Array.from(document.querySelectorAll('button,a,[role="button"],[role="menuitem"]')).filter(visible);
  const matches = nodes.filter((n) => {
    const text = (
      (n.innerText || '') + ' ' +
      (n.getAttribute?.('aria-label') || '') + ' ' +
      (n.getAttribute?.('title') || '')
    ).toLowerCase();
    const testid = (n.getAttribute?.('data-testid') || '').toLowerCase();
    return (
      text.includes('download') ||
      text.includes('save image') ||
      text.includes('save photo') ||
      testid.includes('download')
    );
  });

  if (matches.length > 0) {
    const target = matches[matches.length - 1];
    if (tryClick(target)) {
      return { ok: true, via: 'download_button' };
    }
  }

  return { ok: false, reason: 'download_control_not_found' };
})();
"""

    @staticmethod
    def _chatgpt_open_download_menu_js() -> str:
        return """
(() => {
  const visible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const tryClick = (el) => {
    if (!el) return false;
    try { el.focus?.(); } catch {}
    try { el.click(); return true; } catch {}
    return false;
  };
  const score = (n) => {
    const text = (
      (n.innerText || '') + ' ' +
      (n.getAttribute?.('aria-label') || '') + ' ' +
      (n.getAttribute?.('title') || '') + ' ' +
      (n.getAttribute?.('data-testid') || '')
    ).toLowerCase();
    let s = 0;
    if (text.includes('more')) s += 4;
    if (text.includes('option')) s += 4;
    if (text.includes('action')) s += 4;
    if (text.includes('menu')) s += 2;
    if (text.includes('ellipsis')) s += 2;
    if (text.includes('...')) s += 1;
    return s;
  };

  const buttons = Array.from(document.querySelectorAll('button,[role="button"]')).filter(visible);
  const ranked = buttons
    .map((n, idx) => ({ n, idx, s: score(n) }))
    .filter((x) => x.s > 0)
    .sort((a, b) => (a.s - b.s) || (a.idx - b.idx));

  if (ranked.length > 0) {
    const target = ranked[ranked.length - 1].n;
    if (tryClick(target)) {
      return { ok: true, via: 'more_actions' };
    }
  }

  return { ok: false, reason: 'more_actions_not_found' };
})();
"""

    def _attempt_chatgpt_download(self, retries: int = 6, delay_ms: int = 600) -> None:
        if self._c1_download_request_seen:
            return
        page = self.browser.page()
        if page is None:
            return
        js = self._chatgpt_download_js()

        def _after(result) -> None:
            if self._c1_download_request_seen:
                return
            ok = isinstance(result, dict) and bool(result.get("ok"))
            if ok:
                via = str(result.get("via", "unknown"))
                self._append_log(f"[INFO] Down C1 control clicked ({via}).")
                if retries > 0 and self._awaiting_c1_download and not self._c1_download_request_seen:
                    QTimer.singleShot(delay_ms, lambda: self._attempt_chatgpt_download(retries - 1, delay_ms))
                return

            if retries <= 0:
                if not self._c1_download_request_seen:
                    self._awaiting_c1_download = False
                    self._append_log(
                        "[WARN] Could not auto-click a Download control. Use ChatGPT's download button manually, then click Up C1."
                    )
                return

            if retries in (6, 3):
                self._append_log("[INFO] Looking for download control in ChatGPT...")

            menu_js = self._chatgpt_open_download_menu_js()

            def _after_menu(_menu_result) -> None:
                QTimer.singleShot(delay_ms, lambda: self._attempt_chatgpt_download(retries - 1, delay_ms))

            page.runJavaScript(menu_js, _after_menu)

        page.runJavaScript(js, _after)

    @staticmethod
    def _sanitize_download_filename(filename: str) -> str:
        base = Path(filename).name.strip() or "chatgpt_image.png"
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", base).strip(" .")
        if not cleaned:
            cleaned = "chatgpt_image.png"
        suffix = Path(cleaned).suffix
        if not suffix:
            cleaned += ".png"
        return cleaned

    @staticmethod
    def _unique_download_path(downloads_dir: Path, filename: str) -> Path:
        candidate = downloads_dir / filename
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        i = 1
        while True:
            probe = downloads_dir / f"{stem}_{i}{suffix}"
            if not probe.exists():
                return probe
            i += 1

    @staticmethod
    def _download_state_name(item) -> str:
        try:
            return str(item.state()).split(".")[-1]
        except Exception:
            return "Unknown"

    def _on_download_state_changed(self, item, target_path: Path) -> None:
        item_id = id(item)
        if item_id in self._download_terminal_logged:
            return
        state_name = self._download_state_name(item).lower()
        if "completed" in state_name:
            self._download_terminal_logged.add(item_id)
            if target_path.exists():
                self._append_log(f"[INFO] Download complete: {target_path}")
            else:
                self._append_log(f"[WARN] Download reported complete but file not found: {target_path}")
            return
        if "cancel" in state_name or "interrupt" in state_name:
            self._download_terminal_logged.add(item_id)
            reason = ""
            try:
                reason = f" ({item.interruptReasonString()})"
            except Exception:
                reason = ""
            self._append_log(f"[WARN] Download failed: {target_path}{reason}")

    @staticmethod
    def _chatgpt_focus_composer_js() -> str:
        return """
(() => {
  const selectors = [
    'textarea[data-testid="prompt-textarea"]',
    'textarea#prompt-textarea',
    'textarea[placeholder*="Message"]',
    'textarea',
    'div[contenteditable="true"][data-testid="prompt-textarea"]',
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"]'
  ];

  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (!el) continue;
    try {
      el.focus();
      return { ok: true, via: sel };
    } catch (err) {
      return { ok: false, reason: 'focus_failed', detail: String(err) };
    }
  }

  return { ok: false, reason: 'composer_not_found' };
})();
"""

    def _attach_bw_via_clipboard_paste(self, bw_path: Path) -> None:
        if not bw_path.exists() or not bw_path.is_file():
            return
        image = QImage(str(bw_path))
        if image.isNull():
            self._append_log(
                "[WARN] Clipboard fallback failed: selected image could not be loaded for paste."
            )
            self._set_attach_status("auto-attach failed, use + manually", ok=False)
            return

        QGuiApplication.clipboard().setImage(image)
        page = self.browser.page()
        if page is None:
            return

        self._append_log("[INFO] Auto-open failed; trying clipboard paste fallback (Ctrl+V).")
        self._set_attach_status("trying clipboard paste fallback", ok=None)
        js = self._chatgpt_focus_composer_js()

        def _after_focus(result) -> None:
            target = self.browser.focusProxy()
            if not isinstance(target, QWidget):
                target = self.browser
            target.setFocus(Qt.OtherFocusReason)
            self.browser.setFocus(Qt.OtherFocusReason)
            QTest.keyClick(target, Qt.Key_V, Qt.ControlModifier)

            ok = isinstance(result, dict) and bool(result.get("ok"))
            if ok:
                self._append_log("[INFO] Sent Ctrl+V to ChatGPT composer with selected B&W image on clipboard.")
            else:
                reason = "unknown"
                if isinstance(result, dict):
                    reason = str(result.get("reason", "unknown"))
                self._append_log(
                    "[WARN] Could not focus ChatGPT composer automatically "
                    f"({reason}); Ctrl+V was still sent to the browser."
                )
            self._set_attach_status("paste attempted; confirm thumbnail appears", ok=None)

        page.runJavaScript(js, _after_focus)

    def _fallback_if_upload_not_consumed(self, bw_path: Path) -> None:
        if self.browser_page is None:
            return
        pending = self.browser_page.pending_upload_path
        if pending is None:
            return
        try:
            same_file = pending.resolve() == bw_path.resolve()
        except Exception:
            same_file = str(pending) == str(bw_path)
        if not same_file:
            return
        self._append_log(
            "[WARN] ChatGPT upload control did not consume the selected file; switching to clipboard paste fallback."
        )
        self._attach_bw_via_clipboard_paste(bw_path)

    def _chatgpt_send_enter_fallback(self) -> None:
        page = self.browser.page()
        if page is None:
            return
        js = self._chatgpt_focus_composer_js()

        def _after_focus(result) -> None:
            target = self.browser.focusProxy()
            if not isinstance(target, QWidget):
                target = self.browser
            target.setFocus(Qt.OtherFocusReason)
            self.browser.setFocus(Qt.OtherFocusReason)
            QTest.keyClick(target, Qt.Key_Return, Qt.NoModifier)
            ok = isinstance(result, dict) and bool(result.get("ok"))
            if ok:
                self._append_log("[INFO] Send fallback: pressed Enter in ChatGPT composer.")
            else:
                reason = "unknown"
                if isinstance(result, dict):
                    reason = str(result.get("reason", "unknown"))
                self._append_log(
                    "[WARN] Send fallback used Enter without confirmed composer focus "
                    f"({reason})."
                )

        page.runJavaScript(js, _after_focus)

    def _attempt_chatgpt_send(self, retries: int = 10, delay_ms: int = 700) -> None:
        page = self.browser.page()
        if page is None:
            return
        js = self._chatgpt_send_js()

        def _after_send(result) -> None:
            ok = isinstance(result, dict) and bool(result.get("ok"))
            if ok:
                via = str(result.get("via", "unknown"))
                self._append_log(f"[INFO] Color Step 1 sent to ChatGPT ({via}).")
                return

            reason = "unknown"
            if isinstance(result, dict):
                reason = str(result.get("reason", "unknown"))

            if reason == "send_disabled" and retries > 0:
                if retries in (10, 7, 4, 1):
                    self._append_log("[INFO] Waiting for ChatGPT send control to become ready...")
                QTimer.singleShot(delay_ms, lambda: self._attempt_chatgpt_send(retries - 1, delay_ms))
                return

            self._append_log("[WARN] Send button click failed. Trying Enter-key fallback.")
            self._chatgpt_send_enter_fallback()

        page.runJavaScript(js, _after_send)

    def _color_step_1(self) -> None:
        prompt = self.prompt_box.toPlainText().strip() if hasattr(self, "prompt_box") else ""
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Prompt is empty.")
            return
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            QMessageBox.warning(self, "ChatGPT not open", "Open ChatGPT in the embedded browser first.")
            return

        self._append_log("[INFO] Color Step 1: filling prompt and sending to ChatGPT.")

        def _after_prefill(success: bool) -> None:
            if not success:
                self._append_log(
                    "[WARN] Prompt auto-fill not confirmed; attempting send with current ChatGPT composer contents."
                )
            QTimer.singleShot(220, lambda: self._attempt_chatgpt_send(retries=10, delay_ms=700))

        self._inject_prompt_into_chatgpt(prompt, silent=False, done=_after_prefill)

    def _download_c1_from_chatgpt(self) -> None:
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            QMessageBox.warning(self, "ChatGPT not open", "Open ChatGPT in the embedded browser first.")
            return
        page = self.browser.page()
        if page is None:
            return

        self._append_log("[INFO] Attempting Down C1 from ChatGPT.")
        self._awaiting_c1_download = True
        self._c1_download_request_seen = False
        self._attempt_chatgpt_download(retries=6, delay_ms=600)

    def _attach_selected_bw_to_chatgpt(self) -> None:
        bw_path = Path(self.bw_edit.text().strip()) if hasattr(self, "bw_edit") else Path()
        if not bw_path.exists() or not bw_path.is_file():
            QMessageBox.warning(self, "Missing image", "Select a B&W image first.")
            return
        self._append_log("[INFO] Attempting to attach selected B&W image in ChatGPT.")
        self._auto_upload_bw_to_chatgpt(bw_path, retries=6, delay_ms=700)

    def _auto_upload_bw_to_chatgpt(self, bw_path: Path, retries: int = 3, delay_ms: int = 700) -> None:
        if not bw_path.exists() or not bw_path.is_file():
            return
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            self._append_log("[INFO] Open ChatGPT first to auto-upload the selected B&W image.")
            self._set_attach_status("open ChatGPT first", ok=False)
            return

        if self.browser_page is None:
            return
        self.pending_bw_upload_path = bw_path
        self.browser_page.set_pending_upload_path(bw_path)
        self._set_attach_status("armed (click + in ChatGPT if needed)", ok=None)

        page = self.browser.page()
        if page is None:
            return

        js = self._chatgpt_trigger_upload_js()

        def _after(result) -> None:
            ok = isinstance(result, dict) and bool(result.get("ok"))
            if ok:
                via = str(result.get("via", "unknown"))
                self._append_log(f"[INFO] Triggered ChatGPT upload flow ({via}) using selected B&W image.")
                self._set_attach_status("upload dialog triggered", ok=True)
                QTimer.singleShot(1200, lambda: self._fallback_if_upload_not_consumed(bw_path))
                return
            if retries - 1 <= 0:
                self._append_log(
                    "[WARN] Could not auto-open ChatGPT upload control due browser restrictions. "
                    "Trying clipboard paste fallback now."
                )
                self._attach_bw_via_clipboard_paste(bw_path)
                return
            QTimer.singleShot(delay_ms, lambda: self._auto_upload_bw_to_chatgpt(bw_path, retries - 1, delay_ms))

        page.runJavaScript(js, _after)

    def _on_download_requested(self, item) -> None:
        downloads_dir = Path(self.downloads_edit.text().strip() or self._default_downloads_dir())
        downloads_dir.mkdir(parents=True, exist_ok=True)
        raw_filename = item.downloadFileName() or "chatgpt_image.png"
        safe_filename = self._sanitize_download_filename(raw_filename)
        target_path = self._unique_download_path(downloads_dir, safe_filename)
        item.setDownloadDirectory(str(downloads_dir))
        item.setDownloadFileName(target_path.name)
        try:
            item.stateChanged.connect(lambda _state, it=item, tp=target_path: self._on_download_state_changed(it, tp))
        except Exception:
            pass
        item.accept()
        if self._awaiting_c1_download:
            self._c1_download_request_seen = True
            self._awaiting_c1_download = False
            self._append_log("[INFO] Down C1 succeeded: browser download started.")
        self._append_log(f"[INFO] Download started: {target_path}")

    # ------------------------------------------------------------------
    # Pickers + previews
    # ------------------------------------------------------------------

    def _pick_downloads_dir(self) -> None:
        start = self.downloads_edit.text().strip() or self._default_downloads_dir()
        selected = QFileDialog.getExistingDirectory(self, "Select downloads folder", start)
        if selected:
            self.downloads_edit.setText(selected)

    def _on_pick_bw(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Select B&W image", "", IMAGE_FILTER)
        if selected:
            self.bw_edit.setText(selected)
            self.pending_bw_upload_path = None
            self._set_attach_status("photo selected (click Add Photo)", ok=None)

    def _pick_outdir(self) -> None:
        start = self.outdir_edit.text().strip() or str(self.repo_root / "outputs")
        selected = QFileDialog.getExistingDirectory(self, "Select output directory", start)
        if selected:
            self.outdir_edit.setText(selected)

    @staticmethod
    def _chatgpt_prompt_js(prompt: str) -> str:
        prompt_json = json.dumps(prompt)
        return f"""
(() => {{
  const prompt = {prompt_json};
  const selectors = [
    'textarea[data-testid="prompt-textarea"]',
    'textarea#prompt-textarea',
    'textarea[placeholder*="Message"]',
    'textarea',
    'div[contenteditable="true"][data-testid="prompt-textarea"]',
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"]'
  ];

  let el = null;
  for (const sel of selectors) {{
    const found = document.querySelector(sel);
    if (found) {{
      el = found;
      break;
    }}
  }}

  if (!el) {{
    return {{ ok: false, reason: 'input_not_found' }};
  }}

  try {{
    el.focus();
  }} catch (err) {{
    return {{ ok: false, reason: 'focus_failed' }};
  }}

  if (el.tagName === 'TEXTAREA') {{
    el.value = prompt;
    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
    return {{ ok: true, mode: 'textarea' }};
  }}

  try {{
    if (typeof document.execCommand === 'function') {{
      const range = document.createRange();
      range.selectNodeContents(el);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand('insertText', false, prompt);
    }}
    if (el.textContent !== prompt) {{
      el.textContent = prompt;
    }}
    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
    return {{ ok: true, mode: 'contenteditable' }};
  }} catch (err) {{
    return {{ ok: false, reason: 'inject_failed', detail: String(err) }};
  }}
}})();
"""

    def _inject_prompt_into_chatgpt(
        self,
        prompt: str,
        *,
        silent: bool,
        done=None,
    ) -> None:
        page = self.browser.page()
        if page is None:
            if not silent:
                QMessageBox.warning(self, "Browser unavailable", "Embedded browser is not ready.")
            if done is not None:
                done(False)
            return

        js = self._chatgpt_prompt_js(prompt)

        def _on_done(result) -> None:
            success = isinstance(result, dict) and bool(result.get("ok"))
            if success:
                if not silent:
                    mode = result.get("mode", "unknown")
                    self._append_log(f"[INFO] Prompt transferred to ChatGPT composer ({mode}).")
                if done is not None:
                    done(True)
                return

            if done is not None:
                done(False)
                if silent:
                    return

            reason = "chat_input_not_ready"
            if isinstance(result, dict):
                reason = str(result.get("reason", "unknown"))
            QGuiApplication.clipboard().setText(prompt)
            self._append_log(
                "[WARN] Could not auto-transfer prompt "
                f"({reason}). Prompt copied to clipboard instead. Click inside ChatGPT message box and paste."
            )

        page.runJavaScript(js, _on_done)

    def _import_latest_download(self) -> None:
        downloads_dir = Path(self.downloads_edit.text().strip() or self._default_downloads_dir())
        if not downloads_dir.exists():
            QMessageBox.warning(self, "Missing folder", f"Downloads folder not found:\n{downloads_dir}")
            return

        candidates = [p for p in downloads_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not candidates:
            QMessageBox.warning(self, "No images found", f"No image files in:\n{downloads_dir}")
            return

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        self.color_edit.setText(str(latest))
        self._append_log(f"[INFO] Imported latest download: {latest}")
        self._log_border_ratio_if_possible()
        self._refresh_workflow_status()

    def _log_border_ratio_if_possible(self) -> None:
        bw_path = Path(self.bw_edit.text().strip())
        color_path = Path(self.color_edit.text().strip())
        if not bw_path.exists() or not color_path.exists():
            return

        try:
            bw_w, bw_h = Image.open(bw_path).size
            color_w, color_h = Image.open(color_path).size
            if bw_w <= 0 or bw_h <= 0:
                return
            ratio_w = color_w / bw_w
            ratio_h = color_h / bw_h
            self._append_log(
                f"[INFO] Size ratio color/bw: {ratio_w:.3f}x width, {ratio_h:.3f}x height "
                "(~1.5x means 25% border each side)."
            )
        except Exception as exc:
            self._append_log(f"[WARN] Could not read image sizes: {exc}")

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------

    def _start_run(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Run in progress", "A run is already in progress.")
            return
        if self.prewarm_process is not None and self.prewarm_process.state() != QProcess.NotRunning:
            self._append_log("[INFO] Warmup is running. Waiting briefly for it to finish...")
            if not self.prewarm_process.waitForFinished(10000):
                self._append_log("[INFO] Warmup still running; stopping it and continuing with Colorize now.")
                self._stop_prewarm_if_running()

        if not self.runner_script.exists():
            QMessageBox.critical(self, "Missing runner", f"Could not find:\n{self.runner_script}")
            return

        bw_path = Path(self.bw_edit.text().strip())
        color_path = Path(self.color_edit.text().strip())
        if not bw_path.exists():
            QMessageBox.warning(self, "Missing image", f"B&W original not found:\n{bw_path}")
            return
        if not color_path.exists():
            QMessageBox.warning(self, "Missing image", f"AI colorized image not found:\n{color_path}")
            return

        outdir_text = self.outdir_edit.text().strip()
        if not outdir_text:
            outdir_text = str(self.repo_root / "outputs" / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.outdir_edit.setText(outdir_text)
        outdir = Path(outdir_text)
        outdir.mkdir(parents=True, exist_ok=True)

        python_exec = self._python_executable_for_child_process()

        cmd = [
            str(python_exec),
            "-u",
            str(self.runner_script),
            "--ref",
            str(bw_path),
            "--src",
            str(color_path),
            "--outdir",
            str(outdir),
            "--setting",
            self.setting_combo.currentText(),
            "--num-samples",
            str(self.num_samples_spin.value()),
            "--max-draw",
            str(self.max_draw_spin.value()),
            "--regularize-overlap-thresh",
            str(self.reg_thresh_spin.value()),
            "--regularize-fallback",
            self.reg_fallback_combo.currentText(),
            "--guided-filter-radius",
            str(self.gf_radius_spin.value()),
            "--guided-filter-eps",
            str(self.gf_eps_spin.value()),
            "--chroma-filter-radius",
            str(self.chroma_radius_spin.value()),
            "--color-opacity",
            str(self.color_opacity_spin.value()),
        ]
        if self.compile_check.isChecked():
            cmd.append("--compile")

        self._append_log("")
        self._append_log("=== Colorization Run ===")
        self._append_log(f"B&W Original: {bw_path}")
        self._append_log(f"AI Colorized: {color_path}")
        self._append_log(f"Output: {outdir}")
        self._append_log("Command: " + " ".join(shlex.quote(p) for p in cmd))
        self._clear_preview_tabs()

        proc = QProcess(self)
        proc.setWorkingDirectory(str(self.repo_root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.started.connect(self._on_process_started)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.errorOccurred.connect(self._on_process_error)
        proc.finished.connect(self._on_process_finished)
        self.process = proc
        self._process_partial = ""
        now = time.monotonic()
        self._run_started_monotonic = now
        self._last_process_output_monotonic = now
        self._last_run_watchdog_log_monotonic = now

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status("Running...")
        self._run_watchdog.start()

        proc.start(cmd[0], cmd[1:])

    def _on_process_started(self) -> None:
        if self.process is None:
            return
        try:
            pid = int(self.process.processId())
        except Exception:
            pid = 0
        if pid > 0:
            self._append_log(f"[INFO] Colorize runner started (pid={pid}).")
        else:
            self._append_log("[INFO] Colorize runner started.")

    def _on_process_output(self) -> None:
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not raw:
            return
        self._last_process_output_monotonic = time.monotonic()
        self._process_partial += raw
        while "\n" in self._process_partial:
            line, self._process_partial = self._process_partial.split("\n", 1)
            self._append_log(line.rstrip())

    def _on_process_error(self, error) -> None:
        if self.process is None:
            return
        name = str(error).split(".")[-1]
        self._append_log(f"[ERROR] Runner process error: {name}")
        if self.process.state() == QProcess.NotRunning:
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.process = None
            self._run_watchdog.stop()
            self._run_started_monotonic = None
            self._last_process_output_monotonic = None
            self._last_run_watchdog_log_monotonic = None
            self._set_status("Failed")
            self._refresh_workflow_status()

    def _on_run_watchdog_tick(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self._run_watchdog.stop()
            return

        now = time.monotonic()
        started = self._run_started_monotonic if self._run_started_monotonic is not None else now
        last_out = self._last_process_output_monotonic if self._last_process_output_monotonic is not None else started
        elapsed = max(0.0, now - started)
        idle = max(0.0, now - last_out)
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        self.status_label.setText(f"Running... {elapsed_min:02d}:{elapsed_sec:02d}")

        last_watchdog = (
            self._last_run_watchdog_log_monotonic
            if self._last_run_watchdog_log_monotonic is not None
            else started
        )
        if idle >= 30.0 and (now - last_watchdog) >= 25.0:
            self._append_log(
                "[INFO] Colorize still running "
                f"(elapsed {elapsed_min:02d}:{elapsed_sec:02d}, no new logs for {int(idle)}s)."
            )
            self._append_log(
                "[INFO] This can be normal during first-run model initialization/downloading or large-image matching."
            )
            self._last_run_watchdog_log_monotonic = now

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        if self._process_partial:
            self._append_log(self._process_partial.rstrip())
            self._process_partial = ""

        if exit_code == 0:
            self._append_log("Colorization complete.")
            self._load_outputs()
            self._set_status("Done!")
            self._show_results_tab()
        else:
            self._append_log(f"Pipeline exited with error code {exit_code}.")
            self._set_status("Failed")
            self._show_log_tab()

        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process = None
        self._run_watchdog.stop()
        self._run_started_monotonic = None
        self._last_process_output_monotonic = None
        self._last_run_watchdog_log_monotonic = None
        self._refresh_workflow_status()

    def _stop_run(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self._append_log("[INFO] Stop requested.")
        self._set_status("Stopping...")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_prewarm_if_running()
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self._stop_run()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _clear_preview_tabs(self) -> None:
        while self.output_tabs.count() > 0:
            self.output_tabs.removeTab(0)
        self._result_pixmaps.clear()
        self._current_preview_paths.clear()
        self._result_image_labels.clear()

    def _load_outputs_placeholder(self) -> None:
        placeholder = QLabel("Load images and click Colorize to see results here.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.output_tabs.addTab(placeholder, "Results")

    def _load_outputs(self) -> None:
        outdir = Path(self.outdir_edit.text().strip())
        self._clear_preview_tabs()
        found_any = False

        for title, filename in PREVIEW_FILES:
            path = outdir / filename
            if not path.exists():
                continue
            self._add_preview_tab(title, path)
            found_any = True

        if not found_any:
            self._append_log(f"[WARN] No output images found in: {outdir}")
            self._load_outputs_placeholder()
        else:
            self.output_tabs.setCurrentIndex(0)

        summary_path = outdir / "summary.txt"
        if summary_path.exists():
            self._append_log("")
            self._append_log("--- Summary ---")
            try:
                for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("mae_"):
                        self._append_log(f"  {line}")
            except Exception as exc:
                self._append_log(f"[WARN] Could not read summary: {exc}")

    def _add_preview_tab(self, title: str, image_path: Path) -> None:
        pix = QPixmap(str(image_path))
        if pix.isNull():
            self._append_log(f"[WARN] Could not load preview: {image_path}")
            return

        self._result_pixmaps[title] = pix
        self._current_preview_paths[title] = image_path

        tab = QWidget()
        layout = QVBoxLayout(tab)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setMinimumHeight(320)
        image_label.setPixmap(pix.scaled(900, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image_label, 1)

        path_label = QLabel(str(image_path))
        layout.addWidget(path_label)
        self._result_image_labels[title] = image_label
        self.output_tabs.addTab(tab, title)
        self._refresh_output_previews()

    def _refresh_output_previews(self) -> None:
        if not self._result_image_labels:
            return
        for title, label in self._result_image_labels.items():
            pix = self._result_pixmaps.get(title)
            if pix is None or pix.isNull():
                continue
            target_w = max(120, label.width() - 12)
            target_h = max(120, label.height() - 12)
            scaled = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        super().resizeEvent(event)
        self._refresh_output_previews()


def main() -> int:
    app = QApplication(sys.argv)
    win = PhotoColorizerQt()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
