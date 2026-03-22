from __future__ import annotations

import shlex
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

try:
    from PySide6.QtCore import QProcess, QStandardPaths, Qt, QUrl
    from PySide6.QtGui import QDesktopServices, QGuiApplication, QPixmap
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


DEFAULT_PREP_PROMPT = (
    "Please restore this black-and-white photo with these exact goals:\n"
    "1) Colorize naturally and realistically.\n"
    "2) Enhance detail and clarity while preserving identity and structure.\n"
    "3) Expand the canvas by 25% on every side (left, right, top, bottom),\n"
    "   keeping the original image centered with coherent outpainted borders.\n"
    "4) Return one final image only."
)


def build_dimension_aware_prompt(width: int, height: int) -> str:
    pad_x = max(1, round(width * 0.25))
    pad_y = max(1, round(height * 0.25))
    target_w = width + (2 * pad_x)
    target_h = height + (2 * pad_y)

    return (
        "Please restore this black-and-white photo with these exact requirements.\n"
        f"- Input image size: {width}x{height} pixels.\n"
        f"- Output image size must be exactly: {target_w}x{target_h} pixels.\n"
        f"- Expand canvas by 25% per side: {pad_x}px left, {pad_x}px right, {pad_y}px top, {pad_y}px bottom.\n"
        "- Keep the original image centered.\n"
        "- Colorize naturally and realistically.\n"
        "- Enhance detail and clarity while preserving identity and facial structure.\n"
        "- Keep textures and geometry faithful; do not alter pose, proportions, or landmarks.\n"
        "- Use coherent outpainted borders matching era/style/lighting.\n"
        "- Return one final image only."
    )


class PhotoColorizerQt(QMainWindow):
    """PySide6 GUI with an embedded browser for ChatGPT prep + RoMa colorization."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Photo Colorizer (Qt + In-App Browser)")
        self.resize(1900, 1050)
        self.setMinimumSize(1300, 760)

        self.repo_root = Path(__file__).resolve().parent
        self.runner_script = self.repo_root / "run_romav2_pair.py"
        self.default_outdir = self.repo_root / "outputs" / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.browser_profile_dir = self.repo_root / ".qt_browser_profile"
        self.browser_profile_dir.mkdir(parents=True, exist_ok=True)

        self.process: QProcess | None = None
        self._process_partial = ""
        self._result_pixmaps: dict[str, QPixmap] = {}
        self._current_preview_paths: dict[str, Path] = {}

        self._init_ui()
        self._load_outputs_placeholder()
        self._navigate_chatgpt()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        root_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(root_splitter)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_host = QWidget()
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(8)
        controls_scroll.setWidget(controls_host)

        controls_layout.addWidget(self._build_step0_prep_group())
        controls_layout.addWidget(self._build_input_group())
        controls_layout.addWidget(self._build_controls_group())
        controls_layout.addWidget(self._build_advanced_group())
        controls_layout.addWidget(self._build_action_group())
        controls_layout.addStretch(1)

        left_layout.addWidget(controls_scroll, 4)

        self.output_tabs = QTabWidget()
        left_layout.addWidget(self.output_tabs, 4)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        left_layout.addWidget(log_group, 2)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        right_layout.addWidget(self._build_browser_toolbar())
        self.browser = self._build_browser()
        right_layout.addWidget(self.browser, 1)

        root_splitter.addWidget(left_panel)
        root_splitter.addWidget(right_panel)
        root_splitter.setSizes([760, 1140])

    def _build_step0_prep_group(self) -> QGroupBox:
        group = QGroupBox("Step 0: ChatGPT Prep (In-App Browser)")
        layout = QVBoxLayout(group)

        self.prompt_box = QPlainTextEdit()
        self.prompt_box.setPlainText(DEFAULT_PREP_PROMPT)
        self.prompt_box.setPlaceholderText("Prompt used for the restoration + outpaint step.")
        self.prompt_box.setMinimumHeight(130)
        layout.addWidget(self.prompt_box)

        downloads_row = QHBoxLayout()
        self.downloads_edit = QLineEdit(self._default_downloads_dir())
        downloads_row.addWidget(QLabel("Downloads"))
        downloads_row.addWidget(self.downloads_edit, 1)
        pick_downloads_btn = QPushButton("Browse")
        pick_downloads_btn.clicked.connect(self._pick_downloads_dir)
        downloads_row.addWidget(pick_downloads_btn)
        layout.addLayout(downloads_row)

        btn_row = QHBoxLayout()
        copy_prompt_btn = QPushButton("Copy Prompt")
        copy_prompt_btn.clicked.connect(self._copy_prompt)
        btn_row.addWidget(copy_prompt_btn)

        build_prompt_btn = QPushButton("Build Prompt from B&W")
        build_prompt_btn.clicked.connect(self._build_prompt_from_bw)
        btn_row.addWidget(build_prompt_btn)

        reset_prompt_btn = QPushButton("Reset Prompt")
        reset_prompt_btn.clicked.connect(lambda: self.prompt_box.setPlainText(DEFAULT_PREP_PROMPT))
        btn_row.addWidget(reset_prompt_btn)

        open_chatgpt_btn = QPushButton("Open ChatGPT")
        open_chatgpt_btn.clicked.connect(self._navigate_chatgpt)
        btn_row.addWidget(open_chatgpt_btn)

        import_download_btn = QPushButton("Import Latest Download")
        import_download_btn.clicked.connect(self._import_latest_download)
        btn_row.addWidget(import_download_btn)

        layout.addLayout(btn_row)
        help_label = QLabel(
            "Manual flow: pick B&W, build prompt, sign in, upload, paste/send, download, then Import Latest Download."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        return group

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Step 1: Images")
        layout = QVBoxLayout(group)

        layout.addLayout(self._build_path_row("B&W Original", self._on_pick_bw))
        self.bw_edit = QLineEdit()
        self.bw_edit.textChanged.connect(lambda _: self._update_input_preview(self.bw_edit, self.bw_preview))
        layout.addWidget(self.bw_edit)
        self.bw_preview = self._make_preview_label()
        layout.addWidget(self.bw_preview)

        layout.addSpacing(4)
        layout.addLayout(self._build_path_row("AI Colorized", self._on_pick_color))
        self.color_edit = QLineEdit()
        self.color_edit.textChanged.connect(lambda _: self._update_input_preview(self.color_edit, self.color_preview))
        layout.addWidget(self.color_edit)
        self.color_preview = self._make_preview_label()
        layout.addWidget(self.color_preview)

        pair_btn = QPushButton("Import Pair (B&W first, then Colorized)")
        pair_btn.clicked.connect(self._on_pick_pair)
        layout.addWidget(pair_btn)
        return group

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Step 2: Colorization Settings")
        form = QFormLayout(group)

        self.gf_radius_spin = QSpinBox()
        self.gf_radius_spin.setRange(0, 48)
        self.gf_radius_spin.setValue(16)
        form.addRow("Edge smoothing radius", self.gf_radius_spin)

        self.gf_eps_spin = QDoubleSpinBox()
        self.gf_eps_spin.setDecimals(4)
        self.gf_eps_spin.setRange(0.0001, 1.0)
        self.gf_eps_spin.setSingleStep(0.0005)
        self.gf_eps_spin.setValue(0.001)
        form.addRow("Edge sensitivity", self.gf_eps_spin)

        self.chroma_radius_spin = QSpinBox()
        self.chroma_radius_spin.setRange(0, 128)
        self.chroma_radius_spin.setValue(48)
        form.addRow("Chroma filter radius", self.chroma_radius_spin)

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
        return group

    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Advanced (RoMa Engine)")
        form = QFormLayout(group)

        self.setting_combo = QComboBox()
        self.setting_combo.addItems(list(ROMA_SETTINGS))
        self.setting_combo.setCurrentText("precise")
        form.addRow("Quality preset", self.setting_combo)

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
        self.num_samples_spin.setValue(5000)
        form.addRow("Sample count", self.num_samples_spin)

        self.max_draw_spin = QSpinBox()
        self.max_draw_spin.setRange(1, 200000)
        self.max_draw_spin.setValue(1200)
        form.addRow("Max points drawn", self.max_draw_spin)

        self.compile_check = QCheckBox("Enable torch.compile")
        form.addRow(self.compile_check)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("Step 3: Run")
        layout = QVBoxLayout(group)

        row1 = QHBoxLayout()
        self.run_btn = QPushButton("Colorize")
        self.run_btn.clicked.connect(self._start_run)
        row1.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        row1.addWidget(self.stop_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        new_run_btn = QPushButton("New Run")
        new_run_btn.clicked.connect(self._new_run)
        row2.addWidget(new_run_btn)

        open_folder_btn = QPushButton("Open Folder")
        open_folder_btn.clicked.connect(self._open_output_folder)
        row2.addWidget(open_folder_btn)
        layout.addLayout(row2)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        return group

    def _build_browser_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        back_btn = QPushButton("<")
        back_btn.clicked.connect(lambda: self.browser.back())
        layout.addWidget(back_btn)

        fwd_btn = QPushButton(">")
        fwd_btn.clicked.connect(lambda: self.browser.forward())
        layout.addWidget(fwd_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(lambda: self.browser.reload())
        layout.addWidget(reload_btn)

        home_btn = QPushButton("ChatGPT")
        home_btn.clicked.connect(self._navigate_chatgpt)
        layout.addWidget(home_btn)

        self.url_edit = QLineEdit()
        self.url_edit.returnPressed.connect(self._navigate_to_url_box)
        layout.addWidget(self.url_edit, 1)

        go_btn = QPushButton("Go")
        go_btn.clicked.connect(self._navigate_to_url_box)
        layout.addWidget(go_btn)
        return bar

    def _build_browser(self) -> QWebEngineView:
        profile = QWebEngineProfile("RoMaQtBrowser", self)
        profile.setCachePath(str(self.browser_profile_dir / "cache"))
        profile.setPersistentStoragePath(str(self.browser_profile_dir / "storage"))
        profile.setPersistentCookiesPolicy(QWebEngineProfile.ForcePersistentCookies)
        profile.downloadRequested.connect(self._on_download_requested)

        page = QWebEnginePage(profile, self)
        browser = QWebEngineView()
        browser.setPage(page)
        browser.urlChanged.connect(self._on_browser_url_changed)
        return browser

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _build_path_row(self, label: str, browse_fn) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        btn = QPushButton("Browse")
        btn.clicked.connect(browse_fn)
        row.addWidget(btn)
        return row

    @staticmethod
    def _make_preview_label() -> QLabel:
        label = QLabel("No preview")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(120)
        label.setFrameShape(QFrame.StyledPanel)
        return label

    def _append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

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

    def _navigate_to_url_box(self) -> None:
        raw = self.url_edit.text().strip()
        if not raw:
            return
        if "://" not in raw:
            raw = "https://" + raw
        self.browser.setUrl(QUrl(raw))

    def _on_browser_url_changed(self, url: QUrl) -> None:
        self.url_edit.setText(url.toString())

    def _on_download_requested(self, item) -> None:
        downloads_dir = Path(self.downloads_edit.text().strip() or self._default_downloads_dir())
        downloads_dir.mkdir(parents=True, exist_ok=True)
        filename = item.downloadFileName() or "chatgpt_image.png"
        item.setDownloadDirectory(str(downloads_dir))
        item.setDownloadFileName(filename)
        item.accept()
        self._append_log(f"[INFO] Download started: {downloads_dir / filename}")

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

    def _on_pick_color(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Select colorized image", "", IMAGE_FILTER)
        if selected:
            self.color_edit.setText(selected)

    def _on_pick_pair(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select B&W first, then Colorized",
            "",
            IMAGE_FILTER,
        )
        if len(selected) >= 2:
            self.bw_edit.setText(selected[0])
            self.color_edit.setText(selected[1])
            return
        if len(selected) == 1:
            self.bw_edit.setText(selected[0])

    def _pick_outdir(self) -> None:
        start = self.outdir_edit.text().strip() or str(self.repo_root / "outputs")
        selected = QFileDialog.getExistingDirectory(self, "Select output directory", start)
        if selected:
            self.outdir_edit.setText(selected)

    def _update_input_preview(self, path_edit: QLineEdit, preview_label: QLabel) -> None:
        path = Path(path_edit.text().strip())
        if not path.exists() or not path.is_file():
            preview_label.setText("No preview")
            preview_label.setPixmap(QPixmap())
            return

        pix = QPixmap(str(path))
        if pix.isNull():
            preview_label.setText("No preview")
            preview_label.setPixmap(QPixmap())
            return

        scaled = pix.scaled(420, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        preview_label.setPixmap(scaled)
        preview_label.setText("")

    # ------------------------------------------------------------------
    # Step-0 helpers
    # ------------------------------------------------------------------

    def _copy_prompt(self) -> None:
        prompt = self.prompt_box.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Empty prompt", "Prompt box is empty.")
            return
        QGuiApplication.clipboard().setText(prompt)
        self._append_log("[INFO] Prompt copied to clipboard.")

    def _build_prompt_from_bw(self) -> None:
        bw_path = Path(self.bw_edit.text().strip())
        if not bw_path.exists():
            QMessageBox.warning(self, "Missing image", "Select a valid B&W image first.")
            return

        try:
            with Image.open(bw_path) as img:
                width, height = img.size
            prompt = build_dimension_aware_prompt(width, height)
            self.prompt_box.setPlainText(prompt)
            self._append_log(
                f"[INFO] Built dimension-aware prompt from B&W image: {width}x{height}."
            )
        except Exception as exc:
            QMessageBox.warning(self, "Prompt build failed", f"Could not read image metadata:\n{exc}")

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

        cmd = [
            sys.executable,
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
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        self.process = proc
        self._process_partial = ""

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status("Running...")

        proc.start(cmd[0], cmd[1:])

    def _on_process_output(self) -> None:
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not raw:
            return
        self._process_partial += raw
        while "\n" in self._process_partial:
            line, self._process_partial = self._process_partial.split("\n", 1)
            self._append_log(line.rstrip())

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        if self._process_partial:
            self._append_log(self._process_partial.rstrip())
            self._process_partial = ""

        if exit_code == 0:
            self._append_log("Colorization complete.")
            self._load_outputs()
            self._set_status("Done!")
        else:
            self._append_log(f"Pipeline exited with error code {exit_code}.")
            self._set_status("Failed")

        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process = None

    def _stop_run(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self._append_log("[INFO] Stop requested.")
        self._set_status("Stopping...")

    def _new_run(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Run in progress", "Stop the current run first.")
            return

        self.bw_edit.clear()
        self.color_edit.clear()
        self.outdir_edit.setText(str(self.repo_root / "outputs" / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        self.bw_preview.setText("No preview")
        self.bw_preview.setPixmap(QPixmap())
        self.color_preview.setText("No preview")
        self.color_preview.setPixmap(QPixmap())
        self._clear_preview_tabs()
        self._load_outputs_placeholder()
        self.log_box.clear()
        self._set_status("Ready")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _clear_preview_tabs(self) -> None:
        while self.output_tabs.count() > 0:
            self.output_tabs.removeTab(0)
        self._result_pixmaps.clear()
        self._current_preview_paths.clear()

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
        image_label.setPixmap(pix.scaled(1100, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        image_label.setMinimumHeight(320)
        layout.addWidget(image_label, 1)

        path_label = QLabel(str(image_path))
        layout.addWidget(path_label)
        self.output_tabs.addTab(tab, title)

    # ------------------------------------------------------------------
    # Filesystem actions
    # ------------------------------------------------------------------

    def _open_output_folder(self) -> None:
        outdir = Path(self.outdir_edit.text().strip())
        steps_dir = outdir / "steps"
        target = steps_dir if steps_dir.exists() else outdir
        if not target.exists():
            QMessageBox.warning(self, "Missing folder", f"Directory not found:\n{target}")
            return
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        if not ok:
            QMessageBox.warning(self, "Open failed", f"Could not open folder:\n{target}")


def main() -> int:
    app = QApplication(sys.argv)
    win = PhotoColorizerQt()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
