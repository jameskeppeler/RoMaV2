from __future__ import annotations

import json
import math
import random
import re
import shlex
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, QRect, QRectF, QStandardPaths, QTimer, Qt, QUrl, Signal, QPoint, QPointF
    from PySide6.QtGui import (
        QAction, QBrush, QColor, QDesktopServices, QGuiApplication, QImage,
        QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QRadialGradient,
    )
    from PySide6.QtTest import QTest
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import (
        QApplication,
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGraphicsEllipseItem,
        QGraphicsLineItem,
        QGraphicsPixmapItem,
        QGraphicsPolygonItem,
        QGraphicsScene,
        QGraphicsView,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QScrollArea,
        QSlider,
        QSplitter,
        QStackedWidget,
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


DEFAULT_PREP_PROMPT = """Restore this old photograph in one continuous edit and produce a single final image.

Follow these steps in this exact order:
1. Detect whether the photo includes any visible frame, mount, border card, decorative mat, case edge, or non-image surround.
2. If present, crop/trim away the surround so only the original photographic image area remains.
3. Neutralize the aged sepia, yellow, or brown cast before colorization. Do not leave an overall antique tint unless an object is truly that color.
4. Restore the photo carefully:
   - improve sharpness and fine detail
   - recover facial features, hair, clothing texture, and background detail
   - reduce blur, haze, dust, scratches, stains, cracks, and age damage
   - preserve the original person’s identity and bone structure
   - do not over-smooth skin
   - do not invent new features
   - if there is a person, do not make the face look modern, plastic, or AI-generated
   - if there are no people in the photo, don’t add any that don’t exist
5. Colorize the image in vivid but historically plausible natural color:
   - realistic skin tones
   - natural hair color
   - accurate clothing colors for the era
   - realistic background colors
   - avoid oversaturation
6. Preserve the original pose, expression, framing, composition and photographic realism.
7. The output image MUST match the input image as closely as possible:
    - use the SAME aspect ratio as the input photograph
    - use the SAME framing and field of view — do NOT zoom in, zoom out, pan, or reframe
    - every edge of the output must correspond to the same edge of the input — nothing added, nothing removed
    - do NOT add any border, margin, padding, or letterboxing around the image
    - do NOT crop into the image or shrink the edges of the photograph
    - the output should be a pixel-aligned colorized version of the input, not a reimagined scene
8. Output only the final restored color image.
    - DISPLAY the final image directly in the chat as an embedded image
    - do NOT provide a text download link instead of showing the image"""


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


class DropImageLabel(QLabel):
    """Preview label that accepts image file drag-and-drop."""

    image_dropped = Signal(str)
    clicked = Signal()

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _extract_image_path(self, event) -> Path | None:
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            candidate = Path(url.toLocalFile())
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                return candidate
        return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._extract_image_path(event) is not None:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._extract_image_path(event) is not None:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        path = self._extract_image_path(event)
        if path is None:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self.image_dropped.emit(str(path))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class _CornerHandle(QGraphicsEllipseItem):
    """Draggable corner handle for the manual crop overlay."""

    RADIUS = 8

    def __init__(self, x: float, y: float, index: int, crop_widget: "ManualCropWidget") -> None:
        d = self.RADIUS * 2
        super().__init__(-self.RADIUS, -self.RADIUS, d, d)
        self.setPos(x, y)
        self._index = index
        self._crop_widget = crop_widget
        self.setBrush(QBrush(QColor(0, 180, 255, 200)))
        self.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(10)

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsEllipseItem.ItemPositionChange:
            # Clamp within the image bounds
            rect = self._crop_widget.image_rect()
            if rect is not None:
                x = max(rect.left(), min(value.x(), rect.right()))
                y = max(rect.top(), min(value.y(), rect.bottom()))
                value = QPointF(x, y)
        if change == QGraphicsEllipseItem.ItemPositionHasChanged:
            self._crop_widget.update_polygon()
        return super().itemChange(change, value)


class ManualCropWidget(QGraphicsView):
    """Interactive crop widget with draggable corner handles and polygon overlay.

    Supports 4+ corner points.  The polygon and handles are drawn over the
    image in scene coordinates that match pixel coordinates.
    """

    # Emitted when user confirms the crop
    crop_applied = Signal(object)  # emits list of (x, y) tuples in image-pixel coords

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background: #111111; border: none;")

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._handles: list[_CornerHandle] = []
        self._polygon_item: QGraphicsPolygonItem | None = None
        self._dim_path_item = None  # darkens area outside polygon
        self._image_w: int = 0
        self._image_h: int = 0

    # ── Public API ────────────────────────────────────────────────

    def set_image(self, bgr: np.ndarray) -> None:
        """Load a BGR numpy array as the background image."""
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._image_h, self._image_w = h, w
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._scene.clear()
        self._handles.clear()
        self._polygon_item = None
        self._dim_path_item = None
        self._pixmap_item = self._scene.addPixmap(pix)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_quad(self, points: list[tuple[float, float]]) -> None:
        """Set the initial corner positions (at least 4 points, image-pixel coords)."""
        # Remove old handles
        for h in self._handles:
            self._scene.removeItem(h)
        self._handles.clear()

        for i, (px, py) in enumerate(points):
            handle = _CornerHandle(px, py, i, self)
            self._scene.addItem(handle)
            self._handles.append(handle)
        self.update_polygon()

    def set_default_quad(self) -> None:
        """Place a default rectangle at 10% inset from edges."""
        if self._image_w == 0:
            return
        mx = self._image_w * 0.10
        my = self._image_h * 0.10
        self.set_quad([
            (mx, my),
            (self._image_w - mx, my),
            (self._image_w - mx, self._image_h - my),
            (mx, self._image_h - my),
        ])

    def add_point(self, after_index: int | None = None) -> None:
        """Insert a new point midway between two existing points."""
        n = len(self._handles)
        if n < 2:
            return
        if after_index is None:
            after_index = n - 1
        idx_a = after_index % n
        idx_b = (after_index + 1) % n
        ax, ay = self._handles[idx_a].pos().x(), self._handles[idx_a].pos().y()
        bx, by = self._handles[idx_b].pos().x(), self._handles[idx_b].pos().y()
        mx, my = (ax + bx) / 2, (ay + by) / 2
        insert_at = idx_a + 1
        handle = _CornerHandle(mx, my, insert_at, self)
        self._scene.addItem(handle)
        self._handles.insert(insert_at, handle)
        # Re-index
        for i, h in enumerate(self._handles):
            h._index = i
        self.update_polygon()

    def remove_last_point(self) -> None:
        """Remove the last added point (minimum 4 kept)."""
        if len(self._handles) <= 4:
            return
        handle = self._handles.pop()
        self._scene.removeItem(handle)
        for i, h in enumerate(self._handles):
            h._index = i
        self.update_polygon()

    def get_points(self) -> list[tuple[float, float]]:
        """Return current corner positions in image-pixel coordinates."""
        return [(h.pos().x(), h.pos().y()) for h in self._handles]

    def image_rect(self) -> QRectF | None:
        if self._image_w == 0:
            return None
        return QRectF(0, 0, self._image_w, self._image_h)

    def update_polygon(self) -> None:
        """Redraw the polygon outline and the dimmed-out region."""
        if len(self._handles) < 3:
            return
        pts = [QPointF(h.pos().x(), h.pos().y()) for h in self._handles]
        polygon = QPolygonF(pts)

        # Polygon outline
        if self._polygon_item is not None:
            self._scene.removeItem(self._polygon_item)
        pen = QPen(QColor(0, 180, 255, 220), 2.0)
        pen.setCosmetic(True)
        self._polygon_item = self._scene.addPolygon(
            polygon, pen, QBrush(QColor(0, 180, 255, 25))
        )
        self._polygon_item.setZValue(5)

        # Dim area outside polygon
        if self._dim_path_item is not None:
            self._scene.removeItem(self._dim_path_item)
        outer = QPainterPath()
        outer.addRect(QRectF(0, 0, self._image_w, self._image_h))
        inner = QPainterPath()
        inner.addPolygon(polygon)
        inner.closeSubpath()
        dim_path = outer - inner  # area outside the polygon
        self._dim_path_item = self._scene.addPath(
            dim_path, QPen(Qt.NoPen), QBrush(QColor(0, 0, 0, 140))
        )
        self._dim_path_item.setZValue(3)

    # ── Event overrides ───────────────────────────────────────────

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Double-click on the polygon edge to add a new point."""
        if event.button() == Qt.LeftButton and len(self._handles) >= 3:
            scene_pos = self.mapToScene(event.pos())
            # Find the closest edge to insert after
            best_dist = float("inf")
            best_idx = 0
            n = len(self._handles)
            for i in range(n):
                ax, ay = self._handles[i].pos().x(), self._handles[i].pos().y()
                bx, by = self._handles[(i + 1) % n].pos().x(), self._handles[(i + 1) % n].pos().y()
                # Point-to-segment distance
                dx, dy = bx - ax, by - ay
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-6:
                    continue
                t = max(0, min(1, ((scene_pos.x() - ax) * dx + (scene_pos.y() - ay) * dy) / seg_len_sq))
                px, py = ax + t * dx, ay + t * dy
                dist = ((scene_pos.x() - px) ** 2 + (scene_pos.y() - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            # Only add if reasonably close to an edge (within 30 scene-pixels)
            if best_dist < 30:
                self.add_point(best_idx)
        super().mouseDoubleClickEvent(event)


class ClickableImageLabel(QLabel):
    """Label that emits clicked for preview zoom interactions."""

    clicked = Signal()

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class FullscreenImageDialog(QDialog):
    """Full-screen image viewer. Click or press Esc to close."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.image_label = QLabel("")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #111111;")
        layout.addWidget(self.image_label, 1)

    def show_pixmap(self, pixmap: QPixmap, title: str = "Image Preview") -> None:
        self._pixmap = pixmap
        self.setWindowTitle(title)
        self.showFullScreen()
        self._refresh_scaled()

    def _refresh_scaled(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self.image_label.setPixmap(QPixmap())
            return
        target_w = max(200, self.image_label.width() - 24)
        target_h = max(200, self.image_label.height() - 24)
        scaled = self._pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_scaled()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.close()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)


class ShimmerWidget(QWidget):
    """Shimmer effect over a blurred C1 image.

    Before a C1 image is available: shows cheeky rotating loading messages with
    morphing color blobs and shimmering dots on a dark background.

    After C1 arrives: the blurred C1 image fades in slowly under the dots/morph,
    all effects are clipped to the image rectangle (letterbox areas stay dark).
    """

    _DOT_COUNT = 2000
    _LOADING_MESSAGES = [
        "Fetching paints",
        "Finding a time machine",
        "Picking a color palette",
        "Consulting the color oracle",
        "Mixing the perfect hue",
        "Dusting off the crayons",
        "Rewinding the clock",
        "Asking nicely for pigments",
        "Calibrating the rainbow",
        "Traveling to the past",
        "Selecting vintage tones",
        "Warming up the brush",
    ]

    # Text fade timing constants (seconds)
    _MSG_FADE_DUR = 0.75   # fade in/out duration
    _MSG_LINGER = 4.5      # time at full opacity
    _MSG_CYCLE = _MSG_LINGER + 2 * _MSG_FADE_DUR  # total cycle ~6s

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self._blurred: QPixmap | None = None
        self._blurred_size: tuple[int, int] = (0, 0)
        self._img_rect: QRect = QRect()  # image bounds within widget
        self._phase: float = 0.0
        self._fade_in: float = 0.0  # 0→1 fade for C1 image appearance
        self._dots: list[dict] = []
        self._msg_index: int = 0
        self._msg_timer: float = 0.0
        self._msg_opacity: float = 0.0  # text fade opacity
        self._timer = QTimer(self)
        self._timer.setInterval(50)  # 20 fps — slower
        self._timer.timeout.connect(self._tick)

    def set_source(self, pixmap: QPixmap | None) -> None:
        was_none = self._source_pixmap is None or (self._source_pixmap and self._source_pixmap.isNull())
        self._source_pixmap = pixmap
        self._blurred = None
        self._blurred_size = (0, 0)
        if pixmap and not pixmap.isNull() and was_none:
            self._fade_in = 0.0  # start fade-in from zero
        if pixmap and not pixmap.isNull():
            self._regenerate_dots()
        self.update()

    def start(self) -> None:
        self._phase = 0.0
        self._fade_in = 0.0
        self._msg_index = random.randint(0, len(self._LOADING_MESSAGES) - 1)
        self._msg_timer = 0.0
        self._regenerate_dots()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _regenerate_dots(self) -> None:
        self._dots = []
        for _ in range(self._DOT_COUNT):
            self._dots.append({
                "x": random.random(),
                "y": random.random(),
                "size": random.uniform(0.5, 1.5),
                "speed": random.uniform(0.8, 2.5),
                "phase": random.uniform(0, 2 * math.pi),
            })

    def _tick(self) -> None:
        self._phase += 0.06  # slower overall
        # Rotate loading message with fade in/out transition
        self._msg_timer += 0.05
        if self._msg_timer >= self._MSG_CYCLE:
            self._msg_timer = 0.0
            self._msg_index = (self._msg_index + 1) % len(self._LOADING_MESSAGES)
        # Compute text opacity: fade-in → linger → fade-out
        if self._msg_timer < self._MSG_FADE_DUR:
            self._msg_opacity = self._msg_timer / self._MSG_FADE_DUR
        elif self._msg_timer < self._MSG_FADE_DUR + self._MSG_LINGER:
            self._msg_opacity = 1.0
        else:
            self._msg_opacity = 1.0 - (self._msg_timer - self._MSG_FADE_DUR - self._MSG_LINGER) / self._MSG_FADE_DUR
        self._msg_opacity = max(0.0, min(1.0, self._msg_opacity))
        # Slow fade-in when C1 image is present
        has_source = self._source_pixmap is not None and not self._source_pixmap.isNull()
        if has_source and self._fade_in < 1.0:
            self._fade_in = min(1.0, self._fade_in + 0.008)  # ~6 seconds to fully appear
        self.update()

    def _compute_img_rect(self, w: int, h: int) -> QRect:
        """Compute the aspect-correct image rectangle within the widget."""
        src = self._source_pixmap
        if src and not src.isNull():
            scale = min(w / src.width(), h / src.height())
            img_w = int(src.width() * scale)
            img_h = int(src.height() * scale)
        else:
            img_w, img_h = w, h
        x0 = (w - img_w) // 2
        y0 = (h - img_h) // 2
        return QRect(x0, y0, img_w, img_h)

    def _build_blurred(self, img_rect: QRect) -> QPixmap:
        """Build a heavily blurred pixmap using multi-pass progressive downscale.

        Instead of one extreme 1/28 shrink (which creates blocky pixels),
        we halve the image 4 times, then scale back up. Each SmoothTransformation
        pass acts as a low-pass filter, producing a smooth Gaussian-like blur.
        """
        src = self._source_pixmap
        iw, ih = img_rect.width(), img_rect.height()
        pix = src.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Progressive downscale — 4 halvings ≈ 1/16 each dimension
        for _ in range(4):
            pw = max(1, pix.width() // 2)
            ph = max(1, pix.height() // 2)
            pix = pix.scaled(pw, ph, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        # Progressive upscale — double back up in steps for smooth interpolation
        fw, fh = img_rect.width(), img_rect.height()
        # Scale to fitted dimensions via the source aspect ratio
        fitted_src = src.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        fw, fh = fitted_src.width(), fitted_src.height()
        while pix.width() < fw // 2 and pix.height() < fh // 2:
            pix = pix.scaled(
                min(fw, pix.width() * 2), min(fh, pix.height() * 2),
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
        return pix.scaled(fw, fh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Dark background for the whole widget
        painter.fillRect(0, 0, w, h, QColor("#111"))

        # Compute image bounds
        ir = self._compute_img_rect(w, h)
        self._img_rect = ir

        has_source = self._source_pixmap is not None and not self._source_pixmap.isNull()

        # Clip all effects to the image rectangle
        painter.setClipRect(ir)

        if has_source:
            # Build/cache blurred image
            cache_key = (ir.width(), ir.height())
            if self._blurred is None or self._blurred_size != cache_key:
                self._blurred = self._build_blurred(ir)
                self._blurred_size = cache_key
            # Draw blurred C1 with fade-in opacity
            painter.setOpacity(self._fade_in)
            painter.drawPixmap(ir.x(), ir.y(), self._blurred)
            painter.setOpacity(1.0)

        # Animated morph blobs (clipped to image rect) — stay consistent
        # regardless of whether the C1 source image has arrived.
        for i in range(3):
            cx = ir.x() + ir.width() * (0.5 + 0.25 * math.sin(self._phase * 0.2 + i * 2.1))
            cy = ir.y() + ir.height() * (0.5 + 0.25 * math.cos(self._phase * 0.15 + i * 1.7))
            radius = max(ir.width(), ir.height()) * (0.35 + 0.1 * math.sin(self._phase * 0.1 + i))
            grad = QRadialGradient(cx, cy, radius)
            base_alpha = 0.08 + 0.04 * math.sin(self._phase * 0.12 + i * 3.0)
            hue = (i * 120 + self._phase * 3) % 360
            c1 = QColor.fromHsvF(hue / 360.0, 0.3, 0.6, base_alpha)
            grad.setColorAt(0.0, c1)
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.fillRect(ir, grad)

        # Tiny white shimmering dots (only within image rect)
        painter.setPen(Qt.NoPen)
        white = QColor("#ffffff")
        for dot in self._dots:
            raw_a = 0.5 + 0.5 * math.sin(self._phase * dot["speed"] + dot["phase"])
            if raw_a < 0.25:
                continue
            alpha = raw_a * 0.65
            sx = ir.x() + dot["x"] * ir.width()
            sy = ir.y() + dot["y"] * ir.height()
            sz = dot["size"] * (0.5 + 0.5 * raw_a)
            white.setAlphaF(min(1.0, alpha))
            painter.setBrush(white)
            painter.drawEllipse(int(sx - sz), int(sy - sz), int(sz * 2), int(sz * 2))

        painter.setClipping(False)

        # Loading text centered on image rect — persistent style throughout
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        msg = self._LOADING_MESSAGES[self._msg_index]
        dots_text = "." * (1 + int(self._phase * 0.8) % 3)
        text = f"{msg}{dots_text}"
        text_alpha = int(210 * self._msg_opacity)
        if text_alpha > 2:
            # Shadow
            painter.setPen(QColor(0, 0, 0, int(140 * self._msg_opacity)))
            painter.drawText(ir.adjusted(2, 2, 2, 2), Qt.AlignCenter, text)
            # Main text
            painter.setPen(QColor(255, 255, 255, text_alpha))
            painter.drawText(ir, Qt.AlignCenter, text)

        painter.end()


class BeforeAfterSlider(QWidget):
    """Side-by-side comparison slider showing before (B&W) and after (colorized) images."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._split: float = 0.5  # 0..1 position of divider
        self._dragging = False
        self.setMouseTracking(True)
        self.setCursor(Qt.SplitHCursor)
        self.setMinimumHeight(180)

    def set_images(self, before: QPixmap | None, after: QPixmap | None) -> None:
        self._before = before
        self._after = after
        self._split = 0.5
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._before or not self._after:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # Scale both images to fit the widget while preserving aspect ratio.
        # Use the after image's native aspect ratio as the reference.
        ref = self._after
        scale = min(w / ref.width(), h / ref.height())
        img_w = int(ref.width() * scale)
        img_h = int(ref.height() * scale)
        x0 = (w - img_w) // 2
        y0 = (h - img_h) // 2

        # Dark fill behind letterbox areas
        painter.fillRect(0, 0, w, h, QColor("#111"))

        split_x = x0 + int(img_w * self._split)

        # Scale both preserving aspect ratio (match to the after image dimensions)
        before_scaled = self._before.scaled(img_w, img_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        after_scaled = self._after.scaled(img_w, img_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Center each within the img rect (handles slight size diffs)
        bx = x0 + (img_w - before_scaled.width()) // 2
        by = y0 + (img_h - before_scaled.height()) // 2
        ax = x0 + (img_w - after_scaled.width()) // 2
        ay = y0 + (img_h - after_scaled.height()) // 2

        # Draw "before" (left side)
        painter.setClipRect(x0, y0, split_x - x0, img_h)
        painter.drawPixmap(bx, by, before_scaled)

        # Draw "after" (right side)
        painter.setClipRect(split_x, y0, x0 + img_w - split_x, img_h)
        painter.drawPixmap(ax, ay, after_scaled)

        # Divider line
        painter.setClipping(False)
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(split_x, y0, split_x, y0 + img_h)

        # Divider handle
        handle_y = y0 + img_h // 2
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.setPen(QPen(QColor(0, 0, 0, 100), 1))
        painter.drawEllipse(split_x - 12, handle_y - 12, 24, 24)
        # Arrows
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawLine(split_x - 6, handle_y, split_x - 2, handle_y - 4)
        painter.drawLine(split_x - 6, handle_y, split_x - 2, handle_y + 4)
        painter.drawLine(split_x + 6, handle_y, split_x + 2, handle_y - 4)
        painter.drawLine(split_x + 6, handle_y, split_x + 2, handle_y + 4)

        # Labels
        painter.setPen(QColor(255, 255, 255, 180))
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        if self._split > 0.12:
            painter.drawText(x0 + 8, y0 + 22, "Before")
        if self._split < 0.88:
            painter.drawText(x0 + img_w - 52, y0 + 22, "After")

        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._update_split(event.position().x())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._update_split(event.position().x())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def _update_split(self, mouse_x: float) -> None:
        ref = self._after
        if not ref:
            return
        w, h = self.width(), self.height()
        scale = min(w / ref.width(), h / ref.height())
        img_w = int(ref.width() * scale)
        x0 = (w - img_w) // 2
        raw = (mouse_x - x0) / max(1, img_w)
        self._split = max(0.02, min(0.98, raw))
        self.update()


class PhotoColorizerQt(QMainWindow):
    """PySide6 GUI with an embedded browser for ChatGPT prep + RoMa colorization."""

    BROWSER_PANEL_MIN_WIDTH = 400
    BROWSER_PANEL_MAX_WIDTH = 620

    # Signal for thread-safe crop detection callback
    _crop_detect_done = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._crop_detect_done.connect(self._on_crop_detect_done)
        self.setWindowTitle("Photo Colorizer (Qt + In-App Browser)")
        self.resize(1760, 980)
        self.setMinimumSize(980, 680)

        self.repo_root = Path(__file__).resolve().parent
        self.runner_script = self.repo_root / "run_romav2_pair.py"
        self.outputs_root_dir = self.repo_root / "outputs"
        self.default_outdir = self.outputs_root_dir / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.color_step1_dir = self.repo_root / "color_step_1"
        self.browser_profile_dir = self.repo_root / ".qt_browser_profile"
        self.prewarm_cache_dir = self.repo_root / ".roma_prewarm_cache"
        self.browser_profile_dir.mkdir(parents=True, exist_ok=True)
        self.prewarm_cache_dir.mkdir(parents=True, exist_ok=True)
        self.color_step1_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_root_dir.mkdir(parents=True, exist_ok=True)

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
        self._auto_import_next_download = False
        self._pending_auto_import_download_path: Path | None = None
        self._auto_color_step1_after_attach = False
        self._auto_color_step1_scheduled = False
        self._auto_get_c1_after_send = False
        self._auto_get_c1_polling = False
        self._auto_get_c1_poll_deadline_monotonic: float | None = None
        self._auto_get_c1_poll_attempt = 0
        self._full_workflow_active = False
        self._full_workflow_waiting_for_c1 = False
        self._full_workflow_pending_delete = False
        self._c1_image_nudge_sent = False
        self._c1_workflow_retry_count = 0
        self._c1_prompt_sent_monotonic: float | None = None  # when C1 prompt was sent
        self._delete_target_conv_id: str | None = None
        self._run_started_monotonic: float | None = None
        self._last_process_output_monotonic: float | None = None
        self._last_run_watchdog_log_monotonic: float | None = None
        self._process_partial = ""
        self._result_pixmaps: dict[str, QPixmap] = {}
        self._current_preview_paths: dict[str, Path] = {}
        self._selected_preview_title: str | None = None
        self._input_preview_pixmap: QPixmap | None = None
        self._c1_preview_pixmap: QPixmap | None = None
        self._last_run_outdir: Path | None = None
        self._last_download_result_path: Path | None = None
        self._result_saved: bool = True  # True means no unsaved result
        self._fullscreen_dialog: FullscreenImageDialog | None = None
        self._chat_expanded_width = 500
        self._active_test_preset_id: str | None = None
        self._active_test_extra_flags: list[str] = []
        self._locked_workflow_prompt: str = ""
        self._shimmer_active = False
        self._before_after_active = False
        self._run_watchdog = QTimer(self)
        self._run_watchdog.setInterval(5000)
        self._run_watchdog.timeout.connect(self._on_run_watchdog_tick)

        self.setStyleSheet(
            """
            /* ── Global ── */
            QMainWindow, QWidget { font-size: 13px; }
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 14px;
                padding: 10px 8px 6px 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #aaa;
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
            }
            /* ── Primary action button ── */
            QPushButton#primaryAction {
                font-weight: 700;
                font-size: 15px;
                min-height: 42px;
                border-radius: 6px;
                background: #2563eb;
                color: #fff;
                border: none;
                padding: 4px 20px;
            }
            QPushButton#primaryAction:hover { background: #3b82f6; }
            QPushButton#primaryAction:pressed { background: #1d4ed8; }
            QPushButton#primaryAction:disabled { background: #374151; color: #6b7280; }
            /* ── Stop button ── */
            QPushButton#stopAction {
                font-weight: 600;
                min-height: 36px;
                border-radius: 6px;
                background: #dc2626;
                color: #fff;
                border: none;
                padding: 4px 16px;
            }
            QPushButton#stopAction:hover { background: #ef4444; }
            QPushButton#stopAction:pressed { background: #b91c1c; }
            QPushButton#stopAction:disabled { background: #374151; color: #6b7280; }
            /* ── Regular buttons ── */
            QPushButton {
                min-height: 28px;
                padding: 3px 12px;
                border-radius: 4px;
                border: 1px solid #444;
                background: #2a2a2a;
                color: #ddd;
            }
            QPushButton:hover { background: #363636; border-color: #555; }
            QPushButton:pressed { background: #222; }
            QPushButton:disabled { color: #555; border-color: #333; }
            /* ── Inputs ── */
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 26px;
                padding: 2px 6px;
                border: 1px solid #444;
                border-radius: 4px;
                background: #1e1e1e;
                color: #ddd;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #2563eb;
            }
            QPlainTextEdit {
                border: 1px solid #444;
                border-radius: 4px;
                background: #1a1a1a;
                color: #ddd;
                font-family: 'Consolas', 'Cascadia Code', monospace;
                font-size: 12px;
            }
            QPlainTextEdit:focus { border-color: #2563eb; }
            QCheckBox { spacing: 6px; color: #ccc; }
            /* ── Scroll area ── */
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                width: 8px; background: transparent; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #444; border-radius: 4px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #555; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            /* ── Preview labels ── */
            QLabel#previewLabel {
                border: 1px solid #333;
                border-radius: 6px;
                background: #111;
            }
            QLabel#previewLabel:hover {
                border-color: #2563eb;
            }
            /* ── Status badge ── */
            QLabel#statusBadge {
                padding: 2px 10px;
                border-radius: 10px;
                background: #374151;
                color: #d1d5db;
                font-size: 11px;
                font-weight: 600;
            }
            /* ── Section header labels ── */
            QLabel#sectionHeader {
                font-size: 13px;
                font-weight: 700;
                color: #e5e7eb;
                padding: 2px 0;
            }
            /* ── Splitter handles ── */
            QSplitter::handle { background: #2a2a2a; }
            QSplitter::handle:hover { background: #444; }
            /* ── Form labels ── */
            QFormLayout { }
            QLabel#formHint {
                color: #888;
                font-size: 11px;
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
        self._build_menu_bar()
        self._init_workflow_state_fields()
        self._ensure_advanced_dialog()

        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.setChildrenCollapsible(False)
        root_splitter.setHandleWidth(8)
        self.root_splitter = root_splitter
        self.setCentralWidget(root_splitter)

        # -- Browser panel (slides in from left) --
        browser_panel = QWidget()
        browser_panel.setMinimumWidth(self.BROWSER_PANEL_MIN_WIDTH)
        browser_panel.setMaximumWidth(self.BROWSER_PANEL_MAX_WIDTH)
        self.browser_panel = browser_panel
        browser_layout = QVBoxLayout(browser_panel)
        browser_layout.setContentsMargins(8, 8, 8, 8)
        browser_layout.setSpacing(8)

        browser_layout.addWidget(self._build_browser_toolbar())
        self.browser = self._build_browser()
        self.browser.setMinimumWidth(self.BROWSER_PANEL_MIN_WIDTH - 40)
        self.browser.setMinimumHeight(250)
        browser_layout.addWidget(self.browser, 1)

        # -- Main content: two-column layout --
        main_panel = self._build_main_panel()

        root_splitter.addWidget(browser_panel)
        root_splitter.addWidget(main_panel)
        root_splitter.setStretchFactor(0, 3)
        root_splitter.setStretchFactor(1, 9)
        root_splitter.setSizes([500, 1400])
        self._set_browser_panel_visible(False)
        QTimer.singleShot(0, self._apply_responsive_layout)

    def _init_workflow_state_fields(self) -> None:
        # Hidden state fields retained for existing workflow automation methods.
        self.bw_edit = QLineEdit(self)
        self.bw_edit.setPlaceholderText("Step 1: black-and-white photo path")
        self.bw_edit.textChanged.connect(lambda _: self._refresh_workflow_status())
        self.bw_edit.textChanged.connect(lambda _: self._sync_input_preview_from_field())

        self.color_edit = QLineEdit(self)
        self.color_edit.setPlaceholderText("C1 image path (filled by automation)")
        self.color_edit.textChanged.connect(lambda _: self._refresh_workflow_status())
        self.color_edit.textChanged.connect(lambda _: self._sync_c1_preview_from_field())

        self.prompt_box = QPlainTextEdit(self)
        self.prompt_box.setPlainText(DEFAULT_PREP_PROMPT)
        self.prompt_box.textChanged.connect(self._refresh_workflow_status)
        self.prompt_box.textChanged.connect(self._maybe_prefill_prompt_now)

        self.downloads_edit = QLineEdit(self._default_downloads_dir(), self)

    def _build_menu_bar(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        view_menu = menu.addMenu("View")
        help_menu = menu.addMenu("Help")

        open_bw_action = QAction("Choose B&W Image...", self)
        open_bw_action.triggered.connect(self._on_pick_bw)
        file_menu.addAction(open_bw_action)

        outdir_action = QAction("Choose Output Folder...", self)
        outdir_action.triggered.connect(self._pick_outdir)
        file_menu.addAction(outdir_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        refresh_outputs_action = QAction("Refresh Outputs", self)
        refresh_outputs_action.triggered.connect(self._load_outputs)
        view_menu.addAction(refresh_outputs_action)

        open_folder_action = QAction("Open Output Folder", self)
        open_folder_action.triggered.connect(self._open_output_folder)
        view_menu.addAction(open_folder_action)

        self.toggle_browser_action = QAction("Show ChatGPT Panel", self)
        self.toggle_browser_action.setCheckable(True)
        self.toggle_browser_action.setChecked(False)
        self.toggle_browser_action.toggled.connect(self._set_browser_panel_visible)
        view_menu.addAction(self.toggle_browser_action)

        clear_log_action = QAction("Clear Log", self)
        clear_log_action.triggered.connect(lambda: self.log_box.clear() if hasattr(self, "log_box") else None)
        view_menu.addAction(clear_log_action)

        tips_action = QAction("Workflow Tips", self)
        tips_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "Workflow Tips",
                "1) Pick a B&W image\n"
                "2) Click Colorize to run ChatGPT + RoMa automation\n"
                "3) Review the generated result in the Result Preview panel",
            )
        )
        help_menu.addAction(tips_action)

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.main_layout = layout

        # Two-column splitter: previews (left) | sidebar (right)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(6)
        self.main_sections_splitter = main_splitter

        # -- Left column: previews stacked vertically --
        preview_column = self._build_preview_column()
        main_splitter.addWidget(preview_column)

        # -- Right column: scrollable sidebar --
        sidebar = self._build_sidebar()
        main_splitter.addWidget(sidebar)

        main_splitter.setStretchFactor(0, 6)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setSizes([900, 500])

        layout.addWidget(main_splitter, 1)
        return panel

    def _build_preview_column(self) -> QWidget:
        """Left column: Input and Result previews stacked vertically."""
        column = QWidget()
        col_layout = QVBoxLayout(column)
        col_layout.setContentsMargins(6, 6, 2, 6)
        col_layout.setSpacing(6)

        # Top toolbar: Show Chat + status
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.show_chat_btn = QPushButton("Show Chat")
        self.show_chat_btn.clicked.connect(self._on_show_chat_button_clicked)
        toolbar.addWidget(self.show_chat_btn)

        self.attach_status_label = QLabel("")
        self.attach_status_label.setObjectName("formHint")
        self.attach_status_label.setWordWrap(True)
        toolbar.addWidget(self.attach_status_label, 1)
        col_layout.addLayout(toolbar)

        # Preview splitter (vertical: input on top, result on bottom)
        preview_splitter = QSplitter(Qt.Vertical)
        preview_splitter.setChildrenCollapsible(False)
        preview_splitter.setHandleWidth(5)
        self.preview_splitter = preview_splitter

        # --- Input preview ---
        input_card = QWidget()
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(4)

        input_header = QHBoxLayout()
        input_title = QLabel("Input")
        input_title.setObjectName("sectionHeader")
        input_header.addWidget(input_title)
        self.input_preview_path = QLabel("")
        self.input_preview_path.setObjectName("formHint")
        self.input_preview_path.setWordWrap(True)
        input_header.addWidget(self.input_preview_path, 1)
        input_layout.addLayout(input_header)

        # Stacked widget: 0 = normal drop label, 1 = manual crop widget
        self.input_view_stack = QStackedWidget()

        self.input_preview_label = DropImageLabel("Drop a B&W image here, or click to browse")
        self.input_preview_label.image_dropped.connect(self._on_bw_dropped)
        self.input_preview_label.clicked.connect(self._on_pick_bw)
        self.input_preview_label.setAlignment(Qt.AlignCenter)
        self.input_preview_label.setObjectName("previewLabel")
        self.input_preview_label.setMinimumHeight(180)
        self.input_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_view_stack.addWidget(self.input_preview_label)  # index 0

        self.crop_widget = ManualCropWidget()
        self.crop_widget.setMinimumHeight(180)
        self.crop_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_view_stack.addWidget(self.crop_widget)  # index 1

        self.input_view_stack.setCurrentIndex(0)
        input_layout.addWidget(self.input_view_stack, 1)

        # Crop confirmation bar (hidden until crop review is active)
        crop_bar = QHBoxLayout()
        crop_bar.setSpacing(8)
        self.crop_status_label = QLabel("")
        self.crop_status_label.setObjectName("formHint")
        self.crop_status_label.setWordWrap(True)
        crop_bar.addWidget(self.crop_status_label, 1)

        self.crop_add_pt_btn = QPushButton("+ Point")
        self.crop_add_pt_btn.setToolTip("Add an extra corner (or double-click an edge)")
        self.crop_add_pt_btn.clicked.connect(lambda: self.crop_widget.add_point())
        crop_bar.addWidget(self.crop_add_pt_btn)

        self.crop_remove_pt_btn = QPushButton("- Point")
        self.crop_remove_pt_btn.setToolTip("Remove last added corner (min 4)")
        self.crop_remove_pt_btn.clicked.connect(lambda: self.crop_widget.remove_last_point())
        crop_bar.addWidget(self.crop_remove_pt_btn)

        self.crop_confirm_btn = QPushButton("Confirm Crop")
        self.crop_confirm_btn.setObjectName("primaryAction")
        self.crop_confirm_btn.clicked.connect(self._on_crop_confirmed)
        crop_bar.addWidget(self.crop_confirm_btn)

        self.crop_skip_btn = QPushButton("Skip Crop")
        self.crop_skip_btn.clicked.connect(self._on_crop_skipped)
        crop_bar.addWidget(self.crop_skip_btn)

        self.crop_bar_widget = QWidget()
        self.crop_bar_widget.setLayout(crop_bar)
        self.crop_bar_widget.setVisible(False)
        input_layout.addWidget(self.crop_bar_widget)

        preview_splitter.addWidget(input_card)

        # --- Result preview ---
        result_card = QWidget()
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(4)

        result_header = QHBoxLayout()
        result_title = QLabel("Result")
        result_title.setObjectName("sectionHeader")
        result_header.addWidget(result_title)
        self.result_preview_badge = QLabel("Waiting")
        self.result_preview_badge.setObjectName("statusBadge")
        result_header.addWidget(self.result_preview_badge)
        self.result_preview_path = QLabel("")
        self.result_preview_path.setObjectName("formHint")
        self.result_preview_path.setWordWrap(True)
        result_header.addWidget(self.result_preview_path, 1)
        result_layout.addLayout(result_header)

        # Stacked widget: 0=placeholder label, 1=shimmer, 2=before/after slider
        self.result_stack = QStackedWidget()
        self.result_stack.setMinimumHeight(180)
        self.result_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.result_preview_label = ClickableImageLabel("Run Colorize to generate output.")
        self.result_preview_label.clicked.connect(self._open_result_preview_fullscreen)
        self.result_preview_label.setAlignment(Qt.AlignCenter)
        self.result_preview_label.setObjectName("previewLabel")
        self.result_stack.addWidget(self.result_preview_label)  # index 0

        self.shimmer_widget = ShimmerWidget()
        self.result_stack.addWidget(self.shimmer_widget)  # index 1

        self.before_after_slider = BeforeAfterSlider()
        self.result_stack.addWidget(self.before_after_slider)  # index 2

        self.result_stack.setCurrentIndex(0)
        result_layout.addWidget(self.result_stack, 1)

        result_buttons = QHBoxLayout()
        self.save_result_btn = QPushButton("Save Result")
        self.save_result_btn.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; font-weight: bold; padding: 6px 16px; }"
            "QPushButton:hover { background: #388e3c; }"
            "QPushButton:disabled { background: #444; color: #777; }"
        )
        self.save_result_btn.clicked.connect(self._save_result_dialog)
        self.save_result_btn.setEnabled(False)
        result_buttons.addWidget(self.save_result_btn)
        self.open_result_btn = QPushButton("Open Output Folder")
        self.open_result_btn.clicked.connect(self._open_output_folder)
        self.open_result_btn.setEnabled(False)
        result_buttons.addWidget(self.open_result_btn)
        result_buttons.addStretch(1)
        result_layout.addLayout(result_buttons)

        preview_splitter.addWidget(result_card)
        preview_splitter.setStretchFactor(0, 5)
        preview_splitter.setStretchFactor(1, 5)

        col_layout.addWidget(preview_splitter, 1)
        return column

    def _build_sidebar(self) -> QWidget:
        """Right column: scrollable sidebar with all controls."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        sidebar_layout = QVBoxLayout(container)
        sidebar_layout.setContentsMargins(4, 6, 6, 6)
        sidebar_layout.setSpacing(8)
        self.controls_layout = sidebar_layout

        # 1. Action buttons (Colorize + Stop)
        sidebar_layout.addWidget(self._build_action_group())
        # 2. Settings
        sidebar_layout.addWidget(self._build_controls_group())
        # 3. Prompt
        sidebar_layout.addWidget(self._build_prompt_group())
        # 4. Output folder
        sidebar_layout.addWidget(self._build_output_folder_group())
        # 5. Workflow status
        sidebar_layout.addWidget(self._build_progress_group())
        # 6. Log
        sidebar_layout.addWidget(self._build_log_group())
        sidebar_layout.addStretch(1)

        scroll.setWidget(container)
        return scroll

    def _build_action_group(self) -> QWidget:
        """Primary Colorize + Stop buttons and status."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.run_btn = QPushButton("Colorize")
        self.run_btn.setObjectName("primaryAction")
        self.run_btn.clicked.connect(self._run_default_workflow)
        layout.addWidget(self.run_btn)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        row.addWidget(self.stop_btn)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("formHint")
        row.addWidget(self.status_label, 1)
        layout.addLayout(row)
        return widget

    def _build_prompt_group(self) -> QGroupBox:
        group = QGroupBox("Photo Context")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(4)
        self.prompt_group_layout = layout

        hint = QLabel(
            "Add optional details about the photo to guide colorization "
            "(setting, season, clothing, lighting, era, etc.)"
        )
        hint.setObjectName("formHint")
        hint.setWordWrap(True)
        self.prompt_hint_label = hint
        layout.addWidget(hint)

        # User-facing context field
        self.context_box = QPlainTextEdit()
        self.context_box.setPlaceholderText(
            "e.g. Outdoor summer portrait, 1890s. Subject wears a dark wool suit "
            "with a white collared shirt. Background is a garden with green foliage."
        )
        self.context_box.setMinimumHeight(70)
        self.context_box.setMaximumHeight(150)
        self.context_box.textChanged.connect(self._sync_prompt_from_context)
        layout.addWidget(self.context_box, 1)

        # Hidden prompt_box (still used by all workflow code)
        self.prompt_box.setVisible(False)
        self.prompt_box.setPlaceholderText("")
        self.prompt_box.setMinimumHeight(0)
        self.prompt_box.setMaximumHeight(0)
        # Initialize prompt_box with default prompt
        self._sync_prompt_from_context()

        # Collapsible "Edit Full Prompt" for power users
        btn_row = QHBoxLayout()
        self._show_full_prompt_btn = QPushButton("Edit Full Prompt")
        self._show_full_prompt_btn.setCheckable(True)
        self._show_full_prompt_btn.setChecked(False)
        self._show_full_prompt_btn.toggled.connect(self._toggle_full_prompt_visibility)
        btn_row.addWidget(self._show_full_prompt_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        return group

    def _sync_prompt_from_context(self) -> None:
        """Combine DEFAULT_PREP_PROMPT + user context into prompt_box."""
        context = ""
        if hasattr(self, "context_box"):
            context = self.context_box.toPlainText().strip()
        if context:
            full = DEFAULT_PREP_PROMPT.rstrip() + "\n\nAdditional context about this photo:\n" + context
        else:
            full = DEFAULT_PREP_PROMPT
        # Only update if different to avoid signal loops
        if hasattr(self, "prompt_box") and self.prompt_box.toPlainText() != full:
            self.prompt_box.blockSignals(True)
            self.prompt_box.setPlainText(full)
            self.prompt_box.blockSignals(False)
            self._refresh_workflow_status()

    def _toggle_full_prompt_visibility(self, visible: bool) -> None:
        if hasattr(self, "prompt_box"):
            self.prompt_box.setVisible(visible)
            self.prompt_box.setMinimumHeight(90 if visible else 0)
            self.prompt_box.setMaximumHeight(200 if visible else 0)
        if hasattr(self, "_show_full_prompt_btn"):
            self._show_full_prompt_btn.setText("Hide Full Prompt" if visible else "Edit Full Prompt")

    def _on_show_chat_button_clicked(self) -> None:
        expanded = False
        if hasattr(self, "toggle_browser_action"):
            expanded = bool(self.toggle_browser_action.isChecked())
        elif hasattr(self, "root_splitter"):
            sizes = self.root_splitter.sizes()
            expanded = bool(sizes and sizes[0] > 10)
        self._set_browser_panel_visible(not expanded)

    def _build_status_log_row(self) -> QWidget:
        # Kept for compat - no longer used in the two-column layout but may be
        # referenced by hasattr checks in responsive code.
        host = QWidget()
        row = QBoxLayout(QBoxLayout.LeftToRight, host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.status_log_layout = row
        return host

    def _build_output_folder_group(self) -> QGroupBox:
        group = QGroupBox("Output")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(4)

        self.outdir_edit = QLineEdit(str(self.default_outdir))
        layout.addWidget(self.outdir_edit, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._pick_outdir)
        layout.addWidget(browse_btn)
        return group

    # _build_preview_group and _build_run_controls_group removed —
    # their widgets are now created in _build_preview_column and _build_action_group.

    def _build_log_group(self) -> QWidget:
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Collapsible header button
        self._log_toggle_btn = QPushButton("Log  \u25b6")  # right arrow = collapsed
        self._log_toggle_btn.setCheckable(True)
        self._log_toggle_btn.setChecked(False)
        self._log_toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 8px; font-weight: bold; }"
        )
        self._log_toggle_btn.toggled.connect(self._toggle_log_visibility)
        outer.addWidget(self._log_toggle_btn)

        # Collapsible content
        self._log_content = QWidget()
        content_layout = QVBoxLayout(self._log_content)
        content_layout.setContentsMargins(6, 4, 6, 6)
        content_layout.setSpacing(4)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Pipeline logs will appear here.")
        self.log_box.setMinimumHeight(100)
        self.log_box.setMaximumHeight(220)
        content_layout.addWidget(self.log_box, 1)

        action_row = QHBoxLayout()
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.clicked.connect(self.log_box.clear)
        action_row.addWidget(clear_log_btn)
        action_row.addStretch(1)
        content_layout.addLayout(action_row)

        # Default collapsed
        self._log_content.setVisible(False)
        outer.addWidget(self._log_content)
        return wrapper

    def _toggle_log_visibility(self, expanded: bool) -> None:
        self._log_content.setVisible(expanded)
        self._log_toggle_btn.setText("Log  \u25bc" if expanded else "Log  \u25b6")

    # Pipeline stages for progress tracking (order matters).
    _PIPELINE_STAGES = [
        ("Model init",          "Initializing RoMa"),
        ("Dense match",         "Running dense match"),
        ("Pre-alignment",       "Estimating global pre-alignment"),
        ("Iterative re-match",  "Iterative re-match"),
        ("Warp build",          "Building source->reference warp"),
        ("Guided filter",       "Applying guided-filter"),
        ("Sampling",            "Sampling correspondences"),
        ("Color transfer",      "Applying color"),
        ("Diagnostics",         "Building diagnostics"),
        ("Done",                "Done"),
    ]

    def _build_progress_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QGridLayout(group)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(3)

        for row, (label_text, attr_name, default_text) in enumerate([
            ("B&W", "step1_state", "Waiting"),
            ("C1", "step2_state", "Waiting"),
            ("Run", "step4_state", "---"),
            ("Warmup", "warmup_state", "Checking"),
        ]):
            lbl = QLabel(label_text)
            lbl.setObjectName("formHint")
            layout.addWidget(lbl, row, 0)
            state = QLabel(default_text)
            state.setWordWrap(True)
            setattr(self, attr_name, state)
            layout.addWidget(state, row, 1)

        next_row = 4
        # Progress bar
        self.pipeline_progress = QProgressBar()
        self.pipeline_progress.setRange(0, len(self._PIPELINE_STAGES))
        self.pipeline_progress.setValue(0)
        self.pipeline_progress.setTextVisible(True)
        self.pipeline_progress.setFormat("")  # stage text set manually
        self.pipeline_progress.setFixedHeight(18)
        self.pipeline_progress.setVisible(False)
        layout.addWidget(self.pipeline_progress, next_row, 0, 1, 2)

        self._pipeline_stage_index = 0
        return group


    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Settings")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(4)
        self.controls_group_layout = layout

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)

        self.setting_combo = QComboBox()
        self.setting_combo.addItems(list(ROMA_SETTINGS))
        self.setting_combo.setCurrentText("precise")
        self.setting_combo.currentTextChanged.connect(lambda _: self._on_setting_changed_for_prewarm())
        form.addRow("Quality", self.setting_combo)

        self.accuracy_mode_check = QCheckBox("Accuracy mode")
        self.accuracy_mode_check.setToolTip("Best overlay alignment (slower). Adds multi-pass alignment + iterative re-match.")
        self.accuracy_mode_check.setChecked(True)
        self.accuracy_mode_check.stateChanged.connect(self._on_accuracy_mode_toggled)
        form.addRow("", self.accuracy_mode_check)

        self.color_opacity_spin = QDoubleSpinBox()
        self.color_opacity_spin.setDecimals(2)
        self.color_opacity_spin.setRange(0.0, 1.0)
        self.color_opacity_spin.setSingleStep(0.05)
        self.color_opacity_spin.setValue(1.0)
        form.addRow("Opacity", self.color_opacity_spin)

        layout.addLayout(form)

        # Border exclusion — prominent standalone checkbox
        self.border_exclude_check = QCheckBox("Exclude photo border from colorization")
        self.border_exclude_check.setChecked(False)
        self.border_exclude_check.setToolTip(
            "Detect cardboard mount / frame borders and keep them grayscale.\n"
            "Enable this for CDV, cabinet cards, or phone photos of framed prints."
        )
        self.border_exclude_check.setStyleSheet(
            "QCheckBox { font-weight: 600; padding: 4px 0; color: #e5e7eb; }"
        )
        layout.addWidget(self.border_exclude_check)


        # Helper label (kept for hasattr compat, hidden by default)
        helper = QLabel("")
        helper.setVisible(False)
        self.step3_helper_label = helper
        layout.addWidget(helper)

        # Warmup + Advanced in a compact row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        warm_btn = QPushButton("Warm Up")
        warm_btn.setToolTip("Pre-load model weights so first Colorize is faster")
        warm_btn.clicked.connect(lambda: self._start_background_prewarm(force=True))
        btn_row.addWidget(warm_btn)
        advanced_btn = QPushButton("Advanced...")
        advanced_btn.clicked.connect(self._show_advanced_settings_dialog)
        btn_row.addWidget(advanced_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # Warmup hint (kept for compat, hidden)
        warm_hint = QLabel("")
        warm_hint.setVisible(False)
        self.warm_hint_label = warm_hint

        self._on_accuracy_mode_toggled()
        return group

    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Advanced Settings")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self.gf_radius_spin = QSpinBox()
        self.gf_radius_spin.setRange(0, 48)
        self.gf_radius_spin.setValue(10)
        form.addRow("Edge smoothing radius", self.gf_radius_spin)

        self.gf_eps_spin = QDoubleSpinBox()
        self.gf_eps_spin.setDecimals(4)
        self.gf_eps_spin.setRange(0.0001, 1.0)
        self.gf_eps_spin.setSingleStep(0.0005)
        self.gf_eps_spin.setValue(0.0004)
        form.addRow("Edge sensitivity", self.gf_eps_spin)

        self.chroma_radius_spin = QSpinBox()
        self.chroma_radius_spin.setRange(0, 128)
        self.chroma_radius_spin.setValue(18)
        form.addRow("Chroma filter radius", self.chroma_radius_spin)

        self.chroma_boost_spin = QDoubleSpinBox()
        self.chroma_boost_spin.setDecimals(2)
        self.chroma_boost_spin.setRange(0.10, 4.00)
        self.chroma_boost_spin.setSingleStep(0.05)
        self.chroma_boost_spin.setValue(1.18)
        form.addRow("Color vibrance boost", self.chroma_boost_spin)

        self.adaptive_chroma_check = QCheckBox("Adaptive chroma match to C1")
        self.adaptive_chroma_check.setChecked(True)
        form.addRow(self.adaptive_chroma_check)

        self.chroma_edge_preserve_spin = QDoubleSpinBox()
        self.chroma_edge_preserve_spin.setDecimals(2)
        self.chroma_edge_preserve_spin.setRange(0.00, 1.00)
        self.chroma_edge_preserve_spin.setSingleStep(0.05)
        self.chroma_edge_preserve_spin.setValue(0.94)
        form.addRow("Chroma edge preserve", self.chroma_edge_preserve_spin)

        self.bw_gray_balance_check = QCheckBox("Normalize B&W base before overlay")
        self.bw_gray_balance_check.setChecked(True)
        form.addRow(self.bw_gray_balance_check)

        self.bw_black_clip_spin = QDoubleSpinBox()
        self.bw_black_clip_spin.setDecimals(3)
        self.bw_black_clip_spin.setRange(0.000, 0.190)
        self.bw_black_clip_spin.setSingleStep(0.005)
        self.bw_black_clip_spin.setValue(0.003)
        form.addRow("B&W black clip", self.bw_black_clip_spin)

        self.bw_white_clip_spin = QDoubleSpinBox()
        self.bw_white_clip_spin.setDecimals(3)
        self.bw_white_clip_spin.setRange(0.000, 0.190)
        self.bw_white_clip_spin.setSingleStep(0.005)
        self.bw_white_clip_spin.setValue(0.003)
        form.addRow("B&W white clip", self.bw_white_clip_spin)

        self.bw_midtone_target_spin = QDoubleSpinBox()
        self.bw_midtone_target_spin.setDecimals(2)
        self.bw_midtone_target_spin.setRange(0.05, 0.95)
        self.bw_midtone_target_spin.setSingleStep(0.01)
        self.bw_midtone_target_spin.setValue(0.50)
        form.addRow("B&W midtone target", self.bw_midtone_target_spin)

        self.reg_thresh_spin = QDoubleSpinBox()
        self.reg_thresh_spin.setDecimals(2)
        self.reg_thresh_spin.setRange(0.0, 0.99)
        self.reg_thresh_spin.setSingleStep(0.01)
        self.reg_thresh_spin.setValue(0.33)
        form.addRow("Regularization threshold", self.reg_thresh_spin)

        self.reg_fallback_combo = QComboBox()
        self.reg_fallback_combo.addItems(["identity", "reference", "none"])
        self.reg_fallback_combo.setCurrentText("identity")
        form.addRow("Regularization fallback", self.reg_fallback_combo)

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(1, 200000)
        self.num_samples_spin.setValue(9000)
        form.addRow("Sample count", self.num_samples_spin)

        self.max_draw_spin = QSpinBox()
        self.max_draw_spin.setRange(1, 200000)
        self.max_draw_spin.setValue(1200)
        form.addRow("Max points drawn", self.max_draw_spin)

        self.c1_adherence_combo = QComboBox()
        self.c1_adherence_combo.addItems(["Balanced", "High", "Extreme"])
        self.c1_adherence_combo.setCurrentText("Extreme")
        self.c1_adherence_combo.setToolTip(
            "Controls extra pre-align + iterative rematch effort to make C1 adhere to B&W geometry."
        )
        form.addRow("C1 adherence effort", self.c1_adherence_combo)

        self.compile_check = QCheckBox("Enable torch.compile")
        form.addRow(self.compile_check)
        return group

    def _ensure_advanced_dialog(self) -> None:
        if hasattr(self, "advanced_dialog") and self.advanced_dialog is not None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Settings")
        dialog.resize(620, 520)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(10, 10, 10, 10)
        dialog_layout.setSpacing(8)

        dialog_layout.addWidget(self._build_advanced_group(), 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        close_row.addWidget(close_btn)
        dialog_layout.addLayout(close_row)

        self.advanced_dialog = dialog

    def _show_advanced_settings_dialog(self) -> None:
        self._ensure_advanced_dialog()
        self.advanced_dialog.show()
        self.advanced_dialog.raise_()
        self.advanced_dialog.activateWindow()


    def _build_browser_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

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

    def _ensure_chat_panel_ready(self, *, reason: str, focus: bool) -> None:
        panel_hidden = hasattr(self, "browser_panel") and (not self.browser_panel.isVisible())
        if panel_hidden:
            self._set_browser_panel_visible(True)
            self._append_log(f"[INFO] Chat panel auto-opened for {reason}.")
        if focus and hasattr(self, "browser"):
            self.raise_()
            self.activateWindow()
            self.browser.setFocus(Qt.OtherFocusReason)
            focus_target = self.browser.focusProxy()
            if isinstance(focus_target, QWidget):
                focus_target.setFocus(Qt.OtherFocusReason)

    def _set_browser_panel_visible(self, visible: bool) -> None:
        if not hasattr(self, "browser_panel") or not hasattr(self, "root_splitter"):
            return
        visible = bool(visible)
        sizes_now = self.root_splitter.sizes()
        if sizes_now and sizes_now[0] > 10:
            self._chat_expanded_width = sizes_now[0]

        if visible:
            self.browser_panel.setMinimumWidth(self.BROWSER_PANEL_MIN_WIDTH)
            self.browser_panel.setMaximumWidth(self.BROWSER_PANEL_MAX_WIDTH)
            self.browser_panel.setVisible(True)
            self.root_splitter.setHandleWidth(8)
            total = max(900, self.width())
            left = min(max(self.BROWSER_PANEL_MIN_WIDTH, int(self._chat_expanded_width)), self.BROWSER_PANEL_MAX_WIDTH)
            self.root_splitter.setSizes([left, max(520, total - left)])
        else:
            # Keep chat view technically visible/alive for automation while effectively hidden from the user.
            self.browser_panel.setVisible(True)
            self.browser_panel.setMinimumWidth(1)
            self.browser_panel.setMaximumWidth(1)
            self.root_splitter.setHandleWidth(0)
            total = max(900, self.width())
            self.root_splitter.setSizes([1, max(520, total - 1)])

        if hasattr(self, "toggle_browser_action"):
            self.toggle_browser_action.blockSignals(True)
            self.toggle_browser_action.setChecked(visible)
            self.toggle_browser_action.blockSignals(False)
        if hasattr(self, "show_chat_btn"):
            self.show_chat_btn.setText("Hide Chat" if visible else "Show Chat")
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "preview_splitter"):
            return

        # The two-column layout is naturally responsive via the splitter.
        # We only need minor adjustments for very narrow windows.
        main_width = self.width()
        if hasattr(self, "root_splitter"):
            sizes = self.root_splitter.sizes()
            if len(sizes) >= 2 and sizes[1] > 1:
                main_width = sizes[1]

        compact = main_width < 900
        preview_min_h = 140 if compact else 180
        if hasattr(self, "input_preview_label"):
            self.input_preview_label.setMinimumHeight(preview_min_h)
        if hasattr(self, "result_preview_label"):
            self.result_preview_label.setMinimumHeight(preview_min_h)

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

    def _sync_input_preview_from_field(self) -> None:
        if not hasattr(self, "bw_edit"):
            return
        raw = self.bw_edit.text().strip()
        if not raw:
            self._input_preview_pixmap = None
            if hasattr(self, "input_preview_path"):
                self.input_preview_path.setText("")
            self._refresh_output_previews()
            return
        path = Path(raw)
        if path.exists() and path.is_file():
            pix = QPixmap(str(path))
            if not pix.isNull():
                # If there are unsaved results, prompt the user before clearing
                if self._result_pixmaps and not self._result_saved:
                    reply = QMessageBox.question(
                        self,
                        "Unsaved result",
                        "You have an unsaved colorized result. Save it before loading a new image?",
                        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                        QMessageBox.Save,
                    )
                    if reply == QMessageBox.Save:
                        self._save_result_dialog()
                    elif reply == QMessageBox.Cancel:
                        return
                # Clear previous result when loading a new input image
                self._clear_result_for_new_input()
                self._input_preview_pixmap = pix
                if hasattr(self, "input_preview_path"):
                    self.input_preview_path.setText(str(path))
                self._refresh_output_previews()
                return
        self._input_preview_pixmap = None
        if hasattr(self, "input_preview_path"):
            self.input_preview_path.setText("")
        self._refresh_output_previews()

    def _clear_result_for_new_input(self) -> None:
        """Clear all result previews and state when a new input image is loaded."""
        self._clear_preview_tabs()
        self._c1_preview_pixmap = None
        self._result_saved = True
        self._stop_shimmer()
        self._before_after_active = False
        if hasattr(self, "save_result_btn"):
            self.save_result_btn.setEnabled(False)
        if hasattr(self, "open_result_btn"):
            self.open_result_btn.setEnabled(False)
        if hasattr(self, "color_edit"):
            self.color_edit.clear()
        self._set_selected_output_preview(None)

    def _sync_c1_preview_from_field(self) -> None:
        if not hasattr(self, "color_edit"):
            return
        raw = self.color_edit.text().strip()
        if not raw:
            self._c1_preview_pixmap = None
            if self._selected_preview_title is None and hasattr(self, "result_preview_path"):
                self.result_preview_path.setText("")
            if self._selected_preview_title is None:
                self._set_selected_output_preview(None)
            else:
                self._refresh_output_previews()
            return
        path = Path(raw)
        if path.exists() and path.is_file():
            pix = QPixmap(str(path))
            if not pix.isNull():
                self._c1_preview_pixmap = pix
                if self._selected_preview_title is None and hasattr(self, "result_preview_path"):
                    self.result_preview_path.setText(str(path))
                self._refresh_output_previews()
                if not self._current_preview_paths:
                    self._set_selected_output_preview(None)
                return
        self._c1_preview_pixmap = None
        if self._selected_preview_title is None:
            self._set_selected_output_preview(None)
        else:
            self._refresh_output_previews()

    def _current_selected_preview_path(self) -> Path | None:
        if self._selected_preview_title and self._selected_preview_title in self._current_preview_paths:
            return self._current_preview_paths[self._selected_preview_title]
        if hasattr(self, "color_edit"):
            c1_path = Path(self.color_edit.text().strip())
            if c1_path.exists() and c1_path.is_file():
                return c1_path
        return None

    def _save_result_dialog(self) -> None:
        """Open a Save As dialog for the final colorized image."""
        source = self._get_final_result_path()
        if source is None or not source.exists():
            QMessageBox.warning(self, "No result", "No colorized result available to save.")
            return
        downloads = self._default_downloads_dir()
        # Default filename: original B&W name + "_colorized"
        bw_path = Path(self.bw_edit.text().strip()) if hasattr(self, "bw_edit") else None
        if bw_path and bw_path.stem:
            save_name = f"{bw_path.stem}_colorized.png"
        else:
            save_name = source.name
        suggested = Path(downloads) / save_name
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Colorized Result", str(suggested),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All files (*)",
        )
        if not dest:
            return
        try:
            shutil.copy2(str(source), dest)
            self._result_saved = True
            self._append_log(f"[INFO] Result saved to: {dest}")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save result:\n{exc}")

    def _get_final_result_path(self) -> Path | None:
        """Return the path to the final colorized result image, if it exists."""
        if self._last_run_outdir is not None:
            final = self._last_run_outdir / "final_colorized.png"
            if final.exists():
                return final
        outdir = Path(self.outdir_edit.text().strip()) if hasattr(self, "outdir_edit") else None
        if outdir and (outdir / "final_colorized.png").exists():
            return outdir / "final_colorized.png"
        return None

    def _open_selected_preview_location(self) -> None:
        downloads_dir = Path(self._default_downloads_dir())
        target = self._last_download_result_path.parent if self._last_download_result_path else downloads_dir
        if not target.exists():
            QMessageBox.warning(self, "Downloads missing", f"Downloads folder not found:\n{target}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _open_output_folder(self) -> None:
        if self._last_run_outdir is not None:
            outdir = self._last_run_outdir
        else:
            outdir = Path(self.outdir_edit.text().strip()) if hasattr(self, "outdir_edit") else Path()
        if not outdir.exists() or not outdir.is_dir():
            QMessageBox.warning(self, "Output folder missing", f"Output folder not found:\n{outdir}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(outdir)))

    def _open_result_preview_fullscreen(self) -> None:
        pix: QPixmap | None = None
        title = "Result Preview"
        if self._selected_preview_title and self._selected_preview_title in self._result_pixmaps:
            pix = self._result_pixmaps[self._selected_preview_title]
            title = self._selected_preview_title
        # Don't show C1 fullscreen — the shimmer replaces it
        if pix is None or pix.isNull():
            return
        if self._fullscreen_dialog is None:
            self._fullscreen_dialog = FullscreenImageDialog(self)
        self._fullscreen_dialog.show_pixmap(pix, title=title)

    def _select_primary_output(self) -> None:
        if "Final Colorized" in self._current_preview_paths:
            self._set_selected_output_preview("Final Colorized")
            return
        if self._current_preview_paths:
            first_title = next(iter(self._current_preview_paths.keys()))
            self._set_selected_output_preview(first_title)
            return
        self._set_selected_output_preview(None)

    def _next_run_output_dir(self) -> Path:
        base = self.outputs_root_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base / f"colorize_{stamp}"
        if not candidate.exists():
            return candidate
        i = 1
        while True:
            probe = base / f"colorize_{stamp}{suffix}_{i}"
            if not probe.exists():
                return probe
            i += 1

    def _copy_final_result_to_downloads(self, outdir: Path) -> Path | None:
        final_path = outdir / "final_colorized.png"
        if not final_path.exists() or not final_path.is_file():
            self._append_log(f"[WARN] Final result not found for download copy: {final_path}")
            self._last_download_result_path = None
            return None
        downloads_dir = Path(self._default_downloads_dir())
        try:
            downloads_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._append_log(f"[WARN] Could not access Downloads folder ({downloads_dir}): {exc}")
            self._last_download_result_path = None
            return None

        # Copy only the final colorized image to Downloads (never diagnostics or .npy artifacts).
        target_name = self._sanitize_download_filename(f"{outdir.name}_final_colorized.png")
        target_path = self._unique_download_path(downloads_dir, target_name)
        try:
            copied = Path(shutil.copy2(str(final_path), str(target_path)))
            self._last_download_result_path = copied
            self._append_log(f"[INFO] Copied final result to Downloads: {copied}")
            return copied
        except Exception as exc:
            self._append_log(f"[WARN] Could not copy final result to Downloads: {exc}")
            self._last_download_result_path = None
            return None

    def _set_selected_output_preview(self, title: str | None) -> None:
        if title and title in self._result_pixmaps:
            self._selected_preview_title = title
            path = self._current_preview_paths.get(title)
            self.result_preview_badge.setText(title)
            self.result_preview_badge.setStyleSheet(
                "padding: 2px 10px; border-radius: 8px; background: #2f6c48; color: #def7e8;"
            )
            self.result_preview_path.setText(str(path) if path else "")
            has_outdir = self._last_run_outdir is not None and self._last_run_outdir.exists()
            self.open_result_btn.setEnabled(has_outdir)
        elif self._c1_preview_pixmap is not None:
            self._selected_preview_title = None
            self.result_preview_badge.setText("Generating color...")
            self.result_preview_badge.setStyleSheet(
                "padding: 2px 10px; border-radius: 8px; background: #4a3f8a; color: #e0d8ff;"
            )
            self.result_preview_path.setText("")
            self.open_result_btn.setEnabled(False)
        else:
            self._selected_preview_title = None
            self.result_preview_badge.setText("Waiting")
            self.result_preview_badge.setStyleSheet(
                "padding: 2px 10px; border-radius: 8px; background: #404040; color: #efefef;"
            )
            self.result_preview_path.setText("")
            self.open_result_btn.setEnabled(False)
        self._refresh_output_previews()

    @staticmethod
    def _set_step_label(label: QLabel, text: str, ready: bool) -> None:
        if ready:
            label.setText(f"Ready - {text}")
            label.setStyleSheet("color: #3ea76a;")
        else:
            label.setText(f"Pending - {text}")
            label.setStyleSheet("color: #c7a44b;")

    def _clear_active_test_preset(self) -> None:
        self._active_test_preset_id = None
        self._active_test_extra_flags = []

    def _run_default_workflow(self) -> None:
        self._clear_active_test_preset()
        self._start_full_workflow()

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

    def _effective_roma_setting(self) -> str:
        return self.setting_combo.currentText() if hasattr(self, "setting_combo") else "fast"

    def _c1_adherence_extra_flags(self, level_override: str | None = None) -> list[str]:
        if level_override is None:
            level = self.c1_adherence_combo.currentText().strip() if hasattr(self, "c1_adherence_combo") else "High"
            if not level:
                level = "High"
        else:
            level = str(level_override).strip() or "High"
        level_l = level.lower()
        if level_l == "balanced":
            return [
                "--multi-pass-global-align",
                "--prealign-ransac-iters", "2400",
                "--prealign-min-samples", "5000",
                "--prealign-sample-cap", "18000",
                "--prealign-confidence-quantile", "0.70",
                "--prealign-border-trim-frac", "0.10",
                "--prealign-overlap-margin", "0.0015",
                "--iterative-rematch-passes", "5",
                "--iterative-rematch-overlap-margin", "0.0005",
                "--iterative-rematch-mae-margin", "0.00025",
            ]
        if level_l == "extreme":
            return [
                "--multi-pass-global-align",
                "--prealign-ransac-iters", "4200",
                "--prealign-min-samples", "9000",
                "--prealign-sample-cap", "28000",
                "--prealign-confidence-quantile", "0.75",
                "--prealign-border-trim-frac", "0.12",
                "--prealign-overlap-margin", "0.0010",
                "--iterative-rematch-passes", "9",
                "--iterative-rematch-overlap-margin", "0.00025",
                "--iterative-rematch-mae-margin", "0.00012",
                "--filter-max-megapixels", "28",
            ]
        # High (default)
        return [
            "--multi-pass-global-align",
            "--prealign-ransac-iters", "3200",
            "--prealign-min-samples", "7000",
            "--prealign-sample-cap", "22000",
            "--prealign-confidence-quantile", "0.72",
            "--prealign-border-trim-frac", "0.11",
            "--prealign-overlap-margin", "0.0012",
            "--iterative-rematch-passes", "7",
            "--iterative-rematch-overlap-margin", "0.00035",
            "--iterative-rematch-mae-margin", "0.00018",
            "--filter-max-megapixels", "24",
        ]

    def _on_accuracy_mode_toggled(self, _state=None) -> None:
        enabled = bool(self.accuracy_mode_check.isChecked()) if hasattr(self, "accuracy_mode_check") else False
        if hasattr(self, "setting_combo"):
            if enabled:
                self.setting_combo.setToolTip(
                    "Accuracy mode keeps this preset and adds stronger pre-align + iterative rematch."
                )
            else:
                self.setting_combo.setToolTip("")
        self._refresh_workflow_status()
        QTimer.singleShot(200, self._start_background_prewarm)

    def _start_background_prewarm(self, force: bool = False) -> None:
        setting = self._effective_roma_setting()
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
        if hasattr(self, "accuracy_mode_check") and self.accuracy_mode_check.isChecked():
            cmd.append("--accuracy-mode")

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
        setting = self._effective_roma_setting()
        prewarm_ready = self._is_setting_prewarmed(setting)
        prewarm_running = (
            self.prewarm_process is not None
            and self.prewarm_process.state() != QProcess.NotRunning
            and self._prewarm_setting == setting
        )

        prompt_ok = bool(self.prompt_box.toPlainText().strip()) if hasattr(self, "prompt_box") else False
        runner_ok = self.runner_script.exists()
        run_ready = bw_ok and prompt_ok and runner_ok

        if hasattr(self, "step1_state"):
            self._set_step_label(self.step1_state, "B&W selected" if bw_ok else "Select B&W image", bw_ok)
        if hasattr(self, "step2_state"):
            self._set_step_label(
                self.step2_state,
                "Colorized image imported" if color_ok else "C1 will be generated during Colorize",
                color_ok or (bw_ok and prompt_ok),
            )
        if hasattr(self, "step4_state"):
            self._set_step_label(
                self.step4_state,
                "Ready to run full workflow" if run_ready else "Need B&W image + prompt",
                run_ready,
            )
        warmup_text = f"Warm model cache for '{setting}'"
        warmup_ok = prewarm_ready
        if prewarm_running:
            warmup_text = f"Warming model cache for '{setting}'"
            warmup_ok = False
        if hasattr(self, "warmup_state"):
            self._set_step_label(self.warmup_state, warmup_text, warmup_ok)

        running = self.process is not None and self.process.state() != QProcess.NotRunning
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(run_ready and (not running) and (not self._full_workflow_active))

    def _show_results_tab(self) -> None:
        self._select_primary_output()

    def _show_log_tab(self) -> None:
        if hasattr(self, "log_box"):
            self.log_box.setFocus(Qt.OtherFocusReason)

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

    def _delete_current_chat(self) -> None:
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            QMessageBox.warning(self, "ChatGPT not open", "Open ChatGPT in the embedded browser first.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Chat",
            "Delete the currently open ChatGPT conversation?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self._delete_current_chat_automated()

    def _delete_current_chat_automated(self, *, done=None) -> None:
        self._ensure_chat_panel_ready(reason="Delete Chat automation", focus=False)
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            self._append_log("[WARN] Delete Chat skipped: ChatGPT is not open in the embedded browser.")
            if done is not None:
                done(False)
            return
        self._delete_target_conv_id = self._conversation_id_from_url(current)
        if not self._delete_target_conv_id:
            self._append_log("[WARN] Delete Chat: could not extract conversation ID from URL.")
            if done is not None:
                done(False)
            return
        self._append_log(f"[INFO] Delete Chat: deleting conversation {self._delete_target_conv_id[:12]}... via API.")
        self._attempt_delete_current_chat(retries=3, delay_ms=1500, done=done)

    @staticmethod
    def _conversation_id_from_url(url: str) -> str | None:
        match = re.search(r"/c/([^/?#]+)", str(url or ""), flags=re.IGNORECASE)
        if not match:
            return None
        value = (match.group(1) or "").strip()
        return value or None

    @staticmethod
    def _chatgpt_delete_chat_api_js(conv_id: str) -> str:
        """JS that deletes a ChatGPT conversation via the backend API.

        This is far more reliable than DOM-clicking because it uses the same
        internal API that ChatGPT's own UI calls.  The embedded browser already
        has the session cookies, so same-origin ``fetch()`` is authenticated
        automatically — we just need the access token from the session endpoint.
        """
        safe_id = conv_id.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        return (
            """
(async () => {
  try {
    var convId = '"""
            + safe_id
            + """';
    if (!convId) {
      return JSON.stringify({ ok: false, final: false, reason: "no_conv_id" });
    }

    // 1. Get access token from ChatGPT session endpoint.
    var sessResp = await fetch("/api/auth/session", { credentials: "include" });
    if (!sessResp.ok) {
      return JSON.stringify({ ok: false, final: false, reason: "session_fetch_failed", status: sessResp.status });
    }
    var sessData = await sessResp.json();
    var token = sessData.accessToken || sessData.access_token || "";
    if (!token) {
      return JSON.stringify({ ok: false, final: false, reason: "no_access_token" });
    }

    // 2. PATCH conversation to hide it (this is what "Delete" does in the UI).
    var patchResp = await fetch("/backend-api/conversation/" + convId, {
      method: "PATCH",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
      },
      body: JSON.stringify({ is_visible: false })
    });

    if (patchResp.ok) {
      return JSON.stringify({ ok: true, final: true, via: "api_patch", convId: convId });
    }

    // If 404, the conversation was already deleted.
    if (patchResp.status === 404) {
      return JSON.stringify({ ok: true, final: true, via: "already_deleted", convId: convId });
    }

    var body = "";
    try { body = await patchResp.text(); } catch (e) {}
    return JSON.stringify({
      ok: false,
      final: false,
      reason: "api_error",
      status: patchResp.status,
      detail: body.substring(0, 300)
    });
  } catch (err) {
    return JSON.stringify({
      ok: false,
      final: false,
      reason: "js_exception",
      detail: String(err)
    });
  }
})();
"""
        )

    @staticmethod
    def _decode_js_result_dict(result) -> dict:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            text = result.strip()
            if not text:
                return {
                    "ok": False,
                    "final": False,
                    "reason": "empty_js_result",
                    "_raw_kind": "str",
                    "_raw_preview": "",
                }
            try:
                parsed = json.loads(text)
            except Exception as exc:
                return {
                    "ok": False,
                    "final": False,
                    "reason": "non_json_js_result",
                    "detail": str(exc),
                    "_raw_kind": "str",
                    "_raw_preview": text[:220],
                }
            if isinstance(parsed, dict):
                return parsed
            return {
                "ok": False,
                "final": False,
                "reason": "non_dict_js_result",
                "_raw_kind": "json",
                "_raw_preview": repr(parsed)[:220],
            }
        if result is None:
            return {
                "ok": False,
                "final": False,
                "reason": "empty_js_result",
                "_raw_kind": "none",
                "_raw_preview": "None",
            }
        return {
            "ok": False,
            "final": False,
            "reason": "unsupported_js_result_type",
            "_raw_kind": type(result).__name__,
            "_raw_preview": repr(result)[:220],
        }

    def _attempt_delete_current_chat(self, retries: int = 14, delay_ms: int = 500, done=None) -> None:
        page = self.browser.page()
        if page is None:
            if done is not None:
                done(False)
            return

        conv_id = self._delete_target_conv_id or ""
        if not conv_id:
            self._append_log("[WARN] Delete Chat: no conversation ID to delete.")
            if done is not None:
                done(False)
            return

        js = self._chatgpt_delete_chat_api_js(conv_id)

        def _after(result) -> None:
            parsed = self._decode_js_result_dict(result)
            ok = bool(parsed.get("ok"))
            final = bool(parsed.get("final"))
            via = str(parsed.get("via", "unknown"))
            reason = str(parsed.get("reason", "unknown"))
            detail = str(parsed.get("detail", ""))
            status = parsed.get("status", "")

            if ok and final:
                self._append_log(f"[INFO] Delete Chat succeeded ({via}, conv={conv_id[:12]}).")
                self._delete_target_conv_id = None
                QTimer.singleShot(400, self._navigate_chatgpt)
                if done is not None:
                    done(True)
                return

            # Log the failure reason
            if reason == "no_access_token":
                self._append_log(
                    "[WARN] Delete Chat: could not obtain ChatGPT access token. "
                    "Make sure you are logged in to ChatGPT in the embedded browser."
                )
            elif reason == "session_fetch_failed":
                self._append_log(
                    f"[WARN] Delete Chat: session endpoint returned HTTP {status}. "
                    "Make sure you are logged in to ChatGPT."
                )
            elif reason == "api_error":
                self._append_log(
                    f"[WARN] Delete Chat: API returned HTTP {status}. "
                    f"Detail: {detail[:200]}"
                )
                # Retry on 429 (rate limit) or 5xx
                if retries > 0 and (status == 429 or (isinstance(status, int) and status >= 500)):
                    self._append_log("[INFO] Delete Chat: retrying after server error...")
                    QTimer.singleShot(delay_ms, lambda: self._attempt_delete_current_chat(retries - 1, delay_ms, done=done))
                    return
            elif reason == "js_exception":
                self._append_log(f"[WARN] Delete Chat JS error: {detail}")
            elif reason == "no_conv_id":
                self._append_log("[WARN] Delete Chat: no conversation ID.")
            else:
                self._append_log(f"[WARN] Delete Chat failed: {reason}")

            self._delete_target_conv_id = None
            if done is not None:
                done(False)

        page.runJavaScript(js, 0, _after)

    def _on_browser_load_finished(self, ok: bool) -> None:
        if not ok:
            return
        current = self.browser.url().toString().lower()
        if "chatgpt.com" not in current:
            return
        # Prefill is convenience-only; avoid noisy retries when chat is collapsed.
        if hasattr(self, "toggle_browser_action") and self.toggle_browser_action.isChecked():
            self._auto_prefill_prompt(retries=5, delay_ms=700)
        if (
            self._full_workflow_active
            and self.pending_bw_upload_path is not None
            and self.pending_bw_upload_path.exists()
        ):
            QTimer.singleShot(900, lambda: self._auto_upload_bw_to_chatgpt(self.pending_bw_upload_path, retries=5, delay_ms=800))

    def _on_auto_file_used_for_chatgpt(self, path_str: str) -> None:
        self._set_attach_status("file selected for ChatGPT", ok=True)
        self._append_log(f"[INFO] ChatGPT file chooser auto-filled: {path_str}")
        self._queue_auto_color_step1_after_attach(reason="file attached", delay_ms=650)

    def _queue_auto_color_step1_after_attach(self, *, reason: str, delay_ms: int = 700) -> None:
        if not self._auto_color_step1_after_attach:
            return
        if self._auto_color_step1_scheduled:
            return
        self._auto_color_step1_after_attach = False
        self._auto_color_step1_scheduled = True
        self._append_log(f"[INFO] Auto workflow: attach complete ({reason}). Running Color Step 1...")
        QTimer.singleShot(max(0, int(delay_ms)), self._run_auto_color_step1)

    def _run_auto_color_step1(self) -> None:
        if not self._auto_color_step1_scheduled:
            return
        self._auto_color_step1_scheduled = False
        # Before sending the prompt, verify the B&W image thumbnail is visible
        # in the ChatGPT composer. This prevents sending a text-only message
        # when the image hasn't finished attaching (especially on retries).
        self._wait_for_image_then_send_step1(attempts_left=20, delay_ms=800, paste_retried=False)

    def _wait_for_image_then_send_step1(self, attempts_left: int, delay_ms: int, paste_retried: bool = False) -> None:
        """Poll until an image thumbnail appears in the ChatGPT composer, then send.

        If the thumbnail hasn't appeared after half the attempts, re-paste
        the image from the clipboard as a second chance before giving up.
        """
        if not self._full_workflow_active and not self._auto_color_step1_after_attach:
            # Workflow was cancelled while we were waiting
            return
        page = self.browser.page()
        if page is None:
            return

        def _check(result) -> None:
            parsed = self._decode_js_result_dict(result) if isinstance(result, dict) else {}
            has_image = bool(parsed.get("has_image"))
            if has_image:
                self._append_log("[INFO] Image thumbnail confirmed in ChatGPT composer.")
                self._actually_send_color_step1()
                return
            if attempts_left <= 0:
                self._append_log(
                    "[WARN] Could not confirm image thumbnail in composer after polling. "
                    "Sending prompt anyway (image may not be attached)."
                )
                self._actually_send_color_step1()
                return
            # Mid-poll re-paste: if we're halfway through and still no image,
            # retry the clipboard paste to give the browser another chance.
            if not paste_retried and attempts_left <= 10:
                bw_path = Path(self.bw_edit.text().strip()) if hasattr(self, "bw_edit") else Path()
                if bw_path.exists() and bw_path.is_file():
                    self._append_log("[INFO] Image not yet visible in composer; retrying clipboard paste...")
                    self._attach_bw_via_clipboard_paste_only(bw_path)
                    QTimer.singleShot(1500, lambda: self._wait_for_image_then_send_step1(attempts_left - 1, delay_ms, paste_retried=True))
                    return
            if attempts_left in (20, 15, 10, 5):
                self._append_log("[INFO] Waiting for image thumbnail to appear in ChatGPT composer...")
            QTimer.singleShot(delay_ms, lambda: self._wait_for_image_then_send_step1(attempts_left - 1, delay_ms, paste_retried=paste_retried))

        page.runJavaScript(self._chatgpt_composer_has_image_js(), _check)

    def _attach_bw_via_clipboard_paste_only(self, bw_path: Path) -> None:
        """Paste the B&W image into the composer clipboard without triggering auto-send.

        Unlike ``_attach_bw_via_clipboard_paste``, this does NOT call
        ``_queue_auto_color_step1_after_attach`` — it is only used as a
        mid-poll retry to get the image into the composer.
        """
        if not bw_path.exists() or not bw_path.is_file():
            return
        self._ensure_chat_panel_ready(reason="clipboard re-paste retry", focus=True)
        image = QImage(str(bw_path))
        if image.isNull():
            return

        QGuiApplication.clipboard().setImage(image)
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
            QTest.keyClick(target, Qt.Key_V, Qt.ControlModifier)

        page.runJavaScript(js, _after_focus)

    def _actually_send_color_step1(self) -> None:
        """Send Color Step 1 prompt after image attachment is confirmed."""
        self._auto_get_c1_after_send = True
        started = self._color_step_1(interactive=False)
        if not started:
            self._auto_get_c1_after_send = False
            if self._full_workflow_active:
                self._append_log("[WARN] Workflow aborted: Color Step 1 could not be started after upload.")
                self._full_workflow_active = False
                self._full_workflow_waiting_for_c1 = False
                self._full_workflow_pending_delete = False
                self._locked_workflow_prompt = ""
                self._set_status("Ready")
                self._refresh_workflow_status()

    def _stop_auto_get_c1_poll(self) -> None:
        self._auto_get_c1_polling = False
        self._auto_get_c1_poll_deadline_monotonic = None
        self._auto_get_c1_poll_attempt = 0

    _C1_MIN_WAIT_SEC = 30  # ChatGPT needs at least this long to generate an image

    def _start_auto_get_c1_poll(self, *, timeout_sec: int = 1800, interval_ms: int = 7000) -> None:
        if self._auto_get_c1_polling:
            return
        self._auto_get_c1_polling = True
        self._auto_get_c1_poll_attempt = 0
        self._auto_get_c1_poll_deadline_monotonic = time.monotonic() + max(30, int(timeout_sec))
        self._append_log(
            "[INFO] Auto workflow: waiting for ChatGPT image generation; "
            f"first check in ~{self._C1_MIN_WAIT_SEC}s."
        )
        # Wait at least _C1_MIN_WAIT_SEC before the first poll — ChatGPT needs
        # time to generate the image. Polling too early can capture the uploaded
        # B&W image instead.
        first_delay = max(self._C1_MIN_WAIT_SEC * 1000, int(interval_ms))
        QTimer.singleShot(first_delay, lambda: self._auto_get_c1_poll_tick(interval_ms))

    def _auto_get_c1_poll_tick(self, interval_ms: int) -> None:
        if not self._auto_get_c1_polling:
            return
        if self._c1_download_request_seen or self._pending_auto_import_download_path is not None:
            self._stop_auto_get_c1_poll()
            return
        deadline = self._auto_get_c1_poll_deadline_monotonic
        if deadline is not None and time.monotonic() >= deadline:
            self._stop_auto_get_c1_poll()
            if self._full_workflow_active and self._full_workflow_waiting_for_c1:
                if not self._c1_image_nudge_sent:
                    self._append_log(
                        "[WARN] Auto Get C1 timed out. Sending nudge to ChatGPT..."
                    )
                    self._c1_image_nudge_sent = True
                    self._nudge_chatgpt_for_image()
                    return
                if self._c1_workflow_retry_count < 2:
                    self._c1_workflow_retry_count += 1
                    self._append_log(
                        f"[WARN] Auto Get C1 timed out after nudge. "
                        f"Deleting chat and retrying (attempt {self._c1_workflow_retry_count + 1}/3)..."
                    )
                    self._retry_c1_workflow_from_scratch()
                    return
            self._append_log(
                "[WARN] Auto Get C1 timed out waiting for a downloadable ChatGPT image. "
                "Click ChatGPT's Download button manually; this app will still auto-import the next completed image."
            )
            if self._full_workflow_active and self._full_workflow_waiting_for_c1:
                self._set_status("Waiting for manual ChatGPT download...")
            return

        self._auto_get_c1_poll_attempt += 1
        if self._auto_get_c1_poll_attempt in (1, 4) or self._auto_get_c1_poll_attempt % 8 == 0:
            self._append_log("[INFO] Auto Get C1: checking ChatGPT for download availability...")

        if self._awaiting_c1_download:
            QTimer.singleShot(max(1000, int(interval_ms)), lambda: self._auto_get_c1_poll_tick(interval_ms))
            return

        self._download_and_import_c1(interactive=False, quiet=True)

        QTimer.singleShot(max(1000, int(interval_ms)), lambda: self._auto_get_c1_poll_tick(interval_ms))

    def _auto_prefill_prompt(self, retries: int, delay_ms: int) -> None:
        prompt = self.prompt_box.toPlainText().strip() if hasattr(self, "prompt_box") else ""
        if not prompt:
            return

        def _after(success: bool) -> None:
            if success:
                self._append_log("[INFO] Prompt pre-filled into ChatGPT composer.")
                return
            if retries - 1 <= 0:
                # Non-critical convenience path; avoid false-failure noise.
                return
            QTimer.singleShot(delay_ms, lambda: self._auto_prefill_prompt(retries - 1, delay_ms))

        self._inject_prompt_into_chatgpt(
            prompt,
            silent=True,
            ensure_chat_visible=False,
            done=_after,
        )

    def _maybe_prefill_prompt_now(self) -> None:
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            return
        if hasattr(self, "toggle_browser_action") and (not self.toggle_browser_action.isChecked()):
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
    def _chatgpt_composer_has_image_js() -> str:
        """JS that returns {has_image: true/false} by checking for image thumbnails
        in the ChatGPT composer area (the file-attachment preview chips)."""
        return """
(() => {
  // ChatGPT renders attached image thumbnails inside the composer region.
  // They appear as <img> tags inside thumbnail/preview containers, or as
  // background-image chips.  We look for several known patterns.
  const composer = document.querySelector(
    '#prompt-textarea, [id*="prompt"], [contenteditable="true"], ' +
    'div[class*="composer"], div[class*="chat-input"], div[class*="prosemirror"]'
  );
  if (!composer) return { has_image: false, reason: 'no_composer' };

  // Walk UP from the composer to its parent form/container so we catch
  // thumbnail previews that are siblings of the text area.
  const region = composer.closest('form') || composer.parentElement?.parentElement || composer;

  // Pattern 1: <img> tags with a real src (data: or blob: URLs from pasted images,
  // or https thumbnail URLs from ChatGPT file processing).
  const imgs = region.querySelectorAll('img[src]');
  for (const img of imgs) {
    const src = img.src || '';
    // Ignore tiny UI icons (avatar, logo, etc.)
    if (img.naturalWidth > 30 || img.width > 30 || src.startsWith('blob:') || src.startsWith('data:image')) {
      return { has_image: true, via: 'img_tag' };
    }
  }

  // Pattern 2: file attachment chips — divs with thumbnail background images.
  const chips = region.querySelectorAll(
    '[class*="thumbnail"], [class*="preview"], [class*="attachment"], [class*="file-chip"]'
  );
  for (const chip of chips) {
    const style = window.getComputedStyle(chip);
    const bg = style.backgroundImage || '';
    if (bg && bg !== 'none') {
      return { has_image: true, via: 'chip_bg' };
    }
    // Some chips contain a nested img
    if (chip.querySelector('img[src]')) {
      return { has_image: true, via: 'chip_img' };
    }
  }

  // Pattern 3: any element with role="img" inside composer area
  const roleImgs = region.querySelectorAll('[role="img"]');
  for (const ri of roleImgs) {
    const r = ri.getBoundingClientRect();
    if (r.width > 20 && r.height > 20) {
      return { has_image: true, via: 'role_img' };
    }
  }

  return { has_image: false, reason: 'no_thumbnail_found' };
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
    'button[data-testid="send-button"]',
    'button[data-testid*="submit" i]',
    'button[data-testid*="composer-send" i]',
    'button[data-testid*="send-button" i]',
    'button[data-testid*="send" i]',
    'button[aria-label*="send" i]',
    'button[aria-label*="submit" i]',
    'button[title*="send" i]',
    'button[title*="submit" i]',
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
  const textOf = (n) => (
    ((n.innerText || '') + ' ' +
     (n.getAttribute?.('aria-label') || '') + ' ' +
     (n.getAttribute?.('title') || '')).toLowerCase()
  );

  const directDownloads = Array.from(
    document.querySelectorAll('a[download],button[download],a[href*="/download" i],a[href^="blob:"]')
  ).filter(visible);
  if (directDownloads.length > 0) {
    const direct = directDownloads[directDownloads.length - 1];
    if (tryClick(direct)) {
      return { ok: true, via: 'direct_download_attr' };
    }
  }

  // Only look for visuals inside assistant responses to avoid clicking the uploaded B&W image.
  let visuals = Array.from(
    document.querySelectorAll(
      '[data-message-author-role="assistant"] img,' +
      '[data-message-author-role="assistant"] picture,' +
      '[data-message-author-role="assistant"] canvas,' +
      '.agent-turn img,.agent-turn picture,.agent-turn canvas'
    )
  ).filter(visible);
  if (visuals.length === 0) {
    // Fallback: all visuals except those inside user messages.
    visuals = Array.from(
      document.querySelectorAll('img,picture,canvas,figure [role="img"],div[role="img"]')
    ).filter(el => {
      if (!visible(el)) return false;
      if (el.closest?.('[data-message-author-role="user"]')) return false;
      if (el.closest?.('.user-turn')) return false;
      return true;
    });
  }
  const latestVisual = visuals.length ? visuals[visuals.length - 1] : null;
  if (latestVisual) {
    try {
      latestVisual.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
      latestVisual.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
      latestVisual.dispatchEvent(new MouseEvent('mousemove', { bubbles: true }));
    } catch {}
    const clickTarget = latestVisual.closest?.('button,[role="button"],a') || latestVisual;
    tryClick(clickTarget);
  }

  const nodes = Array.from(
    document.querySelectorAll('button,a,[role="button"],[role="menuitem"],[data-testid]')
  ).filter(visible);

  const ranked = nodes
    .map((n, idx) => {
      const text = textOf(n);
      const testid = (n.getAttribute?.('data-testid') || '').toLowerCase();
      let score = 0;
      if (text.includes('download')) score += 9;
      if (text.includes('save image') || text.includes('save photo') || text.includes('save file')) score += 8;
      if (text.includes('save') && (text.includes('image') || text.includes('photo') || text.includes('file'))) {
        score += 5;
      }
      if (text.includes('export')) score += 2;
      if (testid.includes('download')) score += 7;
      if (text.includes('image') || text.includes('photo')) score += 1;
      return { n, idx, score };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => (a.score - b.score) || (a.idx - b.idx));

  if (ranked.length > 0) {
    const target = ranked[ranked.length - 1].n;
    if (tryClick(target)) {
      return { ok: true, via: 'download_button_ranked' };
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

    @staticmethod
    def _chatgpt_extract_inline_image_js() -> str:
        """JS that finds the last large inline <img> in the ChatGPT response,
        fetches it as a blob, and triggers a download — bypassing any missing
        download-button / broken-link issues.
        """
        return """
(async () => {
  const MIN_DIM = 128;
  // Only look inside assistant message containers to avoid capturing
  // the uploaded B&W image from user messages.
  const assistantSelectors = [
    '[data-message-author-role="assistant"] img',
    '.agent-turn img',
    '[class*="assistant"] img',
  ];
  let imgs = [];
  for (const sel of assistantSelectors) {
    imgs = Array.from(document.querySelectorAll(sel))
      .filter(el => {
        if (!el.src) return false;
        const r = el.getBoundingClientRect();
        return r.width >= MIN_DIM && r.height >= MIN_DIM;
      });
    if (imgs.length > 0) break;
  }
  // Fallback: if no assistant-scoped images found, try all large images
  // but exclude known user-upload containers.
  if (imgs.length === 0) {
    imgs = Array.from(document.querySelectorAll('img'))
      .filter(el => {
        if (!el.src) return false;
        if (el.closest('[data-message-author-role="user"]')) return false;
        if (el.closest('.user-turn')) return false;
        const r = el.getBoundingClientRect();
        return r.width >= MIN_DIM && r.height >= MIN_DIM;
      });
  }
  if (imgs.length === 0) {
    return { ok: false, reason: 'no_large_inline_image' };
  }
  const img = imgs[imgs.length - 1];
  try {
    const resp = await fetch(img.src, { mode: 'cors', credentials: 'include' });
    if (!resp.ok) {
      return { ok: false, reason: 'fetch_failed', status: resp.status };
    }
    const blob = await resp.blob();
    const ext = blob.type === 'image/jpeg' ? '.jpg' : '.png';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chatgpt_image' + ext;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    return { ok: true, via: 'inline_image_extract' };
  } catch (err) {
    return { ok: false, reason: 'extract_error', detail: String(err) };
  }
})();
"""

    def _attempt_chatgpt_download(self, retries: int = 6, delay_ms: int = 600, *, quiet: bool = False) -> None:
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
                if not quiet:
                    self._append_log(f"[INFO] C1 download control clicked ({via}).")
                if retries > 0 and self._awaiting_c1_download and not self._c1_download_request_seen:
                    QTimer.singleShot(
                        delay_ms,
                        lambda: self._attempt_chatgpt_download(retries - 1, delay_ms, quiet=quiet),
                    )
                return

            if retries <= 0:
                if not self._c1_download_request_seen:
                    # Fallback: try extracting the inline image directly from the DOM
                    self._attempt_inline_image_extract(quiet=quiet)
                return

            if not quiet and retries in (6, 3):
                self._append_log("[INFO] Looking for download control in ChatGPT...")

            menu_js = self._chatgpt_open_download_menu_js()

            def _after_menu(_menu_result) -> None:
                QTimer.singleShot(
                    delay_ms,
                    lambda: self._attempt_chatgpt_download(retries - 1, delay_ms, quiet=quiet),
                )

            page.runJavaScript(menu_js, _after_menu)

        page.runJavaScript(js, _after)

    def _attempt_inline_image_extract(self, *, quiet: bool = False) -> None:
        """Fallback: extract the last large inline image from the ChatGPT DOM
        and trigger a download via fetch+blob. Handles cases where ChatGPT
        shows a clickable link instead of a download button."""
        if self._c1_download_request_seen:
            return
        page = self.browser.page() if hasattr(self, "browser") else None
        if page is None:
            self._awaiting_c1_download = False
            return
        if not quiet:
            self._append_log("[INFO] Download button not found — trying to extract inline image from page...")
        js = self._chatgpt_extract_inline_image_js()

        def _after_extract(result) -> None:
            if self._c1_download_request_seen:
                return
            ok = isinstance(result, dict) and bool(result.get("ok"))
            if ok:
                via = str(result.get("via", "unknown"))
                if not quiet:
                    self._append_log(f"[INFO] C1 image extracted from page ({via}).")
                return
            # Truly failed — try workflow recovery if we're in the automated workflow
            self._awaiting_c1_download = False
            reason = str(result.get("reason", "unknown")) if isinstance(result, dict) else "unknown"
            if self._full_workflow_active and self._full_workflow_waiting_for_c1:
                if not self._c1_image_nudge_sent:
                    # First failure: send a follow-up nudge message asking ChatGPT to show the image
                    self._c1_image_nudge_sent = True
                    self._append_log(
                        f"[INFO] No inline image found ({reason}). "
                        "Sending follow-up nudge to ChatGPT..."
                    )
                    self._nudge_chatgpt_for_image()
                    return
                # Nudge already sent and still no image — delete chat and retry
                if self._c1_workflow_retry_count < 2:
                    self._c1_workflow_retry_count += 1
                    self._append_log(
                        f"[WARN] ChatGPT did not display an image after nudge. "
                        f"Deleting chat and retrying (attempt {self._c1_workflow_retry_count + 1}/3)..."
                    )
                    self._retry_c1_workflow_from_scratch()
                    return
                # Exhausted retries
                self._append_log(
                    "[WARN] ChatGPT failed to display an image after 3 attempts. "
                    "Click ChatGPT's download button manually; "
                    "this app will auto-import the next completed image download."
                )
                return
            if not quiet:
                self._append_log(
                    f"[WARN] Could not extract inline image ({reason}). "
                    "Click ChatGPT's download button manually; "
                    "this app will auto-import the next completed image download."
                )

        page.runJavaScript(js, _after_extract)

    _C1_NUDGE_MESSAGE = (
        "Please display the final restored image directly in the chat as an inline image. "
        "Do not provide a download link — show the image itself."
    )

    def _nudge_chatgpt_for_image(self) -> None:
        """Send a short follow-up message asking ChatGPT to re-display the image inline."""
        self._append_log("[INFO] Sending nudge: asking ChatGPT to display the image inline...")
        self._paste_prompt_clipboard_then_send(
            self._C1_NUDGE_MESSAGE,
            send_retries=12,
            send_delay_ms=700,
        )
        # Resume polling to pick up the image after ChatGPT responds
        self._start_auto_get_c1_poll(timeout_sec=120, interval_ms=8000)

    def _retry_c1_workflow_from_scratch(self) -> None:
        """Delete the current chat, navigate to a fresh ChatGPT page, and re-run
        the full workflow (re-attach B&W image + re-send prompt)."""
        self._stop_auto_get_c1_poll()

        def _after_delete(success: bool) -> None:
            if success:
                self._append_log("[INFO] Retry: old chat deleted.")
            else:
                self._append_log("[WARN] Retry: could not confirm chat deletion. Continuing anyway.")
            # Navigate to a fresh ChatGPT chat
            self._append_log("[INFO] Retry: opening fresh ChatGPT chat...")
            self._navigate_chatgpt()
            # Re-run the full workflow after the page loads.
            # Use a longer delay on retries — the fresh page needs time to fully
            # initialise its composer before we can attach an image.
            QTimer.singleShot(6000, self._restart_workflow_after_retry)

        self._delete_current_chat_automated(done=_after_delete)

    def _restart_workflow_after_retry(self) -> None:
        """Re-attach the B&W image and re-send the prompt in a fresh chat."""
        if not self._full_workflow_active:
            return
        bw_path = Path(self.bw_edit.text().strip())
        if not bw_path.exists():
            self._append_log("[WARN] Retry aborted: B&W image not found.")
            self._full_workflow_active = False
            self._set_status("Ready")
            self._refresh_workflow_status()
            return
        self._c1_image_nudge_sent = False
        self._c1_download_request_seen = False
        self._awaiting_c1_download = False
        self._auto_color_step1_after_attach = True
        self._auto_color_step1_scheduled = False
        self._full_workflow_waiting_for_c1 = True
        self._full_workflow_pending_delete = True
        attempt = self._c1_workflow_retry_count + 1
        self._append_log(f"[INFO] Retry: re-running workflow (attempt {attempt}/3)...")
        self._set_status(f"Retrying ChatGPT workflow (attempt {attempt}/3)...")
        self.pending_bw_upload_path = bw_path
        self._attach_selected_bw_to_chatgpt()

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

    def _move_c1_download_to_color_step1_dir(self, source_path: Path) -> Path:
        if not source_path.exists() or not source_path.is_file():
            return source_path
        target_dir = self.color_step1_dir
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._append_log(f"[WARN] Get C1: could not create color_step_1 folder: {exc}")
            return source_path
        target_name = self._sanitize_download_filename(source_path.name)
        target_path = self._unique_download_path(target_dir, target_name)
        try:
            same_path = source_path.resolve() == target_path.resolve()
        except Exception:
            same_path = str(source_path) == str(target_path)
        if same_path:
            return source_path
        try:
            moved = Path(shutil.move(str(source_path), str(target_path)))
            self._append_log(f"[INFO] Get C1: moved download to {moved}")
            return moved
        except Exception as exc:
            self._append_log(f"[WARN] Get C1: move to {target_dir} failed ({exc}); using original file.")
            return source_path

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
                if self._auto_import_next_download:
                    pending = self._pending_auto_import_download_path
                    if pending is None or pending == target_path:
                        import_path = self._move_c1_download_to_color_step1_dir(target_path)
                        self._import_download_path(import_path, source_label="Get C1")
                        self._auto_import_next_download = False
                        self._pending_auto_import_download_path = None
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
            if self._auto_import_next_download and self._pending_auto_import_download_path == target_path:
                self._auto_import_next_download = False
                self._pending_auto_import_download_path = None

    @staticmethod
    def _chatgpt_focus_composer_js() -> str:
        return """
(() => {
  const selectors = [
    'textarea[data-testid="prompt-textarea"]',
    'textarea#prompt-textarea',
    'div#prompt-textarea[contenteditable]',
    '[contenteditable][id="prompt-textarea"]',
    'textarea[aria-label*="Message"]',
    'textarea[placeholder*="Message"]',
    'textarea',
    'div[data-testid="prompt-textarea"][contenteditable]',
    'div[contenteditable="true"][data-testid="prompt-textarea"]',
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="plaintext-only"][role="textbox"]',
    'div[contenteditable="plaintext-only"]',
    'div[contenteditable][role="textbox"]',
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
        self._ensure_chat_panel_ready(reason="clipboard paste fallback", focus=True)
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
        self._attempt_clipboard_paste_with_focus(page, retries_left=4, bw_path=bw_path)

    def _attempt_clipboard_paste_with_focus(
        self, page, *, retries_left: int, bw_path: Path
    ) -> None:
        """Try to focus the composer and paste; retry if focus fails."""
        js = self._chatgpt_focus_composer_js()

        def _after_focus(result) -> None:
            parsed = self._decode_js_result_dict(result)
            ok = bool(parsed.get("ok"))

            if not ok and retries_left > 0:
                reason = str(parsed.get("reason", "unknown"))
                self._append_log(
                    f"[INFO] Composer focus not ready ({reason}); "
                    f"retrying paste in 500ms ({retries_left} left)..."
                )
                QTimer.singleShot(
                    500,
                    lambda: self._attempt_clipboard_paste_with_focus(
                        page, retries_left=retries_left - 1, bw_path=bw_path
                    ),
                )
                return

            target = self.browser.focusProxy()
            if not isinstance(target, QWidget):
                target = self.browser
            target.setFocus(Qt.OtherFocusReason)
            self.browser.setFocus(Qt.OtherFocusReason)
            QTest.keyClick(target, Qt.Key_V, Qt.ControlModifier)

            if ok:
                self._append_log("[INFO] Sent Ctrl+V to ChatGPT composer with selected B&W image on clipboard.")
            else:
                reason = str(parsed.get("reason", "unknown"))
                self._append_log(
                    "[WARN] Could not focus ChatGPT composer automatically "
                    f"({reason}); Ctrl+V was still sent to the browser."
                )
            self._set_attach_status("paste attempted; confirm thumbnail appears", ok=None)
            # Allow extra time for the browser to process the pasted image before
            # we start polling for the thumbnail (clipboard images can take >1s).
            self._queue_auto_color_step1_after_attach(reason="clipboard-paste attach", delay_ms=2000)

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
        self._ensure_chat_panel_ready(reason="Enter-key send fallback", focus=True)
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
            parsed = self._decode_js_result_dict(result)
            ok = bool(parsed.get("ok"))
            if ok:
                self._append_log("[INFO] Send fallback: pressed Enter in ChatGPT composer.")
            else:
                reason = str(parsed.get("reason", "unknown"))
                self._append_log(
                    "[WARN] Send fallback used Enter without confirmed composer focus "
                    f"({reason})."
                )

        page.runJavaScript(js, _after_focus)

    def _paste_prompt_clipboard_then_send(
        self,
        prompt: str,
        *,
        send_retries: int = 12,
        send_delay_ms: int = 700,
    ) -> None:
        self._ensure_chat_panel_ready(reason="prompt paste fallback", focus=True)
        QGuiApplication.clipboard().setText(prompt)
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
            QTest.keyClick(target, Qt.Key_V, Qt.ControlModifier)

            parsed = self._decode_js_result_dict(result)
            ok = bool(parsed.get("ok"))
            if ok:
                self._append_log("[INFO] Prompt fallback: pasted clipboard text into ChatGPT composer.")
            else:
                reason = str(parsed.get("reason", "unknown"))
                self._append_log(
                    "[WARN] Prompt fallback paste used without confirmed composer focus "
                    f"({reason})."
                )
            QTimer.singleShot(
                280,
                lambda: self._attempt_chatgpt_send(retries=send_retries, delay_ms=send_delay_ms),
            )

        page.runJavaScript(js, _after_focus)

    def _attempt_prompt_fill_then_send(
        self,
        prompt: str,
        *,
        retries: int = 12,
        delay_ms: int = 550,
    ) -> None:
        def _attempt(remaining: int) -> None:
            self._inject_prompt_into_chatgpt(
                prompt,
                silent=True,
                done=lambda success: _after_inject(success, remaining),
            )

        def _after_inject(success: bool, remaining: int) -> None:
            if success:
                self._append_log("[INFO] Prompt transferred to ChatGPT composer.")
                QTimer.singleShot(220, lambda: self._attempt_chatgpt_send(retries=12, delay_ms=700))
                return

            if remaining > 0:
                if remaining in (12, 8, 4, 1):
                    self._append_log("[INFO] Waiting for ChatGPT composer to become ready...")
                page = self.browser.page()
                if page is None:
                    return
                page.runJavaScript(
                    self._chatgpt_focus_composer_js(),
                    lambda _r: QTimer.singleShot(delay_ms, lambda: _attempt(remaining - 1)),
                )
                return

            self._append_log(
                "[WARN] Prompt fill retries exhausted; using clipboard paste fallback before send."
            )
            self._paste_prompt_clipboard_then_send(prompt, send_retries=18, send_delay_ms=700)

        _attempt(max(0, int(retries)))

    def _attempt_chatgpt_send(self, retries: int = 10, delay_ms: int = 700) -> None:
        self._ensure_chat_panel_ready(reason="ChatGPT send action", focus=False)
        page = self.browser.page()
        if page is None:
            return
        js = self._chatgpt_send_js()

        def _after_send(result) -> None:
            parsed = self._decode_js_result_dict(result)
            ok = bool(parsed.get("ok"))
            if ok:
                via = str(parsed.get("via", "unknown"))
                self._append_log(f"[INFO] Color Step 1 sent to ChatGPT ({via}).")
                self._c1_prompt_sent_monotonic = time.monotonic()
                if self._auto_get_c1_after_send:
                    self._auto_get_c1_after_send = False
                    self._start_auto_get_c1_poll(timeout_sec=1800, interval_ms=7000)
                return

            reason = str(parsed.get("reason", "unknown"))
            retriable_reasons = {
                "send_disabled",
                "send_button_not_found",
                "composer_not_found",
                "empty_js_result",
                "non_json_js_result",
                "non_dict_js_result",
                "unsupported_js_result_type",
            }
            if retries > 0 and reason in retriable_reasons:
                if retries in (10, 7, 4, 1):
                    self._append_log(
                        f"[INFO] Waiting for ChatGPT send control to become ready ({reason})..."
                    )
                QTimer.singleShot(delay_ms, lambda: self._attempt_chatgpt_send(retries - 1, delay_ms))
                return

            if retries > 0:
                if retries in (10, 7, 4, 1):
                    self._append_log(f"[INFO] Retrying ChatGPT send ({reason})...")
                QTimer.singleShot(delay_ms, lambda: self._attempt_chatgpt_send(retries - 1, delay_ms))
                return

            self._append_log("[WARN] Send button click failed. Trying Enter-key fallback.")
            self._chatgpt_send_enter_fallback()
            self._c1_prompt_sent_monotonic = time.monotonic()
            if self._auto_get_c1_after_send:
                self._auto_get_c1_after_send = False
                self._append_log(
                    "[INFO] Starting Auto Get C1 after Enter fallback (send confirmation unavailable)."
                )
                self._start_auto_get_c1_poll(timeout_sec=1800, interval_ms=7000)

        page.runJavaScript(js, 0, _after_send)

    def _color_step_1(self, *, interactive: bool = True) -> bool:
        if interactive:
            # Manual trigger should cancel any queued auto-send to prevent duplicates.
            self._auto_color_step1_after_attach = False
            self._auto_color_step1_scheduled = False
            self._auto_get_c1_after_send = False
            self._stop_auto_get_c1_poll()

        prompt = ""
        if (not interactive) and self._full_workflow_active and self._locked_workflow_prompt.strip():
            prompt = self._locked_workflow_prompt.strip()
        elif hasattr(self, "prompt_box"):
            prompt = self.prompt_box.toPlainText().strip()
        if not prompt:
            if interactive:
                QMessageBox.warning(self, "Missing prompt", "Prompt is empty.")
            else:
                self._append_log("[WARN] Auto Color Step 1 skipped: prompt is empty.")
            return False
        self._ensure_chat_panel_ready(reason="Color Step 1", focus=True)
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            if interactive:
                QMessageBox.warning(self, "ChatGPT not open", "Open ChatGPT in the embedded browser first.")
            else:
                self._append_log("[WARN] Auto Color Step 1 skipped: ChatGPT is not open in the embedded browser.")
            return False

        self._append_log(f"[INFO] Color Step 1 prompt length: {len(prompt)} chars.")
        self._append_log("[INFO] Color Step 1: filling prompt and sending to ChatGPT.")
        self._attempt_prompt_fill_then_send(prompt, retries=12, delay_ms=550)
        return True

    def _download_and_import_c1(self, *, interactive: bool = True, quiet: bool = False) -> bool:
        if interactive:
            self._stop_auto_get_c1_poll()
            self._auto_get_c1_after_send = False
        self._auto_import_next_download = True
        self._pending_auto_import_download_path = None
        if not quiet:
            self._append_log("[INFO] Get C1: attempting ChatGPT download; will auto-import when complete.")
        started = self._download_c1_from_chatgpt(interactive=interactive, quiet=quiet)
        if not started:
            self._auto_import_next_download = False
            self._pending_auto_import_download_path = None
        return started

    def _download_c1_from_chatgpt(self, *, interactive: bool = True, quiet: bool = False) -> bool:
        self._ensure_chat_panel_ready(reason="Get C1 download automation", focus=False)
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            if interactive:
                QMessageBox.warning(self, "ChatGPT not open", "Open ChatGPT in the embedded browser first.")
            elif not quiet:
                self._append_log("[WARN] Auto Get C1 skipped: ChatGPT is not open in the embedded browser.")
            return False
        page = self.browser.page()
        if page is None:
            return False

        if not quiet:
            self._append_log("[INFO] Attempting C1 download from ChatGPT.")
        self._awaiting_c1_download = True
        self._c1_download_request_seen = False
        self._attempt_chatgpt_download(retries=6, delay_ms=600, quiet=quiet)
        return True

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
        self._ensure_chat_panel_ready(reason="B&W upload automation", focus=True)
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            self._append_log("[INFO] Open ChatGPT first to auto-upload the selected B&W image.")
            self._set_attach_status("open ChatGPT first", ok=False)
            return

        if self.browser_page is None:
            return
        self.pending_bw_upload_path = bw_path
        self.browser_page.set_pending_upload_path(bw_path)

        # Programmatic click on <input type="file"> is blocked by Qt WebEngine's
        # user-activation requirement, so go straight to clipboard paste which
        # is reliable and avoids wasting ~4s on doomed retries.
        self._append_log("[INFO] Attaching B&W image via clipboard paste.")
        self._attach_bw_via_clipboard_paste(bw_path)

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

        # Guard: reject downloads that arrive too soon after the prompt was sent.
        # ChatGPT cannot generate an image in <25s, so any download this early
        # is almost certainly the uploaded B&W image, not the C1 result.
        too_early = False
        if self._c1_prompt_sent_monotonic is not None:
            elapsed = time.monotonic() - self._c1_prompt_sent_monotonic
            if elapsed < self._C1_MIN_WAIT_SEC:
                too_early = True
                self._append_log(
                    f"[INFO] Download ignored (only {elapsed:.0f}s since prompt — "
                    "likely the uploaded B&W image, not the C1 result)."
                )

        if self._awaiting_c1_download and not too_early:
            self._c1_download_request_seen = True
            self._awaiting_c1_download = False
            self._append_log("[INFO] C1 download started from ChatGPT.")
            self._stop_auto_get_c1_poll()
        if self._auto_import_next_download and self._pending_auto_import_download_path is None and not too_early:
            self._pending_auto_import_download_path = target_path
            self._append_log("[INFO] Get C1: captured download request; will import automatically after completion.")
        self._append_log(f"[INFO] Download started: {target_path}")

    # ------------------------------------------------------------------
    # Pickers + previews
    # ------------------------------------------------------------------

    def _pick_downloads_dir(self) -> None:
        start = self.downloads_edit.text().strip() or self._default_downloads_dir()
        selected = QFileDialog.getExistingDirectory(self, "Select downloads folder", start)
        if selected:
            self.downloads_edit.setText(selected)

    def _set_bw_image_path(self, image_path: str, *, source_label: str) -> None:
        selected_path = Path(image_path).expanduser()
        if not selected_path.exists() or not selected_path.is_file():
            self._append_log(f"[WARN] {source_label}: file not found: {selected_path}")
            return
        if selected_path.suffix.lower() not in IMAGE_SUFFIXES:
            self._append_log(f"[WARN] {source_label}: unsupported file type: {selected_path.suffix}")
            return

        resolved = str(selected_path.resolve())
        self.bw_edit.setText(resolved)
        self.pending_bw_upload_path = Path(resolved)
        self._sync_input_preview_from_field()
        self._stop_auto_get_c1_poll()
        self._auto_get_c1_after_send = False
        self._auto_color_step1_after_attach = False
        self._auto_color_step1_scheduled = False
        self._set_attach_status("photo selected (not uploaded yet)", ok=None)
        self._append_log("[INFO] B&W selected. Click Colorize to upload to ChatGPT and run full workflow.")

    def _on_bw_dropped(self, image_path: str) -> None:
        self._set_bw_image_path(image_path, source_label="Drag-and-drop")

    def _on_pick_bw(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Select B&W image", "", IMAGE_FILTER)
        if selected:
            self._set_bw_image_path(selected, source_label="Choose B&W")

    def _pick_outdir(self) -> None:
        start = self.outdir_edit.text().strip() or str(self.outputs_root_dir)
        selected = QFileDialog.getExistingDirectory(self, "Select output directory", start)
        if selected:
            self.outdir_edit.setText(selected)
            self._last_run_outdir = Path(selected)
            self._load_outputs()

    @staticmethod
    def _chatgpt_prompt_js(prompt: str) -> str:
        prompt_json = json.dumps(prompt)
        return f"""
(() => {{
  const prompt = {prompt_json};
  const selectors = [
    'textarea[data-testid="prompt-textarea"]',
    'textarea#prompt-textarea',
    'textarea[aria-label*="Message"]',
    'textarea[placeholder*="Message"]',
    'textarea',
    'div[data-testid="prompt-textarea"][contenteditable]',
    'div[contenteditable="true"][data-testid="prompt-textarea"]',
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="plaintext-only"][role="textbox"]',
    'div[contenteditable="plaintext-only"]',
    'div[contenteditable][role="textbox"]',
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
        ensure_chat_visible: bool = True,
        done=None,
    ) -> None:
        if ensure_chat_visible:
            self._ensure_chat_panel_ready(reason="prompt transfer", focus=False)
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

    def _import_download_path(self, image_path: Path, *, source_label: str) -> None:
        if not image_path.exists() or not image_path.is_file():
            self._append_log(f"[WARN] {source_label}: image not found: {image_path}")
            return
        self._stop_auto_get_c1_poll()
        self.color_edit.setText(str(image_path))
        self._sync_c1_preview_from_field()
        if not self._current_preview_paths:
            self._set_selected_output_preview(None)
        self._append_log(f"[INFO] {source_label}: imported {image_path}")
        self._log_border_ratio_if_possible()
        self._refresh_workflow_status()
        if self._full_workflow_active and self._full_workflow_waiting_for_c1:
            self._full_workflow_waiting_for_c1 = False
            if self._full_workflow_pending_delete:
                self._append_log(
                    "[INFO] Workflow: C1 imported. Deleting ChatGPT conversation before overlay run."
                )
                self._full_workflow_pending_delete = False
                self._delete_current_chat_automated(done=self._on_full_workflow_delete_done)
                return
            self._on_full_workflow_delete_done(False)

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
        self._import_download_path(latest, source_label="Import Latest")

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

    def _start_full_workflow(self) -> None:
        if self._full_workflow_active:
            QMessageBox.warning(self, "Workflow in progress", "Colorize workflow is already in progress.")
            return
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Run in progress", "A run is already in progress.")
            return
        if not self.runner_script.exists():
            QMessageBox.critical(self, "Missing runner", f"Could not find:\n{self.runner_script}")
            return

        bw_path = Path(self.bw_edit.text().strip())
        if not bw_path.exists():
            QMessageBox.warning(self, "Missing image", f"B&W original not found:\n{bw_path}")
            return

        prompt = self.prompt_box.toPlainText().strip() if hasattr(self, "prompt_box") else ""
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Prompt is empty.")
            return

        # Run auto-detect and show manual crop confirmation
        self._append_log("[INFO] Running crop detection on input image...")
        self._set_status("Detecting borders...")
        self.run_btn.setEnabled(False)

        self._pending_workflow_bw_path = bw_path

        def _detect() -> None:
            try:
                import cv2
                from romav2.preprocess import analyze_and_crop_image
                img = cv2.imread(str(bw_path))
                if img is None:
                    self._crop_detect_done.emit(None)
                    return
                result = analyze_and_crop_image(img, enable_crop=False, debug=False)
                self._crop_detect_done.emit(result)
            except Exception:
                self._crop_detect_done.emit(None)

        threading.Thread(target=_detect, daemon=True).start()

    def _on_crop_detect_done(self, result) -> None:
        """Handle crop detection result — show crop confirmation UI."""
        import cv2

        bw_path = self._pending_workflow_bw_path
        img = cv2.imread(str(bw_path))
        if img is None:
            self._append_log("[WARN] Could not read image for crop detection.")
            self.run_btn.setEnabled(True)
            self._set_status("Ready")
            return

        self._crop_source_bgr = img
        self.crop_widget.set_image(img)

        # Try to get a quad from auto-detect
        quad_pts = None
        if result is not None:
            try:
                feats = result.features
                if feats.best_quad is not None:
                    pts = feats.best_quad.reshape(-1, 2).tolist()
                    if len(pts) == 4:
                        from romav2.preprocess.feature_extraction import _order_quad
                        ordered = _order_quad(np.array(pts, dtype=np.float32))
                        quad_pts = [(float(p[0]), float(p[1])) for p in ordered]
                if quad_pts is None and feats.largest_contour is not None:
                    x, y, rw, rh = cv2.boundingRect(feats.largest_contour)
                    quad_pts = [
                        (float(x), float(y)),
                        (float(x + rw), float(y)),
                        (float(x + rw), float(y + rh)),
                        (float(x), float(y + rh)),
                    ]
            except Exception:
                pass

        if quad_pts is not None:
            self.crop_widget.set_quad(quad_pts)
            conf = result.decision.confidence if result else 0
            case = result.decision.case_label.value if result else "unknown"
            self.crop_status_label.setText(
                f"Auto-detected: {case} (confidence {conf:.0%}). "
                "Adjust corners or confirm."
            )
            self._append_log(
                f"[INFO] Crop detection: {case} (confidence={conf:.3f}). "
                "Showing crop preview for confirmation."
            )
        else:
            self.crop_widget.set_default_quad()
            self.crop_status_label.setText(
                "No clear border detected. Adjust corners if needed, or skip."
            )
            self._append_log("[INFO] No clear border detected. Showing default crop for manual adjustment.")

        # Show the crop UI
        self.input_view_stack.setCurrentIndex(1)
        self.crop_bar_widget.setVisible(True)
        self._set_status("Review crop — confirm or skip to continue")

    def _on_crop_confirmed(self) -> None:
        """User confirmed the crop — apply warp and continue workflow."""
        import cv2

        points = self.crop_widget.get_points()
        img = self._crop_source_bgr
        if img is None or len(points) < 3:
            self._on_crop_skipped()
            return

        h, w = img.shape[:2]

        if len(points) == 4:
            from romav2.preprocess.feature_extraction import _order_quad
            pts = np.array(points, dtype=np.float32)
            ordered = _order_quad(pts)

            width_top = float(np.linalg.norm(ordered[1] - ordered[0]))
            width_bot = float(np.linalg.norm(ordered[2] - ordered[3]))
            height_left = float(np.linalg.norm(ordered[3] - ordered[0]))
            height_right = float(np.linalg.norm(ordered[2] - ordered[1]))

            out_w = int(max(width_top, width_bot))
            out_h = int(max(height_left, height_right))
            if out_w < 10 or out_h < 10:
                self._append_log("[WARN] Crop region too small, skipping crop.")
                self._on_crop_skipped()
                return

            dst = np.array(
                [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(ordered, dst)
            cropped = cv2.warpPerspective(img, M, (out_w, out_h))
            self._append_log(f"[INFO] Crop applied: perspective warp to {out_w}x{out_h}")
        else:
            poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            x, y, rw, rh = cv2.boundingRect(poly)
            margin = max(2, int(min(h, w) * 0.003))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + rw + margin)
            y2 = min(h, y + rh + margin)
            cropped = img[y1:y2, x1:x2].copy()
            self._append_log(f"[INFO] Crop applied: {len(points)}-pt polygon to ({x1},{y1})-({x2},{y2})")

        # Save cropped image and update the B&W path
        bw_path = self._pending_workflow_bw_path
        prepped_name = bw_path.stem + "_cropped" + bw_path.suffix
        prepped_path = bw_path.parent / prepped_name
        cv2.imwrite(str(prepped_path), cropped)
        self._pending_workflow_bw_path = prepped_path
        self.bw_edit.setText(str(prepped_path))
        self._append_log(f"[INFO] Saved cropped image: {prepped_path.name}")

        self._exit_crop_review()
        self._continue_workflow_after_crop()

    def _on_crop_skipped(self) -> None:
        """User chose to skip cropping — continue with original image."""
        self._append_log("[INFO] Crop skipped — using original image.")
        self._exit_crop_review()
        self._continue_workflow_after_crop()

    def _exit_crop_review(self) -> None:
        """Hide the crop UI and return to normal input preview."""
        self.input_view_stack.setCurrentIndex(0)
        self.crop_bar_widget.setVisible(False)
        self._crop_source_bgr = None

    def _continue_workflow_after_crop(self) -> None:
        """Resume the colorization workflow after crop review is done."""
        bw_path = self._pending_workflow_bw_path

        prompt = self.prompt_box.toPlainText().strip() if hasattr(self, "prompt_box") else ""
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Prompt is empty.")
            self.run_btn.setEnabled(True)
            self._set_status("Ready")
            return

        self._locked_workflow_prompt = prompt
        self._append_log(f"[INFO] Locked Step 2 prompt for this run ({len(prompt)} chars).")
        self._ensure_chat_panel_ready(reason="Colorize workflow", focus=True)
        current = self.browser.url().toString().lower() if hasattr(self, "browser") else ""
        if "chatgpt.com" not in current:
            self._locked_workflow_prompt = ""
            self._navigate_chatgpt()
            QMessageBox.warning(
                self,
                "ChatGPT loading",
                "Chat panel was opened and ChatGPT is loading. Click Colorize again once the page is ready.",
            )
            self.run_btn.setEnabled(True)
            self._set_status("Ready")
            return

        self._stop_auto_get_c1_poll()
        self._auto_get_c1_after_send = False
        self._auto_color_step1_after_attach = True
        self._auto_color_step1_scheduled = False
        self._full_workflow_active = True
        self._full_workflow_waiting_for_c1 = True
        self._full_workflow_pending_delete = True
        self._c1_image_nudge_sent = False
        self._set_status("Running ChatGPT workflow...")
        self.stop_btn.setEnabled(False)
        # Start shimmer immediately with loading messages (no C1 source yet)
        if hasattr(self, "shimmer_widget") and hasattr(self, "result_stack"):
            self.shimmer_widget.set_source(None)
            self.shimmer_widget.start()
            self.result_stack.setCurrentIndex(1)
            self._shimmer_active = True
            self._before_after_active = False
        self._append_log("")
        self._append_log(
            "[INFO] Colorize workflow started: upload photo -> send prompt -> auto Get C1 -> delete chat -> overlay run."
        )
        if hasattr(self, "accuracy_mode_check") and self.accuracy_mode_check.isChecked():
            self._append_log("[INFO] Accuracy mode is ON (preset + stronger multi-pass pre-align + iterative rematch).")
        self._set_attach_status("workflow running: uploading to ChatGPT", ok=None)
        self.pending_bw_upload_path = bw_path
        self._attach_selected_bw_to_chatgpt()

    def _on_full_workflow_delete_done(self, success: bool) -> None:
        if not self._full_workflow_active:
            return
        if success:
            self._append_log("[INFO] Workflow: ChatGPT conversation deleted.")
        else:
            self._append_log("[WARN] Workflow: delete chat was not confirmed. Continuing to overlay run.")

        self._full_workflow_active = False
        self._full_workflow_waiting_for_c1 = False
        self._full_workflow_pending_delete = False
        self._locked_workflow_prompt = ""
        self._append_log("[INFO] Workflow: launching overlay process.")
        self._start_run()
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self._set_status("Ready")
            self._refresh_workflow_status()

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

        outdir = self._next_run_output_dir()
        self.outdir_edit.setText(str(outdir))
        self._last_run_outdir = outdir
        self._last_download_result_path = None
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
            self._effective_roma_setting(),
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
            "--chroma-boost",
            str(self.chroma_boost_spin.value()),
            "--chroma-edge-preserve",
            str(self.chroma_edge_preserve_spin.value()),
            "--bw-black-point-clip",
            str(self.bw_black_clip_spin.value()),
            "--bw-white-point-clip",
            str(self.bw_white_clip_spin.value()),
            "--bw-midtone-target",
            str(self.bw_midtone_target_spin.value()),
            "--color-opacity",
            str(self.color_opacity_spin.value()),
        ]
        if not self.bw_gray_balance_check.isChecked():
            cmd.append("--disable-bw-gray-balance")
        if self.adaptive_chroma_check.isChecked():
            cmd.append("--adaptive-chroma-match")
        if hasattr(self, "accuracy_mode_check") and self.accuracy_mode_check.isChecked():
            cmd.append("--accuracy-mode")
        if self.compile_check.isChecked():
            cmd.append("--compile")
        if hasattr(self, "border_exclude_check") and self.border_exclude_check.isChecked():
            cmd.append("--border-exclude-mask")
        cmd.extend(self._c1_adherence_extra_flags())

        self._append_log("")
        self._append_log("=== Colorization Run ===")
        if hasattr(self, "c1_adherence_combo"):
            self._append_log(f"C1 adherence effort: {self.c1_adherence_combo.currentText()}")
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
        self._reset_pipeline_progress()

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

    # Map pipeline log keywords to stage indices for progress tracking.
    _STAGE_TRIGGERS = [
        ("Initializing RoMa",                    0),  # Model init
        ("Running dense match (ref -> src)",      1),  # Dense match
        ("Estimating global pre-alignment",       2),  # Pre-alignment
        ("Iterative re-match refinement",         3),  # Iterative re-match
        ("Building source->reference warp",       4),  # Warp build
        ("Applying guided-filter smoothing",      5),  # Guided filter
        ("Sampling correspondences",              6),  # Sampling
        ("B&W base normalized",                   7),  # Color transfer
        ("Building before/after diagnostic",      8),  # Diagnostics
        ("[INFO] Done.",                           9),  # Done
    ]

    def _reset_pipeline_progress(self) -> None:
        self._pipeline_stage_index = 0
        if hasattr(self, "pipeline_progress"):
            self.pipeline_progress.setValue(0)
            self.pipeline_progress.setFormat("Starting...")
            self.pipeline_progress.setVisible(True)

    def _update_pipeline_progress(self, line: str) -> None:
        if not hasattr(self, "pipeline_progress"):
            return
        for trigger_text, stage_idx in self._STAGE_TRIGGERS:
            if trigger_text in line and stage_idx >= self._pipeline_stage_index:
                self._pipeline_stage_index = stage_idx + 1
                label = self._PIPELINE_STAGES[stage_idx][0]
                self.pipeline_progress.setValue(self._pipeline_stage_index)
                self.pipeline_progress.setFormat(f"{label}")
                break
        if "[ERROR]" in line:
            self.pipeline_progress.setFormat("Error")

    def _hide_pipeline_progress(self) -> None:
        if hasattr(self, "pipeline_progress"):
            self.pipeline_progress.setVisible(False)

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
            stripped = line.rstrip()
            self._append_log(stripped)
            self._update_pipeline_progress(stripped)

    def _on_process_error(self, error) -> None:
        if self.process is None:
            return
        name = str(error).split(".")[-1]
        self._append_log(f"[ERROR] Runner process error: {name}")
        if self.process.state() == QProcess.NotRunning:
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.process = None
            self._locked_workflow_prompt = ""
            self._run_watchdog.stop()
            self._run_started_monotonic = None
            self._last_process_output_monotonic = None
            self._last_run_watchdog_log_monotonic = None
            self._set_status("Failed")
            self._hide_pipeline_progress()
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
            if idle >= 120.0:
                self._append_log(
                    "[INFO] If stalls are long, check GPU contention/thermals with `nvidia-smi` "
                    "(other GPU apps can slow RoMa dramatically)."
                )
            self._last_run_watchdog_log_monotonic = now

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        if self._process_partial:
            self._append_log(self._process_partial.rstrip())
            self._process_partial = ""

        if exit_code == 0:
            self._append_log("Colorization complete.")
            self._stop_shimmer()
            self._load_outputs()
            self._result_saved = False
            if hasattr(self, "save_result_btn"):
                self.save_result_btn.setEnabled(True)
            self._set_status("Done!")
            self._show_results_tab()
            self._clear_active_test_preset()
            # Reset ChatGPT to a fresh chat so the user is ready for the next run
            QTimer.singleShot(500, self._navigate_chatgpt)
        else:
            self._append_log(f"Pipeline exited with error code {exit_code}.")
            self._set_status("Failed")
            self._stop_shimmer()
            self._show_log_tab()
            self._clear_active_test_preset()

        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process = None
        self._locked_workflow_prompt = ""
        self._run_watchdog.stop()
        self._run_started_monotonic = None
        self._last_process_output_monotonic = None
        self._last_run_watchdog_log_monotonic = None
        self._hide_pipeline_progress()
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
        self._result_pixmaps.clear()
        self._current_preview_paths.clear()
        self._selected_preview_title = None

    def _load_outputs_placeholder(self) -> None:
        self._set_selected_output_preview(None)
        self._refresh_output_previews()

    def _load_outputs(self) -> None:
        outdir = Path(self.outdir_edit.text().strip())
        if outdir.exists():
            self._last_run_outdir = outdir
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
            self._select_primary_output()

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
        self._refresh_output_previews()

    def _refresh_output_previews(self) -> None:
        if hasattr(self, "input_preview_label"):
            if self._input_preview_pixmap is None or self._input_preview_pixmap.isNull():
                self.input_preview_label.setPixmap(QPixmap())
                self.input_preview_label.setText("Drag and drop a B&W image here.")
            else:
                target_w = max(120, self.input_preview_label.width() - 12)
                target_h = max(120, self.input_preview_label.height() - 12)
                scaled = self._input_preview_pixmap.scaled(
                    target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.input_preview_label.setText("")
                self.input_preview_label.setPixmap(scaled)

        if not hasattr(self, "result_stack"):
            return

        # Determine which result view to show
        has_final = self._selected_preview_title and self._selected_preview_title in self._result_pixmaps
        has_c1 = self._c1_preview_pixmap is not None and not self._c1_preview_pixmap.isNull()

        if has_final and self._input_preview_pixmap is not None:
            # Before/after slider mode
            final_pix = self._result_pixmaps[self._selected_preview_title]
            self.before_after_slider.set_images(self._input_preview_pixmap, final_pix)
            self.result_stack.setCurrentIndex(2)
            self._stop_shimmer()
            self._before_after_active = True
            self.result_preview_label.setToolTip("Click to view full screen")
            self.result_preview_label.setCursor(Qt.CursorShape.PointingHandCursor)
            return

        if has_c1 and not has_final:
            # Shimmer mode — C1 imported but final not yet generated.
            # If shimmer is already running (started at workflow launch), just
            # update the source so the C1 image fades in smoothly.
            if self._shimmer_active:
                self.shimmer_widget.set_source(self._c1_preview_pixmap)
            else:
                self.shimmer_widget.set_source(self._c1_preview_pixmap)
                self.shimmer_widget.start()
            self.result_stack.setCurrentIndex(1)
            self._shimmer_active = True
            self._before_after_active = False
            return

        # If shimmer is running (workflow in progress, no C1 yet), keep it
        if self._shimmer_active and not has_final:
            return

        # Default: placeholder label
        self._stop_shimmer()
        self._before_after_active = False
        self.result_stack.setCurrentIndex(0)

        pix: QPixmap | None = None
        placeholder = "Run Colorize to generate result."
        if has_final:
            pix = self._result_pixmaps[self._selected_preview_title]

        if pix is None or pix.isNull():
            self.result_preview_label.setPixmap(QPixmap())
            self.result_preview_label.setText(placeholder)
            self.result_preview_label.setToolTip("")
            self.result_preview_label.setCursor(Qt.CursorShape.ArrowCursor)
            return

        target_w = max(120, self.result_preview_label.width() - 12)
        target_h = max(120, self.result_preview_label.height() - 12)
        scaled = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_preview_label.setText("")
        self.result_preview_label.setPixmap(scaled)
        self.result_preview_label.setToolTip("Click to view full screen")
        self.result_preview_label.setCursor(Qt.CursorShape.PointingHandCursor)

    def _stop_shimmer(self) -> None:
        if hasattr(self, "shimmer_widget"):
            self.shimmer_widget.stop()
        self._shimmer_active = False

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        super().resizeEvent(event)
        self._apply_responsive_layout()
        self._refresh_output_previews()


def main() -> int:
    app = QApplication(sys.argv)
    win = PhotoColorizerQt()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
