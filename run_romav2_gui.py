from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk


ROMA_SETTINGS = ("turbo", "fast", "base", "precise", "mega1500", "scannet1500", "wxbs", "satast")

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


class PhotoColorizerGui(tk.Tk):
    """GUI for AI-assisted photo colorization workflow.

    Workflow:
      1. Load a B&W original photo.
      2. Load the AI-colorized recreation (e.g. from ChatGPT).
      3. Click Colorize -- the pipeline warps the color image back onto
         the original geometry, then transfers color via LAB.
      4. Review the result and save.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("Photo Colorizer")
        self.geometry("1500x950")
        self.minsize(1100, 700)

        self.repo_root = Path(__file__).resolve().parent
        self.runner_script = self.repo_root / "run_romav2_pair.py"

        # --- Variables ---
        # Main inputs
        self.bw_var = tk.StringVar()
        self.color_var = tk.StringVar()
        self.outdir_var = tk.StringVar(
            value=str(self.repo_root / "outputs" / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )

        # Colorization controls
        self.gf_radius_var = tk.IntVar(value=16)
        self.gf_eps_var = tk.DoubleVar(value=0.001)
        self.chroma_radius_var = tk.IntVar(value=48)
        self.color_opacity_var = tk.DoubleVar(value=1.0)

        # Advanced (collapsed by default)
        self.setting_var = tk.StringVar(value="precise")
        self.num_samples_var = tk.IntVar(value=5000)
        self.max_draw_var = tk.IntVar(value=1200)
        self.compile_var = tk.BooleanVar(value=False)
        self.reg_thresh_var = tk.DoubleVar(value=0.35)
        self.reg_fallback_var = tk.StringVar(value="identity")

        # State
        self.status_var = tk.StringVar(value="Ready")
        self.msg_queue: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_proc: subprocess.Popen[str] | None = None
        self.preview_photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.input_photo_refs: dict[str, ImageTk.PhotoImage] = {}

        # Widget refs
        self.run_btn: ttk.Button
        self.stop_btn: ttk.Button
        self.log_box: ScrolledText
        self.notebook: ttk.Notebook
        self.bw_preview_label: ttk.Label
        self.color_preview_label: ttk.Label

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Subtle.TLabel", font=("Segoe UI", 8), foreground="#666")
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 6))

        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0, minsize=340)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # ---- Left panel: inputs + controls ----
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        self._build_input_section(left)
        self._build_controls_section(left)
        self._build_action_section(left)
        self._build_advanced_section(left)
        self._build_log_section(left)

        # ---- Right panel: results ----
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Placeholder tab
        placeholder = ttk.Frame(self.notebook)
        ttk.Label(
            placeholder,
            text="Load images and click Colorize to see results here.",
            anchor="center",
            font=("Segoe UI", 10),
            foreground="#999",
        ).pack(expand=True)
        self.notebook.add(placeholder, text="Results")

    def _build_input_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="  1. Load Images  ", padding=8)
        section.pack(fill="x", pady=(0, 6))
        section.columnconfigure(0, weight=1)

        # B&W original
        bw_frame = ttk.Frame(section)
        bw_frame.pack(fill="x", pady=(0, 4))
        bw_frame.columnconfigure(0, weight=1)

        bw_top = ttk.Frame(bw_frame)
        bw_top.pack(fill="x")
        ttk.Label(bw_top, text="B&W Original", style="Header.TLabel").pack(side="left")
        ttk.Button(bw_top, text="Browse", command=lambda: self._pick_image(self.bw_var, self.bw_preview_label)).pack(side="right")

        ttk.Entry(bw_frame, textvariable=self.bw_var).pack(fill="x", pady=(2, 2))
        self.bw_preview_label = ttk.Label(bw_frame, anchor="center")
        self.bw_preview_label.pack(fill="x", pady=(2, 4))

        # AI colorized
        color_frame = ttk.Frame(section)
        color_frame.pack(fill="x", pady=(4, 0))
        color_frame.columnconfigure(0, weight=1)

        color_top = ttk.Frame(color_frame)
        color_top.pack(fill="x")
        ttk.Label(color_top, text="AI Colorized Version", style="Header.TLabel").pack(side="left")
        ttk.Button(color_top, text="Browse", command=lambda: self._pick_image(self.color_var, self.color_preview_label)).pack(side="right")

        ttk.Entry(color_frame, textvariable=self.color_var).pack(fill="x", pady=(2, 2))
        self.color_preview_label = ttk.Label(color_frame, anchor="center")
        self.color_preview_label.pack(fill="x", pady=(2, 0))

        # Quick import pair
        ttk.Button(
            section,
            text="Import Pair (B&W first, then Colorized)",
            command=self._pick_pair,
        ).pack(fill="x", pady=(8, 0))

    def _build_controls_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="  2. Colorization Settings  ", padding=8)
        section.pack(fill="x", pady=(0, 6))
        section.columnconfigure(1, weight=1)

        ttk.Label(section, text="Edge smoothing radius").grid(row=0, column=0, sticky="w", pady=2)
        radius_frame = ttk.Frame(section)
        radius_frame.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Scale(
            radius_frame,
            from_=0,
            to=48,
            orient="horizontal",
            variable=self.gf_radius_var,
            command=lambda _: self.gf_radius_var.set(int(float(self.gf_radius_var.get()))),
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(radius_frame, textvariable=self.gf_radius_var, width=3).pack(side="right", padx=(4, 0))

        ttk.Label(section, text="Edge sensitivity").grid(row=1, column=0, sticky="w", pady=2)
        eps_combo = ttk.Combobox(
            section,
            textvariable=self.gf_eps_var,
            values=[0.0001, 0.001, 0.01, 0.1],
            width=8,
        )
        eps_combo.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(
            section,
            text="Warp radius: smooths the warp field (small). "
                 "Lower sensitivity = sharper edges.",
            style="Subtle.TLabel",
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Chrominance filter
        ttk.Label(section, text="Chroma filter radius").grid(row=3, column=0, sticky="w", pady=2)
        chroma_frame = ttk.Frame(section)
        chroma_frame.grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Scale(
            chroma_frame,
            from_=0,
            to=128,
            variable=self.chroma_radius_var,
            command=lambda _: self.chroma_radius_var.set(int(float(self.chroma_radius_var.get()))),
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(chroma_frame, textvariable=self.chroma_radius_var, width=3).pack(side="right", padx=(4, 0))
        ttk.Label(
            section,
            text="Smooths color over misaligned features. Larger = fewer splotches.",
            style="Subtle.TLabel",
            justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Color opacity
        ttk.Label(section, text="Color opacity").grid(row=5, column=0, sticky="w", pady=2)
        opacity_frame = ttk.Frame(section)
        opacity_frame.grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Scale(
            opacity_frame,
            from_=0.0,
            to=1.0,
            variable=self.color_opacity_var,
            command=lambda _: self.color_opacity_var.set(round(float(self.color_opacity_var.get()), 2)),
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(opacity_frame, textvariable=self.color_opacity_var, width=5).pack(side="right", padx=(4, 0))

        # Output dir
        ttk.Label(section, text="Output folder").grid(row=6, column=0, sticky="w", pady=(8, 2))
        out_frame = ttk.Frame(section)
        out_frame.grid(row=6, column=1, sticky="ew", pady=(8, 2))
        out_frame.columnconfigure(0, weight=1)
        ttk.Entry(out_frame, textvariable=self.outdir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_frame, text="...", width=3, command=lambda: self._pick_dir(self.outdir_var)).pack(side="right", padx=(4, 0))

    def _build_action_section(self, parent: ttk.Frame) -> None:
        section = ttk.Frame(parent)
        section.pack(fill="x", pady=(0, 6))
        section.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(section)
        btn_frame.pack(fill="x")

        self.run_btn = ttk.Button(
            btn_frame,
            text="Colorize",
            style="Big.TButton",
            command=self._start_run,
        )
        self.run_btn.pack(side="left", fill="x", expand=True)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_run, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        btn_frame2 = ttk.Frame(section)
        btn_frame2.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_frame2, text="New Run", command=self._new_run).pack(side="left")
        ttk.Button(btn_frame2, text="Open Folder", command=self._open_output_folder).pack(side="left", padx=(6, 0))

        ttk.Label(section, textvariable=self.status_var, style="Subtle.TLabel").pack(anchor="w", pady=(2, 0))

    def _build_advanced_section(self, parent: ttk.Frame) -> None:
        # Collapsible advanced settings
        self._adv_visible = tk.BooleanVar(value=False)

        toggle_frame = ttk.Frame(parent)
        toggle_frame.pack(fill="x", pady=(0, 2))
        self._adv_toggle_btn = ttk.Button(
            toggle_frame,
            text="+ Advanced Settings",
            command=self._toggle_advanced,
        )
        self._adv_toggle_btn.pack(anchor="w")

        self._adv_frame = ttk.LabelFrame(parent, text="  Advanced (RoMa Engine)  ", padding=8)
        # Not packed initially -- collapsed

        self._adv_frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(self._adv_frame, text="Quality preset").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Combobox(
            self._adv_frame,
            textvariable=self.setting_var,
            values=ROMA_SETTINGS,
            state="readonly",
            width=14,
        ).grid(row=row, column=1, sticky="w", pady=2)

        row += 1
        ttk.Label(self._adv_frame, text="Regularization").grid(row=row, column=0, sticky="w", pady=2)
        reg_frame = ttk.Frame(self._adv_frame)
        reg_frame.grid(row=row, column=1, sticky="w", pady=2)
        ttk.Label(reg_frame, text="thresh").pack(side="left")
        ttk.Entry(reg_frame, textvariable=self.reg_thresh_var, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(reg_frame, text="fallback").pack(side="left")
        ttk.Combobox(
            reg_frame,
            textvariable=self.reg_fallback_var,
            values=("identity", "reference", "none"),
            state="readonly",
            width=10,
        ).pack(side="left", padx=(4, 0))

        row += 1
        ttk.Label(self._adv_frame, text="Sampling").grid(row=row, column=0, sticky="w", pady=2)
        samp_frame = ttk.Frame(self._adv_frame)
        samp_frame.grid(row=row, column=1, sticky="w", pady=2)
        ttk.Label(samp_frame, text="samples").pack(side="left")
        ttk.Entry(samp_frame, textvariable=self.num_samples_var, width=8).pack(side="left", padx=(4, 10))
        ttk.Label(samp_frame, text="max draw").pack(side="left")
        ttk.Entry(samp_frame, textvariable=self.max_draw_var, width=8).pack(side="left", padx=(4, 0))

        row += 1
        ttk.Checkbutton(
            self._adv_frame,
            text="torch.compile (slower startup, faster inference)",
            variable=self.compile_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)

    def _toggle_advanced(self) -> None:
        if self._adv_visible.get():
            self._adv_frame.pack_forget()
            self._adv_toggle_btn.configure(text="+ Advanced Settings")
            self._adv_visible.set(False)
        else:
            self._adv_frame.pack(fill="x", pady=(0, 6), after=self._adv_toggle_btn.master)
            self._adv_toggle_btn.configure(text="- Advanced Settings")
            self._adv_visible.set(True)

    def _build_log_section(self, parent: ttk.Frame) -> None:
        log_frame = ttk.LabelFrame(parent, text="  Log  ", padding=4)
        log_frame.pack(fill="both", expand=True, pady=(0, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_box = ScrolledText(log_frame, height=8, wrap="word", font=("Consolas", 8))
        self.log_box.grid(row=0, column=0, sticky="nsew")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------

    def _pick_image(self, var: tk.StringVar, preview_label: ttk.Label) -> None:
        selected = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            var.set(selected)
            self._update_input_preview(selected, preview_label)

    def _pick_pair(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select B&W ORIGINAL first, then AI COLORIZED",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if len(selected) >= 2:
            self.bw_var.set(selected[0])
            self.color_var.set(selected[1])
            self._update_input_preview(selected[0], self.bw_preview_label)
            self._update_input_preview(selected[1], self.color_preview_label)
        elif len(selected) == 1:
            self.bw_var.set(selected[0])
            self._update_input_preview(selected[0], self.bw_preview_label)

    def _pick_dir(self, var: tk.StringVar) -> None:
        initial = var.get().strip() or str(self.repo_root)
        selected = filedialog.askdirectory(title="Select output directory", initialdir=initial)
        if selected:
            var.set(selected)

    def _update_input_preview(self, path_str: str, label: ttk.Label) -> None:
        try:
            img = Image.open(path_str).convert("RGB")
            max_w, max_h = 300, 100
            scale = min(max_w / img.width, max_h / img.height, 1.0)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img_thumb = img.resize(new_size, resample=self._resample_lanczos())
            photo = ImageTk.PhotoImage(img_thumb)
            label.configure(image=photo)
            self.input_photo_refs[path_str] = photo
        except Exception:
            label.configure(image="")

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------

    def _start_run(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("Run in progress", "A run is already in progress.")
            return
        if not self.runner_script.exists():
            messagebox.showerror("Missing runner", f"Could not find:\n{self.runner_script}")
            return

        bw_path = Path(self.bw_var.get().strip())
        color_path = Path(self.color_var.get().strip())
        outdir_text = self.outdir_var.get().strip()
        if not outdir_text:
            outdir_text = str(
                self.repo_root / "outputs" / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.outdir_var.set(outdir_text)
        outdir = Path(outdir_text)

        if not bw_path.exists():
            messagebox.showerror("Missing image", f"B&W original not found:\n{bw_path}")
            return
        if not color_path.exists():
            messagebox.showerror("Missing image", f"AI colorized image not found:\n{color_path}")
            return

        try:
            gf_radius = int(self.gf_radius_var.get())
            gf_eps = float(self.gf_eps_var.get())
            color_opacity = float(self.color_opacity_var.get())
            num_samples = int(self.num_samples_var.get())
            max_draw = int(self.max_draw_var.get())
            reg_thresh = float(self.reg_thresh_var.get())
            if num_samples <= 0 or max_draw <= 0:
                raise ValueError
            if not (0.0 <= reg_thresh < 1.0):
                raise ValueError
            if not (0.0 <= color_opacity <= 1.0):
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid settings", "Check numeric values in settings.")
            return

        outdir.mkdir(parents=True, exist_ok=True)

        # Build command: B&W original = --ref (geometry authority),
        #                AI colorized  = --src (color donor, gets warped)
        cmd = [
            sys.executable,
            str(self.runner_script),
            "--ref", str(bw_path),
            "--src", str(color_path),
            "--outdir", str(outdir),
            "--setting", self.setting_var.get().strip(),
            "--num-samples", str(num_samples),
            "--max-draw", str(max_draw),
            "--regularize-overlap-thresh", str(reg_thresh),
            "--regularize-fallback", self.reg_fallback_var.get().strip(),
            "--guided-filter-radius", str(gf_radius),
            "--guided-filter-eps", str(gf_eps),
            "--chroma-filter-radius", str(self.chroma_radius_var.get()),
            "--color-opacity", str(color_opacity),
        ]
        if self.compile_var.get():
            cmd.append("--compile")

        self._append_log("")
        self._append_log("=== Colorization Run ===")
        self._append_log(f"B&W Original: {bw_path}")
        self._append_log(f"AI Colorized: {color_path}")
        self._append_log(f"Output: {outdir}")
        self._append_log("Command: " + " ".join(shlex.quote(p) for p in cmd))
        self._clear_previews()

        self.status_var.set("Running...")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self.worker_thread = threading.Thread(target=self._worker_run, args=(cmd,), daemon=True)
        self.worker_thread.start()
        self.after(100, self._poll_messages)

    def _worker_run(self, cmd: list[str]) -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.current_proc = proc
            if proc.stdout is not None:
                for line in proc.stdout:
                    self.msg_queue.put(("log", line.rstrip()))
            rc = proc.wait()
            self.msg_queue.put(("done", int(rc)))
        except Exception:
            self.msg_queue.put(("error", traceback.format_exc()))
        finally:
            self.current_proc = None

    def _poll_messages(self) -> None:
        while True:
            try:
                kind, payload = self.msg_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
            elif kind == "error":
                self._append_log("[ERROR] " + str(payload))
                self._on_run_finished(success=False)
            elif kind == "done":
                rc = int(payload)
                if rc == 0:
                    self._append_log("Colorization complete.")
                    self._load_outputs()
                    self._on_run_finished(success=True)
                else:
                    self._append_log(f"Pipeline exited with error code {rc}.")
                    self._on_run_finished(success=False)

        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.after(100, self._poll_messages)

    def _on_run_finished(self, *, success: bool) -> None:
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Done!" if success else "Failed")

    def _stop_run(self) -> None:
        proc = self.current_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            self._append_log("Stop requested.")
            self.status_var.set("Stopping...")

    def _new_run(self) -> None:
        """Reset inputs and previews for a fresh colorization run."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("Run in progress", "Stop the current run first.")
            return

        self.bw_var.set("")
        self.color_var.set("")
        self.outdir_var.set(
            str(self.repo_root / "outputs" / f"colorize_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
        self.bw_preview_label.configure(image="")
        self.color_preview_label.configure(image="")
        self.input_photo_refs.clear()
        self._clear_previews()

        # Reset preview area with placeholder
        placeholder = ttk.Frame(self.notebook)
        ttk.Label(
            placeholder,
            text="Load images and click Colorize to see results here.",
            anchor="center",
            font=("Segoe UI", 10),
            foreground="#999",
        ).pack(expand=True)
        self.notebook.add(placeholder, text="Results")

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.status_var.set("Ready")

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # Output previews
    # ------------------------------------------------------------------

    def _clear_previews(self) -> None:
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        self.preview_photo_refs.clear()

    def _preview_size(self) -> tuple[int, int]:
        width = max(500, self.notebook.winfo_width() - 40)
        height = max(350, self.notebook.winfo_height() - 60)
        return width, height

    @staticmethod
    def _resample_lanczos() -> int:
        if hasattr(Image, "Resampling"):
            return Image.Resampling.LANCZOS
        return Image.LANCZOS

    def _add_preview_tab(self, title: str, image_path: Path) -> None:
        img = Image.open(image_path).convert("RGB")
        max_w, max_h = self._preview_size()
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img_preview = img.resize(new_size, resample=self._resample_lanczos())
        photo = ImageTk.PhotoImage(img_preview)

        frame = ttk.Frame(self.notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        img_label = ttk.Label(frame, image=photo, anchor="center")
        img_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        ttk.Label(frame, text=str(image_path), style="Subtle.TLabel").grid(
            row=1, column=0, sticky="w", padx=6, pady=(0, 6)
        )

        self.notebook.add(frame, text=title)
        self.preview_photo_refs[title] = photo

    def _load_outputs(self) -> None:
        outdir = Path(self.outdir_var.get().strip())

        self._clear_previews()
        found_any = False
        for title, filename in PREVIEW_FILES:
            path = outdir / filename
            if path.exists():
                self._add_preview_tab(title, path)
                found_any = True

        if found_any:
            # Select first tab (Final Colorized)
            self.notebook.select(0)
        else:
            self._append_log(f"[WARN] No output images found in: {outdir}")

        # Show summary in log
        summary_txt = outdir / "summary.txt"
        if summary_txt.exists():
            self._append_log("")
            self._append_log("--- Summary ---")
            for line in summary_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("mae_"):
                    self._append_log(f"  {line}")

    def _open_output_folder(self) -> None:
        outdir = Path(self.outdir_var.get().strip())
        steps_dir = outdir / "steps"
        target = steps_dir if steps_dir.exists() else outdir
        if not target.exists():
            messagebox.showerror("Missing folder", f"Directory not found:\n{target}")
            return
        try:
            os.startfile(str(target))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not open folder:\n{target}\n\n{exc}")


def main() -> int:
    app = PhotoColorizerGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
