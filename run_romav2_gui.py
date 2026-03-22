from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


def _run_tk() -> int:
    from run_romav2_gui_legacy_tk import main as tk_main

    return int(tk_main())


def _try_run_qt() -> int | None:
    try:
        from run_romav2_gui_qt import main as qt_main
    except SystemExit as exc:
        # run_romav2_gui_qt raises SystemExit with install instructions when PySide6 is missing.
        print(f"[INFO] Qt GUI unavailable: {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[INFO] Qt GUI unavailable: {exc}", file=sys.stderr)
        return None
    return int(qt_main())


def _try_launch_qt_external() -> int | None:
    repo_root = Path(__file__).resolve().parent
    qt_script = repo_root / "run_romav2_gui_qt.py"
    if not qt_script.exists():
        return None

    # First try local venv python directly.
    venv_candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ]
    for py in venv_candidates:
        if not py.exists():
            continue
        cmd = [str(py), str(qt_script)]
        print(f"[INFO] Launching Qt GUI via venv python: {py}", file=sys.stderr)
        return subprocess.call(cmd, cwd=str(repo_root))

    # Then try uv run.
    uv_exe = shutil.which("uv")
    if uv_exe:
        cmd = [uv_exe, "run", "python", str(qt_script)]
        print("[INFO] Launching Qt GUI via uv run.", file=sys.stderr)
        return subprocess.call(cmd, cwd=str(repo_root))
    return None


def _parse_args() -> argparse.Namespace:
    default_backend = os.environ.get("ROMAV2_GUI_BACKEND", "auto").lower()
    if default_backend not in {"auto", "qt", "tk"}:
        default_backend = "auto"

    parser = argparse.ArgumentParser(
        description=(
            "RoMaV2 GUI launcher. Defaults to Qt (with in-app browser Step 0) and "
            "falls back to Tkinter if Qt deps are unavailable."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "qt", "tk"),
        default=default_backend,
        help="GUI backend selection. Default: auto.",
    )
    parser.add_argument("--qt", action="store_true", help="Force the Qt GUI backend.")
    parser.add_argument("--tk", action="store_true", help="Force the Tkinter GUI backend.")
    return parser.parse_args()


def _resolve_backend(args: argparse.Namespace) -> str:
    if args.qt and args.tk:
        raise SystemExit("Cannot use both --qt and --tk.")
    if args.qt:
        return "qt"
    if args.tk:
        return "tk"
    return args.backend


def _backend_runner(backend: str) -> Callable[[], int]:
    if backend == "tk":
        return _run_tk
    if backend == "qt":
        def _qt() -> int:
            qt_rc = _try_run_qt()
            if qt_rc is not None:
                return qt_rc
            external_rc = _try_launch_qt_external()
            if external_rc is not None:
                return external_rc
            return 1

        return _qt
    if backend == "auto":
        def _auto() -> int:
            qt_rc = _try_run_qt()
            if qt_rc is not None:
                return qt_rc
            external_rc = _try_launch_qt_external()
            if external_rc is not None:
                return external_rc
            print(
                "[INFO] Falling back to Tkinter GUI. Install Qt GUI deps with: uv sync --extra gui",
                file=sys.stderr,
            )
            return _run_tk()

        return _auto
    raise SystemExit(f"Unsupported backend: {backend}")


def main() -> int:
    args = _parse_args()
    backend = _resolve_backend(args)
    runner = _backend_runner(backend)
    return int(runner())


if __name__ == "__main__":
    raise SystemExit(main())
