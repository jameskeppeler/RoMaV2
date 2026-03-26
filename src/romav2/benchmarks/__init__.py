from .scannet1500 import ScanNet1500 as ScanNet1500
from .mega1500 import Mega1500 as Mega1500

# Optional dependency: wxbs_benchmark is only required for WxBS.
try:
    from .wxbs import WxBSBenchmark as WxBSBenchmark
except ModuleNotFoundError as exc:
    if exc.name != "wxbs_benchmark":
        raise
    WxBSBenchmark = None  # type: ignore[assignment]

# Optional dependency: SatAst benchmark requires eval extras (opencv/matplotlib).
try:
    from .satast import SatAst as SatAst
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2", "matplotlib", "matplotlib.pyplot"}:
        raise
    SatAst = None  # type: ignore[assignment]
