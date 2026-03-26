from romav2 import RoMaV2
from romav2.benchmarks import ScanNet1500
import logging
from pathlib import Path
import pytest

wandb = pytest.importorskip("wandb")

DATA_ROOT = Path("data/scannet/scans")
TEST_SPLIT_FILE = DATA_ROOT / "test.npz"


logger = logging.getLogger(__name__)


def test_scannet1500():
    if not TEST_SPLIT_FILE.exists():
        pytest.skip("ScanNet1500 dataset not available in data/scannet/scans.")

    model = RoMaV2()
    model.apply_setting("scannet1500")
    scannet1500 = ScanNet1500()
    run = wandb.init(project="roma-v2", name="scannet1500", mode="disabled")
    res = scannet1500.benchmark(model)
    wandb.log(res)
    logger.info(f"ScanNet1500 results: {res}")
    run.finish()


if __name__ == "__main__":
    test_scannet1500()
