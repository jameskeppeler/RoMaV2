from romav2 import RoMaV2
from romav2.benchmarks import Mega1500
import logging
from pathlib import Path
import pytest

wandb = pytest.importorskip("wandb")

DATA_ROOT = Path("data/megadepth")
SCENE_FILES = [
    "0015_0.1_0.3.npz",
    "0015_0.3_0.5.npz",
    "0022_0.1_0.3.npz",
    "0022_0.3_0.5.npz",
    "0022_0.5_0.7.npz",
]

logger = logging.getLogger(__name__)


def test_mega1500():
    missing = [name for name in SCENE_FILES if not (DATA_ROOT / name).exists()]
    if missing:
        pytest.skip("Mega1500 dataset not available in data/megadepth.")

    model = RoMaV2()
    model.apply_setting("mega1500")
    mega1500 = Mega1500()
    run = wandb.init(project="roma-v2", name="mega1500", mode="disabled")
    res = mega1500.benchmark(model)
    wandb.log(res)
    logger.info(f"Mega1500 results: {res}")
    run.finish()


if __name__ == "__main__":
    test_mega1500()
