import torch
from romav2 import RoMaV2
from romav2.device import device
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("ROMAV2_RUN_PERF_TESTS", "0") != "1",
    reason="Performance benchmark disabled by default. Set ROMAV2_RUN_PERF_TESTS=1 to run.",
)


def test_unidirectional():
    if not torch.cuda.is_available():
        pytest.skip("Performance benchmark requires CUDA for stable timing thresholds.")

    model = RoMaV2()
    model.apply_setting("base")
    B = 8
    T = 20
    img_A = (
        Image.open("assets/toronto_A.jpg")
        .resize((model.H_lr, model.W_lr))
        .convert("RGB")
    )
    img_B = (
        Image.open("assets/toronto_B.jpg")
        .resize((model.H_lr, model.W_lr))
        .convert("RGB")
    )
    img_A = (
        torch.from_numpy(np.array(img_A))
        .permute(2, 0, 1)
        .to(device)[None]
        .expand(B, -1, -1, -1)
        / 255.0
    )
    img_B = (
        torch.from_numpy(np.array(img_B))
        .permute(2, 0, 1)
        .to(device)[None]
        .expand(B, -1, -1, -1)
        / 255.0
    )
    model.match(img_A, img_B)
    t0 = time.perf_counter()
    for i in tqdm(range(T)):
        model.match(img_A, img_B)
    t1 = time.perf_counter()
    print(f"Time taken: {t1 - t0} seconds")
    fps = T * B / (t1 - t0)
    print(f"FPS: {fps}")
    assert fps > 33, "FPS should be greater than 33"

def test_bidirectional():
    if not torch.cuda.is_available():
        pytest.skip("Performance benchmark requires CUDA for stable timing thresholds.")

    model = RoMaV2()
    model.apply_setting("base")
    model.bidirectional = True
    B = 8
    T = 20
    img_A = (
        Image.open("assets/toronto_A.jpg")
        .resize((model.H_lr, model.W_lr))
        .convert("RGB")
    )
    img_B = (
        Image.open("assets/toronto_B.jpg")
        .resize((model.H_lr, model.W_lr))
        .convert("RGB")
    )
    img_A = (
        torch.from_numpy(np.array(img_A))
        .permute(2, 0, 1)
        .to(device)[None]
        .expand(B, -1, -1, -1)
        / 255.0
    )
    img_B = (
        torch.from_numpy(np.array(img_B))
        .permute(2, 0, 1)
        .to(device)[None]
        .expand(B, -1, -1, -1)
        / 255.0
    )
    model.match(img_A, img_B)
    t0 = time.perf_counter()
    for i in tqdm(range(T)):
        model.match(img_A, img_B)
    t1 = time.perf_counter()
    print(f"Time taken: {t1 - t0} seconds")
    fps = T * B / (t1 - t0)
    print(f"FPS: {fps}")
    assert fps > 23, "FPS should be greater than 23"

if __name__ == "__main__":
    print("Testing unidirectional...")
    test_unidirectional()
    print("Testing bidirectional...")
    test_bidirectional()
