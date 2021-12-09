import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
    load_checkpoint,
    get_test_loader, get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_result_as_numpy,
)

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TEST_IMG_DIR = "Test/img/"


def test_fn():
    test_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                # mean=[0.0, 0.0, 0.0],
                # std=[1.0, 1.0, 1.0],
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=1, out_channels=7).eval().to(DEVICE)  # change out_channel for multi classes

    test_loader = get_test_loader(
        TEST_IMG_DIR,
        BATCH_SIZE,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint.pth.tar"), model)

    save_result_as_numpy(
        test_loader, model, folder="numpy_results", device=DEVICE
    )

if __name__ == "__main__":
    test_fn()