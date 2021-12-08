import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TEST_IMG_DIR = "Test/img/"
TEST_MASK_DIR = "Test/label/"

def test_fn():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
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


    model = UNet(in_channels=1, out_channels=1).to(DEVICE)  # change out_channel for multi classes
    loss_fn = nn.BCEWithLogitsLoss()  # cross entropy loss for multiple classes

    test_loader = get_loaders(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )