import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet , extralayer_UNet
from loss_function import BCEDiceIoUWithLogitsLoss2d
from train_function import (train_fn, train_fn_no_scale)
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters and directories
LEARNING_RATE = 1e-3
step_lr_denominator = 10
gamma = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 160
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/img/"
TRAIN_MASK_DIR = "data/train/label/"
VAL_IMG_DIR = "data/val/img/"
VAL_MASK_DIR = "data/val/label/"


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=1, out_channels=7).to(DEVICE)
    loss_fn = BCEDiceIoUWithLogitsLoss2d()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(NUM_EPOCHS/step_lr_denominator), gamma=gamma)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        loss_fn.train()
        train_fn_no_scale(train_loader, model, optimizer, loss_fn)

        print("Current Epoch : ", epoch+1, "/", NUM_EPOCHS)

        #save
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        model.eval()
        loss_fn.eval()
        scheduler.step()
    print("--------------------* Training Complete! *--------------------")

if __name__ == "__main__":
    main()