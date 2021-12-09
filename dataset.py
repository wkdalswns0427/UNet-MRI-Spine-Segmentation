import os
import pydicom as dcm
import numpy as np
from scipy import io
from torch.utils.data import Dataset
import torchvision
# image in dcm format
# label in matlab dictionary format


class SpineDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".dcm", ".mat"))
        rawimage = dcm.dcmread(img_path).pixel_array
        image_uns = rawimage.astype(float)
        image = (np.maximum(image_uns,0)/image_uns.max())*255
        image = np.uint8(image) # image
        image = np.stack([image]).transpose((1, 2, 0))
        raw_mask = io.loadmat(mask_path)
        mask = raw_mask['label_separated'].argmax(-1) # one hot encoding 해재

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
