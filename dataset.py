import os
from PIL import Image
import pydicom as dcm
import pylibjpeg
import numpy as np
from torch.utils.data import Dataset



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
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".dcm","_mat"))
        rawimage = dcm.dcmread(img_path).pixel_array
        image_uns = rawimage.astype(float)
        image = (np.maximum(image_uns,0)/image_uns.max())*255
        image = np.uint8(image)
