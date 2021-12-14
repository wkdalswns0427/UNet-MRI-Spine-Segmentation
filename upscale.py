import os
import cv2 as cv
import pydicom as dcm
import numpy as np


def upscale_results(data_dir, target_dir, save_dir):
    names = os.listdir(data_dir)
    for name in names:
        data_path = os.path.join(data_dir, name)
        target_path = os.path.join(target_dir, name.replace(".dcm",".npy"))
        save_path = os.path.join(save_dir, name.replace(".dcm", ".npy"))

        rawimage = dcm.dcmread(data_path).pixel_array
        image_uns = rawimage.astype(float)
        image = (np.maximum(image_uns, 0) / image_uns.max()) * 255
        image = np.stack([image]).transpose((1, 2, 0))
        [R, C, D] = image.shape

        target = np.load(target_path)
        processed = cv.resize(target,(C,R),cv.INTER_NEAREST)

        np.save(save_path,processed)
        print('saved')

    print("--------------------* Process Over *--------------------")


if __name__ == "__main__":
    upscale_results('FinalTest/img/','numpy_results','Final/jongang')