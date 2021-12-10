# Spine Segmentation with Pytorch Modified UNet
MinJun Chang, 2021.12.08

[ Data Config ]

input Image  : .dcm 

input label : .mat (num, 'label_separated',[W, H, C])
*****************************************************************

training data files are not in this repository, access google drive below

https://drive.google.com/drive/folders/1u7upLae6Y_rsD3WlLH5w9wmyS72apE3G?usp=sharing

Test data files also not in this repo

https://drive.google.com/drive/folders/1A0BGxDaAugJXfJSYWuOjd8BCV_ZB4RqG?usp=sharing

https://drive.google.com/drive/folders/15jIHjNB47eLC0l0UX1m-m-swh_AgDEur?usp=sharing
*****************************************************************

* if using COLAB Environment must run the following and restart kernel to use updated Numpy

!apt install libjpeg-dev

!apt upgrade

!pip install albumentations==0.4.6

!pip install pydicom

!pip install gdcm

!pip install pylibjpeg

!pip install --upgrade numpy

!pip install pylibjpeg-libjpeg

* install torch cuda

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


## Project Description
segemntation of 6 vertebrae with 7 masks

image input as .dcm MRI images, masks in 7 channel matlab dictionary

training with extra depth UNet [32, 64, 128, 256, 512]

100 images for training, 20 for validation --> Test results in [R, C, D] shape uint8 .npy file


