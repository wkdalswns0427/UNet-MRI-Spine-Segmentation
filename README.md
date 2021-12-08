# Spine Segmentation with Pytorch Modified UNet
MinJun Chang, 2021.12.08

* Data Config

input Image  : .dcm 

input label : .mat (num, 'label_separated',[W, H, C])

data files are not in this repository, access google drive below

https://drive.google.com/drive/folders/1yYqUdpa4-3hOSQmTGSORrrrz4Y71--tK?usp=sharing

* if using COLAB Environment must run the following

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

images will be given as DICOM files

more to vbe continued


