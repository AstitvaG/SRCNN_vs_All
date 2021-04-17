import numpy as np
from PIL import Image
from os import listdir
from tqdm import tqdm

# Set train and val HR and LR paths
train_hr_path = 'data/train_hr/'
train_lr_path = 'data/train_lr/'
val_hr_path = 'data/val_hr/'
val_lr_path = 'data/val_lr/'

# numberOfImagesTrainHR = len(listdir(train_hr_path))


# for i in tqdm(range(numberOfImagesTrainHR)):
#     img_name = listdir(train_hr_path)[i]
#     hr = Image.open(train_hr_path + img_name).convert('RGB')
#     hr = hr.resize((hr.width//3, hr.height//3), resample=Image.BICUBIC)
#     hr.save(train_lr_path + img_name)

numberOfImagesValHR = len(listdir(val_hr_path))

for i in tqdm(range(numberOfImagesValHR)):
    img_name = listdir(val_hr_path)[i]
    hr = Image.open(val_hr_path + img_name).convert('RGB')
    hr = hr.resize((hr.width//3, hr.height//3), resample=Image.BICUBIC)
    hr.save(val_lr_path + img_name)