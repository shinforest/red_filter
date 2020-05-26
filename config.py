import os

# GPU flag, -1 means CPU-only, positive means GPU id
GPU = -1

# Network config
#   Input width and Height
Height = 64
Width = 64
Channel = 3

## Input variable size
##    If you use variable input size, below Variable_input is True.
##    Image is resized satisfying that longer side(width or height) is equal to Max_side.
##    If Variable_input is True, above "Height" and "Width" is ignored.
Variable_input = False
Max_side = 1024

Test_Max_side = 1024 #1536


# input data shape
# channels_last -> [mb, c, h, w] , channels_first -> [mb, h, w, c]
Input_type = 'channels_first'


## Class_label is corresponding to directory "name" including each labeled images.
Class_label = ['sango', 'others',]
Class_num = len(Class_label)

# Directory paths for training
Train_dirs = [
    './train/sango/',
    './train/others/'
]

Test_dirs = [
    './test/sango/',
    './test/others/'
]

File_extensions = ['.jpg']


## Training config
Iteration = 100
Minibatch = 8
Learning_rate = 0.001

Save_train_step = 20
Save_iteration_disp = True

## Test config
##   if Minibatch is None, all data used for test
Test_Minibatch = None


## Data augmentation
Horizontal_flip = True
Vertical_flip = True
Rotate_ccw90 = False

## Save config
Save_dir = 'models'
Model_name = 'model.h5'
Save_model_path = os.path.join(Save_dir, Model_name)

## Other config
##  Randon_seed is used for seed of dataset shuffle in data_loader.py
Random_seed = 0

## Check
variety = ['channels_first', 'channels_last']
if not Input_type in variety:
    raise Exception("unvalid Input_type")

if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
