# Network import
# import your FCN as "network"
from mobilenetv2_unet import mobilenet_v2_unet as network
# Base code import
import sys
from os.path import abspath

sys.path.append(abspath("../../../"))
from BaseCode.model import *
# Dataset import
from cv2 import resize, imread, imwrite, INTER_AREA, IMREAD_GRAYSCALE
from sklearn.model_selection import train_test_split


class CarvanaConfig(Config):
    task_name = 'Carvana'
    name = 'mobilenet_v2_unet'
    alpha = 1.0
    pre_trained_moedl_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' \
                             + str(alpha) + '_224_no_top' + '.h5'
    # Input part
    image_per_gpu = 2
    image_height = 1024
    image_width = 1024
    origin_width = 1918
    origin_height = 1280
    # Output part

    # Train part
    model_pre_trained = True
    verbose = 2

    # Compile part
    monitor = 'val_dice_coef'
    # Loss part

    # Direct part
    pre_trained_model_web_dir = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases' \
                                + pre_trained_moedl_name
    root_dir = '../../../'
