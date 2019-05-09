import numpy as np
import pandas as pd
import os
# model import
from keras.layers import Input
from keras.models import Model as Modelib
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD
import datetime
import re
from losses import *


class Config(object):
    """
    Base configuration class
    Create a sub-class that inherits from this one and override properties
    that need to be changed.
    """

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.batch_size = self.image_per_gpu * self.gpu_count

        # Input image size
        self.image_shape = np.array(
            [self.image_height, self.image_width, 3])
        self.MASK_SHAPE = np.array(
            [self.image_height, self.image_width, 1])

    # ModelName part
    name = 'resnet50'
    task_name = 'MyTask'
    # Input part
    image_width = 512
    image_height = 512
    origin_width = 1024
    origin_height = 1024
    image_per_gpu = 16
    gpu_count = 1
    color_mode = 'rgb'
    # Output part
    class_num = 1

    # Train part
    model_pre_trained = False
    epoch = 1
    verbose = 1
    # Compile part
    optimizer = 'ADAM'
    monitor = 'val_loss'
    learning_rate_start = 1e-4
    learning_momentum = 0.9
    clipnorm = 5.0
    # Loss part
    loss_function = bce_dice_loss
    loss_name = 'bce_dice_loss'
    metrics_function = dice_coef
    metrics_name = 'dice_coef'
    # Direct part
    project_root_dir = '../'
    predict_image_dir = project_root_dir + 'pre_image/'
    model_dir = 'Model/'
    root_dir = '../'

