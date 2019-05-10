# Keras-FCN-template

A FCN template(container) to quickly train/predict a specific FCN from a dataset.

# File introdutction

## Director tree

Architecture inspired by <https://github.com/matterport/Mask_RCNN.>

```None
├── Apply
│   └── Carvana
│       ├── code
│       │   ├── CarvanaConfig.py
│       │   ├── CarvanaDataset.py
│       │   ├── mobilenetv2_unet.py
│       │   ├── prediciton.py
│       │   └── train.py
│       └── sample
└── BaseCode
    ├── config.py
    ├── dataset.py
    ├── losses.py
    └── model.py
```

## 1. BaseCode:basic classes and functions

* config.py: creat a base class to set hyperparameter and director.
* dataset.py: create a class to conduct data preprocessing and provide it to model.
* losses.py: som basic loss function, such as dice loss.
* model.py: create a class(FCNModel) getting data, config and lossfunction from three above python file. The FCNmodel dosen't implement a concrete fully convolution network, but obtain it when we rewrite config and feed a FCN to it

## 2. Apply: use BaseCode to accomplish a specific task

In this part, we provide Carvana Image Masking Challenge(<https://www.kaggle.com/c/carvana-image-masking-challenge)>
 as a sample.

* mobilenetv2_unet: FCN we adopted(<https://github.com/JonathanCMitchell/mobilenet_v2_keras)>
* CarvanaConfig.py: write a class CarvanaConfig extends class config from Config.py to set hyperparameter fitting our task. And import mobilenetv2_unet as our network.
* CarvanaDataset.py: write a class CarvanaDataset extends class dataset from dataset.py to generate data for our model.
* prediciton.py: run model.predict and get the result.
* train.py: run model.train and get the trained network.

# Requirements

Python 3.5, TensorFlow 1.4.0, Keras 2.1.6 and other common packages listed in my_py_envn.txt.

# Start the sample

1. Install required package Requirements.
2. Download dataset from kaggle and set the director(as Carvanadataset)
3. Copy your FCN.py to Apply/Carvana and import it as network in Carvanaconfig.py.(Optional)
4. Run Apply/Carvana/train.py.
5. Run Apply/Carvana/predict.py.
