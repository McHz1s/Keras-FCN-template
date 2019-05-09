from config import *


class DataSet(object):
    """
    The base class for dataset classes.
    Create a new class that adds functions specific to the dataset
    Aim to provide numpy array for keras model
    """
    def __init__(self, config, mode):
        assert mode in ["training", "prediction"]
        self.mode = mode
        self.config = config
        self.step_per_epoch = None
        self.validation_steps = 100

    def train_generator(self, path):
        assert self.mode == 'training'
        from keras.preprocessing.image import ImageDataGenerator
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
        image_generator = image_datagen.flow_from_directory(directory=path,
                                                            target_size=[self.config.image_height,
                                                                         self.config.image_width],
                                                            color_mode=self.config.color_mode,
                                                            classes=self.config.class_num,
                                                            batch_size=self.config.batch_size)
        mask_generator = mask_datagen.flow_from_directory(directory=path,
                                                          target_size=[self.config.image_height,
                                                                       self.config.image_width],
                                                          color_mode=self.config.color_mode,
                                                          classes=self.config.class_num,
                                                          batch_size=self.config.batch_size)
        train_generator = zip(image_generator, mask_generator)
        return train_generator

    def valid_generator(self, path):
        assert self.mode == 'training'
        from keras.preprocessing.image import ImageDataGenerator
        image_datagen = ImageDataGenerator()
        image_generator = image_datagen.flow_from_directory(directory=path,
                                                            target_size=[self.config.image_height,
                                                                         self.config.image_width],
                                                            color_mode=self.config.color_mode,
                                                            classes=self.config.class_num,
                                                            batch_size=self.config.batch_size)
        return image_generator

    def predict_generate(self, path):
        return self.valid_generator(path)

    def saveplot(self, results, save_path):
        from cv2 import imwrite
        for i, item in enumerate(results):
            imwrite(os.path.join(save_path, "%d_predict.png" % i), item)



