from dataset import *


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class FCNModel():
    def __init__(self, config, mode, data, network):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        A abstract model(container?) for a specific FCN
        Reference to https://github.com/matterport/Mask_RCNN
        """
        assert mode in ['training', 'prediction']
        self.mode = mode
        self.config = config
        self.data = data
        self.epoch = 0
        self.network = network
        # Training logs DIR
        self.log_dir = self.config.task_name + '/' + self.config.model_dir + '/' + self.config.name + '/logs'
        # Check point DIR
        self.checkpoint_path = self.config.task_name + '/' + self.config.model_dir + '/' + self.config.name + '/' + self.config.name + '.h5'
        self.keras_model = self.build(mode=mode)
        if mode == "training" and config.model_pre_trained:
            weights_path = self.get_imagenet_weights()
            self.load_weights(weights_path, by_name=True)
            print('Loading weights from ', weights_path)
        else:
            weights_path = self.find_last()
            self.load_weights(weights_path, by_name=True)
            print('Loading weights from ', weights_path)

    def train(self):
        """
        Train the model.
        """
        # Data generators
        cur_lr = self.config.learning_rate_start

        # Callbacks

        def lr_scheduler(epoch):
            if epoch < 20:
                lr = cur_lr
            elif epoch < 32:
                lr = 0.1 * cur_lr
            elif epoch < 45:
                lr = 0.01 * cur_lr
            else:
                lr = 0.005 * cur_lr
            print('\nlr: %f\n' % lr)
            return lr

        callbacks = [
            TensorBoard(log_dir=self.log_dir,
                        histogram_freq=0, write_graph=True, write_images=False),
            ModelCheckpoint(self.checkpoint_path, monitor=self.config.monitor,
                            mode='max', verbose=1, save_weights_only=True, save_best_only=True),
            LearningRateScheduler(lr_scheduler)
        ]
        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, self.config.learning_rate_start))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        print('Number of params', round(
            self.keras_model.count_params() / 10 ** 6, 1), 'million')

        self.compile()

        if os.name is 'nt':
            workers = 0
        else:
            workers = max(self.config.batch_size // 2, 2)

        self.keras_model.fit_generator(generator=self.data.train_generator(),
                                       steps_per_epoch=self.data.step_per_epoch,
                                       epochs=self.config.epoch,
                                       verbose=self.config.verbose,
                                       callbacks=callbacks,
                                       validation_data=self.data.valid_generator(),
                                       validation_steps=self.data.validation_steps,
                                       workers=workers)

    def build(self, mode):
        """
        Build the FCN model
        """
        # Inputs
        input_image = Input(shape=self.config.image_shape.tolist(), name="input_image")
        # Outputs
        output_masks = self.network(input_image, self.config.class_num)
        # Build model
        inputs = [input_image]
        outputs = [output_masks]
        model = Modelib(inputs=inputs,
                        outputs=outputs,
                        name=self.config.name)
        return model

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()
        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /Model/mobilenetv2_unet/logs/Carvana20190514T1741/mobilenetv2_unet_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/NETWORKNAME\w+(\d{4})\.h5"
            regex.replace("NETWORKNAME", self.config.name)
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        self.log_dir = self.config.project_root_dir + self.config.model_dir + \
                       self.config.name + '/logs/' + '{}{:%y%m%dT%H%M}'.format(self.config.task_name,now)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(self.config.name))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def get_imagenet_weights(self):
        """
        If exist, get pre-trained model. Else downloads ImageNet trained weights
        """
        from keras.utils.data_utils import get_file

        weights_path = get_file(self.config.pre_trained_moedl_name,
                                self.config.pre_trained_model_web_dir,
                                cache_subdir='models')
        return weights_path

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        network_dir = self.config.project_root_dir + self.config.model_dir + self.config.name + '/logs/'
        dir_names = next(os.walk(network_dir))[1]
        key = self.config.task_name
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = network_dir + dir_names[-1]
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(self.config.name), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        return dir_name + '/' + checkpoints[-1]

    def load_weights(self, filepath, by_name=False):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
        # Update the log directory
        if self.mode == 'training':
            self.set_log_dir(filepath)

    def compile(self):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        if self.config.optimizer == 'SGD':
            optimizer = SGD(lr=self.config.learning_rate_start,
                            momentum=self.config.learning_momentum, clipnorm=self.config.clipnorm)
        else:
            optimizer = Adam(self.config.learning_rate_start,
                             amsgrad=True, clipnorm=self.config.clipnorm)
        # Compile
        self.keras_model.compile(optimizer=optimizer,
                                 loss=bce_dice_loss,
                                 metrics=[dice_coef])

    def predict(self):
        results = self.keras_model.predict_generator(self.data.predict_generate(),
                                                     len(self.data.ids_test),
                                                     verbose=1)
        self.data.saveplot(results)
        # self.data.saveresult(results)
