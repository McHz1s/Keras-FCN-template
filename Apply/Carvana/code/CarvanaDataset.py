from CarvanaConfig import *


class CarvanaDataset(DataSet):
    """
    Dataset for Cavana
    """

    def __init__(self, config, mode):
        super(CarvanaDataset, self).__init__(config=config, mode=mode)
        if mode == "training":
            df_train = pd.read_csv(os.path.expanduser('~/图片/cavaran/train_masks.csv'))
            ids_train = df_train['img'].map(lambda s: s.split('.')[0])
            self.ids_train, self.ids_valid = train_test_split(ids_train, test_size=0.1)
            self.step_per_epoch = np.ceil(float(len(self.ids_train)) / float(config.batch_size))
            self.validation_steps = np.ceil(float(len(self.ids_valid)) / float(config.batch_size))
        else:
            self.df_test = pd.read_csv(os.path.expanduser('~/图片/cavaran/sample_submission.csv'))
            self.ids_test = self.df_test['img'].map(lambda s: s.split('.')[0])
            self.names = []
            self.rles = []
            self.save_path = config.predict_image_dir

    def train_generator(self, path=None):
        df = self.ids_train
        while True:
            shuffle_indices = np.arange(len(df))
            shuffle_indices = np.random.permutation(shuffle_indices)

            for start in range(0, len(df), self.config.batch_size):
                x_batch = []
                y_batch = []

                end = min(start + self.config.batch_size, len(df))
                ids_train_batch = df.iloc[shuffle_indices[start:end]]

                for _id in ids_train_batch.values:
                    img = imread(os.path.expanduser(
                        '~/图片/cavaran/train_hq/{}.jpg'.format(_id)))

                    img = resize(img, (self.config.image_width, self.config.image_height), interpolation=INTER_AREA)

                    mask = imread(os.path.expanduser(
                        '~/图片/cavaran/train_masks/{}_mask.png'.format(_id)),
                        IMREAD_GRAYSCALE)
                    mask = resize(mask, (self.config.image_width, self.config.image_height), interpolation=INTER_AREA)
                    mask = np.expand_dims(mask, axis=-1)
                    assert mask.ndim == 3

                    # === You can add data augmentations here. === #
                    if np.random.random() < 0.5:
                        img, mask = img[:, ::-1, :], mask[..., ::-1, :]  # random horizontal flip

                    x_batch.append(img)
                    y_batch.append(mask)

                x_batch = np.array(x_batch, np.float32) / 255.
                y_batch = np.array(y_batch, np.float32) / 255.
                yield x_batch, y_batch

    def valid_generator(self, path=None):
        df = self.ids_valid
        while True:
            for start in range(0, len(df), self.config.batch_size):
                x_batch = []
                y_batch = []

                end = min(start + self.config.batch_size, len(df))
                ids_train_batch = df.iloc[start:end]

                for _id in ids_train_batch.values:
                    img = imread(os.path.expanduser(
                        '~/图片/cavaran/train_hq/{}.jpg'.format(_id)))
                    img = resize(img, (self.config.image_width, self.config.image_height), interpolation=INTER_AREA)

                    mask = imread(os.path.expanduser(
                        '~/图片/cavaran/train_masks/{}_mask.png'.format(_id)),
                        IMREAD_GRAYSCALE)
                    mask = resize(mask, (self.config.image_width, self.config.image_height), interpolation=INTER_AREA)
                    mask = np.expand_dims(mask, axis=-1)
                    assert mask.ndim == 3

                    x_batch.append(img)
                    y_batch.append(mask)

                x_batch = np.array(x_batch, np.float32) / 255.
                y_batch = np.array(y_batch, np.float32) / 255.

                yield x_batch, y_batch

    def predict_generate(self, path=None):
        for id in self.ids_test:
            self.names.append('{}.jpg'.format(id))
            img = imread(os.path.expanduser(
                '~/图片/cavaran/test_hq/{}.jpg'.format(id)))
            img = resize(img, (self.config.image_width, self.config.image_height), interpolation=INTER_AREA)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            yield img

    def saveplot(self, results, savepath=None):
        for i, result in enumerate(results):
            mask = resize(result, (self.config.origin_width, self.config.origin_height))
            mask[mask > 0.5] = 1
            mask[mask != 1] = 0
            mask = np.stack([mask, mask, mask], axis=-1)
            img = imread(os.path.expanduser(
                '~/图片/cavaran/test_hq/{}'.format(self.df_test['img'][i])))
            save_img = np.multiply(mask, img)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            imwrite(self.save_path + '{}'.format(self.df_test['img'][i]), save_img)

        def saveresult(self, results):
            for res in results:
                prob = resize(res, (self.config.origin_widthid, self.config.origin_heightg))
                mask = prob > 0.5
                inds = mask.flatten()
                runs = np.where(inds[1:] != inds[:-1])[0] + 2
                runs[1::2] = runs[1::2] - runs[:-1:2]
                rle = ' '.join([str(r) for r in runs])
                self.rles.append(rle)
            df = pd.DataFrame({'img': self.names, 'rle_mask': self.rles})
            df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
