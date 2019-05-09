from CarvanaDataset import *


if __name__ == '__main__':
    config = CarvanaConfig()
    data = CarvanaDataset(config=config, mode="prediction")
    model = FCNModel(config=config, mode="prediction", data=data, network=network)
    model.predict()
