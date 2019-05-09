from CarvanaDataset import *


if __name__ == '__main__':
    config = CarvanaConfig()
    data = CarvanaDataset(config=config, mode='training')
    model = FCNModel(config=config, mode="training", data=data, network=network)
    model.train()

