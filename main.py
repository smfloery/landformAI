import argparse
import yaml
import os
from model import GMM

parser = argparse.ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    model_config = yaml.safe_load(config_file)

gmm = GMM(model_config)
gmm.train_loop()
gmm.plot_train_val_loss()
