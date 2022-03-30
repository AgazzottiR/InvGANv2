import time
from options.train_options import TrainOptions
from models import create_model
from util.visualize import Visualizer

if __name__ == "__main__":
    opt = TrainOptions().parse()
    
