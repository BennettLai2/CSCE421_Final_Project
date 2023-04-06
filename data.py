import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

def load_data(x_path):

    return x


def split_data(x, y, split=0.8):
    train_x, train_y, test_x, test_y = train_test_split(x, y, train_size=split)
    return train_x, train_y, test_x, test_y


def preprocess_x(df):
    # Your code here
    return data
