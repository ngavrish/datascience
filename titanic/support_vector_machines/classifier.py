# coding=utf8

import pandas as pd
from sklearn import svm

pd.set_option('display.width', 256)

train_data = pd.read_csv("./datasets/train.csv")
test_data = pd.read_csv("./datasets/test.csv")

all_data = pd.concat([train_data, test_data])


