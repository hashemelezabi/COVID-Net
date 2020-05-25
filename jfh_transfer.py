import tensorflow as tf

from data import BalanceCovidDataset

BATCH_SIZE = 32
IMG_SIZE = 480
COVID_PERCENT = 0.3
COVID_WEIGHT = 4.
TOP_PERCENT = 0.08

generator = BalanceCovidDataset(data_dir='data',
                                csv_file='train_COVIDx2.txt',
                                batch_size=BATCH_SIZE,
                                input_shape=(IMG_SIZE, IMG_SIZE),
                                covid_percent=COVID_PERCENT,
                                class_weights=[1., 1., COVID_WEIGHT],
                                top_percent=TOP_PERCENT)

batch_x, batch_y, weights = next(generator)
print(batch_x)
print(batch_x.shape)
