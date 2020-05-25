import tensorflow as tf
import os
import numpy as np
# import matplotlib.pyplot as plt
# import torch
from data import BalanceCovidDataset

BATCH_SIZE = 32
IMG_SIZE = 480
COVID_PERCENT = 0.3
COVID_WEIGHT = 4.
TOP_PERCENT = 0.08
CHANNELS = 3

generator = BalanceCovidDataset(data_dir='../../COVID-NET/data',
								csv_file='train_COVIDx2.txt',
								batch_size=BATCH_SIZE,
								input_shape=(IMG_SIZE, IMG_SIZE),
								covid_percent=COVID_PERCENT,
								class_weights=[1., 1., COVID_WEIGHT],
								top_percent=TOP_PERCENT)

batch_x, batch_y, weights = next(generator)
print(batch_x)
print(batch_x.shape)


def load_model():
	IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
	base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
	base_model.trainable = False

	return base_model

if __name__ == '__main__':
	model = load_model()
	print(model.summary())