import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from data import BalanceCovidDataset
from classifier import Classifier
from torch.autograd import Variable
import onnx 
from onnx_tf.backend import prepare

BATCH_SIZE = 32
IMG_SIZE = 480
COVID_PERCENT = 0.3
COVID_WEIGHT = 4.
TOP_PERCENT = 0.08
CHANNELS = 3

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


def load_model():
	trained_model = Classifier()
	ckpt = torch.load('pre_train.pth', map_location=device) # .pth file
	trained_model.load_state_dict(ckpt)
	dummy_input = Variable(torch.randn(1, IMG_SIZE, IMG_SIZE, CHANNELS))
	new_model_name = "jfh_model.onnx"
	torch.onnx.export(trained_model, dummy_input, new_model_name)

	model = onnx.load(new_model_name)
	tf_rep = prepare(model)
	tf_rep.export_graph(new_model_name.strip('.onnx') + '.pb')

	return tf_rep

if __name__ == '__main__':
	model = load_model()
	print(model.summary())
