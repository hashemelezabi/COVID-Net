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

def generate_dataset():
	generator = BalanceCovidDataset(data_dir='data',
									csv_file='train_COVIDx2.txt',
									batch_size=BATCH_SIZE,
									input_shape=(IMG_SIZE, IMG_SIZE),
									covid_percent=COVID_PERCENT,
									class_weights=[1., 1., COVID_WEIGHT],
									top_percent=TOP_PERCENT)

	# batch_x, batch_y, weights = next(generator)
	# return (batch_x, batch_y, weights)
	return generator

def generate_test_dataset():
	generator = BalanceCovidDataset(data_dir='data',
									csv_file='test_COVIDx2.txt',
									batch_size=BATCH_SIZE,
									input_shape=(IMG_SIZE, IMG_SIZE),
									covid_percent=COVID_PERCENT,
									class_weights=[1., 1., COVID_WEIGHT],
									top_percent=TOP_PERCENT,
									is_training=False)

	# batch_x, batch_y, weights = next(generator)
	# return (batch_x, batch_y, weights)
	return generator

def load_base_model():
	IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
	base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
	base_model.trainable = False
	return base_model

def modify_base_model(base_model, learning_rate):
	global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
	dense1_layer = tf.keras.layers.Dense(100)
	dense2_layer = tf.keras.layers.Dense(3)
	prediction_layer = tf.keras.layers.Softmax()
	model = tf.keras.Sequential([
			  base_model,
			  global_average_layer,
			  dense1_layer,
			  dense2_layer,
			  prediction_layer
			])
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
		loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, name='categorical_crossentropy'),
		metrics=['accuracy'])
	return model

def evaluate_model(model, train_gen, test_gen):
	initial_epochs = 10

	loss0,accuracy0 = model.evaluate(test_gen)
	print("initial loss: {:.2f}".format(loss0))
	print("initial accuracy: {:.2f}".format(accuracy0))

	history = model.fit(train_gen,
                 epochs=initial_epochs,
                 validation_data=test_gen)

if __name__ == '__main__':
	# batch_x, batch_y, weights = generate_dataset()
	model = load_base_model()
	# print(model.summary())

	# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
	# feature_batch_average = global_average_layer(batch_x)
	# dense = tf.keras.layers.Dense(100)
	# dense_batch = dense(feature_batch_average)
	# prediction_layer = tf.keras.layers.Softmax()
	# prediction_batch = prediction_layer(dense_batch)
	# print(prediction_batch.shape)

	model = modify_base_model(model, learning_rate=.0001)
	print(model.summary())

	train_generator = generate_dataset()
	test_generator = generate_test_dataset()
	evaluate_model(model, train_generator, test_generator)




