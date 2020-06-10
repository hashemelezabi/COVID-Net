from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os, argparse
import cv2
from tensorflow.python.keras.backend import set_session

from data import process_image_file

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

def eval(sess, model, testfile, testfolder, input_size):
    y_test = []
    pred = []
    pred_tensor = model.layers[-1].output
    in_tensor = model.layers[0].input
    print(pred_tensor)
    print(pred_tensor.shape)
    print(in_tensor.shape)
    for i in range(len(testfile)):
        line = testfile[i].split()
        x = process_image_file(os.path.join(testfolder, line[1]), 0.08, input_size)
        x = x.astype('float32') / 255.0
        y_test.append(mapping[line[2]])
        prediction = np.array(sess.run(pred_tensor, feed_dict={in_tensor: np.expand_dims(x, axis=0)}))
        print(prediction)
        pred.append(prediction.argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)
    #class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))

if __name__ == '__main__':
    
    file = open('test_COVIDx2.txt', 'r')
    testfile = file.readlines()

    sess = tf.Session()
    set_session(sess)
    model = tf.keras.models.load_model('saved_models/chexnet_transfer_model')

    eval(sess, model, testfile, 'data/test', 480)
