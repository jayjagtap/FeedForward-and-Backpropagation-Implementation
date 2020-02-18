#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    #print(test_data[0][0] , test_data[1][0])
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 32, 10])
    # train the network using SGD
    validation_loss , validation_accuracy, training_loss, training_accuracy = model.SGD(
        training_data=train_data,
        epochs=75,
        mini_batch_size=64,
        eta=0.002,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    
    model.save("model0.json")
    model = network2.load("model0.json")
    predictions , prob, vector_base = predict(test_data, model)   
    
    plot_learning_curve(validation_loss, training_loss, validation_accuracy, training_accuracy)
    csvfile = open('predictions.csv','w', newline='')
    obj = csv.writer(csvfile)
    obj.writerows(predictions)
    csvfile.close()
    csvfile = open('raw.csv','w', newline='')
    obj1 = csv.writer(csvfile)
    obj1.writerows(prob)
    csvfile.close()
    
    test_accuracy = 0;
    for x, y in zip(vector_base, test_data[1]):
        if x == y:
            test_accuracy += 1
    print("The accuracy on test set is: " , test_accuracy)
        
    
def plot_learning_curve(validation_loss, training_loss, validation_accuracy, training_accuracy):
    validation_accuracy = [(x/10000)*100 for x in validation_accuracy]
    training_accuracy = [(x/3000)*100 for x in training_accuracy]
    
    fig = plt.figure(figsize=(6,6))
    plt.plot(validation_loss, color='r', label="val_loss")
    plt.plot(training_loss, color='b', label="train_loss")
    plt.legend()
    plt.show()
    fig.savefig('loss.png', bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,6))
    plt.plot(validation_accuracy, color='r', label="val_accuracy")
    plt.plot(training_accuracy, color='b', label="train_accuracy")
    plt.legend()
    plt.show()
    fig.savefig('accuracy.png', bbox_inches='tight')
        
    
def predict(test_data , model):
    predictions = []
    actual_prob = []
    vector_base = []
    for x in test_data[0]:
        raw = model.feedforward(x)
        actual_prob.append(raw)
        vector_base.append(np.argmax(raw))
        vector = network2.vectorized_result(np.argmax(raw))
        vector = [int(x[0]) for x in vector]
        predictions.append(vector)
               
    return (predictions, actual_prob, vector_base)

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
