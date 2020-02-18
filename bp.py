#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    activation = x
    activations = [x]
    us = []
    for b, w in zip(biases, weightsT):
        z = np.dot(w, activation) + b
        us.append(z)
        activation = sigmoid(z)
        activations.append(activation)
        
    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)*sigmoid_prime(us[-1])
    nabla_b[-1] = delta
   # print("nabla_b" , np.array(nabla_b).shape , nabla_b)
    #print("nabla_wT" , np.array(nabla_wT).shape , nabla_wT[0])
    #print("delta" , np.array(delta).shape , delta[0])
    #print("activations" , np.array(activations).shape , activations[0])
    #print(" ",len(nabla),len(nabla_b[0]) ," " ,len(activations[0]), len(nabla_wT[0]) , len(delta[0]))
    nabla_wT[-1] = np.dot(delta, activations[-2].transpose())
    #print(nabla_b.shape() + "   " + nabla_wT.shape() + "   " + delta.shape())
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    
    for l in range(2, num_layers):
        z = us[-l]
        sp = sigmoid_prime(z)
        print(sp)
        print(delta)
        print(weightsT)
        delta = np.dot(weightsT[-l+1].T, delta)*sp
        print(delta)
        nabla_b[-l] = delta
        nabla_wT[-l] = np.dot(delta, activations[-l+1].transpose())
        
    #nabla_b = [x.transpose() for x in nabla_b]
    #nabla_wT = [x.T for x in nabla_wT]
    return (nabla_b, nabla_wT)

