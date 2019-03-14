import sys
import time
import os

import matplotlib.pyplot as plt
import matplotlib
import scipy
from PIL import Image
from scipy import ndimage

import numpy as np
from Globals.FileParser import FileParser
from Globals.Preprocessor import Preprocessor
from Globals.Utils import Utils

"""
author:    Jaswant Singh [developer.jaswant@gmail.com]
algorithm: logistic regression
domain:    classification

"""
class LogisticRegression:

    def initialize_with_zeros(self, features):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        features -- size of the w vector we want (or number of parameters/features in this case)

        Returns:
        weight -- initialized vector of shape (dim, 1)
        bias   -- initialized scalar (corresponds to the bias)
        """
        weight = np.zeros((features, 1))
        bias = 0
        assert (weight.shape == (features, 1))
        assert (isinstance(bias, float) or isinstance(bias, int))
        return weight, bias

    def activation(self, z):
        return Utils.sigmoid_activation(z)

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        """

        m = X.shape[1]

        # FORWARD PROPAGATION (FROM X TO COST)
        A = self.activation(np.dot(X.T, w) + b).T  # compute activation
        cost = -1.0 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = 1 / m * np.dot(X, (A - Y).T)
        db = 1 / m * np.sum(A - Y)

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        """

        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        # Computed vector "A" predicting the probabilities of a cat being present in the picture
        A = self.activation(np.dot(X.T, w) + b).T

        for i in range(A.shape[1]):
            Y_prediction[0][i] = 1 if A[0][i] >= 0.5 else 0

        assert (Y_prediction.shape == (1, m))

        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling helper functions

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """

        w, b = self.initialize_with_zeros(X_train.shape[0])

        parameters, grads, costs = self.optimize(w, b, X_test, Y_test, num_iterations, learning_rate, print_cost)

        w = parameters["w"]
        b = parameters["b"]

        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d

def runAlgorithm():
    start = time.time()
    train_x_orig, train_y, test_x_orig, test_y, classes = FileParser.load_dataset()
    # plt.imshow(train_set_x_orig[25])
    # plt.show(block = True)
    train_x, test_x =  Preprocessor.process(train_x_orig, test_x_orig)
    print("train_x shape: " + str(train_x.shape))   # (n , m)  : n = no of feature, m = no of examples test/train
    print("test_x shape: " + str(test_x.shape))
    algo = LogisticRegression()
    result = algo.model(train_x, train_y, test_x, test_y, num_iterations = 2000, learning_rate = 0.1, print_cost = True)
    end = time.time()
    print("Time taken vector: "+(str)((end - start) * 100) +"ms")

if  __name__ =='__main__':runAlgorithm()
