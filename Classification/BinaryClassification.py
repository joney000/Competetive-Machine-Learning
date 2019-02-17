import sys
import time
import os
import numpy as np

from Globals.FileParser import FileParser
from Globals.Preprocessor import Preprocessor

"""
author:    Jaswant Singh [developer.jaswant@gmail.com]
algorithm: logistic regression
domain:    classification

"""
class BinaryClassification:

    def __init__(self, learning_rate, num_of_features, train_data, test_data, num_iterations):
        self.learning_rate = learning_rate
        self.weight = np.zeros((1, num_of_features)) # 1 * N
        self.num_of_training_examples = len(train_data)
        self.num_of_test_examples = len(test_data)
        self.test_data = test_data
        self.train_data = train_data
        self.num_iterations = num_iterations

    def sigmoid(x):
        return 1./(1. + np.exp(-x))

    def predic(self, x):        # the y^
        p = 1


    def train(self):
        train_input_x  = self.train_data[0]
        train_output_y = self.train_data[1]
        for interation in range(self.num_iterations):
            print("in the iteration " + str(iteration))
            z = np.dot(train_input_x, self.weight.T)  # z = train_input_x [M * N] dot w^T [N * 1] = z[M * 1]
            vectorized_sigmoid = np.vectorize(self.sigmoid)
            z = vectorized_sigmoid(z)   # z[i] = sigmoid(z[i])
    def test(self):
        q = 1#

def main():
    start = time.time()
    data = FileParser.readFile("Data" + os.pathsep + "Classification" + os.pathsep + "Train" + os.pathsep +"train.csv")

    # data = Preprocessor.process(data)
    model = BinaryClassification(0.75, 10, [1, 2], [3, 4], 10)   #
    print(model.learning_rate)
    # model.train()
    end = time.time()
    print("Time taken vector: "+(str)((end - start) * 100) +"ms")

if  __name__ =='__main__':main()
