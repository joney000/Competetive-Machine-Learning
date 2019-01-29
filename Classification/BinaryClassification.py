import sys
import numpy as np
import Globals.Preprocessor
import math

"""
author: Jaswant Singh [developer.jaswant@gmail.com]

"""
class BinaryClassification:
    w = [] *
    def __init__(self, learning_rate, num_of_training_examples, num_of_test_examples, num_of_features):
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        return 1./(1. + math.exp(-x))

    def predic(self, y):
        return sigmoid(w)

def main():
    model = BinaryClassification(1.7)   #
    print(model.learning_rate)
if  __name__ =='__main__':main()
