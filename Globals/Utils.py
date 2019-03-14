import numpy as np

class Utils:

  @staticmethod
  def sigmoid_activation(z):
    return 1. / (1 + np.exp(-z))