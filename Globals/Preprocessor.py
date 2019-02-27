"""
author: Jaswant Singh [developer.jaswant@gmail.com]
desc  : contains a basic data preprocessing utilities like normalization, structuring, filtering, Completing missing data.
"""

class Preprocessor:
    @staticmethod
    def process(train_original_x, test_original_x):
        return Preprocessor.normalize(train_original_x, test_original_x)

    @staticmethod
    def normalize(train_original_x, test_original_x):
        print("in the data preprocessor")
        train_set_x_flatten = train_original_x.reshape(
            (train_original_x.shape[0], train_original_x.shape[1] * train_original_x.shape[1] * 3)  # new shape
        ).T
        test_set_x_flatten = test_original_x.reshape(
            (test_original_x.shape[0], test_original_x.shape[1] * test_original_x.shape[1] * 3)
        ).T

        train_x = train_set_x_flatten/255
        test_x = test_set_x_flatten/255

        return train_x, test_x