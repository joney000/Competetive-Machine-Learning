import numpy as np
from Globals.FileParser import FileParser
from Globals.Preprocessor import Preprocessor
from Globals.Utils import Utils
# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from math import sqrt
import os

"""
author    : Jaswant Singh [developer.jaswant@gmail.com]
algorithm : Random Forest Decision Trees
domain    : Classification and Regression Both : Runs on discrete data
"""

class RandomForest:
  # Random Forest Algorithm
  def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
      sample = self.subsample(train, sample_size)
      tree = self.build_tree(sample, max_depth, min_size, n_features)
      trees.append(tree)
    predictions = [self.bagging_predict(trees, row) for row in test]
    return (predictions)
  
  # Split a dataset into k folds
  def cross_validation_split(self, dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
      fold = list()
      while len(fold) < fold_size:
        index = randrange(len(dataset_copy))
        fold.append(dataset_copy.pop(index))
      dataset_split.append(fold)
    return dataset_split
  
  # Calculate accuracy percentage
  def accuracy_metric(self, actual, predicted):
    correct = 0
    for i in range(len(actual)):
      if actual[i] == predicted[i]:
        correct += 1
    return correct / float(len(actual)) * 100.0
  
  # Evaluate an algorithm using a cross validation split
  def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
    folds = self.cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
      train_set = list(folds)
      train_set.remove(fold)
      train_set = sum(train_set, [])
      test_set = list()
      for row in fold:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
      predicted = algorithm(train_set, test_set, *args)
      actual = [row[-1] for row in fold]
      accuracy = self.accuracy_metric(actual, predicted)
      scores.append(accuracy)
    return scores
  
  # Split a dataset based on an attribute and an attribute value
  def test_split(self, index, value, dataset):
    left, right = list(), list()
    for row in dataset:
      if row[index] < value:
        left.append(row)
      else:
        right.append(row)
    return left, right
  
  # Calculate the Gini index for a split dataset
  def gini_index(self, groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
      size = float(len(group))
      # avoid divide by zero
      if size == 0:
        continue
      score = 0.0
      # score the group based on the score for each class
      for class_val in classes:
        p = [row[-1] for row in group].count(class_val) / size
        score += p * p
      # weight the group score by its relative size
      gini += (1.0 - score) * (size / n_instances)
    return gini
  
  # Select the best split point for a dataset
  def get_split(self, dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
      index = randrange(len(dataset[0]) - 1)
      if index not in features:
        features.append(index)
    for index in features:
      for row in dataset:
        groups = self.test_split(index, row[index], dataset)
        gini = self.gini_index(groups, class_values)
        if gini < b_score:
          b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}
  
  # Create a terminal node value
  def to_terminal(self, group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
  
  # Create child splits for a node or make terminal
  def split(self, node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
      node['left'] = node['right'] = self.to_terminal(left + right)
      return
    # check for max depth
    if depth >= max_depth:
      node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
      return
    # process left child
    if len(left) <= min_size:
      node['left'] = self.to_terminal(left)
    else:
      node['left'] = self.get_split(left, n_features)
      self.split(node['left'], max_depth, min_size, n_features, depth + 1)
    # process right child
    if len(right) <= min_size:
      node['right'] = self.to_terminal(right)
    else:
      node['right'] = self.get_split(right, n_features)
      self.split(node['right'], max_depth, min_size, n_features, depth + 1)
  
  # Build a decision tree
  def build_tree(self, train, max_depth, min_size, n_features):
    root = self.get_split(train, n_features)
    self.split(root, max_depth, min_size, n_features, 1)
    return root
  
  # Make a prediction with a decision tree
  def predict(self, node, row):
    if row[node['index']] < node['value']:
      if isinstance(node['left'], dict):
        return self.predict(node['left'], row)
      else:
        return node['left']
    else:
      if isinstance(node['right'], dict):
        return self.predict(node['right'], row)
      else:
        return node['right']
    
  # Create a random subsample from the dataset with replacement
  def subsample(self, dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
      index = randrange(len(dataset))
      sample.append(dataset[index])
    return sample
  
  # Make a prediction with a list of bagged trees
  def bagging_predict(self, trees, row):
    predictions = [self.predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count) 
  
def runAlgorithm():
  # load and prepare data
  dataset_file = '../../datasets/sonar.all-data.csv'
  dataset = FileParser.load_csv(dataset_file)

  # convert string attributes to integers
  Preprocessor.str_column_to_float(dataset)

  # convert class column to integers
  Preprocessor.str_column_to_int(dataset)
  # Model Parameters :
  n_folds = 5
  max_depth = 10
  min_size = 1
  sample_size = 1.0
  n_features = int(sqrt(len(dataset[0]) - 1))
  model = RandomForest()
  for n_trees in [2, 5, 15]:
    scores = model.evaluate_algorithm(dataset,
                                      model.random_forest,# function pointer
                                      n_folds, max_depth,
                                      min_size, sample_size,
                                      n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

if  __name__ =='__main__':runAlgorithm()