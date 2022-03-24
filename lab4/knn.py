import random
import numpy as np
from scipy import spatial
from scipy import stats

"""
Implementation of the k-nearest neighbors algorithm for classification
and regression problems.
"""
class KNN:
    """
    Takes two parameters.  k is the number of nearest neighbors to use
    to predict the output variable's value for a query point. The
    aggregation_function is either "mode" for classification or
    "average" for regression.

    :param self: The KNN object
    :param int k: Number of neighbors
    :param string aggregation_function: Sets the function of outcome
        "mode" : for classification
        "average" : for regression.

    :return: NONE
    """
    def __init__(self, k, aggregation_function):
        self.k = k
        if aggregation_function == "average":
            self.function = 1
        #Default to mode
        else:
            self.function = 0

    """
    Stores the reference points (X) and their known output values (y).
    
    :param self: The KNN object
    :param Numpy array X: 2D array of training data (n_samples, n_features)
    :param Numpy array y: 1D array of data classification (n_samples)
    
    :return: NONE
    """     
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    """
    Predicts the output variable's values for the query points X.

    :param self: The KNN object
    :param Numpy array X: 2D-array of test samples (n_queries, n_features)

    :return: 1-D Numpy array of the predictions for each sample (n_queries) 
    """
    def predict(self, X):
        # Calculate the distance between the query point and every other point
        # Distance is a 2D numpy array where each row is the query point and 
        # each column is the distance between the other points
        # Shape: (# of query points)x(# of reference points).
        distance = spatial.distance.cdist(X, self.X, 'euclidean')
        
        # Sort the distance array by closest to farthest
        # Each element is the index of the reference point associated with the distance
        # Shape: (# of query points)x(# of reference points).
        nearest = np.argsort(distance)

        # Take the K closest reference points in nearest
        # Since the nearest array contains the index of the reference point, 
        # the classification value can be derivied by calling classification[index]
        # Shape: (# of query points)x(k).
        nearest_k = np.array(self.y[nearest[:,:self.k]])
        
        # Return a 1-D array that contains the predicted type for every query point.
        # Shape: (# of query points)
        if self.function == 1:
            # Stats mode returns a tall single column array so it has to be rotated by 
            # 90 degrees to be useable.
            # When stats.mode is called, it returns the most common value and the count. 
            # We only care about the value so [0] is added
            # When rot90 is called it returns a 2-D array containing only 1 row 
            # so [0] is called to convert to 1-D array
            return np.rot90(stats.mode(nearest_k, axis=1)[0])[0]
        else:
            return np.mean(nearest_k, axis=1)
            
   