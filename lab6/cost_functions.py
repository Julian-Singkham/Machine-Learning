import numpy as np
from sklearn.metrics import mean_squared_error

"""
Implements a cost function for fitting a Gaussian (normal) distribution.
"""
class GaussianCostFunction:

    """
    The constructor takes the feature matrix and true y values for the training data.
    
    :param self: The Gaussian cost function object
    :param Numpy features: 2D array of training data [n_samples, n_features(1)]
    :param numpy y_true: 1D array of data classification [n_samples]

    :return: NONE
    """
    def __init__(self, features, y_true):
        self.features = features
        self.y_true = y_true
        
    """
    Predicts the y values for each data point using the feature matrix and the model
    parameters.

    
    :param self: The Gaussian cost function object
    :param Numpy array features: 2D array of samples [n_samples, n_features(1)]
    :param Numpy array params: 1D array of model parameters (mean, std deviation)
    
    :return: 1-D Numpy array of the predictions for each sample
    """ 
    def predict(self, features, params):
        np_pred = np.zeros(features.size)
        np_pred = 1/(pow(2*np.pi,0.5)*params[1])*np.exp(-0.5*pow(features-params[0],2)/params[1])
        return np_pred


    """
    Calculates the mean-squared error between the predicted and true y values.
    
    :param self: The Gaussian cost function object
    :param Numpy array y_true: 1D array of true classifications [n_samples]
    :param Numpy array pred_y: 1D array of model predictions (n_samples)
    
    :return: Float error of the model  
    """       
    def _mse(self, y_true, pred_y):
        return mean_squared_error(y_true, pred_y)

        
    """
    Calculates the cost function value for the model's predictions using the given params.
    
    :param self: The Gaussian cost function object
    :param Numpy array params: 1D array of model parameters (mean, std deviation)
    
    :return: Float error of the model    
    """        
    def cost(self, params):
        pred = self.predict(self.features, params)
        return self._mse(self.y_true, pred)
    
"""
Implements a cost function for a linear regression model.
"""        
class LinearCostFunction:

    """
    The constructor takes the feature matrix and true y values for the training data.
    
    :param self: The linear cost function object
    :param Numpy features: 2D array of training data [n_samples, n_features]
    :param numpy y_true: 1D array of data classification (n_samples)

    :return: NONE
    """
    def __init__(self, features, y_true):
        self.features = features
        self.y_true = y_true
        
    """
    Predicts the y values for each data point using the feature matrix and the model
    parameters.

    
    :param self: The linear cost function object
    :param Numpy array features: 2D array of samples [n_samples, n_features]
    :param Numpy array params: 1D array of model parameters (n_features)
    
    :return: 1-D Numpy array of the predictions for each sample (n_samples)
    """     
    def predict(self, features, params):
        l_sales = []
        for row in features:
            l_sales.append(params[0] + np.multiply(params[1],row[0]) + np.multiply(params[2],row[1]) + np.multiply(params[3],row[2]))
        return np.array(l_sales)
        
    """
    Calculates the mean-squared error between the predicted and true y values.
    
    :param self: The linear cost function object
    :param Numpy array y_true: 1D array of true classifications [n_samples]
    :param Numpy array pred_y: 1D array of model predictions (n_samples)
    
    :return: Float error of the model  
    """          
    def _mse(self, y_true, pred_y):
        return mean_squared_error(y_true, pred_y)
 
    """
    Calculates the cost function value for the model's predictions using the given params.
    
    :param self: The linear cost function object
    :param Numpy array params: 1D array of model parameters (B0,B1,B2,B3)
    
    :return: Float error of the model    
    """       
    def cost(self, params):
        pred = self.predict(self.features, params)
        return self._mse(self.y_true, pred)