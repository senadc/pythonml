from numpy import exp, array, random, dot
import numpy as np
import math
import scipy

class AnomalyDetector(object):

    def mean(self, X):
        m = np.shape(X)[0]
        return (1 / m) * sum(X)
        
    def variance(self, X):
        m = np.shape(X)[0]
        return (1 / m) * sum((X - self.mean(X))  ** 2)
 
