import numpy as np
import scipy.io

# import matlab.engine


# How to install matlab engine: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html


def genData(jitter):
    """Generates data for use in an Active Search problem

        Parameters
        ----------
        jitter : bool
            Whether or not to add jitter to the points
        
        Returns
        ----------
        sum_to_return : array_like
            the sum of the top (budget) elements from p and q
    """
    
    if jitter:
        points = scipy.io.loadmat("./data/points_jitter_div_by_100.mat")
        points = points['points']
    else:
        points = np.loadtxt('./data/points.txt')
    
    labels_random = np.loadtxt('./data/labels_random.txt')
    labels_deterministic = np.loadtxt('./data/labels_deterministic.txt')
    return labels_random, labels_deterministic, points
