"""Models for active search"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class problem(object):
    def __init__(self, x_pool):
        self.x_train = np.empty(np.shape(x_pool)) # nxd
        self.x_train[:]=np.nan
        self.y_train = np.empty((np.size(x_pool,0),1))#nx1
        self.y_train[:]=np.nan
        self.x_pool = x_pool  #nxd
        #x_train is nxd, everything starts at zeros and values get copied over from x_pool when they are observed
        #y_train is nx1, everything starts as zeros and values get filled in with the oracle function

    def newObservation(self,index,x,y):
        self.x_train[index]=x
        self.y_train[index]=y



class model(object):
    def __init__(self):
        pass
    def predict(self):
        pass
    def update(self):
        pass
    def plot(self):
        pass



class randomModel(model):

    def __init__(self, problem):
        self.problem = problem

    def predict(self):
        #returns a nx1 array of random values between 0 and 1
        return np.random.random_sample((self.problem.x_pool.shape[0],1))


class knnModel(model):
    def __init__(self, problem):
        k = 8
        #k=100
        self.problem = problem
        xs= problem.x_pool
        #calculate euclidean distance between x matrix and itself
        self.dist_matrix = euclidean_distances(xs,xs) # nxn matrix that stores distance between points

        #sorts indices by lowest distance value, stores the k-lowest for each point
        self.knn = np.argsort(self.dist_matrix, axis=1)[:,1:k+1] # nxk matrix
        #print("knns: ",self.knn)
        #self.knn[x] returns array of k closest points to point x

    
    def predict(self):

        #returns probability estimate, Pr(y=1|x,D)

        #equation based off of Bayesian Optimal Active Search and Surveying, Garnett et al. 2012 (Equation 7)
        gamma = 0.1
        #create empty array to return
        predictions = np.zeros((np.size(self.problem.x_pool,0),1))
        #predictions[:]=np.nan
        #iterate over all x values in x_pool
        for i in range(np.size(self.problem.x_pool,0)):
            if np.isnan(self.problem.y_train[i]):

                #a stores the y_train values of the k-nearest neighbors of x
                a = np.take(self.problem.y_train,self.knn[i],axis=0)
                #print("a ",a)
                nans = np.isnan(a)
                #print("nans ",nans)
                #denominator is the number of non-nan y_train entries of the k-nearest neighbors

                denominator = np.sum(np.invert(nans))
                #print("denominator",denominator)
                #numerator is the sum of all non-nan y_train values of the k-nearest neighbors
                #numerator = np.dot(nans,self.problem.y_train)
                numerator = np.nansum(a)
                predictions[i]=(gamma+numerator)/(1+denominator)

        #returns nx1 column vector of probabilities
        return predictions

