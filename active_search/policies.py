"""Policies for active search"""
#from models import *
import numpy as np
import copy
from active_search.models import UnlabelSelector,KnnModel
import sys
from scipy.sparse import find
#import bottleneck as bn
#from cppFolder import merge_sort
import ctypes
import os

#TODO: remove when verify ENS working correctly:
from os.path import dirname, join as pjoin
import scipy.io as sio
import time


def merge_sort(p, q, top_ind, budget):
    """Special custom implementation of the Merge Sort algorithm.

    Parameters
    ----------
    p,q : array_like
        Arrays to merge sort
    top_ind : array_like
        sub-index to pass to p
    budget : int
        remaining budget left for active search problem, only taking top
        (budget) points in total from either p or q arrays
    
    Returns
    ----------
    sum_to_return : array_like
        the sum of the top (budget) elements from p and q
    """
    
    #TODO: re-implement in cython for computational speedup
    n = q.size
    sum_to_return = 0
    i = 0
    j = 0
    while p[top_ind[j]]==0:
        j= j+1
    k = 0

    while i<budget and k<n:
        if p[top_ind[j]]> q[k]:
            sum_to_return += p[top_ind[j]]
            condition1 = True
            while condition1:
                j= j+1
                condition1 = p[top_ind[j]]==0
                    
        else:
            sum_to_return +=  np.take(q,k)
            k=k+1
        i = i+1
    
    while i<budget:
        sum_to_return += p[top_ind[j]]
        condition2 = True
        while condition2:
                j= j+1
                condition2 =  p[top_ind[j]]==0
        i=i+1
    
    return sum_to_return


class Utility(object):
    """
    A base-class used to represent a method of calcuating utilities

    This is one of the principal components of an Active Search algorithm. It
    is used to specify how many steps ahead to look for utility estimation,
    as well as the method used to do so.

    Methods
    -------
    get_scores(self, model, data, test_indices,budget,points)
        calculates and returns n-step utility estimation for all elements of 
        test_indices
    """

    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices,budget,points):
        pass


class OneStep(Utility):
    """
    A class used to represent calcuating one-step utilities

    It is used to calculate the expected utility at the next step for each 
    point in test_indices

    Methods
    -------
    get_scores(self, model, data, test_indices,budget,points)
        Calculates and returns ones-step utility estimation for points in 
        test_indices
    """

    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices,budget,points):
        """Calculates and returns one-step utility estimation for points in 
        test_indices

        Parameters
        ----------
        model : Model
            See models.py
        data : Data
            See models.py
        test_indices : array_like
            Input array of indices that correspond to `points`, must be
            dimensions nx1
        budget : int
            remaining budget left for active search problem
        points : array_like
            Input array of points in the search space, must be nxd
        
        Returns
        ----------
        expected_utilities : array_like
            the expected utilities corresponding to points in test_indices
        """

        expected_utilities = model.predict(data, test_indices) + 
            sum(data.observed_labels)
        return expected_utilities
        # return probability estimation


class TwoStep(Utility):

    def __init__(self):
        self.selector = UnlabelSelector()

        self.ensUtility = ENS()
        pass

    
    def get_scores(self, model,data,test_indices,budget,points):
        """Calculates and returns two-step utility estimation for points in 
        test_indices

        Parameters
        ----------
        model : Model
            See models.py
        data : Data
            See models.py
        test_indices : array_like
            Input array of indices that correspond to `points`, must be
            dimensions nx1
        budget : int
            remaining budget left for active search problem
        points : array_like
            Input array of points in the search space, must be nxd
        
        Returns
        ----------
        expected_utilities : array_like
            the expected utilities corresponding to points in test_indices
        """
        num_test = test_indices.size

        expected_utilities = np.zeros((num_test,1))
        
        unlabeled_ind = self.selector.filter(data, points)

        probabilities = model.predict(data,unlabeled_ind)

        testProbs = model.predict(data,test_indices)

        probs = np.zeros((points.size,1))
        
        probs[data.train_indices] = np.array(data.observed_labels, ndmin=2).T
        
        probs[test_indices] = testProbs

        all_probs = np.tile(probs, (16, 1))
        unlabeled_probs = all_probs[unlabeled_ind]

        reverse_ind = np.ones((points.size,1))*-1
        reverse_ind[unlabeled_ind]= 
            np.arange(0,unlabeled_ind.size).reshape((unlabeled_ind.size,1))

        reverse_ind = reverse_ind.astype(int)


        probabilities[::-1].sort()

        p_max = np.amax(probabilities)

        probabilities_including_negative = np.append(probabilities,
            1-probabilities,axis=1)

        unlabeledWeights = model.weight_matrix.tolil(copy=True)

        unlabeledWeights[data.train_indices,:]=0
        
        fake_data = copy.deepcopy(data)
        
        for i in range(num_test):
            this_test_ind = test_indices[i].astype(int)

            fake_train_ind = np.append(data.train_indices,this_test_ind)

            fake_test_ind = find(unlabeledWeights[:,this_test_ind])[0]

            p = probabilities.copy()

            a = np.take(reverse_ind,this_test_ind)
            b = np.take(reverse_ind,fake_test_ind)
            
            a = a[a>=0].astype(int)
            b = b[b>=0].astype(int)
            
            p[a]=0
            p[b]=0

            pstar = np.zeros((2,1))

            for j in range(2):
                
                fake_data.observed_labels = np.append(data.observed_labels,j)
                fake_data.train_indices = fake_train_ind

                fake_predictions = model.predict(fake_data,fake_test_ind)

                p_max = np.amax(p)

                pstar[1-j][0] = max(p_max, np.amax(fake_predictions))+j+
                    sum(data.observed_labels)

            expected_utilities[i]= np.matmul(probabilities_including_negative[
                reverse_ind[test_indices[i]]],pstar) 
        
        return expected_utilities
        
class ENS(Utility):

    def __init__(self):
        self.selector = UnlabelSelector()
        pass

    def get_scores(self, model,data,test_indices,budget,points,probabilities,
         do_pruning):
        """Calculates and returns (budget)-step utility estimation for points
        in test_indices

        Parameters
        ----------
        model : Model
            See models.py
        data : Data
            See models.py
        test_indices : array_like
            Input array of indices that correspond to `points`, must be
            dimensions nx1
        budget : int
            remaining budget left for active search problem
        points : array_like
            Input array of points in the search space, must be nxd
        probabilities : array_like
            Probabilities for each point in test_indices of being a target,
            must be dimensions nx1
        do_pruning : bool
            Whether or not to perform pruning on the search space

        Returns
        ----------
        utilities : array_like
            the expected utilities corresponding to points in test_indices
        """
        
        print("Starting score function")

        num_test = test_indices.size

        expected_utilities = np.zeros((num_test,1))

        unlabeled_ind = self.selector.filter(data, points)

        probs = np.zeros((points.size,1))

        probs[data.train_indices] = np.array(data.observed_labels, ndmin=2).T
        
        probs[test_indices] = probabilities

        all_probs = np.tile(probs, (16, 1))
        unlabeled_probs = all_probs[unlabeled_ind]

        #create reverse-lookup table
        reverse_ind = np.ones((points.size,1))*-1
        reverse_ind[unlabeled_ind]= np.arange(0,
            unlabeled_ind.size).reshape((unlabeled_ind.size,1))

        top_ind = np.argsort(-unlabeled_probs,axis = 0,kind = 'stable')

        cur_future_utility = np.sum(probabilities[0:budget,:])

        probabilities_including_negative = np.append(probabilities,
            1-probabilities,axis=1)

        fake_data = copy.deepcopy(data)

        unlabeledWeights = model.weight_matrix.tolil(copy=True)

        unlabeledWeights[data.train_indices,:]=0
        
        #begin pruning code
        if do_pruning:
            prob_upper_bound = self.knnbound(model, data, points,
                test_indices, budget)

            future_utility_if_neg = np.sum(probabilities[0:budget])

            max_num_influence = model.k

            if max_num_influence>= budget:
                future_utility_if_pos = np.sum(prob_upper_bound[1:budget])

            else:
                tmp_ind = top_ind[0:(budget-max_num_influence)]
                future_utility_if_pos = np.sum(probabilities[tmp_ind]) +
                    np.sum(prob_upper_bound[0:max_num_influence])

            
            future_utilities= np.zeros((1,2))
            future_utilities[0][0]=future_utility_if_pos
            future_utilities[0][1]=future_utility_if_neg
            
            future_utility = np.matmul(probabilities_including_negative,
                future_utilities.T)


            future_utility_bound=  future_utility - cur_future_utility

            upper_bound_of_score = probabilities + future_utility_bound
            
            pruned = np.zeros((num_test,1))
            current_max =-1
        
        #end pruning code
        utilities = np.zeros(test_indices.shape)

        for i in range(num_test):

            if do_pruning and pruned[i]:
                continue

            this_test_ind = test_indices[i].astype(int)

            fake_train_ind = np.append(data.train_indices,this_test_ind)

            fake_test_ind = np.nonzero(unlabeledWeights[:,this_test_ind])[0]

            pstar = np.zeros((2,1))

            p = unlabeled_probs.copy()

            a = np.take(reverse_ind,this_test_ind)
            b = np.take(reverse_ind,fake_test_ind)
            c = np.take(reverse_ind,test_indices)
            
            a = a[a>=0].astype(int)
            b = b[b>=0].astype(int)
            c = c.astype(int)
            
            p[a]=0
            p[b]=0

            this_test_probs = unlabeled_probs[c]
            this_test_probs = np.append(this_test_probs,1-this_test_probs,
                axis=1)


            fake_data.observed_labels = np.append(data.observed_labels,0)
                
            fake_data.train_indices = fake_train_ind

            fake_predictions = model.predict(fake_data,fake_test_ind)
            
            q = np.sort(fake_predictions, axis=0)
            q = q[::-1]

            pstar[1][0] = merge_sort(p,q,top_ind,budget)


            fake_data.observed_labels = np.append(data.observed_labels,1)
            fake_data.train_indices = fake_train_ind
            fake_predictions = model.predict(fake_data,fake_test_ind)
            q = np.sort(fake_predictions, axis=0)
            q = q[::-1]

            pstar[0][0] = merge_sort(p,q,top_ind,budget)

            for j in range(2):
                
                fake_data.observed_labels = np.append(data.observed_labels,j)
                
                fake_data.train_indices = fake_train_ind

                fake_predictions = model.predict(fake_data,fake_test_ind)
                
                q = np.sort(fake_predictions, axis=0)
                q = q[::-1]

                pstar[1-j][0] = merge_sort(p,q,top_ind,budget)
            
            
            expected_utilities[i]= np.matmul(this_test_probs[i],pstar)

            utilities[i] = probabilities[i]+expected_utilities[i]-
                cur_future_utility
            if do_pruning and utilities[i]>current_max:
                current_max = utilities[i]
                pruned[upper_bound_of_score <= current_max] = True


        return utilities



class Policy(object):
    """
    A base-class used to represent a method of choosing points 

    This is one of the principal components of an Active Search algorithm. It
    is used to choose the point to explore from the points in test_indices
    based on their utility estmation.

    Methods
    -------
    choose_next(self, data, test_indices, budget,points)
        Calls the get_scores utility member function, and then chooses the
        next point based on the scores returned
    """

    def __init__(self, utility, model):
        pass

    def choose_next(self):
        pass


class ArgMaxPolicy(Policy):
    """
    A class used to represent calcuating the point with the highest utility

    It is used to select the point with the highest calculated expected
    utility, based on the utility function used.

    Attributes
    ----------
    model : Model
            See models.py
    utility : Utility
            see policies.py

    Methods
    -------
    choose_next(self, data, test_indices, budget,points)
        Calls the get_scores utility member function, and then chooses the
        next point to be the one corresponding to the highest of the scores
        returned
    """

    def __init__(self, problem, model=None, utility=None):
        if not model:
            model = KnnModel(problem)
        if not utility:
            utility = OneStep()
        self.model = model
        self.utility = utility

    def choose_next(self, data, test_indices, budget,points):
        """Chooses the next point

        Calls the get_scores utility member function, and then chooses the
        next point to be the one corresponding to the highest of the scores
        returned

        Parameters
        ----------
        data : Data
            See models.py
        test_indices : array_like
            Input array of indices that correspond to `points`, must be
            dimensions nx1
        budget : int
            remaining budget left for active search problem
        points : array_like
            Input array of points in the search space, must be nxd
        
        Returns
        ----------
        chosen_x_index : int
            the index to call the oracle function on, corresponds to a point
            in test_indices
        """

        if test_indices.size ==1:
            return test_indices[0]
        scores = self.utility.get_scores(self.model,data,test_indices,
            budget,points)
        
        max_index = np.argmax(scores)

        chosen_x_index = test_indices[max_index]
        
        return chosen_x_index


class ENSPolicy(Policy):
    """
    A class used to represent the optimal ENS policy

    It is used to select the point according to the optimal policy using
    ENS utility estimations from from Bayesian Optimal Active Search and
    Surveying, Garnett et al. 2012

    Attributes
    ----------
    model : Model
            See models.py
    utility : Utility
            see policies.py

    Methods
    -------
    choose_next(self, data, test_indices, budget,points)
        Calls the get_scores utility member function, and then chooses the
        next point to be the one corresponding to the highest of the scores
        returned. A few edge cases are included to speedup code execution
    """

    def __init__(self, problem, model=None, utility=None, do_pruning = True):
        if not model:
            model = KnnModel(problem)
        if not utility:
            utility = ENS()
        self.model = model
        self.utility = utility
        self.do_pruning = do_pruning

    def choose_next(self, data, test_indices,budget,points):
        """Chooses the next point

        Calls the get_scores utility member function, and then chooses the
        next point to be the one corresponding to the highest of the scores
        returned. Includes edge cases for speedups

        Parameters
        ----------
        data : Data
            See models.py
        test_indices : array_like
            Input array of indices that correspond to `points`, must be
            dimensions nx1
        budget : int
            remaining budget left for active search problem
        points : array_like
            Input array of points in the search space, must be nxd
        
        Returns
        ----------
        chosen_x_index : int
            the index to call the oracle function on, corresponds to a point
            in test_indices
        """

        if budget==1:
            probs = self.model.predict(data, test_indices)
            return test_indices[np.argmax(probs)]

        if test_indices.size ==1:
            return test_indices
    
        probabilities = self.model.predict(data,test_indices)
        argsort_ind = (-probabilities).argsort(axis=0)
        probabilities = probabilities[argsort_ind[:,0]]
        test_indices = test_indices[argsort_ind[:,0]]

        scores = self.utility.get_scores(self.model,data,test_indices,budget,
            points,probabilities,self.do_pruning)

        max_index = np.argmax(scores)

        chosen_x_index = test_indices[max_index]
        
        return chosen_x_index