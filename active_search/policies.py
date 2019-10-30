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


MERGE_SORT_FILE_NAME = 'merge_sort.so'
"""
def merge_sort(p, q, top_ind, budget):
    n = q.size
    sum_to_return = 0
    i = 0
    j = 0
    while np.take(p,np.take(top_ind,j))==0:
        j= j+1
    k = 0

    while i<budget and k<n:
        #p_p_ind = np.take(p,top_ind[j])
        if np.take(p,np.take(top_ind,j)) > np.take(q,k):
            sum_to_return += np.take(p,np.take(top_ind,j))
            while True:
                j= j+1
                try:
                    #this is where getting error!
                    if np.take(p,np.take(top_ind,j))!=0:
                        break
                except:
                    import pdb; pdb.set_trace()
        else:
            sum_to_return +=  np.take(q,k)
            k=k+1
        i = i+1
    
    while i<budget:
        sum_to_return += np.take(p,np.take(top_ind,j))
        while True:
                j= j+1
                if np.take(p,np.take(top_ind,j))!=0:
                    break
        i=i+1
    
    return sum_to_return
"""






def merge_sort(p, q, top_ind, budget):
    n = q.size
    sum_to_return = 0
    i = 0
    j = 0
    while p[top_ind[j]]==0:
        j= j+1
    k = 0

    while i<budget and k<n:
        #p_p_ind = np.take(p,top_ind[j])
        if p[top_ind[j]]> q[k]:
            sum_to_return += p[top_ind[j]]
            condition1 = True
            while condition1:
                j= j+1
                #this is where getting error!
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
    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices,budget,points):
        pass


class OneStep(Utility):

    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices,budget,points):
        expected_utilities = model.predict(data, test_indices) + sum(data.observed_labels)
        return expected_utilities
        # return probability estimation


class TwoStep(Utility):

    def __init__(self):
        self.selector = UnlabelSelector()
        pass

    def get_scores(self, model,data,test_indices,budget,points):

        #TODO: do I need to do something if budget <=1?
        
        #compute p
        num_test = test_indices.size
        
        expected_utilities = np.zeros((num_test,1))
        
        #TODO: remove below line
        #probabilitiesStorage = np.zeros((num_test,2))

        unlabeled_ind = self.selector.filter(data, points)

        probabilities = model.predict(data,test_indices)

        probabilities_including_negative = np.append(probabilities,1-probabilities,axis=1)

        unlabeledWeights = model.weight_matrix.tolil(copy=True)



        unlabeledWeights[data.train_indices,:]=0
        
        #TODO: unedit this line!
        fake_data = copy.deepcopy(data)
        
        for i in range(num_test):
            this_test_ind = test_indices[i].astype(int)

            fake_train_ind = np.append(data.train_indices,this_test_ind)
            
            #fake_test_ind = np.delete(unlabeled_ind,i)

            fake_test_ind = find(unlabeledWeights[:,this_test_ind])[0]
            #print("fake_test_indices shape:",fake_test_ind.shape)
            #to_print = np.sort(fake_test_ind)
            #fake_test_ind = fake_test_ind.sort()
            #print("fake_test_indices:",to_print)

            pstar = np.zeros((2,1))

            for j in range(2):
                
                fake_data.observed_labels = np.append(data.observed_labels,j)
                fake_data.train_indices = fake_train_ind

                fake_predictions = model.predict(fake_data,fake_test_ind)

                #np.savetxt("probabilities"+str(i)+str(j), fake_predictions)
                pstar[1-j][0] = np.amax(fake_predictions)
                
            expected_utilities[i]= np.matmul(probabilities_including_negative[i],pstar)
            #print(pstar)
            

        two_step_utilities = np.size(data.train_indices) + probabilities + expected_utilities
        
        #print("number of observed points:", np.size(data.train_indices))
        #print("probabilities: ", probabilities)
        #print("expected_utilities",expected_utilities)
        #

        #test_ind_size = test_indices.size
        #test_indices_reshaped = test_indices.reshape((test_ind_size,1))
        #print("test_indices shape:", test_indices_reshaped.shape)

        #concatenated = np.concatenate((test_indices_reshaped, two_step_utilities), axis=1)
        #np.savetxt('twoStepUtilities.txt', concatenated, fmt='%10.5f', delimiter=' ')
        
        #import pdb; pdb.set_trace()


        
        """print("two-step-utilities shape:", two_step_utilities.shape)
        print("test_indices shape:", test_indices.shape)
        to_save = np.concatenate((test_indices,two_step_utilities))
        np.savetxt('twosteputilities.txt', to_save, delimiter=' ')
        sys.exit()
        """

        
        return two_step_utilities
        
class ENS(Utility):

    def __init__(self):
        self.selector = UnlabelSelector()
        
        pass

    def get_scores(self, model,data,test_indices,budget,points):
        print("Starting score function")
        num_test = test_indices.size

        expected_utilities = np.zeros((num_test,1))

        compareValuesRemoveLater = np.zeros((num_test,2))

        unlabeled_ind = self.selector.filter(data, points)

        probabilities = model.predict(data,test_indices)

        probs = np.zeros((points.size,1))

        probs[data.train_indices] = np.array(data.observed_labels, ndmin=2).T
        
        probs[test_indices] = probabilities

        all_probs = np.tile(probs, (16, 1))
        unlabeled_probs = all_probs[unlabeled_ind]

        reverse_ind = np.ones((points.size,1))*-1
        reverse_ind[unlabeled_ind]= np.arange(0,unlabeled_ind.size).reshape((unlabeled_ind.size,1))

        argsort_ind = (-probabilities).argsort(axis=0)

        probabilities_sorted = probabilities[argsort_ind]
        
        top_ind = np.argsort(-unlabeled_probs,axis = 0,kind = 'stable')

        cur_future_utility = np.sum(probabilities_sorted[0:budget,:])

        probabilities_including_negative = np.append(probabilities,1-probabilities,axis=1)

        fake_data = copy.deepcopy(data)

        unlabeledWeights = model.weight_matrix.tolil(copy=True)

        unlabeledWeights[data.train_indices,:]=0
        
        for i in range(num_test):
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
            this_test_probs = np.append(this_test_probs,1-this_test_probs,axis=1)


            #Begin unrolling For loop
            #J = 0

            fake_data.observed_labels = np.append(data.observed_labels,0)
                
            fake_data.train_indices = fake_train_ind

            fake_predictions = model.predict(fake_data,fake_test_ind)
            
            q = np.sort(fake_predictions, axis=0)
            q = q[::-1]

            pstar[1][0] = merge_sort(p,q,top_ind,budget)

            #J = 1

            fake_data.observed_labels = np.append(data.observed_labels,1)
            fake_data.train_indices = fake_train_ind
            fake_predictions = model.predict(fake_data,fake_test_ind)
            q = np.sort(fake_predictions, axis=0)
            q = q[::-1]

            pstar[0][0] = merge_sort(p,q,top_ind,budget)




            #End unrolling for loop

            for j in range(2):
                
                fake_data.observed_labels = np.append(data.observed_labels,j)
                
                fake_data.train_indices = fake_train_ind

                fake_predictions = model.predict(fake_data,fake_test_ind)
                
                q = np.sort(fake_predictions, axis=0)
                q = q[::-1]

                #import matlab.engine
                #eng = matlab.engine.start_matlab()
                #tf = eng.isprime(37)

                pstar[1-j][0] = merge_sort(p,q,top_ind,budget)

                #if j ==0 and budget ==98 and this_test_ind == 4:
                #    import pdb; pdb.set_trace()
            
            expected_utilities[i]= np.matmul(this_test_probs[i],pstar)
            compareValuesRemoveLater[i]=pstar.ravel()
            
            
        utilities = probabilities + expected_utilities - cur_future_utility

        #np.savetxt('p.txt', p, fmt='%10.5f', delimiter=' ')
        #np.savetxt('q.txt', q, fmt='%10.5f', delimiter=' ')
        #np.savetxt('top_ind.txt', top_ind, fmt='%10.5f', delimiter=' ')
        
        #np.savetxt('expected_utilities.txt', utilities, fmt='%10.5f', delimiter=' ')
        a_ind = np.where(test_indices==1908)

        print("utility_1908: ",utilities[a_ind])


        return utilities

class Policy(object):
    def __init__(self, utility, model):
        pass

    def choose_next(self):
        pass


class ArgMaxPolicy(Policy):
    def __init__(self, problem, model=None, utility=None):
        if not model:
            model = KnnModel(problem)
        if not utility:
            utility = OneStep()
        self.model = model
        self.utility = utility

    def choose_next(self, data, test_indices, budget,points):
        
        if test_indices.size ==1:
            return test_indices
        scores = self.utility.get_scores(self.model,data,test_indices,budget,points)
        max_index = np.argmax(scores)

        chosen_x_index = test_indices[max_index]
        
        #print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.Utility.model.problem.y_train[chosen_x_index])

        #chosen_x = self.model.problem.x_pool[chosen_x_index]
        return chosen_x_index


class ENSPolicy(Policy):
    def __init__(self, problem, model=None, utility=None):
        if not model:
            model = KnnModel(problem)
        if not utility:
            utility = ENS()
        self.model = model
        self.utility = utility

    def choose_next(self, data, test_indices, budget,points):
        
        if budget==1:
            probs = self.model.predict(data, test_indices)
            return test_indices[np.argmax(probs)]

        if test_indices.size ==1:
            return test_indices
        scores = self.utility.get_scores(self.model,data,test_indices,budget,points)
        max_index = np.argmax(scores)

        chosen_x_index = test_indices[max_index]
        
        #print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.Utility.model.problem.y_train[chosen_x_index])

        #chosen_x = self.model.problem.x_pool[chosen_x_index]
        return chosen_x_index