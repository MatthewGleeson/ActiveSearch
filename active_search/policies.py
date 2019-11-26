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

        self.ensUtility = ENS()
        pass

    
    def get_scores(self, model,data,test_indices,budget,points):
        

        #return self.ensUtility.get_scores( model,data,test_indices,2,points)

        
        #compute p
        num_test = test_indices.size
        
        expected_utilities = np.zeros((num_test,1))
        

        unlabeled_ind = self.selector.filter(data, points)

        probabilities = model.predict(data,unlabeled_ind)

        #Begin pasted code
        testProbs = model.predict(data,test_indices)

        probs = np.zeros((points.size,1))

        probs[data.train_indices] = np.array(data.observed_labels, ndmin=2).T
        
        probs[test_indices] = testProbs

        all_probs = np.tile(probs, (16, 1))
        unlabeled_probs = all_probs[unlabeled_ind]

        reverse_ind = np.ones((points.size,1))*-1
        reverse_ind[unlabeled_ind]= np.arange(0,unlabeled_ind.size).reshape((unlabeled_ind.size,1))

        reverse_ind = reverse_ind.astype(int)

        #End pasted code

        

        #TODO: set probability values to zero
        #p(reverse_ind(this_test_ind)) = 0;
        #p(reverse_ind(fake_test_ind)) = 0;

        

        probabilities[::-1].sort()

        p_max = np.amax(probabilities)

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

            #begin pasted code
            p = probabilities.copy()

            a = np.take(reverse_ind,this_test_ind)
            b = np.take(reverse_ind,fake_test_ind)
            
            a = a[a>=0].astype(int)
            b = b[b>=0].astype(int)
            
            p[a]=0
            p[b]=0

            #END pasted code
            #print("fake_test_indices shape:",fake_test_ind.shape)
            #to_print = np.sort(fake_test_ind)
            #fake_test_ind = fake_test_ind.sort()
            #print("fake_test_indices:",to_print)

            pstar = np.zeros((2,1))

            for j in range(2):
                
                fake_data.observed_labels = np.append(data.observed_labels,j)
                fake_data.train_indices = fake_train_ind

                fake_predictions = model.predict(fake_data,fake_test_ind)

                #if len(data.train_indices)==2:
                #import pdb; pdb.set_trace()
                #np.savetxt("probabilities"+str(i)+str(j), fake_predictions)

                #p_max = np.amax(probabilities[1:fake_predictions.size+1])
                #p_max = np.amax(probabilities[0:fake_predictions.size+1])
                #if budget ==98:
                #    import pdb; pdb.set_trace()
                p_max = np.amax(p)

                pstar[1-j][0] = max(p_max, np.amax(fake_predictions))+j+sum(data.observed_labels)

                #pstar[1-j][0] = max(p_max, np.amax(fake_predictions))
                
            expected_utilities[i]= np.matmul(probabilities_including_negative[reverse_ind[test_indices[i]]],pstar)
            #print(pstar)
        

        #two_step_utilities = np.size(data.train_indices) + expected_utilities
        
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
        
        return expected_utilities
        
class ENS(Utility):

    def __init__(self):
        self.selector = UnlabelSelector()
        
        pass


    def knnbound(self, model, data, points, test_indices, budget):
            
        observed_labels = np.asarray(data.observed_labels)
        train_indices = np.asarray(data.train_indices)
        mask = observed_labels == 1
        sparseMatrixColumnIndicesPos = train_indices[mask].astype(int)
        sparseMatrixColumnIndicesNeg = train_indices[~mask].astype(int)
        end = model.weight_matrix.shape[0]
        
        
        
        knn_weights = model.similarities
        in_train = np.isin(model.ind, data.train_indices)
        knn_weights[in_train] =  0
        max_weight = np.max(knn_weights[test_indices.astype(int),0:min(end,len(data.train_indices)+1)] ,axis=1)

        successes = model.weight_matrix[test_indices,:][:,
                                        sparseMatrixColumnIndicesPos].toarray().sum(axis=1)
        
        failures = model.weight_matrix[test_indices,:][:,
                                        sparseMatrixColumnIndicesNeg].toarray().sum(axis=1)

        max_alpha = 0.1 + successes + max_weight

        min_beta = .9 + failures

        #begin pasted code

        bound = np.divide(max_alpha,(max_alpha+min_beta))

        if budget<=1:
            bound = np.amax(bound)
        else:
            bound[::-1].sort()
            bound = bound[0:budget]
            
        return bound



    def get_scores(self, model,data,test_indices,budget,points,probabilities):
        #mport pdb; pdb.set_trace()
        print("Starting score function")

        


        num_test = test_indices.size

        expected_utilities = np.zeros((num_test,1))

        compareValuesRemoveLater = np.zeros((num_test,2))

        unlabeled_ind = self.selector.filter(data, points)

        probs = np.zeros((points.size,1))

        probs[data.train_indices] = np.array(data.observed_labels, ndmin=2).T
        
        probs[test_indices] = probabilities

        all_probs = np.tile(probs, (16, 1))
        unlabeled_probs = all_probs[unlabeled_ind]

        reverse_ind = np.ones((points.size,1))*-1
        reverse_ind[unlabeled_ind]= np.arange(0,unlabeled_ind.size).reshape((unlabeled_ind.size,1))

        
        
        top_ind = np.argsort(-unlabeled_probs,axis = 0,kind = 'stable')

        cur_future_utility = np.sum(probabilities[0:budget,:])

        probabilities_including_negative = np.append(probabilities,1-probabilities,axis=1)

        fake_data = copy.deepcopy(data)

        unlabeledWeights = model.weight_matrix.tolil(copy=True)

        unlabeledWeights[data.train_indices,:]=0
        



        #begin pruning code

        prob_upper_bound = self.knnbound(model, data, points, test_indices, budget)

        future_utility_if_neg = np.sum(probabilities[0:budget])
                                #sum(success_probabilities(... top_ind(1:remaining_budget)));

        max_num_influence = model.k

        if max_num_influence>= budget:
            future_utility_if_pos = np.sum(prob_upper_bound[1:budget])
                                    #np.sum(prob_upper_bound[1:remaining_budget])

        else:
            tmp_ind = top_ind[0:(budget-max_num_influence)]
            #tmp_ind = top_ind[0:(budget-7)]
            future_utility_if_pos = np.sum(probabilities[tmp_ind]) + np.sum(prob_upper_bound[0:max_num_influence])

        
        future_utilities= np.zeros((1,2))
        future_utilities[0][0]=future_utility_if_pos
        future_utilities[0][1]=future_utility_if_neg

        #want sorted probabilities?

        #matches up until future_utility
        future_utility = np.matmul(probabilities_including_negative,future_utilities.T)

        #future_utility = future_utilities*probabilities_including_negative


        future_utility_bound=  future_utility - cur_future_utility


        #TODO: examine order of these things, I think that some are in order of 'test_ind' from matlab(second coefficient), while others are in order of test_ind in python(probabilities)
        upper_bound_of_score = probabilities + future_utility_bound
        
        #np.savetxt('bound.txt', utilities, fmt='%10.5f', delimiter=' ')
        #upper_bound_of_score = upper_bound_of_score[top_ind]
        
        pruned = np.zeros((num_test,1))
        current_max =-1
        
        #end pruning code
        utilities = np.zeros(test_indices.shape)
        #import pdb; pdb.set_trace()

        for i in range(num_test):

            if pruned[i]:
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

            utilities[i] = probabilities[i]+expected_utilities[i]-cur_future_utility
            import pdb; pdb.set_trace()
            if utilities[i]>current_max:
                current_max = utilities[i]
                pruned[upper_bound_of_score <= current_max] = True

            #compareValuesRemoveLater[i]=pstar.ravel()
        
        #TODO: change order that I'm calculating utilities so that I can do it inside for loop


        """
        if expected_utilities(i) > current_max
            current_max = expected_utilities(i);
            query_ind = test_ind(i);
            pruned(upper_bound_of_score < current_max) = true;
        """
        
            
        #utilities = probabilities + expected_utilities - cur_future_utility

        
        #np.savetxt('p.txt', p, fmt='%10.5f', delimiter=' ')
        #np.savetxt('q.txt', q, fmt='%10.5f', delimiter=' ')
        #np.savetxt('top_ind.txt', top_ind, fmt='%10.5f', delimiter=' ')
        
        #np.savetxt('expected_utilities.txt', utilities, fmt='%10.5f', delimiter=' ')
        #np.savetxt('bound.txt', utilities, fmt='%10.5f', delimiter=' ')
        #import pdb; pdb.set_trace()

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
            return test_indices[0]
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
    
        probabilities = self.model.predict(data,test_indices)
        argsort_ind = (-probabilities).argsort(axis=0)
        probabilities = probabilities[argsort_ind[:,0]]
        test_indices = test_indices[argsort_ind[:,0]]


        scores = self.utility.get_scores(self.model,data,test_indices,budget,points,probabilities)
        max_index = np.argmax(scores)

        chosen_x_index = test_indices[max_index]
        
        #print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.Utility.model.problem.y_train[chosen_x_index])

        #chosen_x = self.model.problem.x_pool[chosen_x_index]
        return chosen_x_index