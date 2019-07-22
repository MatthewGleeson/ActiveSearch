"""Policies for active search"""
#from models import *
import numpy as np
import copy
from active_search.models import UnlabelSelector


def merge_sort(p, q, top_ind, budget):
    n = q.size
    sum_to_return = 0
    i = 0
    j = 0
    while np.take(p,np.take(top_ind,j)-1)==0:
        
        j= j+1
    k = 0

    while i<budget and k<n:
        p_p_ind = np.take(p,top_ind[j]-1)
        if p_p_ind > np.take(q,k):
            sum_to_return = sum_to_return + p_p_ind
            while np.take(p,top_ind[j]-1)==0:
                j= j+1
        else:
            sum_to_return = sum_to_return + np.take(q,k)
            k=k+1
        
        i = i+1
    
    while i<budget:
        sum_to_return = sum_to_return + np.take(p,top_ind[j]-1)
        while np.take(p,top_ind[j]-1)==0:
                j= j+1
        i=i+1
    
    return sum_to_return


class Utility(object):
    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices,budget,points,weights):
        pass


class OneStep(Utility):

    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices,budget,points,weights):
        expected_utilities = model.predict(data, test_indices)
        return expected_utilities
        # return probability estimation


class TwoStep(Utility):

    def __init__(self):
        self.selector = UnlabelSelector()
        pass

    def get_scores(self, model,data,test_indices,budget,points,weights):

        #TODO: do I need to do something if budget <=1?
        
        #compute p
        num_test = test_indices.size
        expected_utilities = np.zeros((num_test,1))

        unlabeled_ind = self.selector.filter(data, points)
        probabilities = model.predict(data,unlabeled_ind)

        probabilities_including_negative = np.append(probabilities,1-probabilities,axis=1)
        if budget == 1500:
            np.savetxt('probabilities_including_negative.txt', probabilities_including_negative, delimiter=' ',fmt='%1.3f')
        for i in range(num_test):
            this_test_ind = test_indices[i].astype(int)

            fake_train_ind = np.append(data.train_indices,this_test_ind)
            
            fake_test_ind = np.delete(test_indices,i)

            pstar = np.zeros((2,1))

            for j in range(2):
            
                fake_observed_labels = np.append(data.observed_labels,j)

                fake_data = copy.deepcopy(data)
                fake_data.observed_labels = fake_observed_labels
                fake_data.train_indices = fake_train_ind
                
                pstar[1-j][0] = np.amax(model.predict(fake_data,fake_test_ind))+ j+ np.size(data.train_indices) #TODO: should I add "+j"?
                
            expected_utilities[i]= np.matmul(probabilities_including_negative[i],pstar)
        
        #if budget == 1500:
        #    np.savetxt('twostepfirstprobs.txt', expected_utilities, delimiter=' ',fmt='%1.3f')
        #    np.savetxt('nearest_neighbors',np.nonzero(weights[24,:]),delimiter=' ',fmt='%1.3f')
            
        return expected_utilities

        
        #for x in p:
            #p1 = prob(D u (x,1)) w/o x
            #p1* = max(p1)
            #p0 = prob(D u (x,0)) w/o x
            #p0* = max(p0)
            #expected_utilities[x] = p + p(p1*) + (1-p)p0*
        


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

    def choose_next(self, data, test_indices, budget,points,weights=False):

        scores = self.utility.get_scores(self.model,data,test_indices,budget,points,weights)

        max_index = np.argmax(scores)

        chosen_x_index = test_indices[max_index]
        
        #print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.Utility.model.problem.y_train[chosen_x_index])

        #chosen_x = self.model.problem.x_pool[chosen_x_index]
        return chosen_x_index
