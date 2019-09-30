"""Policies for active search"""
#from models import *
import numpy as np
import copy
from active_search.models import UnlabelSelector
import sys
from scipy.sparse import find
#import bottleneck as bn
#from cppFolder import merge_sort
import ctypes
import os

#TODO: remove when verify ENS working correctly:
from os.path import dirname, join as pjoin
import scipy.io as sio


MERGE_SORT_FILE_NAME = 'merge_sort.so'


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
            sum_to_return = sum_to_return + np.take(p,np.take(top_ind,j))
            while True:
                j= j+1
                if np.take(p,np.take(top_ind,j))!=0:
                    break
                
        else:
            sum_to_return = sum_to_return + np.take(q,k)
            k=k+1
        
        i = i+1
    
    while i<budget:
        sum_to_return = sum_to_return + np.take(p,np.take(top_ind,j))
        while True:
                j= j+1
                if np.take(p,np.take(top_ind,j))!=0:
                    break
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
        expected_utilities = model.predict(data, test_indices)
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

        unlabeledWeights = model.weight_matrix



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
        
        #TODO: do I need to do something if budget <=1?
        budget = 99
        
        num_test = test_indices.size


        np.savetxt('test_indices.txt', test_indices, fmt='%10.5f', delimiter=' ')

        expected_utilities = np.zeros((num_test,1))

        compareValuesRemoveLater = np.zeros((num_test,2))

        unlabeled_ind = self.selector.filter(data, points)

        probabilities = model.predict(data,test_indices)

        probs = np.zeros((points.size,1))

        probs[data.train_indices] = np.array(data.observed_labels, ndmin=2).T
        
        probs[test_indices] = probabilities
        #np.savetxt('probs.txt', probs, fmt='%10.5f', delimiter=' ')
        #import pdb; pdb.set_trace()
        all_probs = np.tile(probs, (16, 1))
        unlabeled_probs = all_probs[unlabeled_ind]


        reverse_ind = np.ones((points.size,1))*-1
        reverse_ind[unlabeled_ind]= np.arange(0,unlabeled_ind.size).reshape((unlabeled_ind.size,1))

        #np.savetxt('probs.txt', reverse_ind, fmt='%10.5f', delimiter=' ')
        
        
        argsort_ind = (-probabilities).argsort(axis=0)

        probabilities_sorted = probabilities[argsort_ind]
        
        #top_ind = np.argsort(unlabeled_probs,axis = 0)
        #top_ind = top_ind[::-1]
        top_ind = np.argsort(-unlabeled_probs,axis = 0,kind = 'stable')
        #np.savetxt('top_ind.txt', top_ind, fmt='%10.5f', delimiter=' ')
        #import pdb; pdb.set_trace()
        
        cur_future_utility = np.sum(probabilities_sorted[0:budget,:])
        #print("cur_future_utility: ",cur_future_utility)

        probabilities_including_negative = np.append(probabilities,1-probabilities,axis=1)
        #np.savetxt('probabilities_including_negative.txt', probabilities_including_negative, fmt='%10.5f', delimiter=' ')
        #TODO: unedit this line!
        fake_data = copy.deepcopy(data)

        unlabeledWeights = model.weight_matrix
        
        #TODO: if numbers aren't matching up, try loading the exact same weight matrix as the matlab program does
        #unlabeledWeights = sio.loadmat("matlab_weight_matrix.mat")
        #unlabeledWeights = unlabeledWeights['weights']

        unlabeledWeights[data.train_indices,:]=0
        
        for i in range(num_test):
            this_test_ind = test_indices[i].astype(int)

            fake_train_ind = np.append(data.train_indices,this_test_ind)
            
            #fake_test_ind = unlabeledWeights[:,this_test_ind]

            fake_test_ind = np.nonzero(unlabeledWeights[:,this_test_ind])[0]

            #if this_test_ind == 726:
                #print(fake_test_ind)
                #np.savetxt('fake_test_ind.txt', fake_test_ind, fmt='%10.5f', delimiter=' ')
            
            

            #print(unlabeledWeights.shape)
            

            pstar = np.zeros((2,1))

            p = unlabeled_probs.copy()


            #p(reverse_ind(fake_test_ind)) = 0;
    
            #np.concatenate((fake_test_ind, this_test_ind), axis=0)


            a = np.take(reverse_ind,this_test_ind)
            b = np.take(reverse_ind,fake_test_ind)
            c = np.take(reverse_ind,test_indices)
            
            a = a[a>=0].astype(int)
            b = b[b>=0].astype(int)
            c = c.astype(int)
            
            #print("YO",this_test_ind)
            #print(a)
            p[a]=0
            p[b]=0

            this_test_probs = unlabeled_probs[c]
            this_test_probs = np.append(this_test_probs,1-this_test_probs,axis=1)

            for j in range(2):
                
                fake_data.observed_labels = np.append(data.observed_labels,j)
                #fake_data.observed_labels[0]=0
                
                fake_data.train_indices = fake_train_ind
                
                #print(fake_data.observed_labels)
                #pstar[1-j][0] = np.sum(model.predict(fake_data,fake_test_ind).argsort()[-budget:][::-1])



                fake_predictions = model.predict(fake_data,fake_test_ind)

                #print(fake_train_ind)

                #print("predictions shape:", fake_predictions.shape)
                
                np.savetxt('fake_predictions.txt', fake_predictions, fmt='%10.5f', delimiter=' ')


                #merge_sort_file = ctypes.CDLL('/Users/matthew/Documents/garnettLabFolder/pythonCodeBase/merge_sort_file.so')
                #print(fake_predictions.shape)
                q = np.sort(fake_predictions, axis=0)

                #q = np.sort(fake_predictions)

                q = q[::-1]
                
                if this_test_ind == 726 and (j ==0):
                    print(fake_data.observed_labels)
                    np.savetxt('fake_data_observed_labels.txt', fake_data.observed_labels, fmt='%10.5f', delimiter=' ')
                    np.savetxt('fake_test_ind.txt', fake_test_ind, fmt='%10.5f', delimiter=' ')
                    np.savetxt('q_with_negative_obs.txt', q, fmt='%10.5f', delimiter=' ')
                    np.savetxt('p_with_negative_obs.txt', q, fmt='%10.5f', delimiter=' ')
                    #np.savetxt('q_positive:'+j+'.txt', q, fmt='%10.5f', delimiter=' ')
                    #import pdb; pdb.set_trace()
            

                #q = np.sort(fake_predictions)[::-1]
                #print(p.shape)
                #print(type(p.A1.ctypes.data))
                #a = p.A1
                #print(p.shape)
                #print(type(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
                #print(type())
                #print(type(p.A1))
                #a = p.A1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                #print(type(a))
                #pstar[1-j][0] = _merge_sort.merge_sort_fxn(a, q.A1.ctypes.data, np.ravel(top_ind).ctypes.data, budget, q.size)
                """
                if this_test_ind == 726:
                    np.savetxt('top_ind.txt', top_ind, fmt='%10.5f', delimiter=' ')
                    np.savetxt('p.txt', p, fmt='%10.5f', delimiter=' ')
                    #import pdb; pdb.set_trace()
                """
                #print(budget)

                pstar[1-j][0] = merge_sort(p,q,top_ind,99)
                if this_test_ind == 726 and (j ==0):
                    print(pstar[1-j][0])

                #print(pstar[1-j][0])
                #import pdb; pdb.set_trace()
                #TODO: below line is old line of code, revert to uncommenting it if cytpes doesn't work
                
                #fake_predictions = fake_predictions.flatten()

                #toSum = budget
                #if budget > fake_predictions.size:
                #    toSum = fake_predictions.size-1
                
                ##ind = np.argpartition(fake_predictions, -toSum,axis=1)[-toSum:]

                #toSum = -bn.partition(-fake_predictions, toSum)[:toSum]
                
                #pstar[1-j][0] = np.sum(toSum)

                #TODO: end old code, delete if not useful


                
            #print(probabilities_including_negative[i])

            #this_test_prob = unlabeled_probs(reverse_ind(this_test_ind), j);


            #c = np.take(reverse_ind,this_test_ind).astype(int)
            

            #this_test_prob = unlabeled_probs[c]

            #this_test_prob = unlabeled_probs[a]
            #this_test_prob = np.append(this_test_prob,1-this_test_prob,axis=1)

            #this_test_probs[i] = np.array([this_test_prob,1-this_test_prob])

            
            expected_utilities[i]= np.matmul(this_test_probs[i],pstar)
            compareValuesRemoveLater[i]=pstar.ravel()
            """
            delta_future_utility = future_utility - cur_future_utility(j);
            average_future_utility = average_future_utility + ...
            sample_weights(j)*delta_future_utility;
            """
            
            
        #TODO: This was the line that was uncommented!!!!
        #utilities = np.size(data.train_indices) + probabilities + expected_utilities
        utilities = probabilities + expected_utilities - cur_future_utility
        
        #np.savetxt('valueToCompare.txt', valueToCompare, fmt='%10.5f', delimiter=' ')
        #np.savetxt('utilities.txt', utilities, delimiter=' ')
        #sys.exit()



        #print("number of observed points:", np.size(data.train_indices))
        #print("probabilities: ", probabilities)
        #print("expected_utilities",expected_utilities)
        #

        #toSave = np.concatenate((test_indices.reshape(utilities.shape),compareValuesRemoveLater,expected_utilities,utilities), axis=1)
        np.savetxt('expected_utilities.txt', utilities, fmt='%10.5f', delimiter=' ')
        
        
        test_ind_size = test_indices.size
        test_indices_reshaped = test_indices.reshape((test_ind_size,1))
        

        
        concatenated = np.concatenate((test_indices_reshaped,  utilities), axis=1)
        
        import pdb; pdb.set_trace()


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

        """
        #scores = scores.flatten()
        test_ind_size = test_indices.size
        test_indices_reshaped = test_indices.reshape((test_ind_size,1))
        print("test_indices shape:", test_indices_reshaped.shape)
        print("scores shape:", scores.shape)
        concatenated = np.concatenate((test_indices_reshaped, scores), axis=1)
        np.savetxt('twoStepUtilities.txt', concatenated, fmt='%i', delimiter=' ')
        sys.exit()
        """

        



        max_index = np.argmax(scores)

        chosen_x_index = test_indices[max_index]
        
        #print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.Utility.model.problem.y_train[chosen_x_index])

        #chosen_x = self.model.problem.x_pool[chosen_x_index]
        return chosen_x_index
