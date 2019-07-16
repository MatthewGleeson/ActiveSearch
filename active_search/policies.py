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

        #TODO: Ask Gustavo if I need this if statement, when will the budget
        #      ever be 0?
        #budget =1 
        num_test = test_indices.size
        num_points = np.size(points,0)
        
        current_found = np.sum(data.observed_labels)
        if budget <= 0:
            expected_utilities = current_found*np.ones((num_test, 1))
            return expected_utilities
 
        unlabeled_ind = self.selector.filter(data, points)
        probabilities = model.predict(data,unlabeled_ind)

        probabilities_including_negative = np.append(probabilities,1-probabilities,axis=1)

        #use numpy take instead of fancy indexing!!! It's faster

        reverse_ind = np.zeros((num_points,1)).astype(int)
        
        np.put(reverse_ind,unlabeled_ind,range(unlabeled_ind.size))

        #reverse_ind[unlabeled_ind]=range(unlabeled_ind.size)
        
        #np.savetxt('twostep.txt', reverse_ind, delimiter=' ',fmt='%1.3f')

        if budget == 1:
            expected_utilities = current_found + np.take(probabilities,np.take
                            (reverse_ind,test_indices.astype(int)))
            return expected_utilities

        #note that this differs from matlab, this version only contains 
        #   probabilities of indices that haven't been examined yet
        
        #np.savetxt('probabilities.txt', -probabilities, delimiter=' ',fmt='%1.3f')
       
        #top_ind = probabilities.argsort()[::-1]
        #import pdb; pdb.set_trace()

        top_ind = np.argsort(-probabilities,axis = 0)
        
        #np.savetxt('top_ind.txt', top_ind, delimiter=' ',fmt='%1.3f') 
        budget = 1
        weights_trimmed = weights.copy()
        weights_trimmed[data.train_indices,:]=0

        #ignore train indices weights

        expected_utilities=np.zeros((num_test,1))
        for i in range(num_test):
            this_test_ind = test_indices[i].astype(int)

            fake_train_ind = np.append(data.train_indices,this_test_ind)


            #problem: knn.ind = a list of the 50 closest indices, not 
            #fake_test_ind: list of this_test_ind's neighbors that haven't been examined yet

            #csr is very efficient for row slicing, do it somewhat similar to how matlab code does it

            #TODO: error in the knn predict function, could the two-step be modifying some aspect
            #      of the data? make sure that we're just making copies

            #start by printing the mask with 1-step vs 2-step, and other aspects of data as well
            
            fake_test_ind = np.nonzero(weights_trimmed[:,this_test_ind])[0]
            #print("FAKE TEST IND",fake_test_ind)

            p = np.copy(probabilities)
            np.put(p,reverse_ind[this_test_ind],0)
            np.put(p,reverse_ind[fake_test_ind],0)
            #p[reverse_ind[this_test_ind]]=0

            
            #np.savetxt('twostep.txt', fake_test_ind, delimiter=' ',fmt='%1.3f')

            #if all of this indice's neighbors have been examined already:
            if fake_test_ind.size == 0:
                #the top (#budget) probability points
                top_bud_ind = top_ind[range(budget)]

                if np.take(reverse_ind,this_test_ind) in top_bud_ind:
                    top_bud_ind = top_ind[range(budget+1)]
                
                baseline = np.sum(np.take(p,top_bud_ind))

                expected_utilities[i]=current_found+np.take(probabilities,np.take
                            (reverse_ind,this_test_ind))+baseline
                continue
            
            #otherwise, do this:

            fake_utilities = np.zeros((2,1))
            for fake_label in range(2):
                fake_observed_labels = np.append(data.observed_labels,fake_label)

                fake_data = copy.deepcopy(data)
                fake_data.observed_labels = fake_observed_labels
                fake_data.train_indices = fake_train_ind

                fake_probabilities =  model.predict(fake_data,fake_test_ind)


                #TODO: is the [::-1] correct?
                q = np.sort(fake_probabilities)[::-1]

                #TODO: top_ind in the matlab code is actually a sorted version of p!!!
                
                
                #print("current_found",current_found)
                #print("fake_label",fake_label)
                merge_sorted = merge_sort(p, q, top_ind, budget)
                #print("merge_sorted",merge_sorted)
                fake_utilities[fake_label]=current_found + fake_label + merge_sorted
                
                
                #should be probabilities of this point being positive or negative multiplied by utilities 
                #  if it's positive or negative, then summed
            #np.savetxt('probabilities.txt', probabilities_including_negative, delimiter=' ',fmt='%1.3f')
            #print("fake_utilities",fake_utilities)
            #print("expected utilities shape: ",expected_utilities.shape)
            #print("fake_utilities shape: ",fake_utilities.shape)
            #print("first value:",np.take(probabilities_including_negative,np.take(reverse_ind,this_test_ind),axis = 0).shape)

            multiply1 = np.take(probabilities_including_negative,np.take(reverse_ind,this_test_ind),axis = 0)
            #print("multiply1",multiply1)
            #print(fake_utilities)
            expected_utilities[i]=np.matmul(multiply1,fake_utilities)




        np.savetxt('expected_utilities.txt', expected_utilities, delimiter=' ',fmt='%1.3f')
        return expected_utilities

        #begin deepcopy attempt!
        #for possible_point in test_indices:
        #    updatedDataNeg = copy.deepcopy(data)
        #    updatedDataNeg.new_observation()

        #updatedDataPos = copy.deepcopy(data)
        




        pass
        
        



        # for a in model.problem.basicSelector():
        #      updatedModelPos = problem.update(x,1)

        #updatedModelPos = problem.update(x,1)
        #updatedModelNeg = problem.update(x,0)

        # secondTerm = [p*OneStep(x_new,updatedModelPos)+(1-p)OneStep(x_new,updatedModelNeg) for x-new in x-pool/x].argmax

        # do I need to calculate x-pool/x here too? if so, then I should probably functionalize it in models.py as a function of the problem
        # return 1 * model.predict()
        # return probability estimation


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
