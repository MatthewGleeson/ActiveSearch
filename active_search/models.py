"""Models for active search"""

import numpy as np
import copy
from active_search.createdata import genData
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KDTree

#TODO: remove this line after validating with matlab code
from scipy.io import savemat
import sys




class Data(object):
    """
        Observed dataset
    """

    def __init__(self, train_indices=None, observed_labels=None):
        train_indices = train_indices or []
        observed_labels = observed_labels or []
        if len(train_indices) != len(observed_labels):
            raise ValueError('Sizes do not match')
        self.train_indices = train_indices
        self.observed_labels = observed_labels

    def new_observation(self, index, label):
        self.train_indices.append(index)
        self.observed_labels.append(label)


class Problem(object):
    """
        Problem captures all relevant information for an active
        learning problem: a pool of points and the oracle
        function.
    """

    def __init__(self):
        self.points = []

    def oracle_function(self, index):
        pass


class ToyProblem(Problem):
    """
        Toy problem from genData
    """

    def __init__(self):
        self.labels_random, self.labels_deterministic, self.points = genData()

    def oracle_function(self, index):
        return self.labels_deterministic[index]


class ToyProblemGarnett2012(Problem):
    """
        Simple implementation of the toy problem from

            Roman Garnett, Yamuna Krishnamurthy, Xuehan Xiong,
            Jeff G. Schneider, and Richard P. Mann;
            Bayesian Optimal Active Search and Surveying. ICML 2012

        - One-step policy chooses point 2 with probability gamma
        - Two-step policy chooses points 0 or 1 with probability
        epsilon for epsilon > gamma
    """

    def __init__(self, epsilon=0.3, gamma=0.8):
        assert gamma > epsilon
        self.points = np.random.randn(3, 2)
        self.probabilities = [epsilon, epsilon, gamma]
        n = 1
        dependent_labels = np.random.binomial(n, epsilon, 1)
        independent_label = np.random.binomial(n, gamma, 1)
        self.labels = [dependent_labels, dependent_labels, independent_label]

    def oracle_function(self, index):
        return self.labels[index]


class Selector(object):
    def __init__(self):
        pass

    def filter(self, pool):
        pass


class UnlabelSelector(Selector):
    def filter(self, data, points):
        test_indices = range(np.size(points, 0))
        test_indices = np.delete(test_indices, data.train_indices)

        return test_indices

class TwoStepPruningSelector(Selector):

    def __init__(self):
        self.selector = UnlabelSelector()
        pass

    
    def filter(self, model, policy, data, points, problem):
        
        def expected_loss_lookahead(model,data, this_test_ind, test_indices, 
                                    probabilities, points):

            true_index = test_indices[this_test_ind]
            loss_probabilities = probabilities[this_test_ind]
            probabilities_including_negative = np.append(loss_probabilities,
                                                    1-loss_probabilities,axis=1)

            fake_train_ind = np.append(data.train_indices,true_index)
            fake_test_ind = np.delete(test_indices,this_test_ind)

            pstar = np.zeros((2,1))

            for j in range(2):
                
                fake_observed_labels = np.append(data.observed_labels,j)
                fake_data = copy.deepcopy(data)
                fake_data.observed_labels = fake_observed_labels
                fake_data.train_indices = fake_train_ind
                    
                pstar[1-j][0] = np.amax(model.predict(fake_data,fake_test_ind)) + j

            expected_utilities= np.matmul(probabilities_including_negative,pstar)
            
            return expected_utilities


        def knnbound(model, data, points, problem, test_indices, num_positives):
            
            observed_labels = np.asarray(data.observed_labels)
            train_indices = np.asarray(data.train_indices)
            mask = observed_labels == 1
            sparseMatrixColumnIndicesPos = train_indices[mask].astype(int)
            sparseMatrixColumnIndicesNeg = train_indices[~mask].astype(int)


            max_weights = np.max(model.weight_matrix, axis = 1).A
            print(max_weights.shape)
            max_weight = np.amax(max_weights[test_indices.astype(int)])

            successes = model.weight_matrix[test_indices,:][:,
                                         sparseMatrixColumnIndicesPos].sum(axis=1)
            
            failures = model.weight_matrix[test_indices,:][:,
                                         sparseMatrixColumnIndicesNeg].sum(axis=1)

            max_alpha = 0.1 + successes + num_positives * max_weight

            min_beta = .9 + failures

            #from knn code

            bound = np.amax(np.divide(max_alpha,(max_alpha+min_beta)))
            print("bound is:",bound)
            
            return bound

        #test_indices = range(np.size(points, 0))
        #test_indices = np.delete(test_indices, data.train_indices)
        
        unlabeled_ind = self.selector.filter(data, points)

        test_indices = unlabeled_ind
        
        probabilities = model.predict(data,unlabeled_ind)

        highest_prob_test_index = np.argmax(probabilities)
        highest_prob_test_ind = test_indices[highest_prob_test_index]

        #dummy_test_ind = np.array([highest_prob_test_ind])

        p_prime = expected_loss_lookahead(model,data,highest_prob_test_index,test_indices,probabilities, points)

        print("p_prime: ",p_prime)

        p_star_zero = knnbound(model, data, points, problem, test_indices, 0)
        p_star_one = knnbound(model, data, points, problem, test_indices, 1)
        optimal_lower_bound = np.divide((p_prime - p_star_zero),(p_star_one-p_star_zero+1))
        
        print("optimal_lower_bound:", optimal_lower_bound)

        true_bound = min(optimal_lower_bound, probabilities[highest_prob_test_index])
        test_ind_mask = probabilities>true_bound
        
        test_ind_mask = np.squeeze(np.asarray(test_ind_mask))

        test_indices = test_indices[test_ind_mask]
        print("test_ind:",test_indices)
        #sys.exit()

        return test_indices


class Model(object):
    def __init__(self):
        pass

    def predict(self):
        pass

    def update(self):
        pass

    def plot(self):
        pass


class RandomModel(Model):
    
    def predict(self, data, test_indices):
        # returns a nx1 array of random values between 0 and 1
        n = len(test_indices)
        return np.random.random_sample((n, 1))


class KnnModel(Model):
    def __init__(self, problem, k=50):
        n = np.size(problem.points, 0)

        self.tree = KDTree(problem.points)
        self.dist, self.ind = self.tree.query(problem.points, k=k + 1)

        self.dist = self.dist[:, 1:]
        self.ind = self.ind[:, 1:]
        self.similarities = np.reciprocal(self.dist)
    
        values = self.similarities.flatten()
        # row = np.tile(np.arange(0,n), (k,1)).T.flatten()
        row = np.kron(np.arange(0, n), np.ones(((1, k)))).flatten()
        column = self.ind.flatten()

        self.weight_matrix = csr_matrix(
            (values, (row, column)), shape=(n, n)
        )

        #TODO: remove below line after done testing
        #savemat('weights', {'weights':self.weight_matrix})
        #print("nearest neighbors:", self.ind[24])
        #savemat('ind', {'ind':self.ind.T})
        #savemat('similarities', {'similarities':self.similarities.T})
        


        # sorts indices by lowest distance value, stores the k-lowest for each point
        # self.knn = np.argsort(self.dist_matrix, axis=1)[:,1:k+1] # nxk matrix
        # print("knns: ",self.knn)
        # self.knn[x] returns array of k closest points to point x

    def predict(self, data, test_indices):
        # returns probability estimate, Pr(y=1|x,D)
        # TODO: Add weights based on distance to other points
        # equation based off of Bayesian Optimal Active Search and Surveying, Garnett et al. 2012 (Equation 7)
        gamma = 0.1
        # create empty array to return

        # want to sum up elements in self.dist,
        #toTest = [i for i in enumerate(test_indices)]
        #print(toTest)
        #predictions = np.zeros((len(test_indices), 1))
        # print("train indices:",self.problem.train_ind)
        # print("neighbor indices:",self.ind[test_ind[0]])
        # predictions[:]=np.nan
        # iterate over all x values in x_pool

        # probabilities = gamma + np.sum()

        # np.multiply(self.problem.train_ind,self.problem.observed_labels)

        # sparseMatrixColumnIndices = np.take(self.problem.train_ind,np.nonzero(self.problem.labels))
        # sparseMatrixColumnIndicesPos = self.problem.train_ind[np.nonzero(self.problem.labels)]

        # make sure that we always use np arrays
        observed_labels = np.asarray(data.observed_labels)
        train_indices = np.asarray(data.train_indices)


        mask = observed_labels == 1
        sparseMatrixColumnIndicesPos = train_indices[mask].astype(int)
        # sparseMatrixColumnIndicesPos=sparseMatrixColumnIndicesPos.astype(int)

        positiveSum = self.weight_matrix[test_indices,:][:,sparseMatrixColumnIndicesPos].sum(axis=1)

        numerator = gamma + positiveSum

        sparseMatrixColumnIndicesNeg = train_indices[~mask].astype(int)

        negativeSum = self.weight_matrix[test_indices,:][:,sparseMatrixColumnIndicesNeg].sum(axis=1)
        denominator = 1 + positiveSum + negativeSum

        predictions = numerator / denominator

        #predictions = np.delete(predictions,train_indices,axis=0)
        
        #begin debug code

        
        #print(np.where(predictions == diffs ,predictions))

        # row-wise sum up the elements in self.weight_matrix whose indices are in (self.problem.train_ind minus
        #np.savetxt('predictions.txt', predictions, delimiter=' ')
        #end debug code
        return predictions
