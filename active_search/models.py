"""Models for active search"""

import numpy as np
from active_search.createdata import genData
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KDTree


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

        # 
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
        predictions = np.zeros((len(test_indices), 1))
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

        positiveSum = self.weight_matrix[:,
                                         sparseMatrixColumnIndicesPos].sum(axis=1)

        numerator = gamma + positiveSum

        sparseMatrixColumnIndicesNeg = train_indices[~mask].astype(int)

        negativeSum = self.weight_matrix[:,
                                         sparseMatrixColumnIndicesNeg].sum(axis=1)
        denominator = 1 + positiveSum + negativeSum

        predictions = numerator / denominator

        predictions = np.delete(predictions,train_indices,axis=0)
        
        #begin debug code

        
        #print(np.where(predictions == diffs ,predictions))

        # row-wise sum up the elements in self.weight_matrix whose indices are in (self.problem.train_ind minus
        #np.savetxt('predictions.txt', predictions, delimiter=' ')
        #end debug code
        return predictions
