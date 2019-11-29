"""Models for active search"""

import numpy as np
import copy
from active_search.createdata import genData
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KDTree

#TODO: remove this line after validating with matlab code
from scipy.io import savemat





class Data(object):
    """
    A class used to represent the observed state of the search space

    The `Data` class keeps track of observed indices and labels

    Attributes
    ----------
    train_indices : list
            List of indices of the `points` array that have been observed
    observed_labels : list
            List of labels corresponding to `train_indices`

    Methods
    -------
    new_observation(index, label)
        Appends the data of a new observation in the search space to its 
        member variables
    """

    def __init__(self, train_indices=None, observed_labels=None):
        train_indices = train_indices or []
        observed_labels = observed_labels or []
        if len(train_indices) != len(observed_labels):
            raise ValueError('Sizes do not match')
        self.train_indices = train_indices
        self.observed_labels = observed_labels

    def new_observation(self, index, label):
        """Appends the data of a new observation in the search space to its 
        member variables

        Parameters
        ----------
        index : int
            index of an element in `points` that was observed by the oracle
            function
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

        self.train_indices.append(index)
        self.observed_labels.append(label)


class Problem(object):
    """
    Problem captures all relevant information for an active
    learning problem: a pool of points and the oracle function.

    Attributes
    ----------
    points : list
            List of points in the search space

    Methods
    -------
    oracle_function(index)
        Simulates observing a point in the search space, e.g. by an
        experiment
    """

    def __init__(self):
        self.points = []

    def oracle_function(self, index):
        pass


class ToyProblem(Problem):
    """Simple implementation of the toy problem from

        Roman Garnett, Yamuna Krishnamurthy, Xuehan Xiong,
        Jeff G. Schneider, and Richard P. Mann;
        Bayesian Optimal Active Search and Surveying. ICML 2012

    - Additionally includes jitter for comparing results with matlab code
    - One-step policy chooses point 2 with probability gamma
    - Two-step policy chooses points 0 or 1 with probability
    epsilon for epsilon > gamma
    
    Attributes
    ----------
    labels_random : array_like
        Array of random labels corresponding to points in the search space
    labels_deterministic : array_like
        Array of deterministic labels corresponding to points in the search space
    points : array_like
        Array of points in the search space

    Methods
    -------
    oracle_function(index)
        Simulates observing a point in the search space, e.g. by an
        experiment
    """

    def __init__(self,jitter = False):
        self.labels_random, self.labels_deterministic, self.points = \
            genData(jitter)

    def oracle_function(self, index):
        """Simulates observing a point in the search space, e.g. by an
        experiment

        Parameters
        ----------
        index : int
            index of an element in `points` that was observed by the oracle
            function
        
        Returns
        ----------
        int
            the label corresponding to the index passed
        """

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

    Attributes
    ----------
    points : array_like
        Array of points in the search space
    probabilities : list
        Prior probability parameters, see class description
    labels : array_like
        Array of random and deterministic labels corresponding to points
        in the search space

    Methods
    -------
    oracle_function(index)
        Simulates observing a point in the search space, e.g. by an
        experiment
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
        """Simulates observing a point in the search space, e.g. by an
        experiment

        Parameters
        ----------
        index : int
            index of an element in `points` that was observed by the oracle
            function
        
        Returns
        ----------
        int
            the label corresponding to the index passed
        """

        return self.labels[index]


class Selector(object):
    """
    A base-class used to represent a method of filtering test points

    The `Selector` class is used to prune points in the search space

    Methods
    -------
    get_scores(self, model, data, test_indices,budget,points)
        calculates and returns n-step utility estimation for all elements of 
        test_indices
    """

    def __init__(self):
        pass

    def filter(self, pool):
        pass


class UnlabelSelector(Selector):
    """
    A class which selects unlabeled points from the search space

    Methods
    -------
    filter(self, data, points)
        Returns all elements in `points` that are not in 
        train_indices
    """

    def filter(self, data, points, *args):
        """Returns all elements in `points` that are not in train_indices

        Parameters
        ----------
        data : Data
            Data object for the Active Search problem
        points : array_like
            Numpy array containing all points in the search space
        Returns
        ----------
        test_indices : array_like
            all elements in `points` that are not in train_indices
        """

        test_indices = range(np.size(points, 0))
        test_indices = np.delete(test_indices, data.train_indices)
        return test_indices

class TwoStepPruningSelector(Selector):
    """
    A class which selects unlabeled points from the search space

    Attributes
    ----------
    selector : Selector
        an instance of the `UnlabelSelector` object

    Methods
    -------
    filter(self, data, points, model, policy, problem, budget)
        This method filters out suboptimal points
    """

    def __init__(self):
        self.selector = UnlabelSelector()
        pass

    
    def filter(self, data, points, model, policy, problem, budget):
        """filter out points in the search space whose expected utilities
        cannot possibly maximize our score.

        Parameters
        ----------
        data : Data
            Data object for the Active Search problem
        points : array_like
            Numpy array containing all points in the search space
        model : Model
        policy : Policy
            An instance of policy to use. See policies.py
        problem : Problem
            The problem space to perform active search over
        
        Returns
        ----------
        test_indices : array_like
            all points in the search space which could possibly maximize the
            utility score
        """

        def expected_loss_lookahead(model,data, this_test_ind, test_indices, 
                                    probabilities, points):
            """Calculates expected increase in utility at every point from
            observing `this_test_ind`

            Parameters
            ----------
            data : Data
                Data object for the Active Search problem
            this_test_ind : int
                Index to simulate observing
            test_indices : array_like
                Input array of indices that correspond to `points`, must be
                dimensions nx1
            probabilities : array_like
                Prior probabilities    
            points : array_like
                Numpy array containing all points in the search space
            
            Returns
            ----------
            expected_utilities : array_like
                utility estimation for every point conditioned on the 
                simulated observation
            """

            true_index = test_indices[this_test_ind]
            loss_probabilities = probabilities[this_test_ind]
            print(loss_probabilities.shape)
            probabilities_including_negative = np.concatenate([
                loss_probabilities,1-loss_probabilities])

            fake_train_ind = np.append(data.train_indices,true_index)
            fake_test_ind = np.delete(test_indices,this_test_ind)

            pstar = np.zeros((2,1))

            for j in range(2):
                
                fake_observed_labels = np.append(data.observed_labels,j)
                fake_data = copy.deepcopy(data)
                fake_data.observed_labels = fake_observed_labels
                fake_data.train_indices = fake_train_ind
                    
                pstar[1-j][0] = np.amax(model.predict(
                    fake_data,fake_test_ind)) + j

            expected_utilities= np.matmul(probabilities_including_negative,
                pstar)
            
            return expected_utilities


        def knnbound(model, data, points, problem, test_indices, num_positives):
            """Calculates a bound on the probability of nay point being a
            target after being conditioned on addtional target point
            observations
            
            Parameters
            ----------
            data : Data
                Data object for the Active Search problem
            points : array_like
                Numpy array containing all points in the search space
            problem : Problem
                The problem space to perform active search over
            test_indices : array_like
                Input array of indices that correspond to `points`, must be
                dimensions nx1
            num_positives : int
                Number of target point observations 
            
            Returns
            ----------
            bound : array_like
                The maximum probability that any point can have of being a
                target, after further conditioning on (num_positives)
                additional target points
            """
            
            observed_labels = np.asarray(data.observed_labels)
            train_indices = np.asarray(data.train_indices)
            mask = observed_labels == 1
            sparseMatrixColumnIndicesPos = train_indices[mask].astype(int)
            sparseMatrixColumnIndicesNeg = train_indices[~mask].astype(int)


            max_weights = np.max(model.weight_matrix, axis = 1).A
            print(max_weights.shape)
            max_weight = np.amax(max_weights[test_indices.astype(int)])

            successes = model.weight_matrix[test_indices,:][:,
                            sparseMatrixColumnIndicesPos].toarray().sum(axis=1)
            
            failures = model.weight_matrix[test_indices,:][:,
                            sparseMatrixColumnIndicesNeg].toarray().sum(axis=1)

            max_alpha = 0.1 + successes + num_positives * max_weight

            min_beta = .9 + failures

            bound = np.amax(np.divide(max_alpha,(max_alpha+min_beta)))
            print("bound is:",bound)
            
            return bound

        
        unlabeled_ind = self.selector.filter(data, points)

        test_indices = unlabeled_ind
        
        probabilities = model.predict(data,unlabeled_ind)
        if budget ==1:
            test_indices = test_indices[np.argmax(probabilities)]
            return test_indices

        highest_prob_test_index = np.argmax(probabilities)
        highest_prob_test_ind = test_indices[highest_prob_test_index]

        p_prime = expected_loss_lookahead(model,data,highest_prob_test_index,
            test_indices,probabilities, points)

        print("p_prime: ",p_prime)

        p_star_zero = knnbound(model, data, points, problem, test_indices, 0)
        p_star_one = knnbound(model, data, points, problem, test_indices, 1)
        optimal_lower_bound = np.divide((p_prime - p_star_zero),
            (p_star_one-p_star_zero+1))
        
        print("optimal_lower_bound:", optimal_lower_bound)

        true_bound = min(optimal_lower_bound, probabilities[
                                                highest_prob_test_index])


        test_ind_mask = probabilities>=true_bound
        
        test_ind_mask = np.squeeze(np.asarray(test_ind_mask))

        test_indices = test_indices[test_ind_mask]
        print("test_ind:",test_indices)
        #sys.exit()

        return test_indices

class ENSPruningSelector(Selector):

    def __init__(self):
        self.selector = UnlabelSelector()
        pass

    
    def filter(self, data, points, model, policy, problem, budget):
        """filter out points in the search space whose expected utilities
        cannot possibly maximize our score.

        Parameters
        ----------
        data : Data
            Data object for the Active Search problem
        points : array_like
            Numpy array containing all points in the search space
        model : Model
        policy : Policy
            An instance of policy to use. See policies.py
        problem : Problem
            The problem space to perform active search over
        
        Returns
        ----------
        test_indices : array_like
            all points in the search space which could possibly maximize the
            utility score
        """

        def expected_loss_lookahead(model,data, this_test_ind, test_indices, 
                                    probabilities, points):
            """Calculates expected increase in utility at every point from
            observing `this_test_ind`

            Parameters
            ----------
            data : Data
                Data object for the Active Search problem
            this_test_ind : int
                Index to simulate observing
            test_indices : array_like
                Input array of indices that correspond to `points`, must be
                dimensions nx1
            probabilities : array_like
                Prior probabilities    
            points : array_like
                Numpy array containing all points in the search space
            
            Returns
            ----------
            expected_utilities : array_like
                utility estimation for every point conditioned on the 
                simulated observation
            """

            true_index = test_indices[this_test_ind]
            loss_probabilities = probabilities[this_test_ind]
            #import pdb; pdb.set_trace()
            probabilities_including_negative = np.zeros((1,2))
            probabilities_including_negative[0][0]=loss_probabilities
            probabilities_including_negative[0][1]=1-loss_probabilities

            fake_train_ind = np.append(data.train_indices,true_index)
            fake_test_ind = np.delete(test_indices,this_test_ind)

            pstar = np.zeros((2,1))

            for j in range(2):
                
                fake_observed_labels = np.append(data.observed_labels,j)
                fake_data = copy.deepcopy(data)
                fake_data.observed_labels = fake_observed_labels
                fake_data.train_indices = fake_train_ind
                    
                pstar[1-j][0] = np.amax(model.predict(fake_data,
                                                        fake_test_ind)) + j

            expected_utilities= np.matmul(probabilities_including_negative,
                                                                        pstar)
            
            return expected_utilities


        def knnbound(model, data, points, problem, test_indices, num_positives):
            """Calculates a bound on the probability of nay point being a
            target after being conditioned on addtional target point
            observations
            
            Parameters
            ----------
            data : Data
                Data object for the Active Search problem
            points : array_like
                Numpy array containing all points in the search space
            problem : Problem
                The problem space to perform active search over
            test_indices : array_like
                Input array of indices that correspond to `points`, must be
                dimensions nx1
            num_positives : int
                Number of target point observations 
            
            Returns
            ----------
            bound : array_like
                The maximum probability that any point can have of being a
                target, after further conditioning on (num_positives)
                additional target points
            """

            observed_labels = np.asarray(data.observed_labels)
            train_indices = np.asarray(data.train_indices)
            mask = observed_labels == 1
            sparseMatrixColumnIndicesPos = train_indices[mask].astype(int)
            sparseMatrixColumnIndicesNeg = train_indices[~mask].astype(int)


            max_weights = np.max(model.weight_matrix, axis = 1).A
            print(max_weights.shape)
            max_weight = np.amax(max_weights[test_indices.astype(int)])

            successes = model.weight_matrix[test_indices,:][:,
                            sparseMatrixColumnIndicesPos].toarray().sum(axis=1)
            
            failures = model.weight_matrix[test_indices,:][:,
                            sparseMatrixColumnIndicesNeg].toarray().sum(axis=1)

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

        #p_star_zero = knnbound(model, data, points, problem, test_indices, 0)
        p_star_one = knnbound(model, data, points, problem, test_indices, 1)
        top_prob_sum = np.sum(probabilities.argsort()[-budget:][::-1])

        
        optimal_lower_bound = np.divide((p_prime - top_prob_sum),(1+ budget*p_star_one-top_prob_sum))
        
        print("optimal_lower_bound:", optimal_lower_bound)

        import pdb; pdb.set_trace()

        true_bound = min(optimal_lower_bound, probabilities[highest_prob_test_index])
        print("true_bound: ", true_bound)
        test_ind_mask = probabilities>=true_bound
        
        test_ind_mask = np.squeeze(np.asarray(test_ind_mask))

        test_indices = test_indices[test_ind_mask]
        print("test_ind:",test_indices)
        #sys.exit()

        return test_indices

class Model(object):
    """
    A base-class used to represent an assumption about the search space,
    such as clustering pattern or a random distribution

    Methods
    -------

    predict(self, data, test_indices)
        predicts probability of points in test_indices of being targets based
        on model definition and corresponding assumptions

    """

    def __init__(self):
        pass

    def predict(self, data, test_indices):
        pass


class RandomModel(Model):
    """
    A class which outputs random probability estimations

    Methods
    -------
    predict(self, data, test_indices)
        Outputs random probability estimates for the points in test_indices
    """

    def predict(self, data, test_indices):
        """Outputs random probability estimates for the points in test_indices

        Parameters
        ----------
        data : Data
            Data object for the Active Search problem
        test_indices : array_like
            Input array of indices that correspond to `points`, must be
            dimensions nx1
        
        Returns
        ----------
        array_like
            nx1 random utility estimations
        """

        # returns a nx1 array of random values between 0 and 1
        n = len(test_indices)
        return np.random.random_sample((n, 1))


class KnnModel(Model):
    """
    A class which outputs probability estimations based on the clustering
    assumption

    Attributes
    ----------
    tree : KDTree
        a tree of
    dist : array_like
        each entry gives the list of distances to the neighbors of the
        corresponding point
    ind : array_like
        each entry gives the list of indices of neighbors of the corresponding
        point
    similarities : array_like
        reciprocal of dist
    weight_matrix : csr_matrix
        Compressed sparse row matrix, used to reduce computational space
    k : int
        The maximum number of times that any point in the weight matrix
        is other points' neighbor

    Methods
    -------
    predict(self, data, test_indices)
        Outputs probability estimates for the points in test_indices based on
        the cluster assumption
    """

    def __init__(self, problem, k=50):
        n = np.size(problem.points, 0)

        self.tree = KDTree(problem.points)
        self.dist, self.ind = self.tree.query(problem.points, k=k + 1)
        
        self.dist = self.dist[:, 1:]
        self.ind = self.ind[:, 1:]
        self.similarities = np.reciprocal(self.dist)
        
        values = self.similarities.flatten()

        row = np.kron(np.arange(0, n), np.ones(((1, k)))).flatten()
        column = self.ind.flatten()

        self.weight_matrix = csr_matrix(
            (values, (row, column)), shape=(n, n)
        )
        self.k = np.amax(np.sum(self.weight_matrix>0,axis=0))

    def predict(self, data, test_indices):
        """Outputs probability estimates for the points in test_indices based
        on the cluster assumption

        Parameters
        ----------
        data : Data
            Data object for the Active Search problem
        test_indices : array_like
            Input array of indices that correspond to `points`, must be
            dimensions nx1
        
        Returns
        ----------
        predictions : array_like
            nx1 utility estimation based on nearest neighbor assumption
        """

        # returns probability estimate, Pr(y=1|x,D)
        # equation based off of Bayesian Optimal Active Search and Surveying, Garnett et al. 2012 (Equation 7)
        gamma = 0.1
        
        observed_labels = np.asarray(data.observed_labels)
        train_indices = np.asarray(data.train_indices)


        mask = (observed_labels == 1)
        sparseMatrixColumnIndicesPos = train_indices[mask].astype(int)
    
        positiveSum = self.weight_matrix[test_indices,:][:,sparseMatrixColumnIndicesPos].toarray().sum(axis=1)

        numerator = gamma + positiveSum

        sparseMatrixColumnIndicesNeg = train_indices[~mask].astype(int)

        negativeSum = self.weight_matrix[test_indices,:][:,sparseMatrixColumnIndicesNeg].toarray().sum(axis=1)
        denominator = 1 + positiveSum + negativeSum
        
        predictions = numerator / denominator

        predictions = predictions.reshape((-1, 1))

        return predictions
