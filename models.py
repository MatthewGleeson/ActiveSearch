"""Models for active search"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix




class problem(object):
    def __init__(self, x_pool):
        
        #self.x_train = np.empty(np.shape(x_pool)) # nxd
        
        #self.x_train[:]=np.nan
        #self.y_train = np.empty((np.size(x_pool,0),1))#nx1
        #self.y_train[:]=np.nan
          #nxd
          
        #x_train is nxd, everything starts at zeros and values get copied over from x_pool when they are observed
        #y_train is nx1, everything starts as zeros and values get filled in with the oracle function
        self.x_pool = x_pool
        self.train_ind = []
        self.observed_labels = []

    def newObservation(self,index,y):
        #self.x_train[index]=x
        #self.y_train[index]=y
        self.train_ind = np.append(self.train_ind,index)
        #print("new obsevation created. To train indices, appended index:",index)
        #print("train indices:",self.train_ind)
        self.observed_labels= np.append(self.observed_labels,y)
    
    def basicSelector(self):
        #most basic version of selector, returns list of unlabeled indices
        test_ind = range(np.size(self.x_pool,0))
        test_ind= np.delete(test_ind,self.train_ind)
        return test_ind



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
        k = 50
        #k=100
        self.problem = problem
        xs= problem.x_pool
        n = np.size(xs,0)
        #calculate euclidean distance between x matrix and itself
        #self.dist_matrix = euclidean_distances(xs,xs) # nxn matrix that stores distance between points

        self.tree = KDTree(xs)
        self.dist, self.ind = self.tree.query(xs, k=k+1)

        self.dist = self.dist[:,1:]
        self.ind = self.ind[:,1:]
        self.similarities = np.reciprocal(self.dist)
        print(self.ind[24])

        data = self.similarities.flatten()
        #row = np.tile(np.arange(0,n), (k,1)).T.flatten()
        row = np.kron(np.arange(0,n), np.ones(((1,k)))).flatten()
       
        
        column = self.ind.flatten()

        

        self.sparseWeightMatrix = csr_matrix((data,(row,column)),shape=(n,n)).toarray()


        np.savetxt("sparseMatrix.csv", self.sparseWeightMatrix ,fmt='%.2f', delimiter=",")
        #sorts indices by lowest distance value, stores the k-lowest for each point
        #self.knn = np.argsort(self.dist_matrix, axis=1)[:,1:k+1] # nxk matrix
        #print("knns: ",self.knn)
        #self.knn[x] returns array of k closest points to point x

    
    def predict(self):

        #returns probability estimate, Pr(y=1|x,D)



        #TODO: Add weights based on distance to other points
        #equation based off of Bayesian Optimal Active Search and Surveying, Garnett et al. 2012 (Equation 7)
        gamma = 0.1
        #create empty array to return

        #want to sum up elements in self.dist,

        test_ind = self.problem.basicSelector()
        predictions = np.zeros((np.size(test_ind,0),1))
        #print("train indices:",self.problem.train_ind)
        #print("neighbor indices:",self.ind[test_ind[0]])
        #predictions[:]=np.nan
        #iterate over all x values in x_pool

        #probabilities = gamma + np.sum()

        #np.multiply(self.problem.train_ind,self.problem.observed_labels)

        #sparseMatrixColumnIndices = np.take(self.problem.train_ind,np.nonzero(self.problem.labels))
        #sparseMatrixColumnIndicesPos = self.problem.train_ind[np.nonzero(self.problem.labels)]
        mask = self.problem.observed_labels == 1
        
        sparseMatrixColumnIndicesPos = self.problem.train_ind[mask].astype(int)
        #sparseMatrixColumnIndicesPos=sparseMatrixColumnIndicesPos.astype(int)
        
        print(sparseMatrixColumnIndicesPos)

        positiveSum = self.sparseWeightMatrix[:,sparseMatrixColumnIndicesPos].sum(axis=1)

        numerator = gamma + positiveSum


        sparseMatrixColumnIndicesNeg = self.problem.train_ind[~mask].astype(int)

        negativeSum = self.sparseWeightMatrix[:,sparseMatrixColumnIndicesNeg].sum(axis=1)
        denominator = 1+positiveSum+negativeSum
        

        
        predictions = numerator/denominator
        
        #import pdb; pdb.set_trace()

        #row-wise sum up the elements in self.sparseWeightMatrix whose indices are in (self.problem.train_ind minus 

        if self.problem.train_ind.size == 1:
            np.savetxt("numerator.csv", numerator,fmt='%.2f', delimiter=",")
            np.savetxt("predictions.csv", predictions,fmt='%.6f', delimiter=",")
            arrayOfPickers = [727,1111,1692,2203,690,1909,2208,2268,222,444,588,2008,278,393,695,1164,1511,1547,1769,2283,159,915,1497,2469,579,990,1823,2320,682,718,1117,1504,1883,1978,2005,2179,158,274,1538,1627,1712,2014,2098,2441,153,722,1625,2010, 70, 149]
            arrayOfPickers = np.asarray(arrayOfPickers)-1
            print("PICKERS",arrayOfPickers)
            nearestprobabilities= np.take(predictions,arrayOfPickers)
            np.savetxt("nearestprobs.csv", nearestprobabilities,fmt='%.6f', delimiter=",")
            nearestNumerator= np.take(numerator,arrayOfPickers)
            np.savetxt("nearestNumerator.csv", nearestNumerator,fmt='%.2f', delimiter=",")



        """
        for i in range(np.size(test_ind,0)):
            neighborIndices=self.ind[test_ind[i]]
            
            #list of indices that were nearest neighbors to this point
            #first, get list of indices in nearest neighbors that have been observed
            
            #predictions[i]= self.ind
            #indicesInception stores the row index of the elements in self.similarities and self.ind that we want
            #print(all3)
            #training indices with positive labels:
            postrainindex_index = np.where(self.problem.observed_labels == 1)[0]

            posTrainIndices = np.take(self.problem.train_ind,postrainindex_index)

            indicesInception_justPos = np.intersect1d(neighborIndices, posTrainIndices, assume_unique=True, return_indices=True)[1]

            indicesInception_posAndNeg = np.intersect1d(neighborIndices, self.problem.train_ind, assume_unique=True, return_indices=True)[1]


            #if indicesInception.size !=0:
            #    print(indicesInception)

            #The way that Shali did it was by making a matrix and using both sets of indices to pick the valuesd
            
            toSum = np.take(self.similarities[test_ind[i]], indicesInception_justPos)

            #should be summing only those with positive labels!!! currently summing distances of positive and negative labels,
            #  which is why it's only exploring near labeled points rather than POSITIVE labeled points

            numerator = gamma + np.sum(toSum)


            
            #denominator = 1+indicesInception_posAndNeg.size #<-paper implementation! TODO: this is wrong, not correct implementation of paper equation
            #I am summing over the number of labeled points, instead of the number of nearest neighbors
            
            #TODO: Decide between versions of denominator!!! Shali normalizes by dividing by the sum of all probabilities for the classes for each point
            denominator = .9+numerator #<-Shali's implementaion!
            #TODO: above is the line for the denominator according to shali's code

            #sum the items in self.similarities whose index is same as the items in self.ind that are also in train_ind


            #goal: sum up similarities of neighbors in train_indices(means they have been observed)

            #next, divide by 1+(# of neighbors in train_indices)

            predictions[i]=np.divide(numerator,denominator)
        """

        #stitched = np.concatenate((test_ind, predictions), axis=1)
        

        return predictions

        """
        for i in range(np.size(self.problem.x_pool,0)):
            if np.isnan(self.problem.y_train[i]):

                #a stores the y_train values of the k-nearest neighbors of x
                a = np.take(self.problem.y_train,self.knn[i],axis=0)
                #print("a ",a)

                multipliers = np.take(self.dist_matrix[i],self.knn[i])

                weighteda = np.multiply(a,multipliers)
                
                nans = np.isnan(a)
                #print("nans ",nans)
                #denominator is the number of non-nan y_train entries of the k-nearest neighbors

                denominator = np.sum(np.invert(nans))
                #print("denominator",denominator)
                #numerator is the sum of all non-nan y_train values of the k-nearest neighbors
                #numerator = np.dot(nans,self.problem.y_train)
                #numerator = np.nansum(a)

                #TODO: for some reason numerator and denominator is always getting evaulated to zero, 
                # there has to be a bug causing this(at least for the neighbors of the first point)

                numerator = np.nansum(weighteda)
                predictions[i]=(gamma+numerator)/(1+denominator)

        #returns nx1 column vector of probabilities
        neighborPredictions = np.take(predictions,self.knn[i],axis=0)

        #knn
        # np.savetxt("foo.csv", neighborPredictions, delimiter=",")
        return predictions
        """

