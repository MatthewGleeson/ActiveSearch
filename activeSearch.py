"""Active search code for various policies"""

from policies  import *
from createdata import *
import matplotlib.pyplot as plt


class activeLearning(object):

    def __init__(self,random=True):
        #TODO: make flexible enough to be able to call multiple data generating scripts, each of which should be in current directory
        self.labels_random, self.labels_deterministic, self.points = genData()
        #self.myData = genData()
        #x_pool = self.mydata[:,0]

        self.problem = problem(self.points)

        if random:
            self.model = randomModel(self.problem)
        else:
            self.model = knnModel(self.problem)

        self.utility = oneStep(self.model)
        self.policy = argMaxPolicy(self.utility)

    def oracle_function(self,x_index):
        return self.labels_deterministic[x_index]


    def run(self, budget):

        #start by giving the system one observation!
        #look at positives, select random point from those

        #TODO: improve efficiency!! consider changing to masked version below
        #positive_indices = self.points[labels_deterministic]
        self.display()
        positive_indices = [i for i,x in enumerate(self.labels_deterministic) if x>0]
        
        
        firstObsIndex = np.random.choice(positive_indices)
        firstPointValue = self.oracle_function(firstObsIndex)
        #print("first point value:",self.oracle_function(firstObsIndex))
        self.problem.newObservation(firstObsIndex,self.problem.x_pool[firstObsIndex],self.oracle_function(firstObsIndex))

        i = 0
        totalrewards = firstPointValue
        while budget >= 0:
            #model.update
            i=i+1
            print("step ",i)
            x_index,x = self.policy.choose_next()
            y = self.oracle_function(x_index)
            self.policy.utility.model.problem.newObservation(x_index,x,y)
            totalrewards += y
            budget = budget-1
            print(totalrewards)
            self.display()
        plt.close()
        return totalrewards

    def display(self):
        plt.clf()
        plt.scatter(self.points[:,0],self.points[:,1],c=self.labels_deterministic)
        plt.scatter(self.policy.utility.model.problem.x_train[:,0],self.policy.utility.model.problem.x_train[:,1],c='red')
        #c=self.policy.utility.model.problem.y_train
        plt.show()
        


class ActiveLearningExperiment(object):
    def __init__(self,random=True):
        self.israndom = random
    
    def run(self,numruns,budget):
        runResults = np.zeros((numruns,1))
        for numrun in range(numruns):
            runResults[numrun] = activeLearning(self.israndom)
        print("avg:",np.mean(runResults))
        return runResults
