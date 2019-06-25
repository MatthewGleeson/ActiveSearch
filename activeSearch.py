"""Active search code for various policies"""

from policies  import *
from createdata import *

class activeLearning(object):

    def __init__(self,random=True):
        #TODO: make flexible enough to be able to call multiple data generating scripts, each of which should be in current directory

        self.myData = genData()
        #x_pool = self.mydata[:,0]

        self.problem = problem(self.myData[:,[0]])

        if random:
            self.model = randomModel(self.problem)
        else:
            self.model = knnModel(self.problem)

        self.utility = oneStep(self.model)
        self.policy = argMaxPolicy(self.utility)

    def oracle_function(self,x_index):
        return self.myData[x_index][1]


    def run(self, budget):

        #start by giving the system one observation!
        firstObsIndex = np.random.randint(0,high = self.myData.shape[0])
        self.problem.newObservation(firstObsIndex,self.problem.x_pool[firstObsIndex],self.oracle_function(firstObsIndex))

        i = 0
        totalrewards = 0
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
        return totalrewards
