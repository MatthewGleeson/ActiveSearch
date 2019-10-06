"""Active search code for various policies"""

from active_search.policies import *
import matplotlib.pyplot as plt
from active_search.models import *
from math import ceil
#import sys

class ActiveLearning(object):

    def __init__(self, random=True, jitter = False, visual=False, problem = ToyProblem,utility = 2):
        self.visual = visual
        self.random = random
        self.problem = problem(jitter)
        self.utilityvalue = utility
        if utility== 1:
            self.utility = OneStep()
            self.selector = UnlabelSelector()
        elif utility == 2:
            self.utility = TwoStep()
            self.random = False
            random = False
            print("two-step utility selected. Overriding random in favor of argmax")
            self.selector = TwoStepPruningSelector()
        elif utility ==3:
            self.utility = ENS()
            self.random = False
            random = False
            print("ENS utility selected. Overriding random in favor of argmax")
            self.selector = UnlabelSelector()
            #self.selector = ENSPruningSelector()
            self.unlabel_selector = UnlabelSelector()

        if random:
            self.model = RandomModel()
        else:
            self.model = KnnModel(self.problem, k = 4)
            print("KNN MODEL SELECTED!!!")
        
        self.iteration = 0
        self.policy = ArgMaxPolicy(self.problem, self.model,  self.utility)

    def run(self, budget):

        # start by giving the system one observation!
        # look at positives, select random point from those

        # TODO: improve efficiency!! consider changing to masked version below
        #       positive_indices = self.points[labels_deterministic]

        #TODO: make this call more general, not all problems will
        #      have labels_deterministic

        starting_budget = budget
        np.random.seed(3)
        positive_indices = [i for i, x in 
            enumerate(self.problem.labels_deterministic) if x > 0]

        #firstObsIndex = np.random.choice(positive_indices)
        print("positive indices list:",positive_indices[0:10])
        is24 = np.where(positive_indices==24)
        print("24th index:",is24)
        firstObsIndex = positive_indices[0]
        
        
        #25;178;221;240

        #TODO: test using random selections of points
        #firstObsIndex = [25,178,221,240]

        # print k-nearest neighbors of first point

        currentData = Data()
        print("K-nearest neighbors indices of first point:",
              self.model.ind[firstObsIndex] + 1)
        print("selected point is index:", firstObsIndex)
        firstPointValue = self.problem.oracle_function(firstObsIndex)
        #print("first point value:",self.oracle_function(firstObsIndex))
        currentData.new_observation(firstObsIndex, firstPointValue)
        budget = budget-1

        if self.visual:
            self.show_problem()
        """
        a = np.array([34, 92, 104])
        for j in a:
            k = self.problem.oracle_function(j)
            currentData.new_observation(j, k)
            if self.visual:
                self.add_points(j, k)
        """
        if self.visual:
            #self.show_problem()
            self.add_points(firstObsIndex, firstPointValue)
        test_points = np.empty((0,0))
        i = 0
        totalrewards = firstPointValue
        while budget > 0:
            self.iteration = self.iteration + 1
            # model.update
            i = i + 1
            print("step ", i)
            print("budget",budget)

            """
            if self.utilityvalue==3 and budget>starting_budget/2:
                test_indices = self.unlabel_selector.filter(currentData, self.problem.points)
                num_ind = test_indices.size
                test_indices_index = ceil(np.random.random()*num_ind)-1
                test_indices = test_indices[test_indices_index]
            else:
                test_indices = self.selector.filter(self.model,self.policy, currentData, 
                                             self.problem.points,self.problem,budget)
           
            """
            test_indices = self.selector.filter(currentData, self.problem.points,
                                    self.model,self.policy, self.problem, budget)
            
            start = time.time()
 
            x_index = self.policy.choose_next(currentData,test_indices, budget,self.problem.points)
            end = time.time()
            print("time for policy choose next: ", end - start)
            #test_points = np.append(test_points,x_index)
            #np.savetxt('test_points.txt', test_points, fmt='%i', delimiter=' ')
            print("selected index: ",x_index)
            y = self.problem.oracle_function(x_index)
            currentData.new_observation(x_index, y)
            #print(currentData.train_indices)
            totalrewards += y
            budget = budget - 1
            
            print("total rewards:", totalrewards)
            if self.visual:
                self.add_points(x_index, y)
        
        
        return totalrewards

    def show_problem(self):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(self.problem.points[:, 0], self.problem.points[:, 1],
                    c=self.problem.labels_deterministic, s=20)
        
        plt.pause(0.001)

    def add_points(self, x_index, y):
        if y == 0:
            plt.scatter(self.problem.points[x_index, 0],
                        self.problem.points[x_index, 1],
                        c='red', s=20)
        else:
            plt.scatter(self.problem.points[x_index, 0],
                        self.problem.points[x_index, 1],
                        c='green', s=20)
        plt.pause(0.001)
        filename = "imgDirectory/progressive"+str(self.iteration)+".png"
        plt.savefig(filename)


class ActiveLearningExperiment(object):
    def __init__(self, random=True):
        self.is_random = random

    def run(self, num_runs, budget):
        run_results = np.zeros((num_runs, 1))
        for run in range(num_runs):
            run_results[run] = ActiveLearning(self.is_random).run(budget)
        print("avg:", np.mean(run_results))
        return run_results
