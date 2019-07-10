"""Active search code for various policies"""

from active_search.policies import *
import matplotlib.pyplot as plt
from active_search.models import *


class ActiveLearning(object):

    def __init__(self, random=True, visual=False, problem = ToyProblem):
        self.visual = visual
        self.problem = problem()

        if random:
            self.model = RandomModel()
        else:
            self.model = KnnModel(self.problem)
            print("KNN MODEL SELECTED!!!")

        self.utility = OneStep()
        self.policy = ArgMaxPolicy(self.problem, self.model,  self.utility)
        self.selector = UnlabelSelector()

    def run(self, budget):

        # start by giving the system one observation!
        # look at positives, select random point from those

        # TODO: improve efficiency!! consider changing to masked version below
        #       positive_indices = self.points[labels_deterministic]

        #TODO: make this call more general, not all problems will
        #      have labels_deterministic
        positive_indices = [i for i, x in 
            enumerate(self.problem.labels_deterministic) if x > 0]

        #firstObsIndex = np.random.choice(positive_indices)
        print("positive indices list:",positive_indices[0:10])
        is24 = np.where(positive_indices==24)
        print("24th index:",is24)
        firstObsIndex = positive_indices[0]

        # print k-nearest neighbors of first point

        currentData = Data()
        print("K-nearest neighbors indices of first point:",
              self.model.ind[firstObsIndex] + 1)
        print("selected point is index:", firstObsIndex)
        firstPointValue = self.problem.oracle_function(firstObsIndex)
        #print("first point value:",self.oracle_function(firstObsIndex))
        currentData.new_observation(firstObsIndex, firstPointValue)

        if self.visual:
            self.show_problem()
            self.add_points(firstObsIndex, firstPointValue)

        i = 0
        totalrewards = firstPointValue
        while budget >= 0:
            # model.update
            i = i + 1
            print("step ", i)
            test_indices = self.selector.filter(currentData, 
                                             self.problem.points)

            x_index = self.policy.choose_next(currentData,test_indices)
            
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


class ActiveLearningExperiment(object):
    def __init__(self, random=True):
        self.is_random = random

    def run(self, num_runs, budget):
        run_results = np.zeros((num_runs, 1))
        for run in range(num_runs):
            run_results[run] = ActiveLearning(self.is_random).run(budget)
        print("avg:", np.mean(run_results))
        return run_results
