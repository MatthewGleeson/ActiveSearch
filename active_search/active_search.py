"""Active search code for various policies"""

from policies import *
import matplotlib.pyplot as plt


class ActiveLearning(object):

    def __init__(self, random=True, visual=False):
        self.visual = visual
        self.problem = Problem(self.points)

        if random:
            self.model = randomModel(self.problem)
        else:
            self.model = knnModel(self.problem)

        self.utility = OneStep()
        self.policy = ArgMaxPolicy(self.model, self.utility)

    def run(self, budget):

        # start by giving the system one observation!
        # look at positives, select random point from those

        # TODO: improve efficiency!! consider changing to masked version below
        #positive_indices = self.points[labels_deterministic]

        positive_indices = [i for i, x in enumerate(
            self.labels_deterministic) if x > 0]

        #firstObsIndex = np.random.choice(positive_indices)
        firstObsIndex = positive_indices[0]
        # print k-nearest neighbors of first point

        #dist, ind = self.model.tree.query(self.model.problem.x_pool[firstObsIndex], k=50)

        print("K-nearest neighbors indices of first point:",
              self.model.knn[firstObsIndex] + 1)
        print("selected point is index:", firstObsIndex)
        firstPointValue = self.oracle_function(firstObsIndex)
        #print("first point value:",self.oracle_function(firstObsIndex))
        self.problem.newObservation(
            firstObsIndex, self.oracle_function(firstObsIndex))

        if self.visual:
            self.show_problem()
            self.add_points(firstObsIndex, firstPointValue)

        i = 0
        totalrewards = firstPointValue
        while budget >= 0:
            # model.update
            i = i + 1
            print("step ", i)
            x_index = self.policy.choose_next()
            y = self.oracle_function(x_index)
            self.policy.model.problem.newObservation(x_index, y)
            totalrewards += y
            budget = budget - 1
            print("total rewards:", totalrewards)
            if self.visual:
                self.add_points(x_index, y)
        return totalrewards

    def show_problem(self):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(self.points[:, 0], self.points[:, 1],
                    c=self.labels_deterministic, s=20)

        plt.pause(0.000001)

    def add_points(self, x_index, y):
        if y == 0:
            plt.scatter(self.policy.model.problem.x_pool[x_index, 0],
                        self.policy.model.problem.x_pool[x_index, 1], c='red', s=20)
        else:
            plt.scatter(self.policy.model.problem.x_pool[x_index, 0],
                        self.policy.model.problem.x_pool[x_index, 1], c='green', s=20)
        plt.pause(0.000001)


class ActiveLearningExperiment(object):
    def __init__(self, random=True):
        self.is_random = random

    def run(self, num_runs, budget):
        run_results = np.zeros((num_runs, 1))
        for run in range(num_runs):
            run_results[run] = ActiveLearning(self.is_random).run(budget)
        print("avg:", np.mean(run_results))
        return run_results
