"""Active search code for various policies"""

from active_search.policies import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from active_search.models import *
from math import ceil
import os
import errno
import time
#import sys

class ActiveLearning(object):
    """
    A class used to represent an active search problem

    It is used to setup the problem, execute an active search policy, 
    send queries to an oracle function, and represent the problem visually

    Attributes
    ----------
    visual : bool
            Whether or not to provide a GUI to display the search problem's
            current state. Only available if d = 2
    random : bool
            Whether or not to use the random Model type, see models.py
    problem : Problem
            The problem space to perform active search over. See models.py
    utilityvalue : int
            The number corresponding to the type of utility function to use.
            See policies.py
    model : Model
            The model of the search space to use. See models.py
    iteration: int
            The number of iterations of calls to the query function
    utility : Utility
            The utility function to use. See policies.py

    Methods
    -------
    run(budget)
        Runs the active search problem
    show_problem()
        Displays a visual representation of the current state of the Active
        Search algorithm
    add_points(x_index, y)
        Adds a point to the visual representation of the current state of the
        Active Search algorithm
    """

    def __init__(self, random=True, jitter = False, visual=False, problem = ToyProblem,utility = 2, do_pruning = True):
        """
        Parameters
        ----------
        random : bool, optional
            Whether or not to use the random Model type, see models.py
        jitter : bool, optional
            Whether or not to add jitter to the 2d problem formulation. See
            createdata.py   
        visual : bool, optional
            Whether or not to provide a GUI to display the search problem's
            current state. Only available if d = 2
        problem : Problem, optional
            The problem space to perform active search over. See models.py
        utility : int, optional
            The type of utility function to use. See policies.py
        selector : Selector
            The type of selector to use for pruning
        unlabel_selector : Selector
            An instance of UnlabelSelector, used to return all unlabeled 
            points
        policy : Policy
            An instance of policy to use. See policies.py
        """

        self.visual = visual
        self.random = random
        self.problem = problem(jitter)
        self.utilityvalue = utility

        if random:
            self.model = RandomModel()
        else:
            self.model = KnnModel(self.problem, k = 4)
            print("Knn Model Selected")
        
        self.iteration = 0

        if utility== 1:
            self.utility = OneStep()
            self.selector = UnlabelSelector()
            self.policy = ArgMaxPolicy(self.problem, self.model,  self.utility)
        elif utility == 2:
            self.utility = TwoStep()
            self.random = False
            random = False
            print("two-step utility selected. Overriding random in favor of argmax")
            self.selector = TwoStepPruningSelector()
            self.policy = ArgMaxPolicy(self.problem, self.model,  self.utility)
        elif utility ==3:
            self.utility = ENS()
            self.random = False
            random = False
            print("ENS utility selected. Overriding random in favor of argmax")
            self.selector = UnlabelSelector()
            self.unlabel_selector = UnlabelSelector()
            self.policy = ENSPolicy(self.problem, self.model,  self.utility, do_pruning= do_pruning)
        

    def run(self, budget):
        """Runs the active search problem

        Parameters
        ----------
        random : bool, optional
            Whether or not to use the random Model type, see models.py
        jitter : bool, optional
            Whether or not to add jitter to the 2d problem formulation. See
            createdata.py   
        visual : bool, optional
            Whether or not to provide a GUI to display the search problem's
            current state. Only available if d = 2
        problem : Problem, optional
            The problem space to perform active search over. See models.py
        utility : int, optional
            The type of utility function to use. See policies.py
        selector : Selector
            The type of selector to use for pruning
        unlabel_selector : Selector
            An instance of UnlabelSelector, used to return all unlabeled 
            points
        policy : Policy
            An instance of policy to use. See policies.py
        
        Returns
        ----------
        totalrewards : int
            The total rewards acquired by the active search function
        """
        
        starting_budget = budget
        np.random.seed(8)
        # TODO: improve efficiency!! consider changing to masked version below
        #       positive_indices = self.points[labels_deterministic]
        positive_indices = [i for i, x in 
            enumerate(self.problem.labels_deterministic) if x > 0]
        print("positive indices list:",positive_indices[0:10])
        is24 = np.where(positive_indices==24)
        print("24th index:",is24)
        firstObsIndex = positive_indices[0]
        firstObsIndex = np.random.randint(0,len(self.problem.points)-1)
        currentData = Data()
        # print k-nearest neighbors of first point
        print("K-nearest neighbors indices of first point:",
              self.model.ind[firstObsIndex] + 1)
        print("selected point is index:", firstObsIndex)
        firstPointValue = self.problem.oracle_function(firstObsIndex)
        currentData.new_observation(firstObsIndex, firstPointValue)
        budget = budget-1

        if self.visual:
            self.show_problem()
            self.add_points(firstObsIndex, firstPointValue)
        i = 0
        totalrewards = firstPointValue
        #iterations of Active Search algorithm
        while budget > 0:
            self.iteration = self.iteration + 1
            i = i + 1
            print("step ", i)
            print("budget",budget)

            test_indices = self.selector.filter(currentData, self.problem.points,
                                    self.model,self.policy, self.problem, budget)
            
            start = time.time()
 
            x_index = self.policy.choose_next(currentData,test_indices, budget,self.problem.points)
            end = time.time()
            print("time for policy choose next: ", end - start)
            print("selected index: ",x_index)
            y = self.problem.oracle_function(x_index)
            currentData.new_observation(x_index, y)
            totalrewards += y
            budget = budget - 1
            
            print("total rewards:", totalrewards)
            if self.visual:
                self.add_points(x_index, y)
        return totalrewards

    def show_problem(self):
        """Displays a visual representation of the current state of the Active
        Search algorithm
        """
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(self.problem.points[:, 0], self.problem.points[:, 1],
                    c=self.problem.labels_deterministic, s=20)
        path = "imgDirectory/"

        try:
            os.mkdir(path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print ("Directory %s already exists" % path)
            else:
                print ("Creation of the directory %s failed" % path)            
        else:
            print ("Successfully created the directory %s " % path)
        plt.pause(0.001)

    def add_points(self, x_index, y):
        """Adds points to visual representation of Active Search problem

        Parameters
        ----------
        x_index : int
            Index corresponding to the observed point, index corresponds to 
            problem.points. See models.py
        y : int
            Value of the observed point
        """

        if y == 0:
            plt.scatter(self.problem.points[x_index, 0],
                        self.problem.points[x_index, 1],
                        c='red', s=20)
        else:
            plt.scatter(self.problem.points[x_index, 0],
                        self.problem.points[x_index, 1],
                        c='green', s=20)
        fontP = FontProperties()
        fontP.set_size('xx-small')
        unobserved_nontarget = mpatches.Patch(color='#440154', label='unobserved non-target')
        unobserved_target = mpatches.Patch(color='#fde725', label='unobserved target')
        observed_nontarget  = mpatches.Patch(color='red', label='observed non-target')
        observed_target = mpatches.Patch(color='green', label='observed target')
        plt.legend(bbox_to_anchor=(1., 1),handles=[unobserved_nontarget,unobserved_target,observed_nontarget,observed_target],prop=fontP)

        plt.pause(0.001)
        filename = "imgDirectory/progressive"+str(self.iteration)+".png"
        plt.savefig(filename)


class ActiveLearningExperiment(object):
    """
    A class used to represent multiple runs of an active search problem

    It creates multiple instances of ActiveLearning objects, and averages
    the results to get more accurate time estimations and result estimations

    Attributes
    ----------
    is_random : bool
            Whether or not to use the random Model type, see models.py


    Methods
    -------
    run(num_runs, budget)
        Runs the active search problem (num_runs) number of times
    """

    def __init__(self, random=True):
        self.is_random = random

    def run(self, num_runs, budget):
        run_results = np.zeros((num_runs, 1))
        for run in range(num_runs):
            #TODO: make this work with new version of ActiveLearning objects
            run_results[run] = ActiveLearning(self.is_random).run(budget)
        print("avg:", np.mean(run_results))
        return run_results
