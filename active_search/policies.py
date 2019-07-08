"""Policies for active search"""


class Utility(object):
    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices):
        pass


class OneStep(Utility):

    def __init__(self):
        pass

    def get_scores(self, model, data, test_indices):
        return model.predict(data, test_indices)
        # return probability estimation


class TwoStep(Utility):

    def __init__(self):
        pass

    def get_scores(self, model):
        pass

        # for a in model.problem.basicSelector():
        #      updatedModelPos = problem.update(x,1)

        #updatedModelPos = problem.update(x,1)
        #updatedModelNeg = problem.update(x,0)

        # secondTerm = [p*OneStep(x_new,updatedModelPos)+(1-p)OneStep(x_new,updatedModelNeg) for x-new in x-pool/x].argmax

        # do I need to calculate x-pool/x here too? if so, then I should probably functionalize it in models.py as a function of the problem
        # return 1 * model.predict()
        # return probability estimation


class Policy(object):
    def __init__(self, utility, model):
        pass

    def choose_next(self):
        pass


class ArgMaxPolicy(Policy):
    def __init__(self, problem, model=None, utility=None):
        if not model:
            model = KnnModel(problem)
        if not utility:
            utility = OneStep()
        self.model = model
        self.utility = utility

    def choose_next(self, data, test_indices):

        scores = self.utility.get_scores(self.model, test_indices)
        max_index = np.argmax(scores)
        chosen_x_index = test_indices[max_index]

        #print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.Utility.model.problem.y_train[chosen_x_index])

        #chosen_x = self.model.problem.x_pool[chosen_x_index]
        return chosen_x_index
