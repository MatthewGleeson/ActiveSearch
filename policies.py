"""Policies for active search"""


from models import *



class utility(object):
    def __init__(self,model):
        pass

    def getScores(self): 
        pass
        

class oneStep(utility):

    def __init__(self, model):
        self.model = model

    def getScores(self):
        return 1*self.model.predict()
        #return probability estimation



class policy(object):
    def __init__(self,utility):
        pass

    def choose_next(self): 
        pass


class argMaxPolicy(policy):
    def __init__(self, utility = oneStep):
        self.utility = utility
    
    def choose_next(self):
        #TODO: decide whether to do all of the argmax stuff in a model function or here
        scores = self.utility.getScores()
        chosen_x_index = np.argmax(scores)
        print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.utility.model.problem.y_train[chosen_x_index])
        chosen_x = self.utility.model.problem.x_pool[chosen_x_index]
        return chosen_x_index, chosen_x