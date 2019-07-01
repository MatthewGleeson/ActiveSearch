"""Policies for active search"""


from models import *



class utility(object):
    def __init__(self):
        pass

    def getScores(self,model): 
        pass
        

class oneStep(utility):

    def __init__(self):
        pass
    def getScores(self,model):
        return 1*model.predict()
        #return probability estimation

class twoStep(utility):

    def __init__(self):
        pass
    def getScores(self,model):
        #updatedModelPos = problem.update(x,1)
        #updatedModelNeg = problem.update(x,0)

        #secondTerm = [p*oneStep(x_new,updatedModelPos)+(1-p)oneStep(x_new,updatedModelNeg) for x-new in x-pool/x].argmax

        #do I need to calculate x-pool/x here too? if so, then I should probably functionalize it in models.py as a function of the problem
        return 1*model.predict()
        #return probability estimation

class policy(object):
    def __init__(self,utility,model):
        pass

    def choose_next(self): 
        pass


class argMaxPolicy(policy):
    def __init__(self,model,utility = oneStep):
        self.model = model
        self.utility = utility
    
    def choose_next(self):
        
        scores = self.utility.getScores(self.model)
        chosen_x_index = np.argmax(scores)
        #print("index value:",chosen_x_index)
        #print("largest score value:",scores[chosen_x_index])
        #print("next index's score value:",scores[chosen_x_index+1])
        #print("chooses x index:",chosen_x_index)
        #print("with y_train value:",self.utility.model.problem.y_train[chosen_x_index])
        chosen_x = self.model.problem.x_pool[chosen_x_index]
        return chosen_x_index, chosen_x