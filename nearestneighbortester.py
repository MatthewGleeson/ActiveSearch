from activeSearch import activeLearning
import numpy as np


rewardsMatrix = np.zeros((20,2))

for i in range(2):
    myFirstRun = activeLearning(False)
    rewardsMatrix[i][0]=i
    rewardsMatrix[i][1]= myFirstRun.run(1000)
    