import numpy as np


# import matlab.engine


# How to install matlab engine: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html


def genData():

    # eng = matlab.engine.start_matlab()

    # # outputs a dictionary object with key values 'points', 'num_points', 'num_classes'
    # labels_random, labels_deterministic, problem = eng.generate_world_driver(
    #     nargout=3)

    # #labels_random = np.array(labels_random._data).reshape(labels_random.size[::-1]).T
    # labels_random = labels_random._data
    # labels_deterministic = labels_deterministic._data
    # myoutput = problem['points']
    # points = np.array(myoutput._data).reshape(myoutput.size[::-1]).T

    # #import pdb; pdb.set_trace()

    # # TODO: uncomment below to graph dataset
    # # plt.scatter(points[:,0],points[:,1],c=labels_deterministic)
    # # plt.show()

    # #import pdb; pdb.set_trace()

    # eng.quit()

    points = np.loadtxt('./data/points.txt')
    labels_random = np.loadtxt('./data/labels_random.txt')
    labels_deterministic = np.loadtxt('./data/labels_deterministic.txt')
    return labels_random, labels_deterministic, points
