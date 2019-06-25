import numpy as np


import matlab.engine
#How to install matlab engine: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html




def genData():

    eng = matlab.engine.start_matlab()

    #outputs a dictionary object with key values 'points', 'num_points', 'num_classes'
    output = eng.generate_world_driver(nargout=1)

    myoutput = output['points']
    mydata = np.array(myoutput._data).reshape(myoutput.size[::-1]).T
    print("converted type: ",type(mydata))

    #import pdb; pdb.set_trace()

    eng.quit()

    return mydata


