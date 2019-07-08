
import numpy as np
#from emukit.model_wrappers import GPyModelWrapper



edge_path = '../efficient_nonmyopic_active_search/data/citeseer/edge_list'
label_path = '../efficient_nonmyopic_active_search/data/citeseer/labels'

#x, y = np.loadtxt(edge_path,unpack=True)
edge_list = np.loadtxt(edge_path)
labels = np.loadtxt(label_path)

labels = labels[:, 1]

num_nodes = np.amax(edge_list)


A = sparse(edge_list(:, 1), edge_list(:, 2), 1, num_nodes, num_nodes)


print(labels.shape, num_nodes)
num_principal_components = 20