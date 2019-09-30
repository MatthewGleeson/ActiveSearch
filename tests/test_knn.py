from active_search.models import Data, ToyProblem, KnnModel, UnlabelSelector
from active_search.policies import *
from unittest.mock import Mock
import pytest
import scipy.io
from scipy.sparse import find


class TestWeightMatrix:
  
  #@pytest.mark.skip(reason="testing nn first")
  def test_weight_matrix_k_4_jitter(self):
    problem = ToyProblem(jitter = True)

    model = KnnModel(problem, k=4)
    nn_weights = scipy.io.loadmat("tests/matlab_variables/weights_4nn_jitter.mat")
    nn_weights = nn_weights['weights']

    #check nnz equal
    assert model.weight_matrix.nnz == nn_weights.nnz
  


    nn_weights = nn_weights.toarray()
    model.weight_matrix = model.weight_matrix.toarray()

    #differences = np.abs(nn_weights-model.weight_matrix)
    differences = nn_weights-model.weight_matrix
    

    print(differences[1165,1266])
    print(differences[1266,1165])
    print(model.weight_matrix[1165,1266])


    print(np.amax(differences))
    print(np.argmax(differences[1165]))


    assert np.allclose(nn_weights,model.weight_matrix, atol=1e-02)


  def test_weight_matrix_k_4(self):
    problem = ToyProblem()

    model = KnnModel(problem, k=4)
    nn_weights = scipy.io.loadmat("tests/matlab_variables/weights_4nn.mat")
    nn_weights = nn_weights['weights']

    #check nnz equal
    assert model.weight_matrix.nnz == nn_weights.nnz

    # remove edge points, messes w/ nearest neighbor comparison
    # first get indices of edge points

    edge_indices_x_max = np.where(problem.points[:, 0] >= 49, 0, 1)
    edge_indices_x_min = np.where(problem.points[:, 0] <= 2, 0, 1)

    edge_indices_x = np.multiply(edge_indices_x_min, edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:, 1] >= 49, 0, 1)
    edge_indices_y_min = np.where(problem.points[:, 1] <= 2, 0, 1)

    edge_indices_y = np.multiply(edge_indices_y_min, edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x, edge_indices_y)

    # next, remove entries from both nearest neighbors

    nn_weights = nn_weights.toarray()
    model.weight_matrix = model.weight_matrix.toarray()

    nn_weights = np.multiply(nn_weights, edge_indices[:, None])
    model.weight_matrix = np.multiply(model.weight_matrix, edge_indices[:, None])

    assert np.allclose(nn_weights,model.weight_matrix)

  def test_weight_matrix_k_8(self):
    problem = ToyProblem()

    model = KnnModel(problem, k=8)
    nn_weights = scipy.io.loadmat("tests/matlab_variables/weights_8nn.mat")
    nn_weights = nn_weights['weights']

    #check nnz equal
    assert model.weight_matrix.nnz == nn_weights.nnz

    # remove edge points, messes w/ nearest neighbor comparison
    # first get indices of edge points

    edge_indices_x_max = np.where(problem.points[:, 0] >= 49, 0, 1)
    edge_indices_x_min = np.where(problem.points[:, 0] <= 2, 0, 1)

    edge_indices_x = np.multiply(edge_indices_x_min, edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:, 1] >= 49, 0, 1)
    edge_indices_y_min = np.where(problem.points[:, 1] <= 2, 0, 1)

    edge_indices_y = np.multiply(edge_indices_y_min, edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x, edge_indices_y)

    # next, remove entries from both nearest neighbors

    nn_weights = nn_weights.toarray()
    model.weight_matrix = model.weight_matrix.toarray()

    nn_weights = np.multiply(nn_weights, edge_indices[:, None])
    model.weight_matrix = np.multiply(model.weight_matrix, edge_indices[:, None])

    assert np.allclose(nn_weights,model.weight_matrix)


  def test_weight_matrix_k_20(self):
    problem = ToyProblem()

    model = KnnModel(problem, k=20)
    nn_weights = scipy.io.loadmat("tests/matlab_variables/weights_20nn.mat")
    nn_weights = nn_weights['weights']

    #check nnz equal
    assert model.weight_matrix.nnz == nn_weights.nnz

    # remove edge points, messes w/ nearest neighbor comparison
    # first get indices of edge points

    edge_indices_x_max = np.where(problem.points[:, 0] >= 49, 0, 1)
    edge_indices_x_min = np.where(problem.points[:, 0] <= 2, 0, 1)

    edge_indices_x = np.multiply(edge_indices_x_min, edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:, 1] >= 49, 0, 1)
    edge_indices_y_min = np.where(problem.points[:, 1] <= 2, 0, 1)

    edge_indices_y = np.multiply(edge_indices_y_min, edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x, edge_indices_y)

    # next, remove entries from both nearest neighbors

    nn_weights = nn_weights.toarray()
    model.weight_matrix = model.weight_matrix.toarray()

    nn_weights = np.multiply(nn_weights, edge_indices[:, None])
    model.weight_matrix = np.multiply(model.weight_matrix, edge_indices[:, None])

    assert np.allclose(nn_weights,model.weight_matrix)
    


  def test_weight_matrix_k_48(self):
    problem = ToyProblem()

    model = KnnModel(problem, k=48)
    nn_weights = scipy.io.loadmat("tests/matlab_variables/weights_48nn.mat")
    nn_weights = nn_weights['weights']

    #check nnz equal
    assert model.weight_matrix.nnz == nn_weights.nnz

    # remove edge points, messes w/ nearest neighbor comparison
    # first get indices of edge points

    edge_indices_x_max = np.where(problem.points[:, 0] >= 47, 0, 1)
    edge_indices_x_min = np.where(problem.points[:, 0] <= 4, 0, 1)

    edge_indices_x = np.multiply(edge_indices_x_min, edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:, 1] >= 47, 0, 1)
    edge_indices_y_min = np.where(problem.points[:, 1] <= 4, 0, 1)

    edge_indices_y = np.multiply(edge_indices_y_min, edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x, edge_indices_y)

    # next, remove entries from both nearest neighbors

    nn_weights = nn_weights.toarray()
    model.weight_matrix = model.weight_matrix.toarray()

    nn_weights = np.multiply(nn_weights, edge_indices[:, None])
    model.weight_matrix = np.multiply(model.weight_matrix, edge_indices[:, None])

    assert np.allclose(nn_weights,model.weight_matrix)





class TestNearestNeighbors:
  #@pytest.mark.skip(reason="testing weight first")
  def test_nearest_neighbors_k_4_jitter(self):
    problem = ToyProblem(jitter=True)

    model = KnnModel(problem,k=4) 
    nn_values = scipy.io.loadmat("tests/matlab_variables/nearest_neighbors_4nn_jitter.mat")
    nn_values = nn_values['nearest_neighbors']
    nn_values = nn_values.T -1
    
    assert np.all(np.sort(nn_values,axis=1) == np.sort(model.ind,axis=1))


  def test_nearest_neighbors_k_4(self):
    problem = ToyProblem()

    model = KnnModel(problem,k=4) 
    nn_values = scipy.io.loadmat("tests/matlab_variables/nearest_neighbors_4nn.mat")
    nn_values = nn_values['nearest_neighbors']
    nn_values = nn_values.T -1

    # remove edge points, messes w/ nearest neighbor comparison

    # first get indices of edge points
    edge_indices_x_max = np.where(problem.points[:,0]==50, 0, 1)
    edge_indices_x_min = np.where(problem.points[:,0]==1, 0, 1)
    edge_indices_x = np.multiply(edge_indices_x_min,edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:,1]==50, 0, 1)
    edge_indices_y_min = np.where(problem.points[:,1]==1, 0, 1)
    edge_indices_y = np.multiply(edge_indices_y_min,edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x,edge_indices_y)

    # next, remove entries from both nearest neighbors
    nn_values = np.multiply(nn_values,edge_indices[:, None])
    model.ind = np.multiply(model.ind,edge_indices[:, None])
    
    assert np.all(np.sort(nn_values,axis=1) == np.sort(model.ind,axis=1))

  def test_nearest_neighbors_k_8(self):
    problem = ToyProblem()
    model = KnnModel(problem,k=8) 
    nn_values = scipy.io.loadmat("tests/matlab_variables/nearest_neighbors_8nn.mat")
    nn_values = nn_values['nearest_neighbors']
    nn_values = nn_values.T -1
    # remove edge points, messes w/ nearest neighbor comparison

    # first get indices of edge points
    edge_indices_x_max = np.where(problem.points[:,0]==50, 0, 1)
    edge_indices_x_min = np.where(problem.points[:,0]==1, 0, 1)
    edge_indices_x = np.multiply(edge_indices_x_min,edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:,1]==50, 0, 1)
    edge_indices_y_min = np.where(problem.points[:,1]==1, 0, 1)
    edge_indices_y = np.multiply(edge_indices_y_min,edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x,edge_indices_y)

    # next, remove entries from both nearest neighbors
    nn_values = np.multiply(nn_values,edge_indices[:, None])
    model.ind = np.multiply(model.ind,edge_indices[:, None])
    
    assert np.all(np.sort(nn_values,axis=1) == np.sort(model.ind,axis=1))

  def test_nearest_neighbors_k_20(self):
    problem = ToyProblem()
    model = KnnModel(problem,k=20) 
    nn_values = scipy.io.loadmat("tests/matlab_variables/nearest_neighbors_20nn.mat")
    nn_values = nn_values['nearest_neighbors']
    nn_values = nn_values.T -1
    # remove edge points, messes w/ nearest neighbor comparison

    # first get indices of edge points
    edge_indices_x_max = np.where(problem.points[:,0]>=49, 0, 1)
    edge_indices_x_min = np.where(problem.points[:,0]<=2, 0, 1)
    edge_indices_x = np.multiply(edge_indices_x_min,edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:,1]>=49, 0, 1)
    edge_indices_y_min = np.where(problem.points[:,1]<=2, 0, 1)
    edge_indices_y = np.multiply(edge_indices_y_min,edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x,edge_indices_y)

    # next, remove entries from both nearest neighbors
    nn_values = np.multiply(nn_values,edge_indices[:, None])
    model.ind = np.multiply(model.ind,edge_indices[:, None])

    assert np.all(np.sort(nn_values,axis=1) == np.sort(model.ind,axis=1))

  def test_nearest_neighbors_k_48(self):
    problem = ToyProblem()
    model = KnnModel(problem,k=48) 
    nn_values = scipy.io.loadmat("tests/matlab_variables/nearest_neighbors_48nn.mat")
    nn_values = nn_values['nearest_neighbors']
    nn_values = nn_values.T -1
    # remove edge points, messes w/ nearest neighbor comparison

    # first get indices of edge points
    edge_indices_x_max = np.where(problem.points[:,0]>=47, 0, 1)
    edge_indices_x_min = np.where(problem.points[:,0]<=4, 0, 1)
    edge_indices_x = np.multiply(edge_indices_x_min,edge_indices_x_max)

    edge_indices_y_max = np.where(problem.points[:,1]>=47, 0, 1)
    edge_indices_y_min = np.where(problem.points[:,1]<=4, 0, 1)
    edge_indices_y = np.multiply(edge_indices_y_min,edge_indices_y_max)

    edge_indices = np.multiply(edge_indices_x,edge_indices_y)

    # next, remove entries from both nearest neighbors
    nn_values = np.multiply(nn_values,edge_indices[:, None])
    model.ind = np.multiply(model.ind,edge_indices[:, None])
    
    assert np.all(np.sort(nn_values,axis=1) == np.sort(model.ind,axis=1))

