from active_search.models import Data, ToyProblem, ToyProblemGarnett2012, KnnModel, ENSPruningSelector, UnlabelSelector, TwoStepPruningSelector
from active_search.policies import *
from unittest.mock import Mock
import pytest
import scipy.io



class TestOneStep:
  def test_one_step_toyProblemGarnett2012(self):
    problem = ToyProblemGarnett2012()
    model = Mock()
    model.predict = Mock(return_value=problem.probabilities)

    utility = OneStep()

    data = Mock()
    test_indices = Mock()

    budget = 100

   
    scores = utility.get_scores(model, data, test_indices,budget,problem.points)
    model.predict.assert_called_once_with(data, test_indices)
    for score, probability in zip(scores, problem.probabilities):
      assert score == probability


class TestTwoStep:
  #@pytest.mark.skip(reason="no way of currently testing this")
  #@pytest.mark.filterwarnings("ignore:PendingDeprecationWarning")
  #@pytest.mark.filterwarnings("ignore:SparseEfficiencyWarning")
   
  def test_two_step_48nn(self):

    budget = 100
    problem = ToyProblem()
    model = KnnModel(problem, k=48)
    currentData = Data()
    #two_step_scores = scipy.io.loadmat("tests/matlab_variables/two_step_scores48nn.mat")
    #nn_weights = two_step_scores['expected_utilities']
    utility = TwoStep()
    selector = TwoStepPruningSelector()

    policy = ArgMaxPolicy(problem, model,utility)
    np.random.seed(3)
    positive_indices = [i for i, x in enumerate(problem.labels_deterministic) if x > 0]

    #firstObsIndex = np.random.choice(positive_indices)
    is24 = np.where(positive_indices==24)
    firstObsIndex = positive_indices[0]

    currentData = Data()
    print("K-nearest neighbors indices of first point:",
          model.ind[firstObsIndex] + 1)
    print("selected point is index:", firstObsIndex)
    firstPointValue = problem.oracle_function(firstObsIndex)
    #print("first point value:",self.oracle_function(firstObsIndex))
    currentData.new_observation(firstObsIndex, firstPointValue)
    #test_indices = np.array([444, 588, 1692, 1909, 2203, 2208, 2268])

    test_indices = selector.filter(currentData, problem.points,model,policy,problem, budget)

    scores = utility.get_scores(model, currentData, test_indices,budget,problem.points)

    expected_scores = np.array([2.11514795905353,
                                2.11514795905353,
                                2.11514795905353,
                                2.11514795905353])

    for score, expected in zip(scores, expected_scores):
      assert score == pytest.approx(expected)


class TestENS:
  @pytest.mark.skip(reason="already passing")
  def test_ENS_4nn(self):

    budget = 99
    problem = ToyProblem(jitter=True)
    model = KnnModel(problem, k=4)
    currentData = Data()

    weight_matrix_matlab = scipy.io.loadmat("tests/matlab_variables/weights_4nn_jitter.mat")
    weight_matrix_matlab = weight_matrix_matlab['weights']
    nearest_neighbors_matlab = scipy.io.loadmat("tests/matlab_variables/nearest_neighbors_4nn_jitter.mat")
    nearest_neighbors_matlab = nearest_neighbors_matlab['nearest_neighbors']-1

    model.weight_matrix = weight_matrix_matlab
    model.ind = nearest_neighbors_matlab.T
    expected_scores = scipy.io.loadmat("tests/matlab_variables/ens_utilities_4nn.mat")
    expected_scores = expected_scores['utilities']
    
    utility = ENS()
    selector = UnlabelSelector()

    policy = ArgMaxPolicy(problem, model,utility)
    np.random.seed(3)
    positive_indices = [i for i, x in enumerate(problem.labels_deterministic) if x > 0]

    firstObsIndex = positive_indices[0]

    currentData = Data()
    
    firstPointValue = problem.oracle_function(firstObsIndex)
    #print("first point value:",self.oracle_function(firstObsIndex))
    currentData.new_observation(firstObsIndex, firstPointValue)
    #test_indices = np.array([444, 588, 1692, 1909, 2203, 2208, 2268])

  
    test_indices = selector.filter(currentData, problem.points,model,policy,problem, budget)


    

    expected_test_indices = scipy.io.loadmat("tests/matlab_variables/ens_test_indices_4nn.mat")
    expected_test_indices = expected_test_indices['test_ind']-1
    
    expected_test_indices = np.sort(expected_test_indices,axis = 0)

    #compare test_indices
    for index, expected_index in zip(test_indices, expected_test_indices):
      assert index == expected_index

    scores = utility.get_scores(model, currentData, test_indices,budget,problem.points)
    print(problem.points)
    for score, expected in zip(scores, expected_scores):
      assert score == pytest.approx(expected)


    def test_ENS_4nn_every_iteration(self):

    budget = 99
    problem = ToyProblem(jitter=True)
    model = KnnModel(problem, k=4)
    currentData = Data()

    weight_matrix_matlab = scipy.io.loadmat("tests/matlab_variables/weights_4nn_jitter.mat")
    weight_matrix_matlab = weight_matrix_matlab['weights']
    nearest_neighbors_matlab = scipy.io.loadmat("tests/matlab_variables/nearest_neighbors_4nn_jitter.mat")
    nearest_neighbors_matlab = nedarest_neighbors_matlab['nearest_neighbors']-1

    model.weight_matrix = weight_matrix_matlab
    model.ind = nearest_neighbors_matlab.T
    expected_scores = scipy.io.loadmat("tests/matlab_variables/ens_utilities_every_iter_4nn.mat")
    expected_scores = expected_scores['utilities']

    utility = ENS()
    selector = UnlabelSelector()

    policy = ArgMaxPolicy(problem, model,utility)
    np.random.seed(3)
    positive_indices = [i for i, x in enumerate(problem.labels_deterministic) if x > 0]

    firstObsIndex = positive_indices[0]

    currentData = Data()
    
    firstPointValue = problem.oracle_function(firstObsIndex)
    #print("first point value:",self.oracle_function(firstObsIndex))
    currentData.new_observation(firstObsIndex, firstPointValue)
    #test_indices = np.array([444, 588, 1692, 1909, 2203, 2208, 2268])



    
    test_indices = selector.filter(currentData, problem.points,model,policy,problem, budget)

    expected_test_indices = scipy.io.loadmat("tests/matlab_variables/ens_test_indices_4nn.mat")
    expected_test_indices = expected_test_indices['test_ind']-1
    
    expected_test_indices = np.sort(expected_test_indices,axis = 0)







    #compare test_indices
    for index, expected_index in zip(test_indices, expected_test_indices):
      assert index == expected_index

    scores = utility.get_scores(model, currentData, test_indices,budget,problem.points)
    print(problem.points)
    for score, expected in zip(scores, expected_scores):
      assert score == pytest.approx(expected)



  """
  def test_ENS_48nn(self):

    budget = 99
    problem = ToyProblem()
    model = KnnModel(problem, k=48)
    currentData = Data()
    expected_scores = scipy.io.loadmat("tests/matlab_variables/ens_48_utilities.mat")
    expected_scores = expected_scores['estimated_expected_utility']
    utility = ENS()
    selector = ENSPruningSelector()

    policy = ArgMaxPolicy(problem, model,utility)
    np.random.seed(3)
    positive_indices = [i for i, x in enumerate(problem.labels_deterministic) if x > 0]

    firstObsIndex = positive_indices[0]

    currentData = Data()
    print("K-nearest neighbors indices of first point:",
          model.ind[firstObsIndex] + 1)
    print("selected point is index:", firstObsIndex)
    firstPointValue = problem.oracle_function(firstObsIndex)
    #print("first point value:",self.oracle_function(firstObsIndex))
    currentData.new_observation(firstObsIndex, firstPointValue)
    #test_indices = np.array([444, 588, 1692, 1909, 2203, 2208, 2268])

    test_indices = selector.filter(model,policy, currentData, 
                                             problem.points,problem,budget)

    expected_test_indices = scipy.io.loadmat("tests/matlab_variables/ens_48_test_indices.mat")
    expected_test_indices = expected_test_indices['test_ind']-1
    

    #compare test_indices
    for index, expected_index in zip(test_indices, expected_test_indices):
      assert index == expected_index

    #next, compare expected utilities
    scores = utility.get_scores(model, currentData, test_indices,budget,problem.points)

    for score, expected in zip(scores, expected_scores):
      assert score == pytest.approx(expected)
    """








class TestMergeSort:
    def test_merge_sort_toyProblemGarnett2012(self):
      
      p = scipy.io.loadmat('tests/matlab_variables/p_values.mat')
      q = scipy.io.loadmat('tests/matlab_variables/q_values.mat')
      top_ind = scipy.io.loadmat('tests/matlab_variables/top_ind_values.mat')
      
      
      p = p['p']
      q = q['q']
      top_ind = top_ind['top_ind_to_save']
      top_ind = top_ind-1
      output = merge_sort(p, q, top_ind, 99)
      assert output == pytest.approx(29.9914722870380)





