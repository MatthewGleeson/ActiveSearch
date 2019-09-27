from active_search.models import Data, ToyProblemGarnett2012, KnnModel, UnlabelSelector
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
  @pytest.mark.xfail
  def test_two_step_toyProblemGarnett2012(self):
    problem = ToyProblemGarnett2012()
    model = Mock()
    model.predict = Mock(return_value=problem.probabilities)

    utility = OneStep()

    data = Mock()
    test_indices = Mock()
    budget = 100
    scores = utility.get_scores(model, data, test_indices,budget,problem.points,model.weight_matrix)

    # computing 2-step score
    epsilon, gamma = problem.probabilities[0], problem.probabilities[2]
    dependent_score = 2 * epsilon + (1 - epsilon) * gamma
    independent_score = epsilon + gamma
    expected_scores = [dependent_score, dependent_score, independent_score]
    for score, expected in zip(scores, expected_scores):
      assert score == expected


class TestMergeSort:
    def test_merge_sort_toyProblemGarnett2012(self):
      """
      try:
        p = np.loadtxt('mergeSortp.txt')
        q = np.loadtxt('mergeSortq.txt')
        top_ind = np.loadtxt('mergeSortTopInd.txt')
      except OSError:
        p = np.loadtxt('tests/mergeSortp.txt')
        q = np.loadtxt('tests/mergeSortq.txt')
        top_ind = np.loadtxt('tests/mergeSortTopInd.txt')
      """
      try:
        p = scipy.io.loadmat('p_values.mat')
        q = scipy.io.loadmat('q_values.mat')
        top_ind = scipy.io.loadmat('top_ind_values.mat')
      except OSError:
        p = scipy.io.loadmat('tests/matlab_variables/p_values.mat')
        q = scipy.io.loadmat('tests/matlab_variables/q_values.mat')
        top_ind = scipy.io.loadmat('tests/matlab_variables/top_ind_values.mat')
      
      
      p = p['p']
      q = q['q']
      top_ind = top_ind['top_ind_to_save']
      top_ind = top_ind-1
      output = merge_sort(p, q, top_ind, 99)
      assert output == pytest.approx(29.9914722870380)





