from active_search.models import Data, ToyProblemGarnett2012, KnnModel, UnlabelSelector
from active_search.policies import *
from unittest.mock import Mock
import pytest


class TestOneStep:
  def test_one_step_toyProblemGarnett2012(self):
    problem = ToyProblemGarnett2012()
    model = Mock()
    model.predict = Mock(return_value=problem.probabilities)

    utility = OneStep()

    data = Mock()
    test_indices = Mock()

    scores = utility.get_scores(model, data, test_indices)
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

    scores = utility.get_scores(model, data, test_indices)

    # computing 2-step score
    epsilon, gamma = problem.probabilities[0], problem.probabilities[2]
    dependent_score = 2 * epsilon + (1 - epsilon) * gamma
    independent_score = epsilon + gamma
    expected_scores = [dependent_score, dependent_score, independent_score]
    for score, expected in zip(scores, expected_scores):
      assert score == expected
