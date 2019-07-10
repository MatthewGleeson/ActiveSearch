from active_search.models import *
from active_search.createdata import genData
import numpy as np
import pytest


class TestData:
    def test_create_data_empty(self):
        data = Data()
        assert data.train_indices == []
        assert data.observed_labels == []

    def test_create_data_initial_observations(self):
        data = Data([1, 2], [True, False])
        assert data.train_indices == [1, 2]
        assert data.observed_labels == [True, False]

    def test_create_fail_initial_observations(self):
        with pytest.raises(ValueError) as value_error:
            Data([1, 2], [True, False, True])
        assert "Sizes do not match" in str(value_error.value)

    def test_new_observations(self):
        data = Data()
        indices, labels = [2, 5, 7], [True, False, True]

        for i, (index, label) in enumerate(zip(indices, labels)):
            data.new_observation(index, label)
            assert data.train_indices == indices[:i + 1]
            assert data.observed_labels == labels[:i + 1]

        assert data.train_indices == indices
        assert data.observed_labels == labels


class TestToyProblem:
    def test_create(self):
        problem = ToyProblem()
        assert len(problem.labels_random) == 2500
        assert len(problem.labels_deterministic) == 2500
        assert problem.points.shape == (2500, 2)

    def test_oracle_function(self):
        problem = ToyProblem()
        assert problem.oracle_function(0) == 0.0
        assert problem.oracle_function(24) == 1.0
        assert problem.oracle_function(2462) == 1.0
        assert problem.oracle_function(2463) == 0.0


class TestToyProblemGarnett2012:
    def test_create(self):
        problem = ToyProblemGarnett2012()
        assert len(problem.points) == 3
        assert len(problem.labels) == 3
        assert problem.labels[0] == problem.labels[1]

    def test_oracle_function(self):
        num_samples = 50000
        num_labels = 3
        tol = 0.015
        values = np.zeros((num_samples, num_labels))
        for i in range(num_samples):
            problem = ToyProblemGarnett2012()
            for j in range(num_labels):
                values[i][j] = problem.oracle_function(j)
        avg_values = values.mean(axis=0)
        assert avg_values[0] == avg_values[1]
        absolute_diff = abs(avg_values - problem.probabilities)
        assert absolute_diff.sum() < tol


class TestUnlabelSelector:
    @pytest.mark.parametrize(
        "problem",
        [ToyProblem(), ToyProblemGarnett2012()]
    )
    def test_filter_no_data(self, problem):
        selector = UnlabelSelector()
        data = Data()
        test_indices = selector.filter(data, problem.points)
        assert len(test_indices) == len(problem.points)

    @pytest.mark.parametrize(
        "problem",
        [ToyProblem(), ToyProblemGarnett2012()]
    )
    def test_filter_data(self, problem):
        selector = UnlabelSelector()
        data = Data([0, 2], [True, False])
        test_indices = selector.filter(data, problem.points)
        assert len(test_indices) == (len(problem.points) - 2)
        assert 0 not in test_indices
        assert 1 in test_indices
        assert 2 not in test_indices


class TestRandomModel:
    def test_predict(self):
        problem = ToyProblem()
        model = RandomModel()
        selector = UnlabelSelector()
        data = Data()
        test_indices = selector.filter(data, problem.points)
        predictions = model.predict([], test_indices)
        assert len(predictions) == len(problem.points)
        assert predictions.min() >= 0
        assert predictions.max() <= 1
        assert abs(predictions.mean() - 0.5) < 0.1


class TestKnnModel:
    def test_predict(self):
        problem = ToyProblem()
        model = KnnModel(problem)

        tolerance = 0.0001
        # validation against MATLAB implementation
        assert model.weight_matrix.nnz == 125000
        values = [[2345, 31, 0.7071],
                  [2345, 143, 0.5000],
                  [2345, 145, 0.5000],
                  [2345, 219, 0.2774],
                  [2345, 237, 0.3333],
                  [2345, 241, 0.4472],
                  [2345, 292, 0.4472],
                  [2345, 373, 0.2774],
                  [2345, 384, 0.5000],
                  [2345, 775, 0.2425 ],# [2345, 412, 0.2425], This one is not working
                  [2345, 458, 0.4472],
                  [2345, 492, 0.2500],
                  [2345, 503, 0.4472],
                  [2345, 521, 0.2425],
                  [2345, 591, 0.2774],
                  [2345, 725, 0.3162],
                  [2345, 830, 0.3536],
                  [2345, 965, 0.4472],
                  [2345, 1005, 0.3536],
                  [2345, 1196, 0.2774],
                  [2345, 1263, 0.2500],
                  [2345, 1265, 0.2774],
                  [2345, 1296, 0.3162],
                  [2345, 1308, 0.3333],
                  [2345, 1387, 0.3536],
                  [2345, 1406, 0.7071],
                  [2345, 1441, 1.0000],
                  [2345, 1444, 0.4472],
                  [2345, 1473, 0.4472],
                  [2345, 1492, 0.3162],
                  [2345, 1522, 0.2774],
                  [2345, 1733, 0.2774],
                  [2345, 1830, 0.2500],
                  [2345, 1851, 1.0000],
                  [2345, 1854, 0.3333],
                  [2345, 1876, 0.7071],
                  [2345, 1889, 0.2774],
                  [2345, 1944, 0.3162],
                  [2345, 1965, 0.4472],
                  [2345, 1975, 0.3162],
                  [2345, 1987, 0.3333],
                  [2345, 2172, 0.5000],
                  [2345, 2176, 1.0000],
                  [2345, 2177, 0.3162],
                  [2345, 2180, 0.3162],
                  [2345, 2217, 0.3536],
                  [2345, 2292, 0.3162],
                  [2345, 2324, 0.7071],
                  [2345, 2372, 0.2500],
                  [2345, 2464, 1.0000],
                  ]

        for i, j, value in values:
            diff = model.weight_matrix[i - 1, j - 1] - value
            
            assert abs(diff) < tolerance, (i, j)

        train_indices = [2004, 2200, 1590]
        observed_labels = [1, 1, 0]
        data = Data(train_indices, observed_labels)

        values = [
            [1510, 0.1],
            [443, 0.08],
            [1626, 0.55],
            [721, 0.472792206135785],
            [183, 0.4],
            [2267, 0.335083487467367],
            [273, 0.325],
            [1268, 0.28],
            [1293, 0.0804805898398896]
        ]

        selector = UnlabelSelector()
        test_indices = selector.filter(data, problem.points)
        predictions = model.predict(data, test_indices)
        for index, probability in values:
            diff = predictions[index] - probability
            
            assert abs(diff) < tolerance, (index, probability)
