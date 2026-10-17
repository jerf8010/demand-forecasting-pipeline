import numpy as np
import pandas as pd
import pytest

from model.scoring.scoring_utils import benchmark_score, maeError, tablas


def frame(rows):
    return pd.DataFrame(rows, index=[f"s{i}" for i in range(len(rows))])


class TestTablas:
    def test_segments_series_by_mean_of_non_zero_sales(self):
        data = frame([[1, 0, 2, 0, 3],      # mean 2   -> (0, 5]
                      [6, 6, 0, 6, 6],      # mean 6   -> (5, 8]
                      [9, 10, 0, 11, 0]])   # mean 10  -> (8, max]
        counts, table = tablas(data)
        assert table == {"(0, 5]": [0], "(5, 8]": [1], "(8, max]": [2]}
        assert counts.loc["Customers"].tolist() == [1, 1, 1]

    def test_boundaries_are_right_closed(self):
        data = frame([[5, 5, 5], [8, 8, 8]])
        _, table = tablas(data)
        assert table["(0, 5]"] == [0] and table["(5, 8]"] == [1]


class TestBenchmark:
    def test_prediction_is_mean_of_last_values(self):
        train = frame([[0, 0, 2, 4, 6]])
        assert benchmark_score(train, benchmark=3, n_pred=2) == [[4.0, 4.0]]


class TestMaeError:
    test = frame([[1, 1, 1], [2, 2, 2]])

    def test_zero_error_for_perfect_predictions(self):
        preds = [[1, 1, 1], [2, 2, 2]]
        assert maeError(self.test, preds, "mean") == 0
        assert maeError(self.test, preds, "median") == 0

    def test_mean_and_median_over_series(self):
        preds = [[2, 2, 2], [4, 4, 4]]             # absolute error 1 for series 0, 2 for series 1
        assert maeError(self.test, preds, "mean") == pytest.approx(1.5)
        assert maeError(self.test, preds, "median") == pytest.approx(1.5)

    def test_invalid_metric_returns_none(self):
        assert maeError(self.test, [[1, 1, 1], [2, 2, 2]], "mode") is None
