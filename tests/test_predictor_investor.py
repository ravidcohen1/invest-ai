from datetime import date
from unittest.mock import patch

import pytest

from invest_ai.investor.prediction_based_investor import PredictionBasedInvestor
from invest_ai.return_predictor.baselines import AbstractReturnPredictor
from invest_ai.simulation.enums import Decision, Status


class DummyPredictor(AbstractReturnPredictor):
    def fit(self, train_df_path, val_df_path=None, **kwargs):
        pass

    def predict(self, features):
        pass  # We will mock this

    def get_labels(self):
        return ["bad", "neutral", "good"]


@pytest.fixture
def investor():
    predictor = DummyPredictor()
    return PredictionBasedInvestor(
        predictor=predictor, buy_labels=["good"], sell_labels=["bad"]
    )


@pytest.fixture
def status():
    return Status(
        date=date(2023, 9, 13),
        cash=1003,
        portfolio={"AAPL": 5},
        total_value=2000,
        total_deposits=1000,
    )


def test_make_decision_good_prediction(investor, status):
    with patch.object(DummyPredictor, "predict", return_value="good"):
        decision = investor.make_decision(status, {"AAPL": 100})
        assert decision.buy == {"AAPL": 10}
        assert decision.sell == {}


def test_make_decision_no_cash(investor, status):
    status.cash = 1
    with patch.object(DummyPredictor, "predict", return_value="good"):
        decision = investor.make_decision(status, {"AAPL": 100})
        assert decision.buy == {}
        assert decision.sell == {}


def test_make_decision_neutral_prediction(investor, status):
    with patch.object(DummyPredictor, "predict", return_value="neutral"):
        decision = investor.make_decision(status, {"AAPL": 100})
        assert decision.buy == {}
        assert decision.sell == {}


def test_make_decision_bad_prediction(investor, status):
    with patch.object(DummyPredictor, "predict", return_value="bad"):
        decision = investor.make_decision(status, {"AAPL": 100})
        assert decision.sell == {"AAPL": 5}
        assert decision.buy == {}


def test_make_decision_bad_prediction_no_stock(investor, status):
    status_no_stock = Status(
        date=date(2023, 9, 13),
        cash=1003,
        portfolio={},
        total_value=2000,
        total_deposits=1000,
    )
    with patch.object(DummyPredictor, "predict", return_value="bad"):
        decision = investor.make_decision(status_no_stock, {"AAPL": 100})
        assert decision.sell == {}
        assert decision.buy == {}


def test_make_decision_multiple_stocks(investor, status):
    with pytest.raises(ValueError):
        investor.make_decision(status, {"AAPL": 100, "GOOGL": 1500})
