from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.analytics.dm import (
    diebold_mariano,
    pairwise_dm_table,
)


def test_dm_returns_exact_tie_for_identical_losses() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)

    losses = pd.Series(
        np.linspace(0.1, 0.5, len(idx)),
        index=idx,
        name="model_a",
    )

    result = diebold_mariano(
        loss_a=losses,
        loss_b=losses.rename("model_b"),
        horizon=21,
    )

    assert result.mean_loss_difference == 0.0
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.maxlags == 20
    assert result.nobs == len(idx)


def test_dm_detects_clearly_dominant_model() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=1_000)

    loss_a = pd.Series(
        rng.gamma(
            shape=2.0,
            scale=0.05,
            size=len(idx),
        ),
        index=idx,
        name="model_a",
    )

    # Model B has systematically higher loss, with enough variation to
    # avoid a degenerate zero-variance differential.
    loss_b = pd.Series(
        loss_a.to_numpy()
        + 0.03
        + rng.normal(
            loc=0.0,
            scale=0.005,
            size=len(idx),
        ),
        index=idx,
        name="model_b",
    )

    result = diebold_mariano(
        loss_a=loss_a,
        loss_b=loss_b,
        horizon=21,
    )

    # d_t = loss_a - loss_b:
    # negative means model A has lower loss.
    assert result.mean_loss_difference < 0.0
    assert result.statistic < 0.0
    assert result.p_value < 0.001


def test_dm_reversing_models_reverses_statistic() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=500)

    loss_a = pd.Series(
        rng.uniform(0.05, 0.20, size=len(idx)),
        index=idx,
        name="model_a",
    )

    loss_b = pd.Series(
        loss_a.to_numpy()
        + rng.normal(
            loc=0.01,
            scale=0.01,
            size=len(idx),
        ),
        index=idx,
        name="model_b",
    )

    ab = diebold_mariano(
        loss_a=loss_a,
        loss_b=loss_b,
        horizon=10,
    )

    ba = diebold_mariano(
        loss_a=loss_b,
        loss_b=loss_a,
        horizon=10,
    )

    assert np.isclose(
        ab.mean_loss_difference,
        -ba.mean_loss_difference,
    )

    assert np.isclose(
        ab.statistic,
        -ba.statistic,
    )

    assert np.isclose(
        ab.p_value,
        ba.p_value,
    )


def test_dm_uses_horizon_minus_one_hac_lags() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=300)

    loss_a = pd.Series(
        rng.uniform(0.05, 0.20, size=len(idx)),
        index=idx,
        name="model_a",
    )

    loss_b = pd.Series(
        rng.uniform(0.05, 0.20, size=len(idx)),
        index=idx,
        name="model_b",
    )

    result = diebold_mariano(
        loss_a=loss_a,
        loss_b=loss_b,
        horizon=7,
    )

    assert result.maxlags == 6


def test_dm_rejects_misaligned_indexes() -> None:
    loss_a = pd.Series(
        [0.1, 0.2, 0.3],
        index=pd.bdate_range(
            "2020-01-01",
            periods=3,
        ),
    )

    loss_b = pd.Series(
        [0.1, 0.2, 0.3],
        index=pd.bdate_range(
            "2020-01-02",
            periods=3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="indexes must match",
    ):
        diebold_mariano(
            loss_a=loss_a,
            loss_b=loss_b,
            horizon=1,
        )


def test_pairwise_dm_table_contains_every_model_pair() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=500)

    losses = pd.DataFrame(
        {
            "naive_ma22": rng.uniform(
                0.10,
                0.30,
                size=len(idx),
            ),
            "har": rng.uniform(
                0.07,
                0.20,
                size=len(idx),
            ),
            "garch": rng.uniform(
                0.08,
                0.22,
                size=len(idx),
            ),
        },
        index=idx,
    )

    table = pairwise_dm_table(
        losses,
        horizon=21,
    )

    assert list(table.index) == [
        ("naive_ma22", "har"),
        ("naive_ma22", "garch"),
        ("har", "garch"),
    ]

    assert list(table.columns) == [
        "mean_loss_difference",
        "statistic",
        "p_value",
        "maxlags",
        "nobs",
    ]

    assert (table["maxlags"] == 20).all()
    assert (table["nobs"] == len(idx)).all()