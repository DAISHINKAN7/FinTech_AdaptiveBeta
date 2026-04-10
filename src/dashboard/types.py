"""Recommended dashboard types/contracts placeholder.

Lightweight constants documenting payload shapes the parent agent may use
when modularizing the Dash application.
"""

STORE_CONTRACTS = {
    "store_returns": {
        "shape": {"strategy_name": {"idx": ["YYYY-MM-DD"], "val": [0.0]}},
        "source": "app.py serialize_returns / deserialize_returns",
    },
    "signal_state": {
        "shape": {
            "signal": "HOLD|REBALANCE|MIN_VARIANCE",
            "mode": "hold|max_sharpe_beta_constrained|min_variance|risk_parity",
            "predicted_betavol": 0.0,
            "vix_level": 0.0,
            "regime": "bull|bear|transition",
            "threshold": 0.0,
        },
        "source": "src.portfolio.signal.SignalState",
    },
}