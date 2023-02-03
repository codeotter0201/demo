import pandas as pd
from fastapi import FastAPI, Response, Depends
import uvicorn
import os
from sentry import Sentry
from importlib import reload

app = FastAPI()

_sentry = None


def get_sentry():
    global _sentry
    if _sentry is None:
        _sentry = Sentry(1)
    return _sentry


@app.put("/reload")
def reload_module(
    test: bool = True, lookback_days: int = 750, _sentry=Depends(get_sentry)
) -> None:
    pw = True
    if pw == pw:
        _sentry.reload_strategies(test=test)
        _sentry.update_data_all()
        # _sentry.subscribe_all()
        _sentry.gen_portfolio_pickle(lookback_days)
        _sentry.gen_signals_pickle()
        _sentry.gen_orders_pickle()
        return dict(
            mothods=list(_sentry.strategy_methods.keys()),
            data_config_list=_sentry.data_config_list,
            subscribtions=_sentry.subscribtion_list,
        )
    else:
        print("set a code")
        return "set a code"


@app.get("/portfolio")
def get_portfolio(_sentry=Depends(get_sentry)) -> dict:
    return dict(
        methods=list(_sentry.strategy_methods.keys()),
        data_config_list=_sentry.data_config_list,
        subscribtions=_sentry.subscribtion_list,
    )


@app.get("/trades")
def get_trades() -> dict:
    temp_trades = pd.read_pickle("history/reports/trades.pkl")
    return temp_trades


@app.get("/signals")
def get_signals() -> dict:
    signals = pd.read_pickle("history/reports/signals.pkl")
    return signals


@app.get("/orders")
def get_orders() -> dict:
    orders = pd.read_pickle("history/reports/orders.pkl")
    return orders


@app.get("/performance")
def get_performance() -> dict:
    temp_performance = pd.read_pickle("history/reports/performance.pkl")
    return temp_performance


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9999)
