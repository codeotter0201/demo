import pandas as pd
from fastapi import FastAPI, Response, Depends
import uvicorn
from sentry import Sentry

app = FastAPI()

sentry = None


def get_sentry():
    global sentry
    if sentry is None:
        sentry = Sentry(1)
    return sentry


@app.get("/strategies_name_list")
def strategies_name_list() -> list:
    return Sentry.get_strategies_name_list()


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


@app.post("/portfolio")
def gen_portfolio(
    lookback_days: int = 750,
    test_all: bool = False,
    pw: str = None,
    sentry=Depends(get_sentry),
) -> dict:
    if pw == "otter":
        sentry.gen_portfolio_pickle(lookback_days, test_all)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9999)
