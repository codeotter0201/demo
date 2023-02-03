import requests
import json
import pandas as pd
import time, os
from plotter import Plotter


class Worker:
    def __init__(self, mode: int = 3) -> None:
        """
        mode 1: container service name
        mode 2: 127.0.0.1
        mode 3: IP
        """
        self.linode_app_ip = os.getenv("LINODE_APP_IP")
        self.linode_bot_ip = os.getenv("LINODE_BOT_IP")
        if mode == 1:
            self.db = "http://database:8888/"
            self.pf = "http://portfolio:9999/"
            self.bot = "http://portfolio:12345/"
        if mode == 2:
            self.db = "http://127.0.0.1:8888/"
            self.pf = "http://127.0.0.1:9999/"
            self.bot = "http://127.0.0.1:12345/"
        if mode == 3:
            self.db = f"http://{self.linode_app_ip}:8888/"
            self.pf = f"http://{self.linode_app_ip}:9999/"
            self.bot = f"http://{self.linode_bot_ip}:12345/"
        if mode == 4:
            self.db = "http://database:8888/"
            self.pf = "http://portfolio:9999/"
            self.bot = f"http://{self.linode_bot_ip}:12345/"
        self.__plot = Plotter()

    @staticmethod
    def data_parser(x):
        if "time" in x.name:
            return pd.to_datetime(x)
        if ("idx" in x.name) | ("index" in x.name):
            return x.astype(int)
        if "direction" in x.name:
            return x.map(lambda y: "L" if (y == 0) | (y == "0") else "S")
        if "status" in x.name:
            return x.map(lambda y: "Open" if (y == 0) | (y == "0") else "Closed")
        if ("sn" == x.name) | ("symbol" == x.name):
            return x
        if "test_mode" == x.name:
            return x == "True"
        return x.astype(float)

    def get_performance(self) -> pd.DataFrame:
        url = self.pf + "performance"
        r = requests.get(
            url,
        )
        if r.status_code == 200:
            return (
                pd.DataFrame(r.json())
                .set_index("index")
                .astype(float)
                .sort_values("recovery_factor")[::-1]
            )
        else:
            return r.status_code

    def get_trades(self, test: bool = False) -> pd.DataFrame:
        url = self.pf + "trades"
        r = requests.get(
            url,
        )
        if r.status_code == 200:
            trades = (
                pd.DataFrame(r.json())
                .apply(self.data_parser)
                .sort_values("exit_time")
                .iloc[:, 1:]
                .reset_index(drop=True)
            )
            return pd.DataFrame(
                dict(
                    direction=trades.direction,
                    entry_time=trades.entry_time,
                    entry_price=trades.entry_price,
                    exit_time=trades.exit_time,
                    exit_price=trades.exit_price,
                    pnl=trades.pnl,
                    pct=trades.pct * 100,
                    mae=trades.mae,
                    g_mfe=trades.g_mfe,
                    mfe=trades.mfe,
                    mae_lv1=trades.mae_lv1,
                    h2c=trades.h2c,
                    l2c=trades.l2c,
                    benchmark=trades.benchmark,
                    sn=trades.sn,
                    symbol=trades.symbol,
                    status=trades.status,
                    # test_mode = trades.test_mode,
                )
            )
        else:
            return r.status_code

    def get_trades_less(self, test: bool = False) -> pd.DataFrame:
        trades = self.get_trades(test)
        drop_col = ["mae", "mfe", "g_mfe", "mae_lv1", "h2c", "l2c"]
        trades.entry_time = trades.entry_time.dt.date
        trades.exit_time = trades.exit_time.dt.date
        return trades.drop(drop_col, axis=1)

    def get_signals(self) -> pd.DataFrame:
        url = self.pf + "signals"
        r = requests.get(
            url,
        )
        if r.status_code == 200:
            return r.json()
        else:
            return r.status_code

    def get_orders(self) -> pd.DataFrame:
        url = self.pf + "orders"
        r = requests.get(
            url,
        )
        if r.status_code == 200:
            return r.json()
        else:
            return r.status_code

    def get_portfolio(self) -> list:
        url = self.pf + "portfolio"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return r.status_code

    def get_trader_info(self) -> dict:
        url = self.bot + "info"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return r.status_code

    def get_orderbook(self) -> dict:
        url = self.bot + "orderbook"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return r.status_code

    def trader(self) -> dict:
        df = self.get_trader_info()

        temp = {
            "subscribe_symbols": {},
            "ignore_tags": [],
            "origin_size": {},
            "stop_orders": [],
            "orderbook": {},
        }

        temp["subscribe_symbols"] = {
            symbol["code"]: symbol["name"] for symbol in df["subscribe_symbols"]
        }
        temp["ignore_tags"] = df["ignore_tags"]
        temp["origin_size"] = df["origin_size"]
        if len(df["stop_orders"]) == 0:
            return temp
        stop_orders = {}
        for temp_order in df["stop_orders"]:
            order = temp_order.copy()
            stop_orders["code"] = order["code"]
            stop_orders["strategy_name"] = order["strategy_name"]
            stop_orders["stop_price"] = order["stop_price"]
            stop_orders["status"] = order["status"]
            stop_orders["direction"] = order["direction"]
            stop_orders["order_kind"] = order["order_kind"]
            stop_orders["tag"] = order["tag"]
            # stop_orders["order_quantity"] = order["order"]["quantity"]
            # stop_orders["order_price"] = order["order"]["price"]
            # stop_orders["order_price_type"] = order["order"]["price_type"]
            # stop_orders["order_action"] = order["order"]["action"]
            # stop_orders["order_time"] = str(pd.Timestamp.fromtimestamp(order["ts"]))
            temp["stop_orders"].append(stop_orders.copy())
        return temp

    def plot_strategy_figs(self, strategy_name: str) -> tuple:
        trades = self.get_trades().query(f"`sn` == @strategy_name")
        # performance = self.get_performance().loc[strategy_name]

        fig = self.__plot.plot_pnl_kbar(trades)
        fig.update_layout(title=strategy_name, title_x=0.5)
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

        fig2 = self.__plot.plot_mafe(trades)
        fig2.update_layout(title=f"{strategy_name} MAE MFE", title_x=0.5)
        return fig, fig2
