import requests
import json
import pandas as pd
import logics
import time
import os
from history.data import DataConfig, DataFile


class Sentry:
    def __init__(self, mode: int = 3) -> None:
        is_test_mode = True if os.getenv('IS_DRYRUN', 'Yes') == 'Yes' else False
        self.strategy_methods = logics.get_strategies(test=is_test_mode)
        self.tradable_data_path = "tradable"
        self.full_data_path = "full"
        self.linode_ip = os.getenv("LINODE_IP")
        if mode == 1:  # production
            self.db = "http://database:8888/"
            self.pf = "http://portfolio:9999/"
        if mode == 2:  # dev
            self.db = "http://127.0.0.1:8888/"
            self.pf = "http://127.0.0.1:9999/"
        if mode == 3:
            self.db = f"http://{self.linode_ip}:8888/"
            self.pf = f"http://{self.linode_ip}:9999/"
        self.setup_directory()
        self.get_symbol_params()

    def reload_strategies(self, test:bool=True) -> None:
        self.strategy_methods = logics.get_strategies(test=test)
        self.get_symbol_params()

    def setup_directory(self):
        db_path = (
            os.getenv("DATABASE_PATH")
            if isinstance(os.getenv("DATABASE_PATH"), str)
            else "history"
        )
        reports_path = os.path.join(db_path, "reports")
        if not os.path.isdir(reports_path):
            os.mkdir(reports_path)

    @staticmethod
    def symbol_transfer(symbol: str) -> str:
        if "MXF" in symbol:
            return "TXF"
        return symbol

    def get_symbol_params(self, lookback_days: int = 90) -> None:
        """
        預設MXF的策略，索取報價的時候，自動替換為TXF，小心使用

        """
        data_config_list = []
        subscribtion_list = []
        for sn, v in self.strategy_methods.items():
            config = dict(v.config)
            config["symbol"] = self.symbol_transfer(config["symbol"])
            if (
                (config["product"] == "future")
                & ("R2" not in config["symbol"])
                & ("R1" not in config["symbol"])
            ):
                config["symbol"] += "R1"
            data_config_dict = {}
            data_config_dict["code"] = config["symbol"]
            data_config_dict["frequency"] = config["freq"]
            data_config_dict["product"] = config["product"]
            # 決定 tradable 資料起點日期
            data_config_dict["start_date"] = (
                str(pd.Timestamp.today().date() - pd.Timedelta(lookback_days, "day"))
                if not isinstance(config["lookback_days"], int)
                else str(
                    pd.Timestamp.today().date()
                    - pd.Timedelta(config["lookback_days"], "day")
                )
            )
            data_config_dict["file_path"] = self.tradable_data_path
            data_config_dict["other"] = "BA"
            if data_config_dict not in data_config_list:
                data_config_list.append(data_config_dict.copy())
            data_config_dict["other"] = None
            if data_config_dict not in data_config_list:
                data_config_list.append(data_config_dict.copy())

            subscribtion_dict = {}
            subscribtion_dict["code"] = config["symbol"]
            subscribtion_dict["product"] = config["product"]
            if subscribtion_dict not in subscribtion_list:
                subscribtion_list.append(subscribtion_dict)
        self.subscribtion_list = subscribtion_list
        self.data_config_list = data_config_list

    def update_data(
        self,
        code: str,
        frequency: str,
        product: str,
        start_date: str = None,
        file_path: str = None,
        data_source: str = "shioaji",
        other: str = None,
    ) -> None:
        """
        c.update_data(code='TXFR1', frequency='1T', product='future', start_date=None, file_path='basic', other='BA')
        c.update_data(code='TXFR1', frequency='1T', product='future', start_date=None, file_path='basic', other=None)
        c.update_data(code='TXFR1', frequency='15T', product='future', start_date=None, file_path='basic', other='BA')
        """
        params = dict(
            code=code,
            frequency=frequency,
            product=product,
            other=other,
            start_date=start_date,
            file_path=file_path,
        )
        url = self.db + "data"
        r = requests.put(url, params=params)
        if r.status_code == 200:
            return r.json()
        else:
            return r.status_code

    def run_database(self) -> None:
        url = self.db + "bot"
        return requests.get(url)

    def stop_database(self) -> None:
        url = self.db + "bot"
        return requests.delete(url)

    def subscribe_all(self) -> None:
        url = self.db + "data"
        for param in self.data_config_list:
            requests.post(url, params=param)

    def unsubscribe_all(self) -> None:
        url = self.db + "data"
        for param in self.data_config_list:
            requests.delete(url, params=param)

    def clean_backadjust_data(self) -> None:
        url = self.db + "backadjust"
        requests.put(url)
        requests.delete(url)

    def update_data_all(self) -> None:
        url = self.db + "data"
        for param in self.data_config_list:
            if (
                (param["product"] == "future")
                & ("R2" not in param["code"])
                & ("R1" not in param["code"])
            ):
                param["code"] += "R1"
            requests.put(url, params=param)

    def get_last_signal(self, strategy_name: str) -> dict:
        method = self.strategy_methods[strategy_name]

        symbol = self.symbol_transfer(method.config.symbol)
        other = "BA" if "future" == method.config.product else None
        if ("R1" not in method.config.symbol) & (method.config.product == "future"):
            symbol += "R1"

        dc = DataConfig(
            file_path=self.tradable_data_path,
            data_source="shioaji",
            code=symbol,
            frequency=method.config.freq,
            product=method.config.product,
            other=other,
        )
        qd = DataFile(dc)
        ohlcv = qd.load_kbar()
        data = method.get_last_signal(ohlcv)
        return data

    def get_last_orders(self, strategy_name: str) -> dict:
        method = self.strategy_methods[strategy_name]

        symbol = self.symbol_transfer(method.config.symbol)
        other = "BA" if "future" == method.config.product else None
        if ("R1" not in method.config.symbol) & (method.config.product == "future"):
            symbol += "R1"

        dc = DataConfig(
            file_path=self.tradable_data_path,
            data_source="shioaji",
            code=symbol,
            frequency=method.config.freq,
            product=method.config.product,
            other=other,
        )
        qd = DataFile(dc)
        ohlcv = qd.load_kbar()
        data = method.get_orders(ohlcv)
        return data

    def get_last_performance(
        self, strategy_name: str = None, lookback_days: int = 750
    ) -> tuple:
        method = self.strategy_methods[strategy_name]

        symbol = self.symbol_transfer(method.config.symbol)
        other = "BA" if "future" == method.config.product else None
        if ("R1" not in method.config.symbol) & (method.config.product == "future"):
            symbol += "R1"

        dc = DataConfig(
            file_path=self.full_data_path,
            data_source="shioaji",
            code=symbol,
            frequency=method.config.freq,
            product=method.config.product,
            other=other,
        )
        df = DataFile(dc)
        self.update_data(**dc.dict())
        start_date_str = str(
            pd.Timestamp.today().date() - pd.Timedelta(lookback_days, "day")
        )
        ohlcv = df.load_kbar().loc[start_date_str:]
        method.create_mafe_trades_report(ohlcv)
        temp_trades = method.trades.assign(sn=method.config.name)
        temp_trades["benchmark"] = ohlcv.close.loc[temp_trades.exit_time].values
        temp_trades["pct"] = temp_trades.direction.map(
            lambda x: -1 if (x == "1") | (x == 1) else 1
        ) * (
            (temp_trades.exit_price - temp_trades.entry_price) / temp_trades.entry_price
        )
        temp_trades["symbol"] = method.config.symbol
        temp_trades["test_mode"] = method.config.test_mode
        temp_performance = method.performance
        return temp_trades, temp_performance, method.config

    def test_method(
        self, method: object, lookback_days: int = 750, fast_mode: bool = False
    ) -> tuple:
        symbol = self.symbol_transfer(method.config.symbol)
        other = "BA" if "future" == method.config.product else None
        if ("R1" not in method.config.symbol) & (method.config.product == "future"):
            symbol += "R1"

        dc = DataConfig(
            file_path=self.full_data_path,
            data_source="shioaji",
            code=symbol,
            frequency=method.config.freq,
            product=method.config.product,
            other=other,
        )
        df = DataFile(dc)
        if not fast_mode:
            self.update_data(**dc.dict())
        start_date_str = str(
            pd.Timestamp.today().date() - pd.Timedelta(lookback_days, "day")
        )
        ohlcv = df.load_kbar().loc[start_date_str:]
        method.ohlcv = ohlcv.copy()
        method.create_mafe_trades_report(ohlcv)
        temp_trades = method.trades.assign(sn=method.config.name)
        temp_trades["benchmark"] = ohlcv.close.loc[temp_trades.exit_time].values
        temp_trades["pct"] = temp_trades.direction.map(
            lambda x: -1 if (x == "1") | (x == 1) else 1
        ) * (
            (temp_trades.exit_price - temp_trades.entry_price) / temp_trades.entry_price
        )
        temp_trades["symbol"] = method.config.symbol
        temp_trades["test_mode"] = method.config.test_mode
        temp_performance = method.performance
        return temp_trades, temp_performance, method.config

    def get_method_ohlcv(
        self, method: object, lookback_days: int = 750, fast_mode: bool = False
    ) -> tuple:
        symbol = self.symbol_transfer(method.config.symbol)
        other = "BA" if "future" == method.config.product else None
        if ("R1" not in method.config.symbol) & (method.config.product == "future"):
            symbol += "R1"

        dc = DataConfig(
            file_path=self.full_data_path,
            data_source="shioaji",
            code=symbol,
            frequency=method.config.freq,
            product=method.config.product,
            other=other,
        )
        df = DataFile(dc)
        return df.load_kbar()

    def gen_signals_pickle(self) -> None:
        temp = {}
        for k, method in self.strategy_methods.items():
            data = self.get_last_signal(k)
            data = {k: v for k, v in data.items() if "note" not in k}
            temp[k] = data
        pd.to_pickle(temp, "history/reports/signals.pkl")

    def gen_orders_pickle(self) -> None:
        temp = {}
        for k, method in self.strategy_methods.items():
            data = self.get_last_orders(k)
            data["signal"] = {
                k: v for k, v in data["signal"].items() if "note" not in k
            }
            temp[k] = data
        pd.to_pickle(temp, "history/reports/orders.pkl")

    def gen_portfolio_pickle(self, lookback_days: int = 750) -> None:
        temp_trades = pd.DataFrame()
        temp_performance = {}
        for k, method in self.strategy_methods.items():
            trades, performance, config = self.get_last_performance(k, lookback_days)
            temp_trades = pd.concat([temp_trades, trades])
            temp_performance[k] = performance
        temp_performance = pd.DataFrame(temp_performance)
        pd.to_pickle(
            temp_trades.reset_index().astype(str).to_dict("records"),
            "history/reports/trades.pkl",
        )
        pd.to_pickle(
            temp_performance.T.reset_index().astype(str).to_dict("records"),
            "history/reports/performance.pkl",
        )

    def run(self) -> None:
        time.sleep(10)
        self.update_data_all()
        self.gen_portfolio_pickle()
        self.gen_signals_pickle()
        self.gen_orders_pickle()

if __name__ == "__main__":
    from vectorbt import ScheduleManager

    con = Sentry(1)

    # initial
    print(con.data_config_list)
    print(con.subscribtion_list)
    
    con.run()

    # start quene
    sm = ScheduleManager()
    sm.every("day", "14:30").do(con.subscribe_all)
    sm.every("day", "14:29").do(con.unsubscribe_all)
    sm.every("day", "14:25").do(con.gen_portfolio_pickle)
    sm.every("day", "14:24").do(con.run_database)
    sm.every("day", "14:23").do(con.stop_database)

    sm.every("day", "08:15").do(con.subscribe_all)
    sm.every("day", "08:14").do(con.unsubscribe_all)
    sm.every("day", "08:10").do(con.gen_portfolio_pickle)
    sm.every("day", "08:09").do(con.run_database)
    sm.every("day", "08:08").do(con.stop_database)

    sm.every(5, "seconds").do(con.update_data_all)
    sm.every(5, "seconds").do(con.gen_orders_pickle)
    sm.start()
