import os, time
import pandas as pd
from datetime import datetime
import shioaji as sj
from shioaji.contracts import BaseContract


class ShioajiRouter:
    def __init__(
        self, api_key: str = None, secret_key: str = None, uid: str = None
    ) -> None:
        self.api = sj.Shioaji()
        self.contracts = None
        self._id = uid if isinstance(uid, str) else os.getenv("SHIOAJI_USER")
        self._api_key = (
            api_key if isinstance(api_key, str) else os.getenv("SHIOAJI_API")
        )
        self._secret_key = (
            secret_key if isinstance(secret_key, str) else os.getenv("SHIOAJI_SECRET")
        )
        self._connected = False
        self._connected_ts = None

    def login(self) -> None:
        self.api.login(
            self._api_key,
            self._secret_key,
        )
        time.sleep(15)
        self._connected = True
        self._connected_ts = pd.Timestamp.now(tz="Asia/Taipei")
        self.contracts = self._update_contracts()

    def logout(self) -> None:
        self._connected = False
        self._connected_ts = None
        self.logout()

    def _update_contracts(self) -> None:
        contracts = {
            code: contract
            for name, iter_contract in self.api.Contracts
            for code, contract in iter_contract._code2contract.items()
        }
        return contracts

    def get_current_stkfut_contract(self, code: str) -> BaseContract:
        current_month = self.contracts["TXFR1"].delivery_month
        return next(
            v
            for k, v in self.contracts.items()
            if (v.delivery_month == current_month)
            & ((code[:3] in k) | (code[:4] in v.underlying_code))
            & ("R1" not in k)
        )

    def get_next_stkfut_contract(self, code: str) -> BaseContract:
        next_month = self.contracts["TXFR2"].delivery_month
        return next(
            v
            for k, v in self.contracts.items()
            if (v.delivery_month == next_month)
            & ((code[:3] in k) | (code[:4] in v.underlying_code))
            & ("R2" not in k)
        )

    def get_R1_stkfut_contract(self, code: str) -> BaseContract:
        current_month = self.contracts["TXFR1"].delivery_month
        return next(
            v
            for k, v in self.contracts.items()
            if (v.delivery_month == current_month)
            & ((code[:3] in k) | (code[:4] in v.underlying_code))
            & ("R1" in k)
        )

    def get_R2_stkfut_contract(self, code: str) -> BaseContract:
        next_month = self.contracts["TXFR2"].delivery_month
        return next(
            v
            for k, v in self.contracts.items()
            if (v.delivery_month == next_month)
            & ((code[:3] in k) | (code[:4] in v.underlying_code))
            & ("R2" in k)
        )

    def get_useful_contracts(self, code: str, product: str) -> dict:
        if product == "future":
            contract_basic = (
                self.get_R1_stkfut_contract(code)
                if "R2" not in code
                else self.get_R2_stkfut_contract(code)
            )
            contract_current = self.get_current_stkfut_contract(contract_basic.code)
            contract_next = self.get_next_stkfut_contract(contract_basic.code)
            if contract_basic.delivery_date.replace("/", "-") == str(
                pd.Timestamp.today().date()
            ):
                print("Todat is settlement date transfer contracts to R2")
                contract_basic = self.get_R2_stkfut_contract(code)
                contract_current = contract_next
        elif product == "stock":
            contract_basic = self.contracts[code]
            contract_current = self.contracts[code]
            contract_next = self.contracts[code]
        else:
            print("Can not find product type in config")
        return dict(
            contract_basic=contract_basic,
            contract_current=contract_current,
            contract_next=contract_next,
        )

    def get_kbar(self, contract: BaseContract, start_date: str = None) -> pd.DataFrame:
        """
        如果傳入的start_date為None，預設只傳回最近一天的資料
        """
        print(f"Download {contract.code} from server since {start_date}.")
        f = lambda x: pd.to_datetime(x) if x.name == "ts" else x
        start_date = (
            start_date
            if isinstance(start_date, str)
            else str(pd.Timestamp.today().date() - pd.Timedelta(1, "day"))
        )
        end_date = str(pd.Timestamp.today().date() + pd.Timedelta(1, "day"))
        diff_dates = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        timeout_int = max(
            diff_dates.days * 75 * 1.2, 30000
        )  # 75是下載1004天資料所需的秒數，1.2是寬容倍數，計算後得出timeout的數值
        temp = pd.DataFrame(
            dict(self.api.kbars(contract, start_date, end_date, timeout=timeout_int))
        )
        temp = temp.apply(f).set_index("ts").drop("Amount", axis=1)
        temp.columns = [i.lower() for i in temp.columns]
        return temp

    def get_tick(self, contract: BaseContract, lookback_days: int = 1) -> pd.DataFrame:
        start_date = str(
            pd.Timestamp.today().date() - pd.Timedelta(lookback_days + 20, "day")
        )
        t = self.get_kbar(contract, start_date).index
        dates = sorted([str(i) for i in set(t.date)])
        dates.append(str(pd.to_datetime(t[-1]).date() + pd.Timedelta(1, "day")))
        df = pd.DataFrame()
        f = lambda x: pd.to_datetime(x) if x.name == "ts" else x

        for d in dates[-lookback_days:]:
            temp = self.api.ticks(contract, d)
            df = pd.concat([df, pd.DataFrame(dict(temp))])

        df = df.apply(f).set_index("ts")
        df["split_BS"] = df.tick_type.apply(lambda x: -1 if x == 2 else 1) * df.volume
        df["split_BS"] *= (df.tick_type != 0).astype(int)
        return df

    def get_amount_rank_200(self, colorful: bool = True) -> pd.DataFrame:
        scanner = self.api.scanners(
            scanner_type=sj.constant.ScannerType.AmountRank, count=200
        )
        scanner_df = pd.DataFrame([dict(i) for i in scanner])
        scanner_df.ts = pd.to_datetime(scanner_df.ts)

        col = [
            "price_range",
            "volume",
            "amount",
            "bid_orders",
            "bid_volumes",
            "ask_orders",
            "ask_volumes",
            "buy_price",
            "buy_volume",
            "sell_price",
            "sell_volume",
            "tick_type",
        ]
        scanner_df = scanner_df.drop(columns=col).assign(
            pct_change=scanner_df.change_price
            / (scanner_df.close - scanner_df.change_price)
        )
        scanner_df = scanner_df.assign(volitility=scanner_df["pct_change"].abs())
        if colorful:
            return scanner_df.sort_values("volitility")[::-1].style.bar(
                "pct_change", align="mid", color=["#5fba8d", "#d65f5f"]
            )
        return scanner_df.sort_values("volitility")[::-1]

    def get_stock_daily_ohlcv(self, n_days: int = 100) -> dict:
        n_days = pd.date_range(
            pd.Timestamp.today().date() - pd.Timedelta(n_days, "day"),
            pd.Timestamp.today().date(),
        )
        temp = pd.DataFrame()
        for d in n_days:
            y = pd.DataFrame(
                dict(self.api.daily_quotes(d.to_pydatetime().date(), timeout=90000))
            )
            temp = pd.concat([temp, y])

        temp_ohlcv = {}
        col = ["Low", "High", "Volume", "Amount", "Open", "Close"]
        for col_name in col:
            temp_ohlcv[col_name[:1].lower()] = temp.dropna().pivot_table(
                col_name, "Date", "Code"
            )
        return temp_ohlcv

    def subscribe(self, contract: BaseContract, quote_type: str = "tick") -> None:
        quote_type = (
            sj.constant.QuoteType.Tick
            if quote_type == "tick"
            else sj.constant.QuoteType.BidAsk
        )
        self.api.quote.subscribe(
            contract, quote_type=quote_type, version=sj.constant.QuoteVersion.v1
        )

    def unsubscribe(self, contract: BaseContract, quote_type: str = "tick") -> None:
        quote_type = (
            sj.constant.QuoteType.Tick
            if quote_type == "tick"
            else sj.constant.QuoteType.BidAsk
        )
        self.api.quote.unsubscribe(
            contract, quote_type=quote_type, version=sj.constant.QuoteVersion.v1
        )
