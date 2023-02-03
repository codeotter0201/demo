from abc import ABC, abstractmethod
import pandas as pd
import vectorbt as vbt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, StrictStr, validator
from typing import Optional, Union, List, Any
from order import OrderPlan, Order, OrderMethod

# import pandas_ta as ta
# from tqdm import tqdm
# from decimal import Decimal


class StrategyConfig(BaseModel):
    """
    symbols: stock id, future id
    freq: "T", "H", "D", "S"
    cash: capital
    direction: "L" or "S"
    excution_price: 'open' or 'close'
    product='future', 'stock', 'digital' artributes are populated with `MarketPeriod` class method
    delay_point: abs(int) | `mafe_performance column`
    delay_type: "up" | "down"
    """

    name: str = None
    freq: str = "1T"
    direction: StrictStr = None
    excution_price: str = "open"
    symbol: str = None
    product: str = None
    delay_point: float = None
    delay_type: str = None
    stop_loss: float = None
    take_profit: float = None
    sl_trailing: bool = False
    order_plans: List[OrderPlan] = []
    test_mode: bool = True
    note: str = None
    entry_note: str = None
    exit_note: str = None
    lookback_days: int = None

    @validator("direction")
    def direction_rules(cls, v, values, **kwargs):
        if v not in ["L", "l", "S", "s"]:
            raise ValueError("direction should be one of ['L', 'l', 'S', 's']")
        return v

    @validator("delay_type")
    def delay_type_rules(cls, v, values, **kwargs):
        if v not in ["up", "down"]:
            raise ValueError("delay_type should be one of ['up', 'down']")
        return v

    @validator("product")
    def product_rules(cls, v, values, **kwargs):
        if v not in ["future", "stock"]:
            raise ValueError("product should be one of ['up', 'down']")
        return v


class IStrategy(ABC):
    def __init__(
        self, config: StrategyConfig = None, order_func: object = None
    ) -> None:
        self.last_backtest_time = pd.Timestamp.now()
        self.config = config
        self.trades = pd.DataFrame()
        self.order_func = order_func

    def __call__(self, populate_enex: object) -> None:
        self.populate_enex = populate_enex
        return self

    def backtest(
        self,
        ohlcv: pd.DataFrame = None,
        entry: pd.Series = None,
        exit: pd.Series = None,
        pf_kwargs: dict = {"sl_stop": None},
    ) -> None:
        if not (isinstance(entry, pd.DataFrame) | isinstance(entry, pd.Series)):
            entry, exit = self.populate_enex(ohlcv)
        self.pf = vbt.Portfolio.from_signals(
            close=ohlcv[self.config.excution_price],
            high=ohlcv.high,
            low=ohlcv.low,
            entries=entry,
            exits=exit,
            fees=0,
            direction="longonly"
            if self.config.direction in ["L", "l"]
            else "shortonly",
            # init_cash=,
            freq=self.config.freq,
            **pf_kwargs,
        )

    def get_last_signal(self, ohlcv: pd.DataFrame) -> dict:
        self.backtest(ohlcv)
        if len(self.pf.trades.records) == 0:
            print(f"{self.config.name} does not have trades")
            return {}
        signal = self.pf.trades.records.iloc[-1]
        signal["entry_time"] = ohlcv.index[int(signal["entry_idx"])]
        signal["exit_time"] = ohlcv.index[int(signal["exit_idx"])]
        signal["entry_price"] = ohlcv[self.config.excution_price].iloc[
            int(signal["entry_idx"])
        ]
        signal["exit_price"] = ohlcv[self.config.excution_price].iloc[
            int(signal["exit_idx"])
        ]
        et = signal["entry_time"]
        xt = signal["exit_time"]
        signal["rise_max"] = ohlcv[et:xt].high.max()
        signal["fall_max"] = ohlcv[et:xt].low.min()
        signal["status"] = "Closed" if signal["status"] == 1 else "Open"
        d = -1 if signal["direction"] == 1 else 1
        signal["pnl"] = (signal["exit_price"] - signal["entry_price"]) * d
        cols = [
            "entry_time",
            "entry_price",
            "rise_max",
            "fall_max",
            "exit_price",
            "exit_time",
            "pnl",
            "direction",
            "status",
        ]
        signal = signal[cols].astype(str).to_dict()
        signal.update(self.config.dict())
        return signal

    def get_orders(self, ohlcv: pd.DataFrame) -> dict:
        signal = self.get_last_signal(ohlcv)
        if signal == {}:
            return {"signal": self.config.dict()}
        temp = Order(signal).gen_orders()
        temp["signal"] = signal
        return temp

    def create_mafe_trades_report(
        self,
        ohlcv: pd.DataFrame,
        atr: pd.Series = None,
        mafe_mode: bool = True,
        pf: vbt.Portfolio = None,
    ) -> pd.DataFrame:
        if not pf:  # 給自定義的PF，方便測試
            self.backtest(ohlcv)
            portfolio = self.pf
        else:
            portfolio = pf

        if mafe_mode:  # 如果關閉，就只是普通的回測，不產生報表
            ret = self.clean_portfolio_records(portfolio)
            raw_mafe = self.mafe(
                ohlcv.close,
                ret.entry_idx,
                ret.exit_idx,
                ret.direction,
                ohlcv.high,
                ohlcv.low,
            )
            if isinstance(atr, pd.Series):
                ret_mafe = self.convert_mafe_to_atr(raw_mafe, ret["entry_price"], atr)
            else:
                ret_mafe = self.convert_mafe_to_atr(
                    raw_mafe, ret["entry_price"], np.full(ohlcv.close.shape, 1)
                )
            self.trades = pd.concat([ret, ret_mafe], axis=1)
            self.performance = self.pnl_performance(
                self.trades.pnl, self.trades.exit_time
            )
            self.last_backtest_time = pd.Timestamp.now()
            self.config.order_plans = (
                self.order_func(self.trades)
                if self.order_func
                else self.config.order_plans
            )

    def create_mafe_trades_report_level2(
        self,
        ohlcv: pd.DataFrame,
        atr: pd.Series = None,
        mafe_mode: bool = True,
        pf: vbt.Portfolio = None,
    ) -> pd.DataFrame:
        if not pf:  # 給自定義的PF，方便測試
            self.backtest(ohlcv)
            portfolio = self.pf
        else:
            portfolio = pf

        if mafe_mode:  # 如果關閉，就只是普通的回測，不產生報表
            ret = self.clean_portfolio_records(portfolio)
            raw_mafe = self.mafe(
                ohlcv.close,
                ret.entry_idx,
                ret.exit_idx,
                ret.direction,
                ohlcv.high,
                ohlcv.low,
            )
            if isinstance(atr, pd.Series):
                ret_mafe = self.convert_mafe_to_atr(raw_mafe, ret["entry_price"], atr)
            else:
                ret_mafe = self.convert_mafe_to_atr(
                    raw_mafe, ret["entry_price"], np.full(ohlcv.close.shape, 1)
                )
            self.level2_trades = pd.concat([ret, ret_mafe], axis=1)
            self.level2_performance = self.pnl_performance(
                self.level2_trades.pnl, self.level2_trades.exit_time
            )
            self.level2_last_backtest_time = pd.Timestamp.now()

    @staticmethod
    def bar_pattens(ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        ohlcv = ohlcv_data.copy()
        # 工具
        ohlcv["ALWAYS_TRUE"] = ohlcv.index == ohlcv.index

        # KBAR TYPE
        ohlcv["FULL_BAR_RANGE"] = ohlcv.high - ohlcv.low
        ohlcv["RISE_BAR"] = ohlcv.close > ohlcv.open
        ohlcv["FALL_BAR"] = ohlcv.close < ohlcv.open

        # KBAR PATTEN
        ohlcv["FALL_BAR_N_LOW_SHAOW"] = (ohlcv.close - ohlcv.low) > (
            ohlcv.FULL_BAR_RANGE * 0.25
        )
        ohlcv["RISE_BAR_N_LOW_SHAOW"] = (ohlcv.open - ohlcv.low) > (
            ohlcv.FULL_BAR_RANGE * 0.25
        )
        ohlcv["FALL_BAR_N_HIGH_SHAOW"] = (ohlcv.high - ohlcv.open) > (
            ohlcv.FULL_BAR_RANGE * 0.25
        )
        ohlcv["RISE_BAR_N_HIGH_SHAOW"] = (ohlcv.high - ohlcv.close) > (
            ohlcv.FULL_BAR_RANGE * 0.25
        )

        # VOLUME PATTEN
        ohlcv["MORE_VOLUME"] = ohlcv.volume > ohlcv.volume.shift()  # 比前一根成交量還多
        ohlcv["LESS_VOLUME"] = ohlcv.volume < ohlcv.volume.shift()  # 比前一根成交量還少

        # MARKET TIME FILTER 開盤後30分，收盤前30分，不下單
        ENTRY_TIME_FILTER = (
            ((ohlcv.index.hour == 9) & (ohlcv.index.minute <= 30))
            | ((ohlcv.index.hour == 21) & (ohlcv.index.minute >= 30))
            | ((ohlcv.index.hour == 22) & (ohlcv.index.minute <= 1))
            | ((ohlcv.index.hour == 15) & (ohlcv.index.minute <= 30))
            | ((ohlcv.index.hour == 4) & (ohlcv.index.minute >= 30))
        )

        ohlcv["ENTRY_TIME_FILTER"] = ~ENTRY_TIME_FILTER
        return ohlcv

    @staticmethod
    def pnl_performance(
        pnl: pd.Series, entry_time: pd.Series = None, print_result: bool = False
    ) -> pd.Series:
        ret = pnl.copy()
        if isinstance(entry_time, pd.Series):
            ret.index = entry_time
        ret.index = pd.to_datetime(ret.index)
        temp = {}
        temp["trades"] = int(len(ret))
        temp["win_ratio"] = (ret > 0).sum() / len(ret)
        temp["profit_factor"] = abs(ret[ret > 0].sum() / ret[ret <= 0].sum())
        temp["recovery_factor"] = (
            ret.sum() / (ret.cumsum().cummax() - ret.cumsum()).max()
        )
        temp["expect_payoff"] = ret[(ret > 0)].mean() * temp["win_ratio"] + ret[
            ~(ret > 0)
        ].mean() * (1 - temp["win_ratio"])
        temp["kelly"] = temp["win_ratio"] - (
            (1 - temp["win_ratio"]) / temp["expect_payoff"]
        )

        # temp['sharp_ratio'] = ret[ret > 0].mean() / ret.std() # ??
        temp["sharp_ratio"] = ret.mean() / ret.std()

        temp["SQN"] = (
            temp["expect_payoff"]
            * (min(ret.groupby(ret.index.year).size().mean().round(), 100) ** 0.5)
        ) / ret.std()
        temp = pd.Series(temp)
        if print_result:
            print(temp.to_string())
        return temp

    @staticmethod
    def mafe(
        tick: pd.Series,
        entry_idx: pd.Series,
        exit_idx: pd.Series,
        direction_arr: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None,
    ) -> pd.DataFrame:
        mafe = {
            "entry_time": [],
            "exit_time": [],
            "mae_idx": [],
            "mae": [],
            "mae_lv1_idx": [],
            "mae_lv1": [],
            "mfe_idx": [],
            "mfe": [],
            "g_mfe_idx": [],
            "g_mfe": [],
            "h2c_idx": [],
            "h2c": [],
            "l2c_idx": [],
            "l2c": [],
            "edge": [],
            "ask_price": [],
            "ask_volume": [],
            "bid_price": [],
            "bid_volume": [],
            "delay_time": [],
            "delay_price": [],
            "delay_idx": [],
            "direction": [],
        }
        high = high if isinstance(high, pd.Series) else tick
        low = low if isinstance(low, pd.Series) else tick

        for le, lx, direction in zip(entry_idx, exit_idx, direction_arr):
            temp_high = high[le : lx + 1]
            temp_low = low[le : lx + 1]

            temp_h2c = temp_high.cummax() - temp_low
            cond = temp_h2c == temp_h2c.max()
            h2c_idx = temp_h2c.reset_index()[cond.values].index[0] + le
            temp_l2c = temp_high - temp_low.cummin()
            cond = temp_l2c == temp_l2c.max()
            l2c_idx = temp_l2c.reset_index()[cond.values].index[0] + le

            mafe["h2c"].append(temp_h2c.max())
            mafe["h2c_idx"].append(h2c_idx)
            mafe["l2c"].append(temp_l2c.max())
            mafe["l2c_idx"].append(l2c_idx)

            if direction == 0:
                cond = temp_low == temp_low.min()
                mae_idx = temp_low.reset_index()[cond.values].index[0] + le
                cond = temp_high == temp_high.max()
                g_mfe_idx = temp_high.reset_index()[cond.values].index[0] + le

                temp_mfe = high[le:mae_idx]
                cond = temp_mfe == temp_mfe.max()
                try:
                    mfe_idx = temp_mfe.reset_index()[cond.values].index[0] + le
                except:
                    mfe_idx = le

                temp_mae_lv1 = low[le:mfe_idx]
                cond = temp_mae_lv1 == temp_mae_lv1.min()
                try:
                    mae_lv1_idx = temp_mae_lv1.reset_index()[cond.values].index[0] + le
                except:
                    mae_lv1_idx = le

                mafe["mae"].append(low[mae_idx])
                mafe["mae_idx"].append(mae_idx)

                mafe["g_mfe"].append(high[g_mfe_idx])
                mafe["g_mfe_idx"].append(g_mfe_idx)

                mafe["mfe"].append(high[mfe_idx])
                mafe["mfe_idx"].append(mfe_idx)

                mafe["mae_lv1"].append(low[mae_lv1_idx])
                mafe["mae_lv1_idx"].append(mae_lv1_idx)
            else:
                cond = temp_high == temp_high.max()
                mae_idx = temp_high.reset_index()[cond.values].index[0] + le
                cond = temp_low == temp_low.min()
                g_mfe_idx = temp_low.reset_index()[cond.values].index[0] + le

                temp_mfe = low[le:mae_idx]
                cond = temp_mfe == temp_mfe.min()
                try:
                    mfe_idx = temp_mfe.reset_index()[cond.values].index[0] + le
                except:
                    mfe_idx = le

                temp_mae_lv1 = high[le:mfe_idx]
                cond = temp_mae_lv1 == temp_mae_lv1.max()
                try:
                    mae_lv1_idx = temp_mae_lv1.reset_index()[cond.values].index[0] + le
                except:
                    mae_lv1_idx = le

                mafe["mae"].append(high[mae_idx])
                mafe["mae_idx"].append(mae_idx)

                mafe["g_mfe"].append(low[g_mfe_idx])
                mafe["g_mfe_idx"].append(g_mfe_idx)

                mafe["mfe"].append(low[mfe_idx])
                mafe["mfe_idx"].append(mfe_idx)

                mafe["mae_lv1"].append(high[mae_lv1_idx])
                mafe["mae_lv1_idx"].append(mae_lv1_idx)

            for i, v in list(mafe.items()):
                if len(v) == 0:
                    del mafe[i]
        return pd.DataFrame(mafe)

    @staticmethod
    def clean_portfolio_records(portfolio: vbt.Portfolio):
        drop_col = [
            "id",
            "col",
            "size",
            "entry_fees",
            "exit_fees",
            "return",
            "parent_id",
        ]
        ret = portfolio.trades.records.drop(drop_col, axis=1)
        ret["entry_time"] = portfolio.close.index[ret.entry_idx]
        ret["exit_time"] = portfolio.close.index[ret.exit_idx]
        ret["pnl"] = (ret["exit_price"] - ret["entry_price"]) * ret["direction"].map(
            lambda x: 1 if x == 0 else -1
        )
        # time_diff = (portfolio.close.index[-1] - portfolio.close.index[-2]).total_seconds() / 60
        # print(f'data frequency: {time_diff}/min')
        return ret[sorted(ret.columns)]

    @staticmethod
    def convert_mafe_to_atr(ret_mafe, entry_price: pd.Series, atr: pd.Series = None):
        mafe_point = ret_mafe[[i for i in ret_mafe.columns if not "idx" in i]].apply(
            lambda x: abs(x - entry_price) if not x.name in ["h2c", "l2c"] else x
        )
        mafe_idx_atr = ret_mafe[[i for i in ret_mafe.columns if "idx" in i]].applymap(
            lambda x: atr[x]
        )
        mafe_atr = mafe_point.copy() * 0
        mafe_atr += mafe_point.values / mafe_idx_atr.values
        mafe_atr = round(mafe_atr, 3)
        return mafe_atr

    @staticmethod
    def plot_pnl_kbar(ret: pd.DataFrame, plot_duration_vol=True, show_xaxis=False):

        pnl_df = pd.DataFrame(
            dict(
                open=ret.pnl.shift().cumsum().fillna(0),
                low=-ret.mae + ret.pnl.shift().cumsum().fillna(0),
                high=ret.g_mfe + ret.pnl.shift().cumsum().fillna(0),
                close=ret.pnl.cumsum().fillna(0),
                volume=(ret.exit_time - ret.entry_time).dt.total_seconds() / (60 * 60)
                if plot_duration_vol
                else ret.pnl * 0,
            )
        ).set_index(ret.exit_time)

        fig = pnl_df.vbt.ohlc().plot(xaxis=dict(type="category"))
        fig.add_trace(
            go.Scatter(
                x=pnl_df.index, y=ret.pnl.cumsum().rolling(20).mean(), name="20MA"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pnl_df.index,
                y=ret.pnl.cumsum().rolling(20).mean()
                + ret.pnl.cumsum().rolling(20).std() * 2,
                name="UB",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pnl_df.index,
                y=ret.pnl.cumsum().rolling(20).mean()
                - ret.pnl.cumsum().rolling(20).std() * 2,
                name="DB",
            )
        )
        layout_dict = dict(
            width=1500,
            height=750,
            xaxis=dict(type="category"),
            xaxis2=dict(type="category"),
            title_x=0.5,
            xaxis_visible=True,
        )
        fig.update_layout(**layout_dict)

        xaxis_dict = dict(visible=show_xaxis)
        fig.update_xaxes(**xaxis_dict)
        return fig

    @staticmethod
    def plot_trades_ohlcv(
        ohlcv, vbt_portfolio: vbt.Portfolio, plot_vol=True, show_xaxis=False
    ):
        trades = vbt_portfolio.trades.plot()
        data = ohlcv.copy()
        bb = vbt.BBANDS.run(data.close)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
            ],
            vertical_spacing=0,
            row_heights=[0.8, 0.1, 0.1],
        )

        # include candlestick with rangeselector
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                showlegend=False,
                name="Price",
                opacity=0.75,
            ),
            secondary_y=False,
        )

        if plot_vol:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data["volume"],
                    showlegend=False,
                    opacity=0.25,
                    name="Volume",
                ),
                secondary_y=True,
            )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=vbt.ATR.run(data.high, data.low, data.close, 20).atr,
                showlegend=False,
                name="ATR",
                line=dict(width=1),
            ),
            row=2,
            col=1,
            # secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=vbt.ATR.run(data.high, data.low, data.close, 60).atr,
                showlegend=False,
                name="ATR60",
                line=dict(width=1),
            ),
            row=2,
            col=1,
            # secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bb.upper,
                showlegend=False,
                name="ub",
                marker=dict(color="gray"),
                opacity=0.25,
            ),
            row=1,
            col=1,
            # secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bb.lower,
                showlegend=False,
                name="db",
                marker=dict(color="gray"),
                opacity=0.25,
            ),
            row=1,
            col=1
            # secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bb.middle,
                showlegend=False,
                name="mb",
                marker=dict(color="gray"),
                opacity=0.25,
            ),
            row=1,
            col=1
            # secondary_y=True
        )

        for i in trades.data:
            if i.name != "Close":
                fig.add_trace(i, row=1, col=1)

        ret = vbt_portfolio.trades.records
        pnl = (data.close * 0).reset_index(drop=True)
        for i, order in ret.iterrows():
            pnl[order["exit_idx"]] = order["return"]
        pnl.index = data.index
        benchmark_pnl = data.close.pct_change()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=pnl.cumsum(),
                showlegend=False,
                name="strategy",
                marker=dict(color="red"),
                opacity=0.75,
                line=dict(width=1),
            ),
            row=3,
            col=1,
            # secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=benchmark_pnl.cumsum(),
                showlegend=False,
                name="benchmark",
                marker=dict(color="gray"),
                opacity=0.5,
                line=dict(width=1),
            ),
            row=3,
            col=1,
            # secondary_y=True
        )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis=dict(type="category"),
            xaxis2=dict(type="category"),
            height=800,
            margin={"b": 30, "l": 30, "r": 30, "t": 30},
            shapes=trades.layout.shapes,
        )
        fig.update_xaxes(visible=show_xaxis)
        return fig

    @staticmethod
    def plot_mafe(
        all_trades: pd.DataFrame = None,
        benchmarket_ohlcv: pd.DataFrame = None,
        bins=500,
    ):
        """required columns
        ['mfe', 'mae', 'g_mfe', 'pnl']
        """
        trades = all_trades.copy()
        drop_col = [
            "direction",
            "entry_idx",
            "entry_price",
            "exit_idx",
            "exit_price",
            "status",
        ]
        print("-" * 100)
        print(trades.describe().drop(drop_col, axis=1).round(4).to_string())
        print("-" * 100)
        print(
            trades.query("pnl > 0")
            .describe()
            .drop(drop_col, axis=1)
            .round(4)
            .to_string()
        )
        print("-" * 100)
        print(
            trades.query("pnl <= 0")
            .describe()
            .drop(drop_col, axis=1)
            .round(4)
            .to_string()
        )
        print("-" * 100)

        winTrades = trades[trades.pnl > 0]
        lostTrades = trades[trades.pnl <= 0]
        try:
            daily_ret = (
                trades.set_index("Exit Timestamp")
                .resample("D")
                .pnl.sum()
                .fillna(0)
                .cumsum()
            )
        except:
            trades["Exit Timestamp"] = trades["exit_time"]
            daily_ret = (
                trades.set_index(trades["exit_time"])
                .resample("D")
                .pnl.sum()
                .fillna(0)
                .cumsum()
            )
        if benchmarket_ohlcv:
            benchmarket_ret = (
                benchmarket_ohlcv.close.diff()[daily_ret.index[0] :]
                .cumsum()
                .loc[daily_ret.index]
            )
        else:
            benchmarket_ret = daily_ret

        fig = make_subplots(
            4,
            3,
            subplot_titles=(
                "MFE MAE",
                "G_MFE MAE",
                "MAE G_MFE TIMESERIES",
                "MAE W/L COUNT",
                "MFE W/L COUNT",
                "G_MFE W/L COUNT",
                "MAE/RETURN",
                "MFE/RETURN",
                "G_MFE/RETURN",
                "Profitloss",
            ),
            specs=[
                [{}, {}, {}],
                [{}, {}, {}],
                [{}, {}, {}],
                [{"rowspan": 1, "colspan": 3}, {}, {}],
            ],
            shared_xaxes=False,
        )

        ### Row 1 mae mfe scatter plot
        fig.add_trace(
            go.Scatter(
                x=winTrades.mae,
                y=winTrades.mfe,
                text=winTrades.pnl,
                mode="markers",
                marker=dict(
                    # size=winTrades.pnl_ranking,
                    # sizemode='area',
                    # sizemin=5,
                    # showscale=True,
                    color="cornflowerblue"
                ),
                marker_symbol="star",
                name="[MAE MFE SCATTER] Win",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=lostTrades.mae,
                y=lostTrades.mfe,
                text=lostTrades.pnl,
                marker=dict(
                    # size=lostTrades.pnl_ranking,
                    # sizemode='area',
                    # sizemin=5,
                    # showscale=True,
                    color="coral",
                ),
                opacity=0.35,
                mode="markers",
                marker_symbol="x",
                name="[MAE MFE SCATTER] Lost",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=winTrades.mae,
                y=winTrades.g_mfe,
                text=winTrades.pnl,
                mode="markers",
                marker=dict(
                    # size=winTrades.pnl_ranking,
                    # sizemode='area',
                    # sizemin=5,
                    # showscale=True,
                    color="cornflowerblue"
                ),
                marker_symbol="star",
                name="[MAE G_MFE SCATTER] Win",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=lostTrades.mae,
                y=lostTrades.g_mfe,
                text=lostTrades.pnl,
                marker=dict(
                    # size=lostTrades.pnl_ranking,
                    # sizemode='area',
                    # sizemin=5,
                    # showscale=True,
                    color="coral",
                ),
                opacity=0.35,
                mode="markers",
                marker_symbol="x",
                name="[MAE G_MFE SCATTER] Lost",
            ),
            row=1,
            col=2,
        )

        ### mae mfe line plot
        fig.add_trace(
            go.Scatter(
                x=trades["Exit Timestamp"],
                y=trades.g_mfe,
                text=trades.pnl,
                #  marker=dict(
                #     size=lostTrades.pnl_ranking,
                #     sizemode='area',
                #     sizemin=5,
                #  ),
                mode="lines",
                name="g_mfe",
                marker_color="cornflowerblue",
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=trades["Exit Timestamp"],
                y=trades.mae,
                text=trades.pnl,
                #  marker=dict(
                # size=lostTrades.pnl_ranking,
                # sizemode='area',
                # sizemin=5,
                # color='colar'
                #  ),
                mode="lines",
                marker_color="coral",
                name="mae",
                opacity=0.75,
            ),
            row=1,
            col=3,
        )

        ### Row2 win lose mfe
        fig.add_trace(
            go.Histogram(
                name="win mfe",
                x=winTrades.mfe,
                nbinsx=bins,
                marker_color="cornflowerblue",
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Histogram(
                name="lose mfe",
                x=lostTrades.mfe,
                nbinsx=bins,
                marker_color="coral",
                opacity=0.75,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Histogram(
                name="win g_mfe",
                x=winTrades.g_mfe,
                nbinsx=bins,
                marker_color="cornflowerblue",
            ),
            row=2,
            col=3,
        )

        fig.add_trace(
            go.Histogram(
                name="lose g_mfe",
                x=lostTrades.g_mfe,
                nbinsx=bins,
                marker_color="coral",
                opacity=0.75,
            ),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Histogram(
                name="win mae",
                x=winTrades.mae,
                nbinsx=bins,
                marker_color="cornflowerblue",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                name="lose mae",
                x=lostTrades.mae,
                nbinsx=bins,
                marker_color="coral",
                opacity=0.75,
            ),
            row=2,
            col=1,
        )

        ### Row 3 MAE MFE G_MFE
        fig.add_trace(
            go.Scatter(
                #  name='win mae',
                x=trades.pnl,
                y=trades.mae,
                #    marker=dict(
                #     size=trades.mae,
                #     sizemode='area',
                #     sizemin=1,
                #    ),
                marker_color="coral",
                mode="markers",
                marker_symbol="circle-open",
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                #  name='win mfe',
                x=trades.pnl,
                y=trades.mfe,
                #    marker=dict(
                #     size=trades.mfe,
                #     sizemode='area',
                #     sizemin=1,
                #    ),
                marker_color="LightSkyBlue",
                mode="markers",
                marker_symbol="circle-open",
            ),
            row=3,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                #  name='win g_mfe',
                x=trades.pnl,
                y=trades.g_mfe,
                #    marker=dict(
                #     size=trades.g_mfe,
                #     sizemode='area',
                #     sizemin=1,
                #    ),
                marker_color="LightSkyBlue",
                mode="markers",
                marker_symbol="circle-open",
            ),
            row=3,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                name="Profitloss",
                x=daily_ret.index,
                y=daily_ret,
                #    marker=dict(
                #     size=trades.g_mfe,
                #     sizemode='area',
                #     sizemin=1,
                #    ),
                marker_color="red",
                mode="lines",
                marker_symbol="circle",
            ),
            row=4,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                name="Profitloss[B]",
                x=daily_ret.index,
                y=benchmarket_ret,
                #    marker=dict(
                #     size=trades.g_mfe,
                #     sizemode='area',
                #     sizemin=1,
                #    ),
                marker_color="black",
                mode="lines",
                marker_symbol="circle",
            ),
            row=4,
            col=1,
        )

        fig.update_layout(
            title_x=0.5,
            margin=dict(l=20, r=50, t=50, b=20),
            showlegend=False,
            height=2000,
        )
        return fig

    @staticmethod
    def plot_trade(ohlcv: pd.DataFrame = None, temp_oredrs: pd.Series = None, bars=100):
        et, ex = temp_oredrs["entry_time"], temp_oredrs["exit_time"]
        # temp_ohlcv = ohlcv.loc[str(order.entry_time.date())].between_time('08:45', '13:45')
        # time_delta = pd.Timedelta(data_days_range, 'day')
        # temp_ohlcv = ohlcv.loc[str(temp_oredrs.entry_time - time_delta):str(temp_oredrs.exit_time + time_delta)].copy()
        et_idx, ex_idx = temp_oredrs["entry_idx"] - bars, temp_oredrs["exit_idx"] + bars
        temp_ohlcv = ohlcv.iloc[et_idx : ex_idx + 1].copy()
        min_values = temp_ohlcv.low.min()
        diff_ranger = (temp_ohlcv.close.max() - temp_ohlcv.close.min()) // 10
        pnl_type = "Win" if temp_oredrs["pnl"] > 0 else "Lost"
        fig = temp_ohlcv.vbt.ohlc.plot(
            title=f'({et} {ex}) | `{pnl_type}` `{round(temp_oredrs["pnl"], 3)}`'
        )

        slope = temp_ohlcv.open + np.nan
        slope.loc[temp_oredrs.entry_time] = temp_oredrs.entry_price
        slope.loc[temp_oredrs.exit_time] = temp_oredrs.exit_price
        fig.add_trace(
            go.Scatter(
                x=slope.index,
                y=slope,
                connectgaps=True,
                line_dash="dash",
                line_width=1,
                showlegend=False,
                line_color="#F54349" if temp_oredrs.pnl > 0 else "black",
            ),
        )

        # entry exit
        temp_ohlcv["entry_price"] = np.nan
        temp_ohlcv["exit_price"] = np.nan

        temp_ohlcv.loc[et, "entry_price"] = temp_oredrs["entry_price"]
        temp_ohlcv.loc[ex, "exit_price"] = temp_oredrs["exit_price"]
        fig.add_trace(
            go.Scatter(
                x=temp_ohlcv.index,
                y=temp_ohlcv["entry_price"],
                mode="markers",
                name="Entry",
                marker_symbol="triangle-up",
                marker=dict(color="blue", size=15, opacity=0.75),
                showlegend=False,
            ),
        )
        exit_color = "red" if temp_oredrs["pnl"] > 0 else "black"
        fig.add_trace(
            go.Scatter(
                x=temp_ohlcv.index,
                y=temp_ohlcv["exit_price"],
                mode="markers",
                name="Exit",
                marker=dict(color=exit_color, size=15, opacity=0.75),
                marker_symbol="triangle-down",
                showlegend=False,
            ),
        )

        temp_oredrs = temp_oredrs[
            [i for i in temp_oredrs[4:].index if ("time" in i)]
        ].sort_values()
        for k, v in temp_oredrs.items():
            min_values -= diff_ranger
            fig.add_vline(
                x=temp_oredrs[k], line_width=0.2, line_dash="dash", line_color="green"
            )  # , annotation_text=k, annotation_position="bottom right")
            fig.add_annotation(
                x=temp_oredrs[k],
                y=min_values,
                text=k,
                yshift=10,
            )
            # fig.add_trace(go.Scatter(x=temp_ohlcv.index,
            #                             y=temp_ohlcv['up'],
            #                             # mode='lines',
            #                             name='band-up',
            #                             line_width=0.2,
            #                             showlegend=False
            #                         )
            #             )

            # fig.add_trace(go.Scatter(x=temp_ohlcv.index,
            #                             y=temp_ohlcv['db'],
            #                             # mode='lines',
            #                             name='band-db',
            #                             line_width=0.2,
            #                             showlegend=False
            #                         ),

            #             )

            # fig.add_trace(go.Scatter(x=temp_ohlcv.index,
            #                             y=temp_ohlcv['mb'],
            #                             # mode='lines',
            #                             name='band-mb',
            #                             line_width=0.2,
            #                             showlegend=False
            #                         ),

            #             )

            # fig.add_trace(go.Scatter(x=temp_ohlcv.index,
            #                             y=temp_ohlcv['sma'],
            #                             # mode='lines',
            #                             name='sma',
            #                             line_width=0.2,
            #                             showlegend=False,
            #                             line_color='blue'
            #                         ),

            #             )

        step = max(round(len(temp_ohlcv) * 0.1), 5)

        fig.update_layout(
            xaxis=dict(type="category", tickvals=temp_ohlcv[::step]),
            xaxis2=dict(type="category", tickvals=temp_ohlcv[::step]),
            title_x=0.5,
        )
        return fig

    # def delay_entry(ohlcv, entry_idx, exit_idx, entry_price, delay_type='down', stop_point_series:pd.Series=None):
    #     if isinstance(ohlcv, pd.DataFrame):
    #         # print('delay test on ohlcv mode')
    #         price = ohlcv.low.copy() if delay_type == 'down' else ohlcv.high.copy()
    #     else:
    #         # print('delay test on tick mode')
    #         price = ohlcv.copy()

    #     mafe = {
    #         "delay_idx":[],
    #         "delay_time":[],
    #         "delay_price":[],
    #         "entry_idx":[],
    #         "exit_idx":[],
    #         "entry_price":[]
    #     }

    #     for et, ex, ep in zip(entry_idx, exit_idx, entry_price):
    #         tick_price = price[et:ex+1].copy() # ex 切片取資料需+1
    #         tick_stop_price = stop_point_series[et:ex+1].copy()
    #         stop_price = (ep + tick_stop_price) if delay_type == 'up' else (ep - tick_stop_price)

    #         delay_cond = (tick_price >= stop_price) if delay_type == 'up' else (tick_price <= stop_price)
    #         if delay_cond.sum() != 0:
    #             delay_idx = delay_cond.reset_index()[delay_cond.values].index[0] + et
    #             mafe['delay_idx'].append( delay_idx )
    #             mafe['delay_time'].append( price.index[delay_idx] )
    #             mafe['delay_price'].append( round(stop_price[delay_idx - et]) ) # 觸價後的price 不是當根最低價的price
    #         else:
    #             mafe['delay_idx'].append(np.nan)
    #             mafe['delay_time'].append(np.nan)
    #             mafe['delay_price'].append(np.nan)
    #         mafe['entry_idx'].append(et)
    #         mafe['exit_idx'].append(ex)
    #         mafe['entry_price'].append(ep)
    #     return pd.DataFrame(mafe)

    # def adjust_delay_entry_ohlcv(ohlcv, delay_trades):
    #     delay_entries = pd.Series(False, range(ohlcv.shape[0]))
    #     delay_exits = pd.Series(False, range(ohlcv.shape[0]))

    #     delay_entries.loc[delay_trades.delay_idx] = True
    #     delay_exits.loc[delay_trades.exit_idx] = True

    #     delay_strategy_ohlcv = ohlcv.copy()
    #     delay_strategy_ohlcv['entry'] = delay_entries.values
    #     delay_strategy_ohlcv['exit'] = delay_exits.values
    #     delay_prices = pd.Series(delay_strategy_ohlcv['open'].values)
    #     for idx, price in zip(delay_trades.delay_idx, delay_trades.delay_price):
    #         delay_prices[idx] = price
    #     delay_prices.index = delay_strategy_ohlcv.index
    #     # delay_strategy_ohlcv['open'] = delay_prices.values
    #     delay_strategy_ohlcv['enter_price'] = delay_prices
    #     # delay_strategy_ohlcv['open'] = ohlcv.open.copy()
    #     return delay_strategy_ohlcv

    # def create_delay_ohlcv(ohlcv, trades_idx, delay_type, stop_point_series):
    #     delay_ohlcv = ohlcv.copy()
    #     ret_delay = delay_entry(delay_ohlcv, trades_idx.entry_idx, trades_idx.exit_idx, trades_idx.entry_price, delay_type, stop_point_series).dropna()
    #     df = adjust_delay_entry_ohlcv(delay_ohlcv, ret_delay).copy()
    #     return df

    # def create_delay_portfolio(ohlcv, trades_idx, delay_type, stop_point_series, **pf_kwargs):
    #     delay_ohlcv = ohlcv.copy()
    #     ret_delay = delay_entry(delay_ohlcv, trades_idx.entry_idx, trades_idx.exit_idx, trades_idx.entry_price, delay_type, stop_point_series).dropna()
    #     delay_ohlcv = adjust_delay_entry_ohlcv(delay_ohlcv, ret_delay).copy()
    #     freq_str = str(round((ohlcv.index[-1] - ohlcv.index[-2]).total_seconds() / 60)) + 'T'
    #     pf = vbt.Portfolio.from_signals(
    #         delay_ohlcv.open,
    #         delay_ohlcv.entry,
    #         delay_ohlcv.exit,
    #         # adjust_sl_func_nb=adjust_sl_func_nb,
    #         # adjust_sl_args=(vbt.Rep('atr_arr'), vbt.Rep('high'), vbt.Rep('low'), vbt.Rep('close')),
    #         high=delay_ohlcv.high,
    #         low=delay_ohlcv.low,
    #         # broadcast_named_args=dict(
    #         #     # atr_arr=atr,
    #         #     long_num_arr=entry,
    #         # ),
    #         size=np.inf,
    #         # accumulate='addonly',
    #         direction='shortonly' if trades_idx.direction.sum() != 0 else 'longonly',
    #         upon_opposite_entry='ignore',
    #         freq=freq_str,
    #         **pf_kwargs
    #         )
    #     return pf
