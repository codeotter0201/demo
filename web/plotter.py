from abc import ABC, abstractmethod
import pandas as pd
import vectorbt as vbt
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Plotter(ABC):
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
    def plot_mafe(
        all_trades: pd.DataFrame = None,
        benchmarket_ohlcv: pd.DataFrame = None,
        bins=500,
    ):
        """Require Coloumns
        ['mfe', 'mae', 'g_mfe', 'pnl']
        """
        trades = all_trades.copy()
        # drop_col = ['direction', 'entry_idx', 'entry_price', 'exit_idx', 'exit_price', 'status']
        # print('-'*100)
        # print(trades.describe().drop(drop_col, axis=1).round(4).to_string())
        # print('-'*100)
        # print(trades.query('pnl > 0').describe().drop(drop_col, axis=1).round(4).to_string())
        # print('-'*100)
        # print(trades.query('pnl <= 0').describe().drop(drop_col, axis=1).round(4).to_string())
        # print('-'*100)

        winTrades = trades[trades.pnl > 0]
        lostTrades = trades[trades.pnl <= 0]
        try:
            daily_ret = (
                trades.set_index("Exit Timestamp")
                .resample("D")
                .sum()
                .pnl.fillna(0)
                .cumsum()
            )
        except:
            trades["Exit Timestamp"] = trades["exit_time"]
            daily_ret = (
                trades.set_index(trades["exit_time"])
                .resample("D")
                .sum()
                .pnl.fillna(0)
                .cumsum()
            )
        # benchmarket_ret = ohlcv.close.diff()[daily_ret.index[0]:].cumsum().loc[daily_ret.index]
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

        """
        benchmark_ret = method1.ohlcv.resample('D').close.last().dropna().diff().cumsum().fillna(0)
        strategy_ret = method1.trades.set_index('Exit Timestamp').pnl.reindex(method1.ohlcv.index).resample('D').sum().fillna(0).cumsum()
        figs.add_trace(go.Scatter(x=benchmark_ret.index,
                                y=benchmark_ret,
                                text=benchmark_ret,
                                mode='lines',
                                marker_color='black',
                                name='benchmark ret',
                                ), row=3, col=1
                    )
        figs.add_trace(go.Scatter(x=benchmark_ret.index,
                                y=strategy_ret,
                                text=strategy_ret,
                                mode='lines',
                                marker_color='red',
                                name='strategy ret',
                                ), row=3, col=1
                    )
        """

        fig.update_layout(
            title_x=0.5, margin=dict(l=20, r=50, t=50, b=20), showlegend=False
        )
        # fig.update_traces(hoverinfo='skip', hovertemplate=None)
        return fig

    @staticmethod
    def plot_linebar_chart(line_data: pd.Series, bar_data: pd.Series):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=line_data.index,
                y=line_data,
                name="Cumulative",
            )
        )

        fig.add_trace(
            go.Bar(
                x=line_data.index,
                y=bar_data,
                name="Monthly Profit",
            )
        )
        return fig
