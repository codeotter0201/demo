from collections import namedtuple
import pandas as pd
import streamlit as st
from worker import Worker
import plotly.graph_objects as go


def plot_bar_line_chart(line_data: pd.Series, bar_data: pd.Series):
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


@st.cache
def get_worker():
    return Worker(1)


@st.experimental_singleton
def figs():
    temp_fig = {}
    name_list = list(get_worker().get_trades().sn.unique())
    for sn in name_list[:]:
        temp_fig[sn] = {}
        (
            temp_fig[sn]["pnl_kbar"],
            temp_fig[sn]["mafe"],
        ) = get_worker().plot_strategy_figs(sn)
    return temp_fig


###############################################################################################################################################################################


def portfolio_summary(symbol: str, trades: pd.DataFrame, trades_less: pd.DataFrame):
    if (symbol == "") | (symbol == "end"):
        return

    # mode_list = ['Online', 'Testing', 'ALL']

    # for mode, col in zip(mode_list, st.columns([1 for i in mode_list])):
    #     with col:
    #         if mode == 'Online':
    #             if st.checkbox(mode, True):
    #                 ret = mode
    #         else:
    #             if st.checkbox(mode, False):
    #                 ret = mode

    # if ret == 'Online':
    #     trades = get_worker().get_trades().query(f'`symbol` == @symbol').drop(['symbol'], axis=1).reset_index(drop=True)[::-1]
    #     trades_less = get_worker().get_trades_less().query(f'`symbol` == @symbol').drop(['symbol'], axis=1).reset_index(drop=True)[::-1]
    #     trades = trades[trades.test_mode]
    #     trades_less = trades_less[trades_less.test_mode]
    # elif ret == 'Testing':
    #     trades = get_worker().get_trades().query(f'`symbol` == @symbol').drop(['symbol'], axis=1).reset_index(drop=True)[::-1]
    #     trades_less = get_worker().get_trades_less().query(f'`symbol` == @symbol').drop(['symbol'], axis=1).reset_index(drop=True)[::-1]
    #     trades = trades[~trades.test_mode]
    #     trades_less = trades_less[~trades_less.test_mode]
    # else:
    #     trades = get_worker().get_trades().query(f'`symbol` == @symbol').drop(['symbol'], axis=1).reset_index(drop=True)[::-1]
    #     trades_less = get_worker().get_trades_less().query(f'`symbol` == @symbol').drop(['symbol'], axis=1).reset_index(drop=True)[::-1]

    trades = (
        trades.query(f"`symbol` == @symbol")
        .drop(["symbol"], axis=1)
        .reset_index(drop=True)[::-1]
    )
    trades_less = (
        trades_less.query(f"`symbol` == @symbol")
        .drop(["symbol"], axis=1)
        .reset_index(drop=True)[::-1]
    )

    plot_strategies = list(
        trades.groupby("sn")
        .apply(lambda x: x.g_mfe.sum() / x.mae.sum())
        .sort_values()[::-1]
        .index
    )
    temp = list(
        trades.groupby("sn")
        .apply(lambda x: x.g_mfe.sum() / x.mae.sum())
        .sort_values()[::-1]
        .index
    )

    for sn, col in zip(plot_strategies, st.columns([1 for _ in plot_strategies])):
        with col:
            if not st.checkbox(sn, True):
                temp.remove(sn)

    if len(temp) == 0:
        st.markdown("### Please select a strategy above.👆")
        return
    else:
        trades = trades[trades["sn"].isin(temp)]
        trades_less = trades_less[trades_less["sn"].isin(temp)]

    pnl, pct = (
        trades.set_index("exit_time")["pnl"],
        trades.set_index("exit_time")["pct"],
    )
    tab1, tab2 = st.tabs(["🙏Performance", "🎈Trades"])
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            month = [f"{y}-{m}" for y, m in zip(pnl.index.year, pnl.index.month)]
            pnl = pnl.groupby(month).sum()
            pnl.index = pd.to_datetime(pnl.index)
            pnl = pnl.sort_index()
            fig3 = plot_bar_line_chart(line_data=pnl.cumsum(), bar_data=pnl)
            fig3.update_layout(title="Monthly Return[Point]", title_x=0.5)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            performance = dict(
                start_date=trades.entry_time.iloc[-1],
                monthly_avg_trades=trades.groupby(
                    [trades.entry_time.dt.month, trades.entry_time.dt.year]
                )
                .size()
                .mean(),
                monthly_avg_profit=trades.groupby(
                    [trades.exit_time.dt.month, trades.exit_time.dt.year]
                )
                .pnl.sum()
                .mean(),
                long_strategies=len([i for i in trades.sn.unique() if "LONG" in i]),
                short_strategies=len(
                    [i for i in trades.sn.unique() if "LONG" not in i]
                ),
            )
            st.dataframe(
                pd.Series(performance, name=str(symbol)),
                use_container_width=True,
                width=200,
            )
    with tab2:
        pw = st.text_input("space", label_visibility="hidden")
        if str(pw) == "otter":
            trades = trades.set_index("sn").style.format(
                subset=[
                    "pnl",
                    "entry_price",
                    "exit_price",
                    "mae",
                    "mae_lv1",
                    "mfe",
                    "g_mfe",
                    "h2c",
                    "l2c",
                    "pct",
                ],
                formatter="{:.2f}",
            )
            st.dataframe(trades, use_container_width=True)
        else:
            trades = trades_less.set_index("sn").style.format(
                subset=["pnl", "entry_price", "exit_price", "pct"], formatter="{:.2f}"
            )
            st.dataframe(trades, use_container_width=True)


###############################################################################################################################################################################


def strategy_summary(
    sn: str, trades: pd.DataFrame, trades_less: pd.DataFrame, performance: pd.DataFrame
):
    if (sn == " ") | (sn == "end "):
        return
    fig, fig2 = figs()[sn]["pnl_kbar"], figs()[sn]["mafe"]
    trades = (
        trades.query(f"`sn` == @sn").drop(["sn"], axis=1).reset_index(drop=True)[::-1]
    )
    trades_less = (
        trades_less.query(f"`sn` == @sn")
        .drop(["sn"], axis=1)
        .reset_index(drop=True)[::-1]
    )
    symbol = trades.symbol.unique()
    pnl, pct = (
        trades.set_index("exit_time")["pnl"],
        trades.set_index("exit_time")["pct"],
    )
    performance = performance.loc[sn]
    performance["point_return"] = pnl.sum()
    performance["percent_return"] = pct.sum()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💵Kbar PnL", "📊MAE MFE", "🙏Performance", "🎈Trades"]
    )
    with tab1:
        fig.update_layout(height=450, width=700, title="")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig2.update_layout(height=1800, width=700, title="")
        st.plotly_chart(fig2, use_container_width=True)
    with tab3:
        col1, col2 = st.columns([3, 1])
        with col1:
            month = [f"{y}-{m}" for y, m in zip(pnl.index.year, pnl.index.month)]
            pnl = pnl.groupby(month).sum()
            pnl.index = pd.to_datetime(pnl.index)
            pnl = pnl.sort_index()
            fig3 = plot_bar_line_chart(line_data=pnl.cumsum(), bar_data=pnl)
            fig3.update_layout(title="Monthly Return[Point]", title_x=0.5)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            st.dataframe(
                pd.Series(performance.to_dict(), name=str(symbol)),
                use_container_width=True,
                width=200,
            )

    with tab4:
        pw = st.text_input("space2", label_visibility="hidden")
        if str(pw) == "otter":
            trades = trades.style.format(
                subset=[
                    "pnl",
                    "entry_price",
                    "exit_price",
                    "mae",
                    "mae_lv1",
                    "mfe",
                    "g_mfe",
                    "h2c",
                    "l2c",
                    "pct",
                ],
                formatter="{:.2f}",
            )
            st.dataframe(trades, use_container_width=True)
        else:
            trades = trades_less.style.format(
                subset=["pnl", "entry_price", "exit_price", "pct"], formatter="{:.2f}"
            )
            st.dataframe(trades, use_container_width=True)


###############################################################################################################################################################################

st.set_page_config(
    page_title="⚡Portfolio",
    # page_icon=":otter:",  # https://icon-sets.iconify.design/fa6-solid/otter/
    layout="wide",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

st.title("Welcome to Fincode! 🎉")


about_tag, portfolio_tag, strategy_tag = st.tabs(["😎About", "📈Portfolio", "🎯Strategy"])

with about_tag:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        """
        ### 😎About
        > 提供**策略組合**與**個別策略**的可視化結果，藉由圖表的觀察，調整策略組、修正開發策略邏輯
        * 使用的價格資料集來自`Shioaji API`
        * `Redis`緩存 tick data，並即時更新
        * `Vectorbt`、`Plotly`策略運算與績效可視化圖
        * 後端利用`FastAPI`搭配`RESTFul`架構
        * 前端渲染使用`Streamlit`
        * 最後藉由`Docker`、`Linode`完成雲端部屬
        """
    with col2:
        """
        ### 📈Portfolio
        * `Performance`
            * 月績效(單利計算)
            * 策略組數據，評估策略組整體概況
            * 自定義策略組合
        * `Trades`
            * 近期交易數據
        """
    with col3:
        """
        ### 🎯Strategy
        * `Kbar PnL`
            * 顯示出每筆交易持有期間的開高低收，並且呈現出 Kbar 圖，用以分析策略整體的交易情況
        * `MAE MFE`
            * 將策略每次交易表現用可視化的方式呈現，用以分析策略整體的交易情況
        * `Performance`
            * 月績效(單利計算)
            * 策略各項數據指標，數據指標用來評價策略的好壞
        * `Trades`
            * 近期交易數據
        """

    """
    ---
    > If you have any questions, checkout my [github](https://github.com/codeotter0201) or email me codeotter0201@gmail.com 🌞
    """

trades = get_worker().get_trades()
trades_less = get_worker().get_trades_less()
performance = get_worker().get_performance()

with portfolio_tag:
    # symbols = sorted(trades.symbol.unique())
    # ret = symbols[0]
    # for symbol, col in zip(symbols, st.columns([1 for _ in symbols])):
    #     with col:
    #         if st.button(symbol):
    #             ret = symbol

    symbols = sorted(trades.symbol.unique())
    symbols = [""] + list(symbols) + ["end"]
    ret = st.select_slider("", symbols, value=symbols[1], help="\n\n".join(symbols))

    if len(symbols) != 0:
        portfolio_summary(ret, trades, trades_less)

    """
    ---
    > If you have any questions, checkout my [github](https://github.com/codeotter0201) or email me codeotter0201@gmail.com 🌞
    """

with strategy_tag:
    sns = sorted(trades.sn.unique())
    ret = sns[0]
    for sn, col in zip(sns, st.columns([1 for _ in sns])):
        with col:
            if st.button(sn):
                ret = sn

    if len(sns) != 0:
        strategy_summary(ret, trades, trades_less, performance)

    """
    ---
    > If you have any questions, checkout my [github](https://github.com/codeotter0201) or email me codeotter0201@gmail.com 🌞
    """
