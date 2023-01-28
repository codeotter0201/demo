from Strategy import IStrategy, Config, OrderMethod, OrderPlan
import pandas as pd
import vectorbt as vbt
import numpy as np

bar_pattens = IStrategy.bar_pattens


@IStrategy(
    Config(
        name="trend_rsi_up_15T_STOP_FALLING_LONG",
        symbol="MXF",
        product="future",
        freq="15T",
        direction="L",
        excution_price="open",
        delay_point=1,
        delay_type="down",
        stop_loss=None,
        take_profit=None,
        sl_trailing=False,
        test_mode=False,
        note="運用mafe分析，大停損位置，但因為出場手段不好，所以封存, all_mae_75",
        order_plans=[
            OrderPlan(
                entry_order=OrderMethod(
                    order_type="touch",
                    price_type="market",
                    stop_followed_price="entry_price",
                ),
                # sp_exit_order=OrderMethod(order_type='touch', price_type='market', stop_followed_price='rise_max'),
                # tp_exit_order=OrderMethod(order_type='touch', price_type='market', stop_followed_price='entry_price'),
                exit_order=OrderMethod(
                    order_type="touch",
                    price_type="market",
                    stop_followed_price="exit_price",
                ),
            )
        ],
    )
)
def run(ohlcv: pd.DataFrame):
    window = 480
    brk_window = 320
    quantile_n = 0.65
    rsi = vbt.RSI.run(ohlcv.close, window).rsi

    RISE_GAP_FILTER = (ohlcv.open > (ohlcv.high.shift() * 1.005)).rolling(4).sum() == 0
    LONG_TREND = rsi > rsi.rolling(window).quantile(quantile_n)
    # SHORT_TREND = (ohlcv['rsi'] < ohlcv['rsi'].rolling(window).quantile(.35))
    STOP_FALLING = (ohlcv.low == ohlcv.low.rolling(brk_window).min()).rolling(
        brk_window
    ).sum() == 0
    STOP_RISING = (ohlcv.high == ohlcv.high.rolling(brk_window).max()).rolling(
        brk_window
    ).sum() == 0

    entry = STOP_FALLING & LONG_TREND & RISE_GAP_FILTER
    exit = STOP_RISING  # | ~LONG_TREND

    entry, exit = vbt.signals.nb.clean_enex_1d_nb(np.array(entry), np.array(exit), True)
    return entry, exit
