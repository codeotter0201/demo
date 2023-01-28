from backend.shioaji_router import ShioajiRouter
from shioaji import TickFOPv1, TickSTKv1, BidAskSTKv1, BidAskFOPv1, Exchange
import os, redis, json
import pandas as pd
from datetime import datetime


class ShioajiDatabase(ShioajiRouter):
    def __init__(self, api_key: str = None, secret_key: str = None, dev=False) -> None:
        super().__init__(api_key, secret_key, dev)
        redis_host = (
            os.getenv("REDIS_HOST")
            if isinstance(os.getenv("REDIS_HOST"), str)
            else "localhost"
        )
        redis_port = (
            os.getenv("REDIS_PORT")
            if isinstance(os.getenv("REDIS_PORT"), str)
            else "6379"
        )
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.login()
        self.api.set_context(self.redis)
        self._redis_minid_timestamp = datetime.now().timestamp() * 1000

        @self.api.on_tick_fop_v1(bind=True)
        def Quote_callback_fop_v1_tick(self, exchange: Exchange, tick: TickFOPv1):
            channel = tick.code
            data = tick.to_dict(raw=True)
            data = dict(
                code=data["code"],
                datetime=data["datetime"],
                close=data["close"],
                volume=data["volume"],
                tick_type=data["tick_type"],
                simtrade=data["simtrade"],
            )
            data = json.dumps(data)
            self.xadd(channel, {"tick": data}, maxlen=50000)

        @self.api.on_tick_stk_v1(bind=True)
        def Quote_callback_stk_v1_tick(self, exchange: Exchange, tick: TickSTKv1):
            channel = tick.code
            data = tick.to_dict(raw=True)
            data = dict(
                code=data["code"],
                datetime=data["datetime"],
                close=data["close"],
                volume=data["volume"],
                tick_type=data["tick_type"],
                simtrade=data["simtrade"],
            )
            data = json.dumps(data)
            self.xadd(channel, {"tick": data}, maxlen=50000)

        @self.api.on_bidask_fop_v1(bind=True)
        def Quote_callback_fop_v1_bidask(self, exchange: Exchange, bidask: BidAskFOPv1):
            channel = bidask.code
            self.xadd(
                channel, {"bidask": json.dumps(bidask.to_dict(raw=True))}, maxlen=50000
            )

        @self.api.on_bidask_stk_v1(bind=True)
        def Quote_callback_stk_v1_bidask(self, exchange: Exchange, bidask: BidAskSTKv1):
            channel = bidask.code
            data = bidask.to_dict(raw=True)
            data = json.dumps(data)
            self.xadd(channel, {"bidask": data}, maxlen=50000)

    def get_redis_data(
        self, channel_key: str, quote_type: str = "tick", timestamp_id: int = None
    ) -> pd.DataFrame:
        """
        timestamp ID 不含小數點，所以會轉換成整數，單位為毫秒，用當下時間轉換需*100。設定0取得所有資料。
        """
        start_timestamp = (
            timestamp_id
            if isinstance(timestamp_id, int)
            else self._redis_minid_timestamp
        )
        raw_data = self.redis.xread({channel_key: f"{int(start_timestamp)}-0"})
        if len(raw_data) != 0:
            data = [json.loads(i[-1][quote_type]) for i in raw_data[-1][-1]]
            data = pd.DataFrame(data).set_index("datetime")
            data.index = pd.to_datetime(data.index)
            data.close = data.close.astype(float)
            return data
        else:
            print(
                f"{channel_key} dose not have enough data in redis since |{datetime.fromtimestamp(start_timestamp/1000)}|, return empty pd.DataFrame()"
            )
            return pd.DataFrame()

    def dp_ticks_to_1T(
        self,
        data: pd.DataFrame,
        drop_firtst_kbar: bool = True,
        drop_simtrade: bool = True,
    ) -> pd.DataFrame:
        """
        預設第一根K去除，simtrade也去除
        """
        data = data[data["simtrade"] != 1].copy() if drop_simtrade else data.copy()
        r = data.resample("1T", label="right")
        data = r.close.ohlc().assign(volume=r.volume.sum(), trades=r.size())
        data = data.iloc[1:] if drop_firtst_kbar else data
        return data

    def get_redis_kbar_data(
        self, channel_key: str, quote_type: str = "tick", timestamp_id: int = None
    ) -> pd.DataFrame:
        raw_data = self.get_redis_data(channel_key, quote_type, timestamp_id)
        if len(raw_data) == 0:
            return pd.DataFrame()
        kbar_data = self.dp_ticks_to_1T(raw_data)
        if len(kbar_data) == 0:
            return pd.DataFrame()
        return kbar_data
