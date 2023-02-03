from pydantic import BaseModel
from typing import Literal
import pandas as pd
import os


class DataConfig(BaseModel):
    data_source: str
    file_path: str
    product: Literal["future", "stock", "option"]
    code: str
    frequency: str
    other: str = None


class DataFile:
    def __init__(self, dc: DataConfig = None) -> None:
        self.dc = dc
        self.db_path = os.path.dirname(os.path.abspath(__file__))
        self.fn = self.create_file_full_path()
        self.fn_1T = self.create_file_full_path(get_kbar_1T_path=True)
        self.tradable_data_path = "tradable"
        self.full_data_path = "full"
        self._setup_directory()

    def create_file_full_path(self, get_kbar_1T_path: bool = False) -> str:
        dc = dict(self.dc.copy())
        if get_kbar_1T_path:
            dc["frequency"] = "1T"
        file_path = dc.pop("file_path")
        file_path_string = (
            "_".join([v for k, v in dict(dc).items() if (isinstance(v, str))]) + ".pkl"
        )
        file_path_string = os.path.join(self.db_path, file_path, file_path_string)
        return file_path_string

    def _setup_directory(self) -> None:
        dirname = os.path.join(self.db_path, self.dc.file_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def is_kbarfile_exist(self) -> bool:
        return os.path.isfile(self.fn)

    def is_kbarfile_1T_exist(self) -> bool:
        return os.path.isfile(self.fn_1T)

    def is_need_to_resample(self) -> bool:
        return self.dc.frequency != "1T"

    def is_tw_future_option(self) -> bool:
        return (self.dc.data_source == "shioaji") & (self.dc.product != "stock")

    def store_data(self, data: pd.DataFrame) -> None:
        # 儲存的資料點設在前一天的夜盤之前(t-1 ~14:00)
        # last_update_time_point = (data.index[-1].date() - pd.Timedelta(1, 'day'))
        # last_update_time_point = pd.to_datetime(last_update_time_point) + pd.Timedelta(14, 'hours')
        # last_update_time_point = str(last_update_time_point)
        # pd.to_pickle(data[:last_update_time_point], self.fn)

        # 只要call就存資料，這樣讀取時會包含未完成的kbar，但注意拼接資料的時候，本地資料需跳過未完成的kbar，再進行新資料的拼接
        if len(data) != 0:
            pd.to_pickle(data, self.fn)
        else:
            raise ValueError("Dataframe is empty stop to store data.")

    def store_1T_data(self, data: pd.DataFrame = None, rebuild: bool = False) -> None:
        if not self.is_kbarfile_1T_exist():
            pd.to_pickle(data, self.fn_1T)
        else:
            if rebuild:
                pd.to_pickle(data, self.fn_1T)
            else:
                print(
                    f"{self.fn_1T} is existed. If you want to rebuild it make rebuild arg as True"
                )

    def update_data(
        self,
        data: pd.DataFrame = None,
        start_date: str = None,
    ) -> None:
        """
        * 建立檔案，可自動改頻率
        如果有設定start_date，可依照此參數修改資料起始點
        如果資料夾存放在 basic 不接受修剪資料長度 維持最完整的資料
        """
        if self.is_need_to_resample() & self.is_tw_future_option():
            data = self.dp_resample_tw_future_data(data)

        if self.dc.file_path == self.full_data_path:
            start_date = None

        data = (
            self.dp_merge_data(self.load_kbar(), data)
            if self.is_kbarfile_exist()
            else data
        )
        data = data.loc[start_date:]
        self.store_data(data)

    def load_kbar(self, lookback_kbars: int = 0) -> pd.DataFrame:
        """
        * 沒有更新資料庫功能，但如果需要轉換頻率，會自動從1T轉換並儲存
        取得資料庫的data，最新資料為 store_data 設定的資料點，起始資料為從最新資料往前 lookback_kbars
        可自動轉換頻率，有1T資料優先選擇
        """
        lookback_kbars *= -1
        if self.dc.frequency == "1T":
            data = pd.read_pickle(self.fn)
            return data[lookback_kbars:]
        if self.is_need_to_resample() & self.is_kbarfile_1T_exist():
            # 情境假設為有1T data，需要轉換頻率，直接讀1T data，轉換後儲存，需注意欲讀取的文件大小，過大會影響效能
            print(
                f"|{self.dc.file_path}|{self.dc.code}|{self.dc.frequency}|{self.dc.other}| Read 1T data and resample"
            )
            data = self.load_1T_kbar()
            data = self.dp_resample_tw_future_data(data)  # 轉換頻率
            self.store_data(data)
            return data[lookback_kbars:]
        else:
            print(f"{self.fn} does not have file return empty df")
            return pd.DataFrame()

    def load_1T_kbar(self) -> pd.DataFrame:
        try:
            return pd.read_pickle(self.fn_1T)
        except:
            print(f"Can not read {self.fn_1T} return empty df")
            return pd.DataFrame()

    def dp_resample_tw_future_data(self, data: pd.DataFrame) -> pd.DataFrame:
        market_data, after_market_data = self.dp_split_txf_market_time(data)
        market_data.index += pd.Timedelta(60 * 15 - 1, "sec")
        market_data = self.dp_resample_from_kbar(market_data, self.dc.frequency)
        market_data.index -= pd.Timedelta(60 * 15, "sec")
        after_market_data = self.dp_resample_from_kbar(
            after_market_data, self.dc.frequency
        )
        data = pd.concat([market_data, after_market_data]).sort_index()
        return data

    @staticmethod
    def dp_split_txf_market_time(data: pd.DataFrame) -> tuple:
        market_data = data.between_time("08:00", "14:00")
        after_market_data = data[(data.index.hour < 8) | (data.index.hour > 14)]
        return market_data, after_market_data

    @staticmethod
    def dp_resample_from_kbar(data: pd.DataFrame, freq: str):
        if freq == "1T":
            return data
        df = data.copy()
        df.index = df.index - pd.Timedelta(1, "sec")
        df = (
            df.resample(freq, label="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        return df

    @staticmethod
    def dp_merge_data(old_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        目前設計資料庫資料是實時更新，最新一筆KBAR不一定會收完，所以拼接資料時，取倒數第二筆資料，這樣拼接進來的新資料會覆蓋本地最後一根K
        """
        # print(old_data[-1:], new_data[-1:])
        last_index = old_data.index[-2]
        data_concat_idx = (
            next(i for i, v in enumerate(new_data.index) if v >= last_index) + 1
        )
        new_data = new_data.iloc[data_concat_idx:].copy()
        data = pd.concat([old_data, new_data])
        return data[~data.index.duplicated(keep="last")].sort_index()
