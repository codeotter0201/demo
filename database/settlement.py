import pandas as pd
import requests, pickle, os
from bs4 import BeautifulSoup


class Settlement:
    def __init__(self) -> None:
        self.file_name = "settlement_dates.pkl"

    @staticmethod
    def clean_multichart_data(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [i.lower() for i in df.columns]
        index = pd.to_datetime(df.date + " " + df.time)
        df = df.iloc[:, 2:]
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = index
        return df

    def gen_settlement_dates(self) -> dict:
        data = dict(
            start_year="2010", start_month="01", end_year="2030", end_month="12"
        )
        r = requests.post("https://www.taifex.com.tw/cht/5/futIndxFSP", data=data)
        s = BeautifulSoup(r.text, "html.parser")
        df = {}
        for row in s.find_all("tr", {"bgcolor": "#FFFFFF", "class": "12bk"}):
            tse = row.find("td", dict(align="right", width="10%")).get_text(strip=True)
            if "-" == tse:
                continue
            date = (
                row.find("td", dict(align="middle", width="14%"))
                .get_text(strip=True)
                .replace("/", "-")
            )
            month = row.find("td", dict(align="middle", width="10%")).get_text(
                strip=True
            )
            df[date] = month
        with open(self.file_name, "wb") as f:
            pickle.dump(df, f)

    def read_dates(self, build: bool = False) -> dict:
        try:
            if build:
                self.gen_settlement_dates()
            return pd.read_pickle(self.file_name)
        except:
            self.gen_settlement_dates()
            return pd.read_pickle(self.file_name)

    def dp_back_adjust(self, data: pd.DataFrame) -> pd.DataFrame:
        settlement_dates = [i for i, v in self.read_dates().items() if "W" not in v]
        data_dates = set(data.index.date.astype(str))
        diff_values = data.open.shift(-1) - data.close
        temp = {}
        for sd in settlement_dates:
            if sd in data_dates:
                timestamp_str = sd + " " + "13:30:00"
                settlement_data = diff_values.loc[sd].between_time("08:45", "13:30")
                if len(settlement_data) == 0:
                    print(f"WARNING {sd} backadjust failed")
                    continue
                else:
                    timestamp_str = str(
                        settlement_data.index[-1]
                    )  # 寫比較複雜是為了避免13:30:00有時會沒有價格導致無法搜尋
                    temp[timestamp_str] = diff_values[timestamp_str]
        st_price = (
            pd.Series(temp).sort_index()[::-1].cumsum()
        )  # 加總所有的價差，因為是向過去價格調整，因此需要排序後，倒序累加
        st_price.index = pd.to_datetime(st_price.index)  # 轉換成時間格式
        st_price = st_price.reindex(data.index).bfill().fillna(0)  # 合併進ohlcv，補齊資料
        price_cols = [
            "open",
            "high",
            "low",
            "close",
        ]
        for col in price_cols:
            data.loc[:, col] += st_price
        return data
