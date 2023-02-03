from history.data import DataConfig, DataFile
from database import ShioajiDatabase
from settlement import Settlement
import pandas as pd


class Accessor:
    """
    檢查 1T、1T_BA ，如果沒有就自動建立這兩個檔案
    更新 1T 資料
    """

    def __init__(self, df: DataFile, api: ShioajiDatabase) -> None:
        self.df = df
        self.api = api
        self.settlement = Settlement()
        self.contracts = api.get_useful_contracts(df.dc.code, df.dc.product)
        self.setup_BA_data_by_original_1T_data()
        self.setup_data()

    def setup_data(self) -> None:
        if not self.df.is_kbarfile_1T_exist():
            print(f"{self.df.fn_1T} does not exist. creating data...")
            self.create_data()

    def setup_BA_data_by_original_1T_data(self) -> None:
        """
        如果這個商品需要backadjust，先嘗試搜尋有沒有original 1T 可以提供修改，若沒有就 pass
        """
        if (
            (self.df.dc.other == "BA")
            & (self.df.dc.data_source == "shioaji")
            & (self.df.dc.product == "future")
        ):
            dc_1T = self.df.dc.copy()
            dc_1T.other = None
            df_1T = DataFile(dc_1T)
            if df_1T.is_kbarfile_1T_exist():
                # 取出原本的1T data，檢查是否有重複，backadjust，最後儲存
                data = df_1T.load_1T_kbar()
                data = self.kbar_duplicated_clean(data)
                data = self.settlement.dp_back_adjust(data)
                self.df.store_1T_data(data)
                self.df.load_kbar()

    def kbar_duplicated_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # 檢查是否有重複kbar資料
        duplicated_kbars = data.index.duplicated().sum()
        if duplicated_kbars != 0:
            print(
                f"WARNING {self.df.fn_1T} data download from server has duplicated {duplicated_kbars} kbars"
            )
        data = self.df.dp_merge_data(data[:-1000], data)
        return data

    def create_data(self, start_date: str = "2018-01-01") -> None:
        """
        強制下載並建立資料，預設起始日期為'2018-01-01'
        """
        data = self.api.get_kbar(self.contracts["contract_basic"], start_date)
        data = self.kbar_duplicated_clean(data)

        # 如果資料需要backadjust 且為期貨，就執行BA，最後再儲存
        if (
            (self.df.dc.other == "BA")
            & (self.df.dc.data_source == "shioaji")
            & (self.df.dc.product == "future")
        ):
            data = self.settlement.dp_back_adjust(data)

        self.df.store_1T_data(data, True)

    def download_n_update_data(self, start_date: str = None) -> None:
        """
        下載資料，如果redis有足夠的1T資料，而且redis資料第一筆[0]比本地資料倒數第二筆[-2]還舊，就從redis抓資料，反之就從Server下載資料，最後儲存
        """
        contract = self.contracts["contract_current"]
        local_1T_data = self.df.load_1T_kbar()
        new_data = self.api.get_redis_kbar_data(contract.code)
        if len(new_data) == 0:
            new_data = self.api.get_kbar(
                self.contracts["contract_basic"],
                str(local_1T_data.index[-1].date() - pd.Timedelta(1, "day")),
            )
        else:
            if new_data.index[0] > local_1T_data.index[-2]:
                new_data = self.api.get_kbar(
                    self.contracts["contract_basic"],
                    str(local_1T_data.index[-1].date() - pd.Timedelta(1, "day")),
                )
            else:
                # 不做任何動作，使用redis取出來的資料
                pass
        data = self.df.dp_merge_data(old_data=local_1T_data, new_data=new_data)
        data = data.loc[start_date:].copy()
        self.df.store_1T_data(data, rebuild=True)
