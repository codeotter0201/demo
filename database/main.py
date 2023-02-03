import pandas as pd
from fastapi import FastAPI, Response, Depends
import os, gc
import uvicorn, json
from pydantic import BaseModel
import redis
from history.data import DataConfig, DataFile
from database import ShioajiDatabase
from accessor import Accessor

app = FastAPI()
logged_in_client = None


def get_client():
    global logged_in_client
    if logged_in_client is None:
        logged_in_client = ShioajiDatabase()
    return logged_in_client


def del_client():
    global logged_in_client
    if isinstance(logged_in_client, ShioajiDatabase):
        logged_in_client.logout()
        logged_in_client = None
        gc.collect()
    return logged_in_client


@app.get("/bot")
def startbot(api=Depends(get_client)):
    if not api._connected:
        api.logout()
        api.login()
        return f"Bot is working.Bot Started at {api._connected_ts}"
    else:
        return f"Bot is activated at {api._connected_ts}"


@app.delete("/bot")
def stopbot(api=Depends(del_client)):
    del_client()
    return "Stop Bot"


@app.put("/data")
def update_data(
    code: str,
    frequency: str,
    product: str,
    other: str = None,
    file_path: str = None,
    data_source: str = "shioaji",
    start_date: str = None,
    api=Depends(get_client),
):
    """
    更新下載資料，資料來源為redis或server
    """

    if isinstance(api, ShioajiDatabase):
        dc = DataConfig(
            file_path=file_path,
            data_source=data_source,
            code=code,
            frequency=frequency,
            product=product,
            other=other,
        )
        df = DataFile(dc)
        data_api = Accessor(df, api)
        data_api.download_n_update_data(start_date=start_date)
        data_1 = df.load_kbar(3)
        return f"{data_1.index[-1]} {data_1.close.iloc[-1]}"
    else:
        return f"Bot is not running"


@app.post("/data")
def subscribe_data(code: str, product: str, api=Depends(get_client)):
    """
    訂閱tick
    """
    if isinstance(api, ShioajiDatabase):
        contracts = api.get_useful_contracts(code=code, product=product)
        api.subscribe(contracts["contract_current"])
        return f"subscribe_data ok"
    else:
        return f"Bot is not running"


@app.delete("/data")
def unsubscribe_data(code: str, product: str, api=Depends(get_client)):
    """
    取消訂閱tick
    """
    if isinstance(api, ShioajiDatabase):
        contracts = api.get_useful_contracts(code=code, product=product)
        api.unsubscribe(contracts["contract_current"])
        return f"unsubscribe_data ok"
    else:
        return f"Bot is not running"


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888)
