## Env
避免明碼請自行建立.env檔案並存放於./env裡面(./env/.env)。

```bash 
mkdir env
touch env/.env
```

`.env` 檔案中含下列設定
```
REDIS_HOST = 'redis'
REDIS_PORT = 6379

SHIOAJI_USER = "YOUR_USERNAME"
SHIOAJI_API = "YOUR_API_KEY"
SHIOAJI_SECRET = "YOUR_SECRET_KEY"
SHIOAJI_ORDERBOOK_PATH = 'history/orderbook'
```

## Demo
![](https://github.com/codeotter0201/demo/blob/master/demo.gif)
