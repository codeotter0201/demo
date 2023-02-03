## Env
避免明碼請自行建立.env檔案並存放於./env裡面(./env/.env)。

```bash 
mkdir env && touch env/.env
```

`.env` 檔案中含下列設定
```
SHIOAJI_USER = "YOUR_USERNAME"
SHIOAJI_API = "YOUR_API_KEY"
SHIOAJI_SECRET = "YOUR_SECRET_KEY"

### Default ###
REDIS_HOST = 'redis'
REDIS_PORT = 6379
SHIOAJI_ORDERBOOK_PATH = 'history/orderbook'
DATABASE_PATH = 'history'
IS_DRYRUN = 'Yes'
### Default ###
```

## Usage
```bash
source create_data.sh
docker-compose up -d
```

### Dashboard
[https://127.0.0.1](https://127.0.0.1)

### Backend
[https://127.0.0.1:9999/docs](https://127.0.0.1:9999/docs)

## Demo
![](https://github.com/codeotter0201/demo/blob/master/demo.gif)
