FROM python:3.8.6-slim

#這行之後代表volume映射的./app:/app --> . 從此開始當下資料夾都是在/app以內了
WORKDIR /home/work

# command line的起點是database，這裡有docker-compose，up 啟動時若沒有images
# docker開始構建images，這時的起點是映射後的/app，且這時是空資料夾，如果要使用檔案
# 就想成docker正在宿主機/app裡面，複製檔案進去容器內的/app
ADD requirements.txt /home/work

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# RUN useradd -ms /bin/bash bot

# USER bot