FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt-get update && apt-get upgrade -y && \
    apt-get install libgl1 libgomp1 libglib2.0-0 ffmpeg vim wget curl zip unzip g++ build-essential procps -y && \
    mkdir /app

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到工作目录
COPY . /app

RUN uv sync --frozen
EXPOSE 1995
CMD ["uv", "run", "python", "src/run.py"]