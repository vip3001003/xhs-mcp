# 1. 使用 Python 基础环境
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 复制依赖清单并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制所有代码文件
COPY . .

# 5. 启动命令
CMD ["python", "server.py"]