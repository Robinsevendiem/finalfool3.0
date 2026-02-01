#!/bin/bash

# 获取脚本所在目录的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "🚀 正在启动量化回测系统..."

# 检查 Anaconda 是否安装
if [ -f "/opt/anaconda3/bin/activate" ]; then
    source /opt/anaconda3/bin/activate base
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
    source "$HOME/anaconda3/bin/activate" base
elif [ -f "$HOME/opt/anaconda3/bin/activate" ]; then
    source "$HOME/opt/anaconda3/bin/activate" base
else
    echo "⚠️ 未找到 Anaconda 环境，尝试使用系统 Python..."
fi

# 检查依赖是否安装
echo "📦 检查依赖环境..."
pip install -r requirements.txt > /dev/null 2>&1

# 启动 Streamlit
echo "✨ 启动应用中，请稍候..."
streamlit run app.py

# 防止窗口立刻关闭
read -p "按回车键退出..."
