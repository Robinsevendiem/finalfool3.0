#!/bin/bash
# 获取脚本所在目录的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=================================================="
echo "   正在启动 花姑娘 2.0 AI 投顾系统..."
echo "=================================================="

# 检测并激活虚拟环境
if [ -d ".venv" ]; then
    echo "✅ 检测到虚拟环境 (.venv)，正在激活..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "✅ 检测到虚拟环境 (venv)，正在激活..."
    source venv/bin/activate
else
    echo "ℹ️ 未检测到本地虚拟环境目录，将使用系统默认 Python 环境。"
fi

# 检查 Streamlit 是否已安装
if ! command -v streamlit &> /dev/null; then
    echo "⚠️ 未找到 'streamlit' 命令。"
    echo "正在尝试自动安装依赖 (需要网络)..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 安装失败。请检查网络或手动运行 'pip install -r requirements.txt'"
        read -p "按回车键退出..."
        exit 1
    fi
fi

# 运行应用
echo "🚀 正在启动服务..."
echo "正在为您自动打开浏览器，请稍候..."
echo "如果浏览器没有自动弹出，请手动在浏览器地址栏输入：http://127.0.0.1:8501"
echo "--------------------------------------------------"

# 强制禁用 Streamlit 欢迎弹窗和统计收集，确保启动顺畅
export STREAMLIT_CONFIG_DIR="$DIR/.streamlit"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 启动应用
# 使用 echo "" 自动跳过首次启动的邮箱询问弹窗
# 强制使用 127.0.0.1 确保本地访问稳定性
echo "" | streamlit run app.py --server.address 127.0.0.1 --server.port 8501
