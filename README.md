# 🌸 花姑娘 2.0 AI 投顾系统

基于 **AutoGluon** 的智能量化投资决策系统。通过“行为克隆”学习历史优秀策略逻辑，并结合风险调整后的绩效优化模型，为全球核心资产提供每日交易建议。

---

## 🚀 功能特性

- **多模型集成决策**: 自动集成 CatBoost, XGBoost, LightGBM, NeuralNet 等多种顶尖算法。
- **三重模型版本**:
  - 🏆 **最强王者**: 经典行为克隆版本，稳健复刻动量趋势策略。
  - 🥀 **进化失败**: 历史实验版本（仅供对比）。
  - 🚀 **绩效优化版**: 针对未来 10 日风险调整后收益（Sharpe 逻辑）进行专项优化的最强版本。
- **自动化数据中心**: 一键同步 Tushare 最新行情，支持自动复权计算。
- **全功能回测引擎**: 模拟真实交易环境，提供年化收益、最大回撤、夏普比率、卡玛比率等专业金融指标。
- **交互式可视化**: 基于 Streamlit 打造，支持多模型横向对比及净值曲线实时渲染。

---

## 🛠️ 快速开始

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. 环境安装
建议使用 Python 3.9+ 环境：
```bash
pip install -r requirements.txt
```

### 3. 配置数据源 (Tushare)
在项目根目录创建 `.streamlit/secrets.toml`：
```toml
TS_TOKEN = "你的Tushare_Token"
```

### 4. 启动应用
```bash
streamlit run app.py
```

---

## ☁️ 网页部署 (Streamlit Cloud)

本项目已针对 Streamlit Cloud 进行优化，部署步骤如下：

1. **上传 GitHub**: 确保 `.gitignore` 允许上传 `AutogluonModels/` 下的必需模型文件夹。
2. **连接 Streamlit Cloud**: 在 Streamlit 控制台连接你的 GitHub 仓库。
3. **配置 Secrets**: 在 Streamlit Cloud 的设置界面（Advanced settings -> Secrets）中添加：
   ```toml
   TS_TOKEN = "你的Tushare_Token"
   ```
4. **部署**: 点击 Deploy 即可。

---

## 📂 目录说明

- `app.py`: Web 交互终端主程序。
- `update_data.py`: 市场数据自动同步引擎。
- `build_performance_dataset.py`: 绩效优化版数据集构建。
- `train_performance_optimized.py`: 绩效优化版模型训练脚本。
- `market_data/`: 历史行情数据存储。
- `AutogluonModels/`: 预训练好的模型权重。

---

## ⚠️ 免责声明

*   本系统仅供量化研究与辅助决策参考，**不构成任何投资建议**。
*   金融市场存在固有风险，AI 预测基于历史数据，不代表未来表现。
*   投资有风险，入市需谨慎。盈亏自负。
