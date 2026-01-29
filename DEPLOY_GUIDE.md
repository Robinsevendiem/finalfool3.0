# 🚀 花姑娘 2.0 部署指南 (GitHub + Streamlit Cloud)

本指南将帮助您将 AI 投顾系统发布到互联网，让您的朋友也能访问。

## 第一步：准备 GitHub 仓库

1.  登录 [GitHub](https://github.com/)。
2.  点击右上角的 **+** 号 -> **New repository**。
3.  **Repository name** 填入 `huaguniang-ai` (或您喜欢的名字)。
4.  选择 **Public** (公开) 或 **Private** (私有) 均可。
    *   *注意：如果选择 Public，请确保代码中没有硬编码密码（我们已经处理了代码，现在是安全的）。*
5.  点击 **Create repository**。

## 第二步：上传代码 (在您的电脑终端操作)

在您当前的终端中，依次执行以下命令：

```bash
# 1. 初始化 Git
git init

# 2. 添加所有文件
git add .

# 3. 提交更改
git commit -m "Initial commit of Huaguniang AI 2.0"

# 4. 关联远程仓库 (请将下面的 URL 换成您刚才创建的仓库地址)
# 例如: git remote add origin https://github.com/YourUsername/huaguniang-ai.git
git remote add origin <您的GitHub仓库地址>

# 5. 推送到 GitHub
git branch -M main
git push -u origin main
```

## 第三步：在 Streamlit Cloud 部署

1.  访问 [Streamlit Cloud](https://share.streamlit.io/) 并使用 GitHub 账号登录。
2.  点击右上角的 **New app**。
3.  **Repository**: 选择您刚才上传的 `huaguniang-ai`。
4.  **Branch**: 选择 `main`。
5.  **Main file path**: 选择 `app.py`。
6.  👇 **关键步骤：配置 Tushare Token** 👇
    *   点击下方的 **Advanced settings**。
    *   在 **Secrets** 文本框中，粘贴以下内容：
    
    ```toml
    TS_TOKEN = "e5e7ab8532e5d39159a7a47fe439348a68844653e1b9cf5b1f7426ea"
    ```
    
    *(这是为了保护您的 Token 不直接暴露在公开代码中)*

7.  点击 **Deploy!**

## 第四步：等待起飞 🛫

*   Streamlit 会自动安装依赖（`autogluon` 等）并启动应用。
*   首次启动可能需要几分钟（安装依赖较慢）。
*   一旦成功，您将获得一个专属网址（如 `https://huaguniang-ai.streamlit.app`），可以直接发给朋友使用！

---

### 常见问题

*   **Q: 启动失败，提示内存不足？**
    *   A: AutoGluon 比较占内存。如果遇到此问题，我们可能需要精简 `requirements.txt` 或仅加载轻量级模型。目前的模型文件仅 67MB，应该没问题。
    
*   **Q: 数据如何更新？**
    *   A: 您在网页上点击“更新市场数据”按钮时，云端应用会使用您配置的 Secrets Token 去拉取最新数据。注意：云端数据的更新是临时的，应用重启后会重置为 GitHub 仓库里的初始数据。建议定期在本地更新数据后，重新 `git push` 到仓库，以固化最新的历史数据。
