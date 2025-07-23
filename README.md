# 📄 Multi-modal Document RAG

这是一个基于 Streamlit 的多模态文档问答系统，支持 PDF / 图片文档的 **页面级** 或 **布局级** 检索与问答。后端集成了 MiniCPM + VisRAG + DocLayout-YOLO 等组件。后续还会支持更多模型，敬请期待......

---

## 📦 环境安装

### 1. 创建 Conda 环境（推荐 Python 3.10）

```bash
conda create -n docrag python=3.10 -y
conda activate docrag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 VisRAG 所需依赖（⚠️ 必须）

请参考 [VisRAG](https://github.com/OpenBMB/VisRAG) 项目安装说明 进行安装，并确保依赖（如 MiniCPM）安装完整。

## ⚙️ 配置说明（config.py）
请根据模板复制 `config.template.py` 为 `config.py` 并填写你自己的路径和密钥

本项目使用 Box 封装了配置信息，位于 config.py 中，配置格式如下：
```bash
from box import Box

RAG_CONFIG = Box({
    'api_key': 'your-api-key-here',
    'base_url': 'https://your-api-url.com',
    ...
})
```

| 参数名                | 类型     | 说明                                                |
|-----------------------|----------|-----------------------------------------------------|
| api_key               | str      | 用于访问 Gemini 或其他云服务接口的 API 密钥         |
| base_url              | str      | 云服务的基础 URL，通常用于构建请求地址             |
| templates             | dict     | 包含 query / response 模板字符串的字典             |
| topk                  | int      | 检索返回的文档数量（图像数）                       |
| device                | str      | 模型推理使用的设备，例如 "cuda:0" 或 "cpu"         |
| retrievers            | dict     | 支持的检索器模型及其本地路径配置                   |
| generators            | dict     | 支持的生成模型及其本地路径或在线服务配置           |
| layout_model          | dict     | DocLayout-YOLO 模型路径配置                        |

⚠️ 请根据你本地模型路径修改对应字段，否则会报错。

## 🚀 启动项目

```bash
streamlit run app.py
```

## 🧾 输入说明
- 上传文件：
  - 支持 `.pdf`, `.jpg`, `.png`
  - 系统会根据粒度（页面级 / 布局级）提取图像
- 检索粒度（granularity）：
  - `page-level`：按整页图像进行检索
  - `layout-level`：基于文档结构块级检索
- 检索器（retriever）：
  - 当前支持 `VisRAG-Retriever(MiniCPM-V2.0)`
  - 需配置本地模型路径
- 生成器（generator）：
  - `MiniCPM-V2.6`（本地部署）
  - `gemini-2.5-pro`（需配置 API Key 和 base_url）
- 用户提问：
  - 可输入任意自然语言问题
  - 系统基于检索图像与生成器返回回答