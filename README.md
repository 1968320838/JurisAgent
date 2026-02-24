# JurisAgent - 法律智能体

专业的法律AI助手，提供合同审查、法律咨询、案例分析功能，基于RAG检索增强技术提供准确的法律依据。

## 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 📄 合同审查 | ✅ 已完成 | 上传PDF/Word合同，AI自动识别风险条款并提供修改建议 |
| 💬 法律咨询 | ✅ 已完成 | RAG增强的智能法律问答，自动引用相关法条 |
| 📊 案例分析 | ✅ 已完成 | 争议焦点提取、法律适用分析、裁判倾向预测 |

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      JurisAgent 架构                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  合同审查   │  │  法律咨询   │  │  案例分析   │  Web层  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│  ┌──────┴────────────────┴────────────────┴──────┐         │
│  │              Agent 层 (BaseAgent)              │         │
│  └──────────────────────┬────────────────────────┘         │
│                         │                                   │
│  ┌──────────────────────┴────────────────────────┐         │
│  │                 LLM 层 (GLM-4)                 │         │
│  └──────────────────────┬────────────────────────┘         │
│                         │                                   │
│  ┌──────────────────────┴────────────────────────┐         │
│  │    RAG 层 (ChromaDB + 智谱Embedding API)      │         │
│  │    - 向量存储: ChromaDB                       │         │
│  │    - 向量维度: 1024                           │         │
│  │    - 法条数量: 23,748条                       │         │
│  └───────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/JurisAgent.git
cd JurisAgent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API Key

创建 `.env` 文件：

```bash
# .env
ZHIPU_API_KEY=your_api_key_here
```

> 获取API Key: 访问 [智谱AI开放平台](https://open.bigmodel.cn/) 注册并获取

### 3. 构建RAG知识库

```bash
# 方式一：使用示例数据（14条常用法条）
python scripts/import_laws.py

# 方式二：从公开法律数据源导入（推荐）
# 1. 首先下载数据（需要自行获取公开数据）
# 2. 然后解析并导入
python scripts/parse_local_laws.py --data-dir /path/to/law/data --all
```

### 4. 启动Web界面

```bash
streamlit run web/app.py
```

浏览器访问 http://localhost:8501

---

## RAG知识库构建指南

### 数据来源说明

本项目的RAG知识库数据来源于**公开可获取的法律文本**，用户需自行获取数据。推荐以下合法数据来源：

#### 推荐数据来源

| 来源 | 说明 | 获取方式 |
|------|------|---------|
| **国家法律法规数据库** | 官方法律法规发布平台 | https://flk.npc.gov.cn/ |
| **中国裁判文书网** | 公开的裁判文书 | https://wenshu.court.gov.cn/ |
| **北大法宝** | 部分免费法律文本 | https://www.pkulaw.com/ |
| **威科先行** | 法律信息服务 | https://law.wkinfo.com.cn/ |
| **GitHub开源项目** | 社区整理的法律文本 | 各开源仓库 |

#### 重要说明

1. **法律文本本身**：根据《中华人民共和国著作权法》第五条，法律法规、国家机关的决议、决定、命令等不属于著作权保护对象，可以自由使用。

2. **数据库/汇编作品**：他人整理的法律数据库可能享有汇编作品著作权，使用时需注意：
   - 仅使用原始法律文本，不使用他人的编排结构
   - 自行解析和处理数据
   - 注明数据来源

3. **本项目不提供**：出于版权考虑，本项目不直接提供法律数据文件，用户需自行从合法渠道获取。

### RAG构建流程

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 知识库构建流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 数据获取                                                │
│     └── 从公开渠道获取法律文本 (.md, .txt, .json)           │
│                                                             │
│  2. 数据解析 (scripts/parse_local_laws.py)                  │
│     ├── 读取法律文件                                        │
│     ├── 提取法条内容（第X条）                               │
│     ├── 提取元数据（法律名称、条款号、分类）                │
│     └── 生成唯一ID                                          │
│                                                             │
│  3. 向量化 (src/rag/embeddings.py)                          │
│     ├── 调用智谱AI Embedding API                            │
│     ├── 文本 → 1024维向量                                   │
│     └── 批量处理（50条/批）                                 │
│                                                             │
│  4. 存储 (src/rag/vector_store.py)                          │
│     ├── ChromaDB向量数据库                                  │
│     ├── 持久化到 ./data/vector_db/                          │
│     └── 支持增量更新                                        │
│                                                             │
│  5. 检索 (src/rag/retriever.py)                             │
│     ├── 查询向量化                                          │
│     ├── 余弦相似度检索                                      │
│     └── 返回Top-K相关法条                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 数据格式要求

解析脚本支持以下格式的法律文本：

```markdown
# 中华人民共和国民法典

## 第一编 总则

### 第一章 基本规定

第一条 为了保护民事主体的合法权益...（法条内容）

第二条 民法调整平等主体的自然人...（法条内容）
```

**必需元素**：
- 法条编号：`第X条`（支持中文数字和阿拉伯数字）
- 法条内容：条款的具体文字

**可选元数据**：
- 法律名称
- 编/章/节分类
- 生效日期

### 向量数据库配置

```python
# src/rag/retriever.py 配置示例
retriever = LegalRetriever(
    api_key="your_api_key",
    persist_directory="./data/vector_db",  # 数据库路径
    collection_name="legal_documents"      # 集合名称
)

# 查看统计
stats = retriever.get_stats()
# {'total_documents': 23748, 'embedding_dimension': 1024}
```

### 增量更新

```python
# 添加新法条
new_laws = [
    {
        "id": "unique_id",
        "text": "法条内容...",
        "metadata": {
            "law_name": "法律名称",
            "article_number": "第X条",
            "doc_type": "law"
        }
    }
]
retriever.index_documents(new_laws)
```

---

## 项目结构

```
JurisAgent/
├── config.py                     # 全局配置
├── glm_client.py                 # GLM-4 API客户端
├── requirements.txt              # 依赖清单
│
├── src/
│   ├── core/
│   │   ├── base_agent.py         # Agent基类
│   │   └── exceptions.py         # 异常定义
│   │
│   ├── agents/
│   │   ├── contract_agent.py     # 合同审查Agent
│   │   ├── legal_qa_agent.py     # 法律咨询Agent
│   │   └── case_agent.py         # 案例分析Agent
│   │
│   ├── rag/
│   │   ├── embeddings.py         # 向量嵌入（智谱API）
│   │   ├── vector_store.py       # ChromaDB存储
│   │   └── retriever.py          # 检索器
│   │
│   ├── parsers/
│   │   ├── pdf_parser.py         # PDF解析
│   │   └── docx_parser.py        # Word解析
│   │
│   └── prompts/
│       ├── contract_review.py    # 合同审查提示词
│       ├── legal_qa.py           # 法律咨询提示词
│       └── case_analysis.py      # 案例分析提示词
│
├── scripts/
│   ├── import_laws.py            # 导入示例法条
│   ├── parse_local_laws.py       # 解析本地法律文件
│   └── fetch_laws_from_github.py # 从GitHub获取（需网络）
│
├── data/
│   ├── vector_db/                # ChromaDB向量数据库
│   ├── parsed_laws.json          # 解析后的法条缓存
│   └── raw/                      # 原始法律文件
│
└── web/
    └── app.py                    # Streamlit Web界面
```

## 代码示例

### 1. 合同审查

```python
from glm_client import get_glm_client
from src.agents.contract_agent import ContractReviewAgent

llm = get_glm_client()
agent = ContractReviewAgent(llm)

# 审查合同文本
result = agent.execute({
    "contract_text": "甲方与乙方于2024年签订本合同..."
})

print(result.content)
print(f"合规评分: {result.metadata.get('compliance_score')}")
print(f"风险数量: {result.metadata.get('risk_count')}")
```

### 2. 法律咨询（带RAG）

```python
from glm_client import get_glm_client
from src.agents.legal_qa_agent import LegalQAAgent
from src.rag.retriever import LegalRetriever

llm = get_glm_client()
retriever = LegalRetriever(api_key="your_key", persist_directory="./data/vector_db")
agent = LegalQAAgent(llm, retriever=retriever)

# RAG增强的问答
result = agent.execute({
    "question": "违约金过高怎么办？"
})

print(result.content)
print("引用法条:", result.sources)
```

### 3. 案例分析

```python
from glm_client import get_glm_client
from src.agents.case_agent import CaseAnalysisAgent

llm = get_glm_client()
agent = CaseAnalysisAgent(llm)

result = agent.execute({
    "case_content": "张某向李某借款10万元，约定年利率24%...",
    "analysis_type": "full"  # full/focus/prediction
})

print(result.content)
```

### 4. 直接使用RAG检索

```python
from src.rag.retriever import LegalRetriever

retriever = LegalRetriever(
    api_key="your_api_key",
    persist_directory="./data/vector_db"
)

# 检索相关法条
results = retriever.retrieve("民间借贷利息上限是多少", n_results=5)

for r in results:
    print(f"《{r['metadata']['law_name']}》{r['metadata']['article_number']}")
    print(f"相似度: {r['score']:.4f}")
    print(f"内容: {r['text'][:100]}...")
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM | GLM-4 (智谱AI) | 中文能力强，性价比高 |
| Web框架 | Streamlit | 快速原型开发 |
| 向量数据库 | ChromaDB | 轻量级，零配置 |
| 向量嵌入 | 智谱Embedding API | 1024维向量 |
| 文档解析 | pdfplumber, python-docx | PDF/Word解析 |
| 语言 | Python 3.8+ | - |

## 性能指标

| 指标 | 目标值 | 实际值 |
|------|--------|--------|
| 合同审查响应时间 | < 30秒 | ~20秒 |
| 法律咨询响应时间 | < 10秒 | ~8秒 |
| RAG检索时间 | < 2秒 | ~1秒 |
| 法条覆盖率 | > 80% | 常用法条覆盖 |

## 后续计划

- [ ] FastAPI服务化，提供RESTful API
- [ ] 支持更多LLM后端（OpenAI、本地模型等）
- [ ] 知识图谱集成，增强法律关系推理
- [ ] 多轮对话支持，上下文记忆
- [ ] 用户系统，使用记录保存

## 常见问题

### Q: RAG显示"未启用"？

A: 需要先构建向量数据库：
```bash
python scripts/import_laws.py
# 或
python scripts/parse_local_laws.py --all --data-dir /path/to/laws
```

### Q: API调用失败？

A: 检查以下几点：
1. `.env` 文件中的API Key是否正确
2. 网络是否能访问智谱AI服务
3. API账户是否有余额

### Q: 如何添加更多法条？

A: 将法律文本放入 `data/raw/` 目录，然后运行：
```bash
python scripts/parse_local_laws.py --import --data-dir data/raw
```

## 许可证

MIT License

## 免责声明

本项目仅供学习和研究使用，不构成任何法律建议。AI生成的分析结果可能存在错误，具体法律问题请咨询专业律师。

## 数据来源声明

本项目的RAG知识库需要用户自行获取法律数据。推荐使用官方渠道获取法律文本：
- 国家法律法规数据库 (https://flk.npc.gov.cn/)
- 中国裁判文书网 (https://wenshu.court.gov.cn/)

法律文本本身不受著作权保护，但数据库编排可能享有汇编作品权利，使用时请遵守相关法律法规。
