# GameLens 🎮

一个基于大模型和 RAG 工具链构建的海外手游竞品洞察助手，帮助游戏产品经理从 App Store 评论中快速提取用户痛点、竞品差距和产品决策建议。

## 核心功能

- **评论抓取与分析**：基于 iTunes RSS API 抓取多国家 App Store 评论，自动完成情感分析和主题分类
- **多游戏竞品对比**：支持跨游戏用户痛点横向对比，识别竞品优势与差距
- **深度竞品分析 Agent**：输入自然语言问题，Agent 自动检索真实评论、对比竞品差距、输出结构化分析报告
- **产品决策建议**：基于用户反馈自动生成 P0 / P1 / P2 优先级产品建议
- **Rule-based Fallback**：LLM 不可用时自动降级为规则引擎，功能不中断

## 系统架构

```text
用户输入游戏名
    |
    v
data/fetcher.py
    |  iTunes RSS API 抓取评论
    v
analysis/
    |  情感分析 + 规则主题分类
    v
llm/
    |  LLM 主题归纳 + 痛点抽象 + 建议生成
    v
llm/rag.py
    |  构建游戏专属 FAISS 索引
    v
llm/agent.py
    |  Deep Dive Agent
    |  关键词提取 -> RAG 检索 -> 跨游戏对比 -> 结构化输出
    v
app.py
    |  Streamlit UI 展示
```

## 数据来源

| 来源 | 方式 | 说明 |
| --- | --- | --- |
| iTunes RSS API | 实时抓取 | App Store 用户评论（多国家） |
| 本地 FAISS 索引 | 分析后自动构建 | 游戏专属向量检索库 |
| Azure OpenAI | GPT-4o 推理 | 主题归纳、建议生成、竞品对比 |

## 目录结构

```text
gamelens/
├── app.py              # Streamlit 主界面
├── data/               # 评论抓取（iTunes RSS）
├── analysis/           # 情感分析 + 规则主题分类
├── llm/                # LLM 调用层、RAG 索引、Agent
├── insights/           # 分析流水线 + 跨游戏对比
├── utils/              # 缓存、导出
├── indices/            # 游戏专属 FAISS 索引（自动生成）
└── docs/               # 设计决策文档
```

## 快速开始

1. 配置环境变量：`AZURE_OPENAI_KEY`、`AZURE_OPENAI_ENDPOINT`
2. 安装依赖并启动：

```bash
pip install -r requirements.txt
cd gamelens
streamlit run app.py
```

## Deep Dive Agent 使用示例

输入：

```text
为什么用户在广告体验上的抱怨比竞品更多？
```

Agent 执行步骤：

1. 从问题中提取关键词
2. 在当前游戏的评论索引中检索相关评论
3. 在竞品游戏的评论索引中检索相关评论
4. 调用 LLM 生成结构化竞品差距分析

输出示例：

```json
{
  "gap_summary": "当前游戏在广告打断感和误触跳转上的负反馈明显高于竞品。",
  "root_causes": [
    "强制广告触发频率偏高",
    "广告关闭路径不清晰",
    "去广告付费承诺与实际体验不一致"
  ],
  "priority_score": 5
}
```

## 技术栈

| 类别 | 内容 |
| --- | --- |
| 后端 | Python 3.11+，Azure OpenAI，FAISS |
| 前端 | Streamlit |
| 数据 | iTunes RSS API，本地向量索引 |
| 架构 | 自研分析流水线，非 LangChain 套壳 |

## 设计决策

见 [docs/design-decisions.md](docs/design-decisions.md)
