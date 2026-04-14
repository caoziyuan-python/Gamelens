# ============================================================
# Module: config.py
# ============================================================
# 【业务作用】集中管理 GameLens 的环境变量、抓取参数、主题规则和 LLM 阈值
# 【上游】app.py、data/、analysis/、insights/、llm/ 等模块直接读取
# 【下游】不调用业务模块，本文件只向外提供配置常量
# 【缺失影响】系统会失去统一配置来源，抓取、分析、LLM 调用会出现口径不一致
# ============================================================

import os
from dotenv import load_dotenv

# 启动时立刻加载 .env，是为了让整套分析链路都使用同一份运行参数，
# 避免界面层、抓取层、LLM 层各自读取到不同环境状态。
load_dotenv(override=True)

AZURE_OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-5-mini")
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")

# 默认关闭 Foundry，是为了让本地调试先走更稳定的 Azure 路径，
# 否则团队成员只要少配一个 Foundry 参数，界面就会在启动后才暴露问题。
USE_FOUNDRY = os.getenv("USE_FOUNDRY", "false").strip().lower() in {"1", "true", "yes"}
USE_AZURE = not USE_FOUNDRY

# 这里放的是内置演示游戏，目的是让产品侧一打开页面就能跑通完整分析，
# 不必先理解 App ID、数据源格式等技术前置知识。
ALL_GAMES = {
    "Block Blast":   {"app_id": "1617391485"},
    "Vita Mahjong":  {"app_id": "6468921495"},
    "Arrows Puzzle": {"app_id": "6748397500"}
}

GAMES = {
    "Vita Mahjong": {"app_id": "6468921495"}
}

# 多国抓取不是为了“抓得越多越好”，而是为了尽量覆盖主要英语市场和增长市场，
# 避免洞察只代表单一区域用户情绪。
COUNTRIES = ["us", "gb", "au", "ca", "in", "br", "id", "mx", "de", "fr"]

# 每国只抓 5 页是速度和代表性的折中：
# 太少会让样本波动大，太多会显著拉长首次分析时间，影响产品体验。
PAGES_PER_COUNTRY = 5

FAST_MODE_COUNTRIES = ["us", "gb", "ca"]
FAST_MODE_PAGES_PER_COUNTRY = 2

# 请求间隔保留 0.5 秒，是为了降低被接口限流的概率；
# 对评论抓取这类离线分析场景来说，稳定性比极限速度更重要。
REQUEST_SLEEP = 0.5

# 只重试 3 次，是为了把“临时网络抖动”和“真实不可用”区分开；
# 次数再高只会拖慢页面反馈，且对持续性错误没有帮助。
MAX_RETRY = 3

# 主题关键词是规则引擎的兜底语义地图。
# 当 LLM 不可用、超时或输出异常时，系统仍能给出基本可读的主题判断，
# 避免整个洞察页出现空白。
TOPIC_KEYWORDS = {
    "Ads":          ["ad","ads","advertisement","commercial",
                     "annoying","popup","every level","too many"],
    "Gameplay":     ["level","game","play","puzzle","stage",
                     "mechanic","controls","design"],
    "Monetization": ["pay","purchase","subscription","coin",
                     "gem","expensive","worth","price"],
    "UX_Issues":    ["crash","bug","slow","freeze","update",
                     "loading","performance"],
    "Positive":     ["love","fun","great","amazing","enjoy",
                     "addictive","relaxing","satisfying"]
}

# 这些阈值本质上是在给“哪些问题值得被业务优先处理”设门槛。
# 统一收口在这里，能避免规则引擎、页面告警和验证模块各说各话。
THRESHOLDS = {
    "high_priority_topic_ratio":  0.40,
    "critical_keyword_sentiment": -0.30,
    "low_agreement_warning":      0.70,
    "polarization_high":          0.65,
    "llm_rule_overlap_warning":   0.40,
    "min_sample_high":            100,
    "min_sample_medium":          50
}

LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 1500
LLM_TEMPERATURE = 0.3

# 只抽 80 条评论喂给 LLM，是成本、速度和代表性的折中。
# 如果把数百上千条评论直接送进模型，既容易超 token，也会让单次分析成本迅速上升。
LLM_MAX_INPUT_REVIEWS = 80

# 输入 token 上限单独配置，是为了把“评论样本量”和“模型上下文预算”分开控制，
# 这样后续换模型时不必连采样规则一起改。
LLM_MAX_INPUT_TOKENS = 2000
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# 预检查超时给 5 秒，是为了尽快判断“网络根本不通”这类硬故障，
# 避免页面一直卡在模型调用阶段，让用户误以为系统还在思考。
LLM_PREFLIGHT_TIMEOUT_SECONDS = int(os.getenv("LLM_PREFLIGHT_TIMEOUT_SECONDS", "5"))

# RAG configuration
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "local").strip().lower()
RAG_LOCAL_EMBED_MODEL = os.getenv("RAG_LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
RAG_TOP_K = 3
RAG_SIM_THRESHOLD = 0.35
RAG_KNOWLEDGE_DIR = "knowledge"
RAG_INDEX_PATH = "cache/rag_index.faiss"
RAG_META_PATH = "cache/rag_meta.json"
RAG_MAX_CONTEXT = 900
