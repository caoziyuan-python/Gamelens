# ============================================================
# Module: schema.py
# ============================================================
# 【业务作用】定义评论与洞察的统一数据结构，确保抓取、分析、展示使用同一套字段
# 【上游】data、analysis、insights、llm、visualization、tests 全部依赖
# 【下游】不调用其他业务模块，只输出结构定义
# 【缺失影响】模块之间会用各自的 dict 拼字段，系统很容易出现字段缺失和口径不一致
# ============================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
# 评论结构用 dataclass 而不是散落的 dict，
# 是为了让抓取、清洗、情感分析、主题识别都围绕同一份业务对象协作。
class ReviewSchema:
    review_id:      str
    game_name:      str
    country:        str
    rating:         int
    text:           str
    date:           str
    # 这些派生字段默认给出中性值，是为了保证抓取完成后即使分析尚未执行，
    # 页面和后续模块也不会因为字段缺失直接崩掉。
    vader_score:    float = 0.0
    vader_label:    str   = "Neutral"
    textblob_score: float = 0.0
    textblob_label: str   = "Neutral"
    rule_topics:    List[str] = field(default_factory=list)
    llm_topics:     List[str] = field(default_factory=list)
    agreement:      bool  = False


@dataclass
# 洞察结构单独建模，是为了把“证据、优先级、动作建议”固定下来，
# 这样页面卡片、导出、反馈学习都能复用同一份结果。
class InsightSchema:
    insight_id:        str
    game_name:         str
    source:            str
    confidence:        str
    rule_trigger:      str   = ""
    evidence:          str   = ""
    evidence_count:    int   = 0
    evidence_ratio:    float = 0.0
    avg_sentiment:     float = 0.0
    priority:          str   = "Medium"
    impact_metric:     str   = "Retention"
    action:            str   = ""
    validation_status: str   = "pending"
    feedback:          str   = "pending"
