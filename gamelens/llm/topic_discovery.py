# ============================================================
# Module: topic_discovery.py
# ============================================================
# 【业务作用】调用 LLM 从评论中归纳高层主题，供主题页和后续洞察复用
# 【上游】insights.engine.run_analysis_pipeline 调用
# 【下游】依赖 llm._base 的调用、采样和日志能力
# 【缺失影响】系统只能退回规则主题，失去更抽象的用户需求概括能力
# ============================================================

from typing import List, Optional

from data.schema import ReviewSchema
from llm._base import (
    call_llm,
    call_llm_json,
    get_last_llm_error,
    get_last_llm_error_type,
    log_validation,
    sample_reviews_for_llm,
)

LAST_TOPIC_DISCOVERY_ERROR = ""


# 单独暴露最近一次错误，是为了让页面和流水线能解释“为什么降级了”，
# 而不是只给用户一个模糊的失败状态。
def get_last_topic_discovery_error() -> str:
    return LAST_TOPIC_DISCOVERY_ERROR


# 这个函数的业务价值是把大量碎片化评论浓缩成几张“主题卡”，
# 帮产品经理快速看到用户最关心的体验块，而不是逐条读原始评论。
def discover_topics(reviews: List[ReviewSchema], game_name: str) -> Optional[List[dict]]:
    """Topic discovery with strict JSON retry."""
    global LAST_TOPIC_DISCOVERY_ERROR
    LAST_TOPIC_DISCOVERY_ERROR = ""

    samples = sample_reviews_for_llm(reviews)
    if not samples:
        LAST_TOPIC_DISCOVERY_ERROR = "No usable review samples"
        log_validation("topic_discovery", "failed", LAST_TOPIC_DISCOVERY_ERROR)
        return None

    reviews_text = "\n".join(samples)

    prompt = f"""
Given these user reviews of {game_name}:
{reviews_text}

Return ONLY a JSON array of 5 items with fields:
- topic_name (English)
- core_demand (Chinese, one sentence)
- keywords (3 strings)
- sentiment (positive/negative/mixed)
- representative_review (original quote)
""".strip()

    response = call_llm(prompt)
    parsed = call_llm_json(prompt)

    if parsed is None:
        # 先修复再放弃，是为了把“模型格式不稳定”与“模型完全不可用”区分开，
        # 很多时候不是理解失败，而只是输出格式多了说明文字。
        repair_prompt = f"""
Fix the following output into a valid JSON array only (no extra text):
{response or ''}

If it is not fixable, regenerate from these reviews:
{reviews_text}

Schema per item:
{{
  "topic_name": "...",
  "core_demand": "...",
  "keywords": ["...","...","..."],
  "sentiment": "positive|negative|mixed",
  "representative_review": "..."
}}
""".strip()
        repair_response = call_llm(repair_prompt)
        parsed = call_llm_json(repair_prompt)

    if parsed is None:
        preview = (repair_response if 'repair_response' in locals() and repair_response else response or "").strip()
        preview = preview[:200].replace("\n", " ")
        last_error = get_last_llm_error()
        if last_error and get_last_llm_error_type() == "network_error":
            reason = last_error
        else:
            reason = last_error or f"LLM output is not a valid JSON array | preview={preview}"
        LAST_TOPIC_DISCOVERY_ERROR = reason
        log_validation("topic_discovery", "failed", reason)
        return None

    log_validation("topic_discovery", "success")
    return parsed
