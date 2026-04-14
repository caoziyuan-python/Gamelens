# ============================================================
# Module: decision_generation.py
# ============================================================
# 【业务作用】把问题模式转成可执行的产品动作建议
# 【上游】insights.engine.run_analysis_pipeline 调用
# 【下游】依赖 llm._base 的调用、日志和文本清洗能力
# 【缺失影响】系统只能停留在“发现问题”，无法进一步给出业务可执行动作
# ============================================================

from typing import List, Optional

from llm._base import (
    call_llm,
    call_llm_json,
    get_last_llm_error,
    get_last_llm_error_type,
    log_validation,
    sanitize_text_for_llm,
)

LAST_DECISION_GENERATION_ERROR = ""


def get_last_decision_generation_error() -> str:
    return LAST_DECISION_GENERATION_ERROR


# 这个函数的价值在于把“用户抱怨”转换成“产品动作”，
# 否则分析结果会停留在描述层，很难直接支持优先级讨论和需求落地。
def generate_decisions(complaints: List[dict], sentiment_stats: dict, game_name: str) -> Optional[List[dict]]:
    """Generate product decisions in strict JSON format."""
    global LAST_DECISION_GENERATION_ERROR
    LAST_DECISION_GENERATION_ERROR = ""

    if not complaints:
        LAST_DECISION_GENERATION_ERROR = "No complaint data, cannot generate decisions"
        log_validation("decision_generation", "skipped", LAST_DECISION_GENERATION_ERROR)
        return None

    summary = []
    for i, item in enumerate(complaints, 1):
        ctype = sanitize_text_for_llm(str(item.get("complaint_type", "Unknown")))
        demand = sanitize_text_for_llm(str(item.get("core_demand", "")))
        ratio = item.get("estimated_ratio", "")
        summary.append(f"{i}. {ctype} | demand={demand} | ratio={ratio}")
    complaints_summary = "\n".join(summary)

    avg_score = float(sentiment_stats.get("avg_score", 0.0)) if sentiment_stats else 0.0
    neg_ratio = float(sentiment_stats.get("neg_ratio", 0.0)) if sentiment_stats else 0.0

    prompt = f"""
Based on {game_name} user data, generate exactly 3 product decisions.

Complaints:
{complaints_summary}

Data:
- avg_score: {avg_score:.2f}/5
- negative_ratio: {neg_ratio:.2f}%

Return ONLY JSON array:
[
  {{
    "problem": "...",
    "action": "...",
    "expected_impact": "...",
    "impact_metric": "Retention|Conversion|Rating",
    "priority": "High|Medium|Low"
  }}
]
Replace sensitive words with [SENSITIVE].
""".strip()

    response = call_llm(prompt)
    parsed = call_llm_json(prompt)

    if parsed is None:
        repair_prompt = f"""
Convert this output into valid JSON array only:
{response or ''}

If not fixable, regenerate from:
{complaints_summary}
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
        LAST_DECISION_GENERATION_ERROR = reason
        log_validation("decision_generation", "failed", reason)
        return None

    log_validation("decision_generation", "success")
    return parsed
