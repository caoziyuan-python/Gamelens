# ============================================================
# Module: problem_abstraction.py
# ============================================================
# 【业务作用】从低分评论中提炼出可操作的问题模式，供后续建议生成使用
# 【上游】insights.engine.run_analysis_pipeline 调用
# 【下游】依赖 llm._base 的调用、日志和敏感词清洗能力
# 【缺失影响】系统会看到很多负面评论，但难以把它们归并成产品可以处理的问题桶
# ============================================================

from typing import List, Optional

from data.schema import ReviewSchema
from llm._base import (
    call_llm,
    call_llm_json,
    get_last_llm_error,
    get_last_llm_error_type,
    log_validation,
    sanitize_text_for_llm,
)

LAST_PROBLEM_ABSTRACTION_ERROR = ""


def get_last_problem_abstraction_error() -> str:
    return LAST_PROBLEM_ABSTRACTION_ERROR


# 这个函数专盯低星评论，是因为产品真正优先要修的，往往不是“大家都在聊的点”，
# 而是“导致差评和流失的点”。
def abstract_problems(reviews: List[ReviewSchema], game_name: str) -> Optional[List[dict]]:
    """Complaint abstraction with strict JSON retry."""
    global LAST_PROBLEM_ABSTRACTION_ERROR
    LAST_PROBLEM_ABSTRACTION_ERROR = ""

    low_rating_reviews = [r for r in reviews if isinstance(r.rating, int) and r.rating <= 2]
    # 至少要求 10 条低分样本，是为了降低“偶发吐槽”被模型误判成系统性问题的风险。
    if len(low_rating_reviews) < 10:
        LAST_PROBLEM_ABSTRACTION_ERROR = f"Insufficient low-rating samples: {len(low_rating_reviews)} (<10)"
        log_validation("problem_abstraction", "skipped", LAST_PROBLEM_ABSTRACTION_ERROR)
        return None

    samples = low_rating_reviews[:50]
    reviews_text = "\n".join([f"[{r.rating}星] {sanitize_text_for_llm((r.text or '').strip())}" for r in samples])

    prompt = f"""
Given these negative reviews for {game_name}:
{reviews_text}

Return ONLY a JSON array of top 3 complaint patterns:
[
  {{
    "complaint_type": "...",
    "core_demand": "...",
    "typical_quote": "...",
    "estimated_ratio": 0.35
  }}
]
Rules:
- Merge semantically similar complaints
- estimated_ratio must be float in [0,1]
- Replace sensitive words with [SENSITIVE]
""".strip()

    response = call_llm(prompt)
    parsed = call_llm_json(prompt)

    if parsed is None:
        repair_prompt = f"""
Convert this output into a valid JSON array only:
{response or ''}

If not fixable, regenerate from reviews:
{reviews_text}

Required fields: complaint_type, core_demand, typical_quote, estimated_ratio
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
        LAST_PROBLEM_ABSTRACTION_ERROR = reason
        log_validation("problem_abstraction", "failed", reason)
        return None

    log_validation("problem_abstraction", "success")
    return parsed
