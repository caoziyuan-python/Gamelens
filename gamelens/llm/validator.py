# ============================================================
# Module: validator.py
# ============================================================
# 【业务作用】为 LLM 产出的主题和建议提供基础可信度验证
# 【上游】insights.engine.run_analysis_pipeline 调用
# 【下游】依赖 config.THRESHOLDS 和 schema 数据结构
# 【缺失影响】页面只能展示模型结果，却无法提示哪些地方需要人工复核
# ============================================================

from dataclasses import dataclass
from typing import List

from config import THRESHOLDS
from data.schema import InsightSchema, ReviewSchema


@dataclass
# 单独建结构，是为了让每一层验证结果都能统一输出状态、置信度和备注，
# 页面展示时不必针对不同验证器做特殊判断。
class ValidationResult:
    """验证结果结构。"""

    status: str
    confidence: str
    note: str = ""


# Level 1 关注“模型说负面，数据是不是也真的偏负面”，
# 它解决的是主题描述看起来合理、但情绪方向可能跑偏的问题。
def validate_level1_sentiment(
    llm_topics: List[dict],
    reviews: List[ReviewSchema]
) -> ValidationResult:
    """
    Level 1：情感一致性检查。

    逻辑：
    - 找出 llm_topics 中 sentiment="negative" 的主题。
    - 对每个负面主题，以其 keywords 在评论文本中匹配对应评论。
    - 计算匹配评论的平均 vader_score。
    - 均分 < -0.1 视为一致，返回 passed/High。
    - 均分 >= -0.1 视为不一致，返回 warning/Medium，并给出复核提示。
    - 若无负面主题，返回 passed/High。

    参数：
    - llm_topics: LLM 主题列表。
    - reviews: 评论列表。

    返回：
    - ValidationResult: Level 1 验证结果。
    """
    try:
        negative_topics = [
            t for t in (llm_topics or []) if str(t.get("sentiment", "")).lower() == "negative"
        ]

        if not negative_topics:
            return ValidationResult(status="passed", confidence="High")

        all_scores = []
        for topic in negative_topics:
            keywords = [str(k).lower().strip() for k in topic.get("keywords", []) if str(k).strip()]
            if not keywords:
                continue
            for rv in reviews or []:
                text = (rv.text or "").lower()
                if any(kw in text for kw in keywords):
                    all_scores.append(float(rv.vader_score))

        if not all_scores:
            return ValidationResult(status="warning", confidence="Medium", note="负面主题缺少可匹配样本")

        avg_score = sum(all_scores) / len(all_scores)
        if avg_score < -0.1:
            return ValidationResult(status="passed", confidence="High")
        return ValidationResult(
            status="warning",
            confidence="Medium",
            note="LLM标记为负面但情感分偏高，建议人工复核",
        )
    except Exception as exc:
        return ValidationResult(status="failed", confidence="Low", note=f"Level1异常: {exc}")


# Level 2 看的是“模型主题和规则主题之间有没有基本重合”，
# 它的价值是提醒团队：当前模型是发现了新主题，还是可能在胡说。
def validate_level2_overlap(
    llm_topics: List[dict],
    rule_topic_stats: dict
) -> ValidationResult:
    """
    Level 2：LLM主题与规则主题关键词重叠率。

    逻辑：
    - 合并 llm_topics 的 keywords 为集合 llm_kw。
    - 提取 rule_topic_stats 的主题名（key）为集合 rule_kw。
    - 计算 Jaccard 相似度 overlap。
    - overlap > 0.6: passed/High。
    - 0.4 <= overlap <= 0.6: warning/Medium，提示规则未覆盖主题。
    - overlap < 0.4: warning/Low，提示可能存在幻觉。

    参数：
    - llm_topics: LLM 主题列表。
    - rule_topic_stats: 规则主题统计字典。

    返回：
    - ValidationResult: Level 2 验证结果。
    """
    try:
        llm_kw = set()
        for topic in llm_topics or []:
            for kw in topic.get("keywords", []) or []:
                k = str(kw).strip().lower()
                if k:
                    llm_kw.add(k)

        rule_kw = {str(k).strip().lower() for k in (rule_topic_stats or {}).keys() if str(k).strip()}

        union = llm_kw | rule_kw
        if not union:
            return ValidationResult(status="passed", confidence="High")

        overlap = len(llm_kw & rule_kw) / len(union)

        if overlap > 0.6:
            return ValidationResult(status="passed", confidence="High")
        if 0.4 <= overlap <= 0.6:
            return ValidationResult(
                status="warning",
                confidence="Medium",
                note="LLM发现了规则未覆盖的新主题",
            )
        return ValidationResult(
            status="warning",
            confidence="Low",
            note="主题重叠率低，可能存在幻觉，建议人工复核",
        )
    except Exception as exc:
        return ValidationResult(status="failed", confidence="Low", note=f"Level2异常: {exc}")


# Level 3 不要求模型和规则完全一致，而是检查“优先级分布别偏得太离谱”，
# 这样既保留模型创造性，也给业务侧一个基本安全边界。
def validate_level3_suggestions(
    llm_suggestions: List[dict],
    rule_insights: List[InsightSchema]
) -> ValidationResult:
    """
    Level 3：建议优先级分布对齐（调试用途）。

    逻辑：
    - 比较 LLM 建议与规则洞察中 High 优先级占比。
    - 占比差异 < 30% 时 passed。
    - 占比差异 >= 30% 时 warning，并输出差异说明。
    - 任一列表为空时，返回 passed（无法比较）。

    参数：
    - llm_suggestions: LLM 生成建议列表。
    - rule_insights: 规则引擎生成洞察列表。

    返回：
    - ValidationResult: Level 3 验证结果。
    """
    try:
        if not llm_suggestions or not rule_insights:
            return ValidationResult(status="passed", confidence="High")

        llm_high = sum(
            1
            for s in llm_suggestions
            if str(s.get("priority", "")).strip().lower() == "high"
        )
        rule_high = sum(1 for r in rule_insights if str(r.priority).strip().lower() == "high")

        llm_ratio = llm_high / len(llm_suggestions)
        rule_ratio = rule_high / len(rule_insights)
        diff = abs(llm_ratio - rule_ratio)

        if diff < 0.30:
            return ValidationResult(status="passed", confidence="High")
        return ValidationResult(
            status="warning",
            confidence="Medium",
            note=f"LLM建议与规则引擎优先级差异较大({diff:.0%})",
        )
    except Exception as exc:
        return ValidationResult(status="failed", confidence="Low", note=f"Level3异常: {exc}")


# 样本量分级是业务沟通里的“信心翻译器”：
# 同样一句建议，来自 20 条评论和来自 200 条评论，可信度应该明显不同。
def get_confidence_from_sample(sample_size: int) -> str:
    """
    基于样本量返回置信度等级。

    规则：
    - sample_size > THRESHOLDS["min_sample_high"]: High
    - sample_size > THRESHOLDS["min_sample_medium"]: Medium
    - 其余: Low

    参数：
    - sample_size: 样本数量。

    返回：
    - str: "High" / "Medium" / "Low"。
    """
    high = int(THRESHOLDS.get("min_sample_high", 100))
    medium = int(THRESHOLDS.get("min_sample_medium", 50))

    if sample_size > high:
        return "High"
    if sample_size > medium:
        return "Medium"
    return "Low"
