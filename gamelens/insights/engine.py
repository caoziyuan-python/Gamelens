import os
# ============================================================
# Module: engine.py
# ============================================================
# 【业务作用】串联情感分析、主题识别、LLM 归纳、建议生成和验证，输出页面可直接消费的完整分析结果
# 【上游】app.py 点击分析后调用
# 【下游】依赖 analysis、llm、data.schema 等模块
# 【缺失影响】系统将失去统一分析流水线，只能拿到零散中间结果，无法稳定渲染洞察页
# ============================================================

import sys
import uuid
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.rule_topics import classify_topics, get_topic_stats
from analysis.sentiment import analyze_sentiment, get_sentiment_by_rating, get_sentiment_stats
from config import THRESHOLDS
from data.schema import InsightSchema, ReviewSchema
from llm._base import log_validation
from llm.decision_generation import (
    generate_decisions,
    get_last_decision_generation_error,
)
from llm.problem_abstraction import (
    abstract_problems,
    get_last_problem_abstraction_error,
)
from llm.rag import index_game_reviews
from llm.topic_discovery import discover_topics, get_last_topic_discovery_error
from llm.validator import (
    ValidationResult,
    get_confidence_from_sample,
    validate_level1_sentiment,
    validate_level2_overlap,
    validate_level3_suggestions,
)

TOPIC_DISPLAY_META = {
    "Ads": {
        "topic_name": "Ads & Monetization Friction",
        "core_demand": "用户希望广告频次更低、时长更短，并且付费去广告承诺能够被兑现。",
        "keywords": ["ads", "redirects", "remove ads"],
    },
    "Gameplay": {
        "topic_name": "Gameplay Depth & Fairness",
        "core_demand": "用户希望玩法更有趣、更公平，关卡和机制变化不要破坏核心体验。",
        "keywords": ["level design", "difficulty", "mechanics"],
    },
    "Monetization": {
        "topic_name": "Pricing & Value Perception",
        "core_demand": "用户希望付费点更透明，付费后的收益与预期一致。",
        "keywords": ["price", "purchase", "value"],
    },
    "UX_Issues": {
        "topic_name": "Stability & UX Issues",
        "core_demand": "用户希望应用更稳定、不卡顿、不误触，整体使用流程更顺畅。",
        "keywords": ["bug", "crash", "performance"],
    },
    "Positive": {
        "topic_name": "Core Delight",
        "core_demand": "用户认可轻松、上手快、能持续带来放松感的核心体验。",
        "keywords": ["fun", "relaxing", "satisfying"],
    },
}


# 这个函数的核心价值是提供规则兜底建议，
# 保证 LLM 不可用时产品侧依然能看到“该先修什么”的方向。
def generate_rule_insights(
    reviews: List[ReviewSchema],
    game_name: str,
    topic_stats: dict
) -> List[InsightSchema]:
    """
    使用规则引擎生成 Fallback Insights。

    处理逻辑：
    - 基于 topic_stats 中 is_high_priority=True 的主题生成 InsightSchema。
    - 若不存在高优先级主题，则返回空列表。
    - 每条洞察包含规则触发依据、证据摘要、影响指标与建议动作。

    参数：
    - reviews: 评论列表。
    - game_name: 游戏名称。
    - topic_stats: 主题统计结果。

    返回：
    - List[InsightSchema]: 规则生成的洞察列表。
    """
    insights: List[InsightSchema] = []
    # 规则建议也要带置信度，是为了让页面在展示时能和 LLM 建议用同一把尺子解释可信度。
    confidence = get_confidence_from_sample(len(reviews or []))
    ratio_threshold = float(THRESHOLDS.get("high_priority_topic_ratio", 0.40))

    action_map = {
        "Ads": "优化广告触发逻辑，将强制广告改为激励式广告",
        "Gameplay": "优化关卡难度曲线，提升玩法深度",
        "Monetization": "调整付费点设计，引入更多免费获取途径",
        "UX_Issues": "修复崩溃和性能问题，提升应用稳定性",
        "Positive": "强化现有优势特性，扩大正面口碑传播",
    }

    for topic, stats in (topic_stats or {}).items():
        try:
            is_high = bool(stats.get("is_high_priority", False))
            if not is_high:
                continue

            count = int(stats.get("count", 0))
            ratio = float(stats.get("ratio", 0.0))
            avg_sentiment = float(stats.get("avg_sentiment", 0.0))
            low_star_ratio = float(stats.get("low_star_ratio", 0.0))

            if topic == "Ads":
                impact_metric = "Retention"
            elif topic == "UX_Issues":
                impact_metric = "Rating"
            elif topic == "Monetization":
                impact_metric = "Conversion"
            else:
                impact_metric = "Retention"

            insights.append(
                InsightSchema(
                    insight_id=str(uuid.uuid4())[:8],
                    game_name=game_name,
                    source="rule_fallback",
                    confidence=confidence,
                    rule_trigger=(
                        f"{topic}主题在1-2星占比{low_star_ratio:.0%}，"
                        f"超过{ratio_threshold:.0%}阈值"
                    ),
                    evidence=f"影响{count}条评论，占比{ratio:.0%}，均分{avg_sentiment:.3f}",
                    evidence_count=count,
                    evidence_ratio=ratio,
                    avg_sentiment=avg_sentiment,
                    priority="High" if is_high else "Medium",
                    impact_metric=impact_metric,
                    action=action_map.get(topic, "围绕该主题制定专项优化方案并持续监控"),
                    validation_status="passed",
                )
            )
        except Exception as exc:
            print(f"[Warning] 规则洞察生成失败(topic={topic}): {exc}")
            continue

    return insights


# 规则主题卡片存在的意义，是让主题页在模型失败时仍然可读，而不是整块空白。
def build_fallback_topics(reviews: List[ReviewSchema], topic_stats: dict, top_k: int = 5) -> List[dict]:
    """Build deterministic topic cards from rule-topic stats when LLM topics are unavailable."""
    # 先按数量再按低星比排序，是为了把“更常见且更痛”的主题优先推到页面前排。
    ranked_topics = sorted(
        (topic_stats or {}).items(),
        key=lambda item: (
            -float(item[1].get("count", 0)),
            -float(item[1].get("low_star_ratio", 0.0)),
            str(item[0]),
        ),
    )

    cards: List[dict] = []
    for topic, stats in ranked_topics[:top_k]:
        matched_reviews = [r for r in (reviews or []) if topic in (r.rule_topics or [])]
        representative_review = ""
        if matched_reviews:
            matched_reviews = sorted(
                matched_reviews,
                key=lambda r: (
                    int(getattr(r, "rating", 0) or 0),
                    float(getattr(r, "vader_score", 0.0) or 0.0),
                    -len((getattr(r, "text", "") or "").strip()),
                ),
            )
            representative_review = (matched_reviews[0].text or "").strip()

        meta = TOPIC_DISPLAY_META.get(
            topic,
            {
                "topic_name": str(topic),
                "core_demand": f"用户对 {topic} 相关体验有持续反馈，值得作为重点跟进主题。",
                "keywords": [str(topic).lower()],
            },
        )

        avg_sentiment = float(stats.get("avg_sentiment", 0.0))
        low_star_ratio = float(stats.get("low_star_ratio", 0.0))
        if avg_sentiment >= 0.2 and low_star_ratio < 0.25:
            sentiment = "positive"
        elif avg_sentiment <= -0.05 or low_star_ratio >= 0.4:
            sentiment = "negative"
        else:
            sentiment = "mixed"

        cards.append(
            {
                "topic_name": meta["topic_name"],
                "core_demand": meta["core_demand"],
                "keywords": list(meta["keywords"]),
                "sentiment": sentiment,
                "representative_review": representative_review,
                "topic_source": "rule_fallback",
                "count": int(stats.get("count", 0)),
                "ratio": float(stats.get("ratio", 0.0)),
            }
        )

    return cards


# 这是整套分析能力的总入口，负责把原始评论一步步变成页面可展示的洞察、验证和日志。
def run_analysis_pipeline(
    reviews: List[ReviewSchema],
    game_name: str,
    run_level2: bool = False,
    run_level3: bool = False,
    enable_llm: bool = True
) -> dict:
    """
    执行完整分析流水线（Step 1-8）。

    参数：
    - reviews: 原始评论列表。
    - game_name: 游戏名称。
    - run_level2: 是否执行 Level2 验证。
    - run_level3: 是否执行 Level3 验证。
    - enable_llm: 是否启用 LLM 相关步骤；关闭后仅保留规则分析与规则洞察。

    返回：
    - dict: 流水线产物，包含评论、LLM结果、洞察、统计、验证与日志。
    """
    # 用结构化日志记录每一步，是为了后面能解释“分析成功了什么、降级了什么、失败在哪”。
    pipeline_log: List[str] = []
    validation_results: dict = {}
    error_details: dict = {}

    # Step 1 先补情感字段，因为后面的优先级判断和验证都依赖情绪强弱信号。
    reviews = analyze_sentiment(reviews or [])
    pipeline_log.append("Step1: 情感分析完成")

    # Step 2
    sentiment_stats = get_sentiment_stats(reviews)
    by_rating_stats = get_sentiment_by_rating(reviews)

    # Step 3 提前做规则主题，是为了给后续 LLM 失败场景准备稳定地基。
    reviews = classify_topics(reviews)
    topic_stats = get_topic_stats(reviews)
    pipeline_log.append("Step3: 规则主题分类完成")

    if enable_llm:
        # Step 4 优先尝试更抽象的 LLM 主题，但一旦失败就立刻切回规则卡片，避免页面断层。
        llm_topics = discover_topics(reviews, game_name)
        topic_cards = llm_topics
        topic_source = "llm"
        if llm_topics is not None:
            validation_results["level1"] = validate_level1_sentiment(llm_topics, reviews)
            if run_level2:
                validation_results["level2"] = validate_level2_overlap(llm_topics, topic_stats)
            pipeline_log.append("Step4: LLM主题发现成功")
        else:
            llm_topics = None
            topic_cards = build_fallback_topics(reviews, topic_stats)
            topic_source = "rule_fallback"
            reason = get_last_topic_discovery_error() or "未知错误"
            error_details["step4_topic_discovery"] = reason
            pipeline_log.append(f"Step4: LLM失败，使用规则主题卡片Fallback（原因：{reason}）")

        # Step 5
        complaints = abstract_problems(reviews, game_name)
        if complaints is not None:
            pipeline_log.append("Step5: LLM痛点总结成功")
        else:
            complaints = None
            reason = get_last_problem_abstraction_error() or "未知错误"
            error_details["step5_problem_abstraction"] = reason
            pipeline_log.append(f"Step5: LLM失败，痛点数据不可用（原因：{reason}）")

        # Step 6
        suggestions = generate_decisions(complaints, sentiment_stats, game_name)
        if suggestions is not None:
            if run_level3:
                rule_insights_for_check = generate_rule_insights(reviews, game_name, topic_stats)
                validation_results["level3"] = validate_level3_suggestions(suggestions, rule_insights_for_check)
            pipeline_log.append("Step6: LLM建议生成成功")
        else:
            suggestions = None
            reason = get_last_decision_generation_error() or "未知错误"
            error_details["step6_decision_generation"] = reason
            pipeline_log.append(f"Step6: LLM失败，使用规则引擎生成Insights（原因：{reason}）")
    else:
        llm_topics = None
        topic_cards = build_fallback_topics(reviews, topic_stats)
        topic_source = "rule_fallback"
        complaints = None
        suggestions = None
        pipeline_log.append("Step4: 已跳过LLM主题发现（快速模式）")
        pipeline_log.append("Step5: 已跳过LLM痛点总结（快速模式）")
        pipeline_log.append("Step6: 已跳过LLM建议生成，直接使用规则洞察（快速模式）")

    # Step 7 统一把不同来源的建议都装配成 InsightSchema，
    # 这样展示层不需要分别兼容 LLM 和规则两套结构。
    insights: List[InsightSchema] = []
    if suggestions is not None:
        confidence = get_confidence_from_sample(len(reviews))
        level1_status = validation_results.get("level1", ValidationResult("passed", confidence)).status

        for idx, item in enumerate(suggestions):
            try:
                matched_complaint = complaints[idx] if idx < len(complaints or []) else {}
                evidence_ratio = float(matched_complaint.get("estimated_ratio", 0.0) or 0.0)
                evidence_ratio = min(max(evidence_ratio, 0.0), 1.0)
                evidence_count = int(round(evidence_ratio * len(reviews)))
                insights.append(
                    InsightSchema(
                        insight_id=str(uuid.uuid4())[:8],
                        game_name=game_name,
                        source="llm",
                        confidence=confidence,
                        rule_trigger=item.get("problem", ""),
                        evidence=item.get("expected_impact", ""),
                        evidence_count=evidence_count,
                        evidence_ratio=evidence_ratio,
                        avg_sentiment=float(sentiment_stats.get("avg_score", 0.0)),
                        priority=item.get("priority", "Medium"),
                        impact_metric=item.get("impact_metric", "Retention"),
                        action=item.get("action", ""),
                        validation_status=level1_status,
                    )
                )
            except Exception as exc:
                print(f"[Warning] LLM Insight组装失败: {exc}")
                continue
    else:
        insights = generate_rule_insights(reviews, game_name, topic_stats)

    pipeline_log.append(f"Step7: 生成 {len(insights)} 条Insight")

    # Step 8
    pipeline_log.append("Step8: 分析完成，准备渲染")

    log_validation("run_analysis_pipeline", "success", f"insights={len(insights)}")

    try:
        review_texts = [r.text for r in reviews if r.text and r.text.strip()]
        if review_texts:
            index_game_reviews(game_name, review_texts)
            pipeline_log.append(f"RAG索引构建完成，共索引{len(review_texts)}条评论")
    except Exception as e:
        pipeline_log.append(f"RAG索引构建失败（不影响主流程）: {e}")

    return {
        "reviews": reviews,
        "llm_topics": llm_topics,
        "topic_cards": topic_cards,
        "topic_source": topic_source,
        "complaints": complaints,
        "suggestions": suggestions,
        "insights": insights,
        "sentiment_stats": sentiment_stats,
        "by_rating_stats": by_rating_stats,
        "topic_stats": topic_stats,
        "validation_results": validation_results,
        "error_details": error_details,
        "pipeline_log": pipeline_log,
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from insights.feedback import get_feedback_stats, record_feedback
    from utils.cache import load_reviews

    reviews = load_reviews("Block Blast")
    if not reviews:
        print("请先运行 Task 1 生成缓存")
        exit()

    print("=== 运行完整分析流水线 ===")
    result = run_analysis_pipeline(
        reviews,
        "Block Blast",
        run_level2=True,
        run_level3=False,
    )

    print("\n📋 Pipeline 执行日志：")
    for log in result["pipeline_log"]:
        print(f"  {log}")

    print(f"\n✅ LLM主题: {'成功' if result['llm_topics'] else '使用Fallback'}")
    print(f"✅ 痛点总结: {'成功' if result['complaints'] else '不可用'}")
    print(f"✅ 产品建议: {'成功' if result['suggestions'] else '使用Fallback'}")
    print(f"✅ Insights数量: {len(result['insights'])}")

    print("\n📊 Insights 列表：")
    for ins in result["insights"]:
        print(f"  [{ins.priority}][{ins.source}][{ins.confidence}] {ins.action[:40]}...")

    print("\n🔍 验证结果：")
    for level, vr in result["validation_results"].items():
        print(f"  {level}: {vr.status} ({vr.confidence}) {vr.note}")

    # 测试反馈记录
    if result["insights"]:
        first_insight = result["insights"][0]
        record_feedback(
            first_insight.insight_id,
            "Block Blast",
            first_insight.source,
            first_insight.priority,
            "useful",
        )
        stats = get_feedback_stats("Block Blast")
        print(f"\n💬 反馈统计：{stats}")


# 这个函数是页面层的防御性适配器，
# 目的是让任何一步缺字段时都尽量保持页面还能渲染，而不是直接抛 KeyError。
def normalize_result(result: dict) -> dict:
    """
    统一 result 字典的 schema，防止 KeyError
    所有页面通过此函数访问 result，不直接用 result[key]
    """
    result = result or {}
    sentiment_stats = dict(result.get("sentiment_stats", {}) or {})
    sentiment_stats.setdefault("positive_ratio", 0.0)
    sentiment_stats.setdefault("negative_ratio", float(sentiment_stats.get("neg_ratio", 0.0) or 0.0))
    sentiment_stats.setdefault("neg_ratio", float(sentiment_stats.get("negative_ratio", 0.0) or 0.0))
    sentiment_stats.setdefault("avg_vader_score", 0.0)
    sentiment_stats.setdefault("agreement_rate", 0.0)
    sentiment_stats.setdefault("low_agreement_warning", False)

    return {
        "reviews": result.get("reviews", []),
        "insights": result.get("insights", []),
        "topics": result.get("llm_topics") or result.get("topic_cards") or [],
        "complaints": result.get("complaints") or [],
        "suggestions": result.get("suggestions") or [],
        "sentiment_stats": sentiment_stats,
        "by_rating_stats": result.get("by_rating_stats", {}),
        "topic_stats": result.get("topic_stats", {}),
        "validation_results": result.get("validation_results", {}),
        "pipeline_log": result.get("pipeline_log", []),
        "llm_available": result.get("llm_topics") is not None,
        "error_details": result.get("error_details", {}),
        "topic_cards": result.get("topic_cards") or result.get("llm_topics") or [],
        "topic_source": result.get("topic_source", "llm" if result.get("llm_topics") else "rule_fallback"),
        "game_name": result.get("game_name", ""),
    }
