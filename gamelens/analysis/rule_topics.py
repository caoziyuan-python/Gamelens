# ============================================================
# Module: rule_topics.py
# ============================================================
# 【业务作用】在 LLM 失败或不可用时，用规则关键词为评论打主题标签并输出主题统计
# 【上游】insights.engine 在完整分析流水线中调用
# 【下游】依赖 config.TOPIC_KEYWORDS 和 data.schema.ReviewSchema
# 【缺失影响】系统会失去最关键的规则兜底，LLM 失效时主题分析和洞察生成会明显降级
# ============================================================

from typing import Dict, List

from config import TOPIC_KEYWORDS
from data.schema import ReviewSchema


# 这个函数解决的是“模型失败时系统仍要有主题结果”的问题。
# 没有它，主题页和部分 Insight 生成就会直接断档，产品侧无法继续看趋势和优先级。
def detect_rule_topics(
    reviews: List[ReviewSchema],
    topic_keywords: Dict[str, List[str]]
) -> List[ReviewSchema]:
    """
    基于关键词规则对评论进行主题识别，并原地写回 rule_topics 字段。

    处理逻辑：
    - 对每条评论执行不区分大小写的关键词匹配。
    - 某主题任一关键词命中即视为该主题命中。
    - 每条评论的 rule_topics 去重后写回；无命中则写空列表。
    - 单条评论处理异常时仅打印 warning，不中断整体流程。

    参数：
    - reviews: 待处理评论列表。
    - topic_keywords: 主题关键词字典，格式为 {topic: [keyword1, keyword2, ...]}。
      若传入为空，则回退使用 config.TOPIC_KEYWORDS。

    返回：
    - List[ReviewSchema]: 写回 rule_topics 后的同一批评论对象。
    """
    if not reviews:
        return reviews

    # 优先允许外部传入关键词，是为了后续做 A/B 规则调优时不必改全局配置。
    keyword_map = topic_keywords or TOPIC_KEYWORDS

    for review in reviews:
        try:
            text = (review.text or "").lower()
            # list 而不是 set，是为了先保留命中顺序，后面再去重，
            # 这样页面展示时更接近用户文本里真实出现的关注顺序。
            matched_topics: List[str] = []

            for topic, keywords in keyword_map.items():
                if not keywords:
                    continue
                for keyword in keywords:
                    key = str(keyword).strip().lower()
                    if key and key in text:
                        matched_topics.append(topic)
                        break

            # 按匹配顺序去重，保证输出稳定。
            review.rule_topics = list(dict.fromkeys(matched_topics))
        except Exception as exc:
            print(f"[Warning] 规则主题识别失败(review_id={review.review_id}): {exc}")
            review.rule_topics = []

    return reviews


# 这个包装函数保留了单一入口，让流水线只关心“做规则分类”这件事，
# 不必知道具体关键词配置来自哪里，方便后续替换规则来源。
def classify_topics(reviews: List[ReviewSchema]) -> List[ReviewSchema]:
    """
    兼容入口：使用 config.TOPIC_KEYWORDS 对评论执行规则主题识别。

    参数：
    - reviews: 评论列表。

    返回：
    - List[ReviewSchema]: 写回 rule_topics 后的评论列表。
    """
    return detect_rule_topics(reviews, TOPIC_KEYWORDS)


# 这个函数把“主题命中了多少条评论、是否足够严重”统一沉淀成统计口径，
# 供洞察页、Fallback 规则建议和优先级判断共同复用。
def get_topic_stats(reviews: List[ReviewSchema]) -> dict:
    """
    统计规则主题分布与优先级信号。

    指标说明：
    - count: 命中主题的评论数。
    - ratio: 命中评论占总评论比例。
    - avg_sentiment: 主题下评论平均 vader_score。
    - low_star_ratio: 主题下 1-2 星评论占比。
    - is_high_priority: low_star_ratio 是否超过配置阈值。

    参数：
    - reviews: 已完成 classify_topics 的评论列表。

    返回：
    - dict: 以主题名为 key 的统计结果字典。
    """
    from config import THRESHOLDS

    topic_buckets = {}
    total = len(reviews or [])
    # 用低星占比而不是单纯评论数做高优判断，是为了把“抱怨强度”放在“声量”之前，
    # 否则高频但中性的讨论会挤掉真正伤害评分和留存的问题。
    ratio_threshold = float(THRESHOLDS.get("high_priority_topic_ratio", 0.40))

    for rv in reviews or []:
        topics = rv.rule_topics or []
        for topic in topics:
            if topic not in topic_buckets:
                # dict 比 dataclass 更适合这里的中间累加态，
                # 因为字段少、组装快，也方便后续直接转成页面展示结构。
                topic_buckets[topic] = {
                    "count": 0,
                    "sent_sum": 0.0,
                    "low_star_count": 0,
                }
            topic_buckets[topic]["count"] += 1
            topic_buckets[topic]["sent_sum"] += float(rv.vader_score)
            try:
                if int(rv.rating) <= 2:
                    topic_buckets[topic]["low_star_count"] += 1
            except Exception:
                pass

    result = {}
    for topic, raw in topic_buckets.items():
        count = int(raw["count"])
        ratio = (count / total) if total > 0 else 0.0
        avg_sentiment = (raw["sent_sum"] / count) if count > 0 else 0.0
        low_star_ratio = (raw["low_star_count"] / count) if count > 0 else 0.0
        result[topic] = {
            "count": count,
            "ratio": ratio,
            "avg_sentiment": avg_sentiment,
            "low_star_ratio": low_star_ratio,
            "is_high_priority": low_star_ratio > ratio_threshold,
        }

    return result
