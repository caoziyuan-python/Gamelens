# ============================================================
# Module: sentiment.py
# ============================================================
# 【业务作用】为评论补充双模型情感标签，并产出整体情感统计口径
# 【上游】insights.engine 在分析流水线开始阶段调用
# 【下游】依赖 TextBlob、VADER、config.THRESHOLDS、data.schema.ReviewSchema
# 【缺失影响】系统会失去评分之外的语义情绪判断，很多主题优先级和验证能力都会变弱
# ============================================================

from typing import List

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import THRESHOLDS
from data.schema import ReviewSchema


# 这个函数的价值不是单纯“打标签”，而是把用户文本里的情绪强弱转成后续可计算的信号。
# 没有它，系统只能看星级，无法识别“高星但抱怨多”或“低星但文本温和”这类细微差异。
def analyze_sentiment(reviews: List[ReviewSchema]) -> List[ReviewSchema]:
    """
    对评论列表执行双模型情感分析，并原地写回情感字段。

    处理逻辑：
    - VADER 使用 compound 分数：
      compound > 0.05  -> vader_label = "Positive"
      compound < -0.05 -> vader_label = "Negative"
      其余             -> vader_label = "Neutral"
      同时写入 vader_score。
    - TextBlob 使用 polarity 分数：
      polarity > 0.1  -> textblob_label = "Positive"
      polarity < -0.1 -> textblob_label = "Negative"
      其余            -> textblob_label = "Neutral"
      同时写入 textblob_score。
    - 任意单条评论分析异常时，仅打印 warning，并将该条评论相关字段置为默认中性值。
      不中断整体流程。

    参数：
    - reviews: 待分析的评论对象列表。

    返回：
    - List[ReviewSchema]: 写回情感结果后的同一批评论对象。
    """
    if not reviews:
        return reviews

    # 阈值改成可配置，是为了后续能按游戏品类微调“多大情绪波动才算明显正负面”，
    # 而不用去改分析代码本身。
    vader_pos_threshold = float(THRESHOLDS.get("vader_positive", 0.05))
    vader_neg_threshold = float(THRESHOLDS.get("vader_negative", -0.05))
    textblob_pos_threshold = float(THRESHOLDS.get("textblob_positive", 0.1))
    textblob_neg_threshold = float(THRESHOLDS.get("textblob_negative", -0.1))

    try:
        analyzer = SentimentIntensityAnalyzer()
    except Exception as exc:
        print(f"[Warning] VADER 初始化失败: {exc}")
        analyzer = None

    for review in reviews:
        try:
            text = (review.text or "").strip()

            if analyzer is None:
                vader_score = 0.0
            else:
                vader_score = float(analyzer.polarity_scores(text).get("compound", 0.0))

            if vader_score > vader_pos_threshold:
                vader_label = "Positive"
            elif vader_score < vader_neg_threshold:
                vader_label = "Negative"
            else:
                vader_label = "Neutral"

            textblob_score = float(TextBlob(text).sentiment.polarity)
            if textblob_score > textblob_pos_threshold:
                textblob_label = "Positive"
            elif textblob_score < textblob_neg_threshold:
                textblob_label = "Negative"
            else:
                textblob_label = "Neutral"

            review.vader_score = vader_score
            review.vader_label = vader_label
            review.textblob_score = textblob_score
            review.textblob_label = textblob_label
        except Exception as exc:
            print(f"[Warning] 评论情感分析失败(review_id={review.review_id}): {exc}")
            # 出错时统一回落到中性值，是为了让后续统计还能继续跑完，
            # 不会因为个别脏数据把整批评论分析中断。
            review.vader_score = 0.0
            review.vader_label = "Neutral"
            review.textblob_score = 0.0
            review.textblob_label = "Neutral"

    return reviews


# 这个函数给管理层和页面 KPI 提供最简情感摘要，
# 用很少的数字快速回答“这款游戏口碑大体是好是坏”。
def get_sentiment_stats(reviews: List[ReviewSchema]) -> dict:
    """
    统计评论整体情感指标。

    处理逻辑：
    - avg_score 使用 rating 字段均值（0-5）。
    - neg_ratio 使用 VADER Negative 标签占比（百分比，0-100）。
    - 空列表时返回默认值，不抛异常。

    参数：
    - reviews: 评论列表。

    返回：
    - dict: 包含 avg_score 与 neg_ratio 的统计结果。
    """
    if not reviews:
        return {"avg_score": 0.0, "neg_ratio": 0.0}

    try:
        avg_score = sum(int(r.rating) for r in reviews) / len(reviews)
    except Exception:
        avg_score = 0.0

    try:
        neg_count = sum(1 for r in reviews if str(r.vader_label).lower() == "negative")
        neg_ratio = (neg_count / len(reviews)) * 100
    except Exception:
        neg_ratio = 0.0

    return {"avg_score": round(float(avg_score), 4), "neg_ratio": round(float(neg_ratio), 4)}


# 这个函数把“不同星级用户为什么给出不同评价”拆开来看，
# 是为了发现体验断层，而不是只看一个整体平均分。
def get_sentiment_by_rating(reviews: List[ReviewSchema]) -> dict:
    """
    按星级统计情感分布与均值。

    处理逻辑：
    - 按 1-5 星分组统计评论数与平均 vader_score。
    - 空分组的平均值记为 0.0。
    - 任意异常不抛出，返回已累计的结果。

    参数：
    - reviews: 评论列表。

    返回：
    - dict: 形如 {1: {"count": int, "avg_vader": float}, ..., 5: {...}}。
    """
    stats = {star: {"count": 0, "avg_vader": 0.0} for star in range(1, 6)}
    sums = {star: 0.0 for star in range(1, 6)}

    for rv in reviews or []:
        try:
            star = int(rv.rating)
            if star not in stats:
                continue
            stats[star]["count"] += 1
            sums[star] += float(rv.vader_score)
        except Exception:
            continue

    for star in range(1, 6):
        cnt = stats[star]["count"]
        stats[star]["avg_vader"] = (sums[star] / cnt) if cnt > 0 else 0.0

    return stats
