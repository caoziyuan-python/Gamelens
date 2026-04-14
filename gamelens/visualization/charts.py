import base64
# ============================================================
# Module: charts.py
# ============================================================
# 【业务作用】把评论分析结果转成图表和词云，帮助产品经理快速读懂趋势和差异
# 【上游】app.py 各个 Tab 调用
# 【下游】依赖 Plotly、Matplotlib、WordCloud 和 ReviewSchema
# 【缺失影响】分析结果只能以表格和文字呈现，跨游戏比较和趋势识别会更吃力
# ============================================================

import io
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import nltk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud

from data.schema import ReviewSchema

# 统一配色是为了让用户在不同图表之间形成稳定映射，看到颜色就能快速联想到正负面和优先级含义。
# 配色规范
COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
    "ads": "#E67E22",
    "high": "#E74C3C",
    "medium": "#E67E22",
    "low": "#95A5A6",
}
CHART_SIZE = {"width": 800, "height": 400}


# 评分分布图负责回答“口碑是集中稳定还是明显分裂”。
def plot_rating_distribution(reviews: List[ReviewSchema]) -> go.Figure:
    """
    评分分布柱状图。

    处理逻辑：
    - x轴展示 1-5 星。
    - y轴展示评论数量。
    - 柱体显示百分比标注。
    - 颜色从 5 星绿色渐变到 1 星红色。
    - 标题使用双语。

    参数：
    - reviews: 评论列表。

    返回：
    - go.Figure: Plotly 柱状图对象。
    """
    counts = {i: 0 for i in range(1, 6)}
    for rv in reviews or []:
        try:
            star = int(rv.rating)
            if star in counts:
                counts[star] += 1
        except Exception:
            continue

    total = sum(counts.values())
    x = [1, 2, 3, 4, 5]
    y = [counts[i] for i in x]
    percents = [(v / total * 100) if total > 0 else 0 for v in y]

    # 1星到5星：红 -> 绿
    bar_colors = ["#E74C3C", "#E67E22", "#F1C40F", "#82E0AA", "#2ECC71"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in percents],
            textposition="outside",
            hovertemplate="%{x}星: %{y}条 (%{text})<extra></extra>",
        )
    )
    fig.update_layout(
        title="评分分布 / Rating Distribution",
        xaxis_title="星级 / Rating",
        yaxis_title="评论数 / Count",
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# 饼图适合给非技术读者快速建立整体印象，先看大盘情绪，再决定是否深挖细节。
def plot_sentiment_pie(sentiment_stats: dict) -> go.Figure:
    """
    情感分布饼图。

    处理逻辑：
    - 展示 Positive/Neutral/Negative 三个扇区。
    - 使用 COLORS 规定配色。
    - 显示数量和百分比。
    - 标题使用双语。

    参数：
    - sentiment_stats: 情感统计字典，支持正负中占比或计数。

    返回：
    - go.Figure: Plotly 饼图对象。
    """
    pos_ratio = float(sentiment_stats.get("positive_ratio", 0.0))
    neg_ratio = float(sentiment_stats.get("neg_ratio", 0.0))
    if neg_ratio > 1:
        neg_ratio = neg_ratio / 100.0

    neu_ratio = float(sentiment_stats.get("neutral_ratio", 1 - pos_ratio - neg_ratio))
    # 防御性修正
    if neu_ratio < 0:
        neu_ratio = 0.0

    values = [pos_ratio, neu_ratio, neg_ratio]
    labels = ["Positive", "Neutral", "Negative"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(
                    colors=[
                        COLORS["positive"],
                        COLORS["neutral"],
                        COLORS["negative"],
                    ]
                ),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="情感分布 / Sentiment Distribution",
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# 这个图把评分和文本情绪放在一起，是为了发现“打分”和“说法”是否一致。
def plot_sentiment_by_rating(by_rating_stats: dict) -> go.Figure:
    """
    分星级情感得分柱状图。

    处理逻辑：
    - x轴为1-5星。
    - y轴为平均VADER分，范围[-1,1]。
    - 正值绿色、负值红色。
    - 添加 y=0 参考线。
    - 标题使用双语。

    参数：
    - by_rating_stats: 分星级统计字典。

    返回：
    - go.Figure: Plotly 柱状图对象。
    """
    x = [1, 2, 3, 4, 5]
    y = []
    colors = []
    for star in x:
        stat = by_rating_stats.get(star, by_rating_stats.get(str(star), {}))
        score = float(stat.get("avg_vader", 0.0))
        y.append(score)
        colors.append(COLORS["positive"] if score >= 0 else COLORS["negative"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker_color=colors,
            text=[f"{v:.2f}" for v in y],
            textposition="outside",
            hovertemplate="%{x}星: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_layout(
        title="分星级情感得分 / Sentiment Score by Rating",
        xaxis_title="星级 / Rating",
        yaxis_title="平均VADER分 / Avg VADER",
        yaxis=dict(range=[-1, 1]),
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# 热力图用来同时回答“哪些主题最常见”和“它们集中在几星评论里”。
def plot_topic_heatmap(topic_stats: dict, reviews: List[ReviewSchema]) -> go.Figure:
    """
    主题×评分交叉热力图。

    处理逻辑：
    - x轴为1-5星。
    - y轴为主题名称。
    - 颜色表示各星级中该主题评论数。
    - 标题使用双语。

    参数：
    - topic_stats: 主题统计信息（用于确定主题集合）。
    - reviews: 评论列表。

    返回：
    - go.Figure: Plotly 热力图对象。
    """
    topics = list((topic_stats or {}).keys())
    if not topics:
        topics = ["NoTopic"]

    stars = [1, 2, 3, 4, 5]
    matrix = []

    for topic in topics:
        row = []
        for star in stars:
            count = 0
            for rv in reviews or []:
                try:
                    if int(rv.rating) == star and topic in (rv.rule_topics or []):
                        count += 1
                except Exception:
                    continue
            row.append(count)
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=stars,
            y=topics,
            colorscale="YlOrRd",
            colorbar=dict(title="Count"),
            hovertemplate="主题=%{y}<br>星级=%{x}<br>数量=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="主题×评分热力图 / Topic × Rating Heatmap",
        xaxis_title="星级 / Rating",
        yaxis_title="主题 / Topic",
        width=CHART_SIZE["width"],
        height=max(CHART_SIZE["height"], 300 + 28 * len(topics)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# 词云更像快速扫描器，帮助产品经理先抓高频词，再回到主题和原评论里精读。
def plot_keyword_cloud(reviews: List[ReviewSchema]) -> str:
    """
    生成关键词词云图片并返回 base64 字符串。

    处理逻辑：
    - 正面词来自 vader_score > 0.1 的评论。
    - 负面词来自 vader_score < -0.1 的评论。
    - 进行停用词过滤（nltk stopwords）。
    - 生成 PNG 后转 base64，供 st.image 渲染。

    参数：
    - reviews: 评论列表。

    返回：
    - str: PNG 图片的 base64 编码字符串。
    """
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        try:
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))
        except Exception:
            stop_words = {
                "the", "a", "an", "is", "are", "to", "and", "of", "for", "in", "on", "it", "this", "that"
            }

    pos_tokens = []
    neg_tokens = []

    for rv in reviews or []:
        text = (rv.text or "").lower()
        tokens = [w.strip(".,!?;:'\"()[]{}") for w in text.split()]
        tokens = [w for w in tokens if w and w.isascii() and w not in stop_words and len(w) > 2]

        if float(rv.vader_score) > 0.1:
            pos_tokens.extend(tokens)
        elif float(rv.vader_score) < -0.1:
            neg_tokens.extend(tokens)

    pos_counter = Counter(pos_tokens)
    neg_counter = Counter(neg_tokens)

    if not pos_counter and not neg_counter:
        pos_counter = Counter({"no": 1, "keywords": 1, "available": 1})

    merged = {}
    for k, v in pos_counter.items():
        merged[k] = merged.get(k, 0) + v
    for k, v in neg_counter.items():
        merged[k] = merged.get(k, 0) + v

    def color_func(word, **kwargs):
        if word in neg_counter:
            return COLORS["negative"]
        return COLORS["positive"]

    wc = WordCloud(width=1200, height=600, background_color="white", max_words=120)
    wc.generate_from_frequencies(merged)
    wc.recolor(color_func=color_func)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("关键词词云 / Keyword Cloud")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# 多游戏对比图的价值在于把“哪款更好”从感觉问题变成可并排比较的数字问题。
def plot_multi_game_comparison(all_stats: dict) -> go.Figure:
    """
    多游戏评分与正面情感占比对比图。

    处理逻辑：
    - each game 一组分组柱。
    - 展示 avg_rating 与 positive_ratio（按百分比）。
    - 标题使用双语。

    参数：
    - all_stats: 多游戏统计字典。

    返回：
    - go.Figure: Plotly 分组柱状图对象。
    """
    games = list((all_stats or {}).keys())
    avg_ratings = [float(all_stats[g].get("avg_rating", 0.0)) for g in games]
    pos_ratios = [float(all_stats[g].get("positive_ratio", 0.0)) * 100 for g in games]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="均分 / Avg Rating", x=games, y=avg_ratings, marker_color="#3498DB"))
    fig.add_trace(
        go.Bar(
            name="正面占比% / Positive Ratio%",
            x=games,
            y=pos_ratios,
            marker_color=COLORS["positive"],
        )
    )
    fig.update_layout(
        barmode="group",
        title="多游戏对比 / Multi-Game Comparison",
        xaxis_title="游戏 / Game",
        yaxis_title="值 / Value",
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
