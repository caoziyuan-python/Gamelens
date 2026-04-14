# ============================================================
# Module: cross_game.py
# ============================================================
# 【业务作用】提供跨游戏分析入口，输出每款游戏的独特优势、共性问题和机会点
# 【上游】app.py 后续可在跨游戏分析视图中调用
# 【下游】依赖 insights.engine.normalize_result 和 llm._base.call_llm_json
# 【缺失影响】系统只能逐款查看结果，无法稳定沉淀“哪些优势值得借鉴、哪些问题是共性的”
# ============================================================

from typing import Dict, List
import json

from insights.engine import normalize_result
from llm._base import call_llm_json


def _resolve_game_name(
    cache_key: str,
    result: dict
) -> str:
    """
    从缓存中取真实游戏名
    优先用 result["game_name"]
    否则去掉缓存键的技术后缀
    """
    name = result.get("game_name", "")
    if name:
        return name
    # 去掉 __l2_0 / __l2_1 等技术后缀
    return cache_key.split("__l2_")[0].strip()


def _extract_game_summary(
    game_name: str,
    r: dict
) -> dict:
    """
    提取单游戏结构化摘要
    严格使用 normalize_result 的真实字段名
    """
    stats = r.get("sentiment_stats", {})

    # avg_sentiment 双字段回退，防止取不到
    avg_sentiment = float(
        stats.get(
            "avg_vader_score",
            stats.get("avg_score", 0)
        )
    )

    topics = []
    for t in r.get("topics", []):
        # 从 topic_stats 取 ratio（真实统计值）
        topic_name = t.get("topic_name", "")
        ratio = float(
            r.get("topic_stats", {})
             .get(topic_name, {})
             .get("ratio", 0.0)
        )
        topics.append({
            "topic_name":    topic_name,
            "sentiment":     t.get("sentiment", ""),
            "core_demand":   t.get("core_demand", ""),
            "quote":         t.get(
                "representative_review", ""
            )[:80],
            "ratio":         ratio
        })

    complaints = []
    for c in r.get("complaints", []):
        complaints.append({
            "complaint_type":  c.get(
                "complaint_type", ""
            ),
            "estimated_ratio": float(
                c.get("estimated_ratio", 0)
            ),
            "typical_quote":   c.get(
                "typical_quote", ""
            )[:80]
        })

    return {
        "game_name":      game_name,
        "review_count":   len(r.get("reviews", [])),
        "positive_ratio": float(
            stats.get("positive_ratio", 0)
        ),
        "negative_ratio": float(
            stats.get("negative_ratio", 0)
        ),
        "avg_sentiment":  avg_sentiment,
        "topics":         topics,
        "complaints":     complaints,
        "llm_available":  r.get("llm_available", False)
    }


def _rule_based_strengths(
    all_summaries: dict
) -> dict:
    """
    规则引擎版闪光点提取
    三类 + 兜底，保证每款游戏至少1个闪光点
    """
    strengths = {
        game: [] for game in all_summaries
    }

    # ── 类型一：局部优势 ──
    # 正面主题 ratio 相对其他游戏 uplift >= 1.3
    # 且 sample_size >= 10

    for game_name, summary in all_summaries.items():
        review_count = summary.get("review_count", 0)
        if review_count < 50:
            continue

        for topic in summary.get("topics", []):
            if topic.get("sentiment") != "positive":
                continue

            current_ratio = float(
                topic.get("ratio", 0)
            )
            topic_name = topic.get("topic_name", "")

            # 先找同名 topic 的 peer 值
            peer_ratios_exact = []
            for other_game, other_sum in (
                all_summaries.items()
            ):
                if other_game == game_name:
                    continue
                for t in other_sum.get("topics", []):
                    if (t.get("topic_name") == topic_name
                            and t.get("sentiment")
                            == "positive"):
                        peer_ratios_exact.append(
                            float(t.get("ratio", 0))
                        )

            # 区分数据来源：同名精确对比 vs 基线均值
            if peer_ratios_exact:
                peer_avg = (
                    sum(peer_ratios_exact) /
                    len(peer_ratios_exact)
                )
                source_tag = (
                    "rule_cross_game_exact_topic"
                )
            else:
                # 回退：用所有游戏正面主题均值作基线
                all_pos = []
                for other_sum in all_summaries.values():
                    for t in other_sum.get("topics", []):
                        if t.get("sentiment") == "positive":
                            all_pos.append(
                                float(t.get("ratio", 0))
                            )
                peer_avg = (
                    sum(all_pos) / len(all_pos)
                    if all_pos else 0.1
                )
                source_tag = (
                    "rule_cross_game_positive_baseline"
                )

            uplift = (
                current_ratio / max(peer_avg, 0.01)
            )
            if uplift < 1.3:
                continue

            sample_size = int(
                current_ratio * review_count
            )
            if sample_size < 10:
                continue

            strengths[game_name].append({
                "strength":      topic_name,
                "strength_type": "局部优势",
                "mechanism":     topic.get(
                    "core_demand", ""
                ),
                "data_evidence": {
                    "source":        source_tag,
                    "metric":        "topic_positive_ratio",
                    "current_value": round(
                        current_ratio, 3
                    ),
                    "peer_avg":      round(peer_avg, 3),
                    "uplift":        round(uplift, 2),
                    "sample_size":   sample_size,
                    "matched_topic": topic_name,
                    "quote":         topic.get("quote", "")
                },
                "data_backed":    True,
                "confidence":     (
                    "High" if uplift >= 2.0
                    else "Medium"
                ),
                "transferable_pattern": topic.get(
                    "core_demand", ""
                ),
                "how_to_apply":
                    f"参考{game_name}在「{topic_name}」"
                    f"上的产品设计（uplift={uplift:.1f}x）",
                "applicability":  (
                    "高" if uplift >= 2.0 else "中"
                ),
                "why_valuable":   "",
                "pm_apply_tip":   ""
            })

    # ── 类型二：整体优势 ──
    # 情感极差最高且至少领先均值5%
    # 只给1款游戏

    sentiment_gaps = {}
    for game_name, summary in all_summaries.items():
        if summary.get("review_count", 0) < 50:
            continue
        pos = summary.get("positive_ratio", 0)
        neg = summary.get("negative_ratio", 0)
        sentiment_gaps[game_name] = pos - neg

    if len(sentiment_gaps) >= 2:
        best_game = max(
            sentiment_gaps,
            key=lambda x: sentiment_gaps[x]
        )
        best_gap = sentiment_gaps[best_game]
        other_gaps = [
            v for k, v in sentiment_gaps.items()
            if k != best_game
        ]
        peer_avg_gap = (
            sum(other_gaps) / len(other_gaps)
            if other_gaps else 0
        )

        if best_gap > peer_avg_gap + 0.05:
            strengths[best_game].append({
                "strength":      "整体用户口碑最佳",
                "strength_type": "整体优势",
                "mechanism":
                    f"正负情感差值{best_gap*100:.1f}%，"
                    f"领先其他游戏均值"
                    f"{(best_gap-peer_avg_gap)*100:.1f}%",
                "data_evidence": {
                    "source":        "rule_cross_game",
                    "metric":        "sentiment_gap",
                    "current_value": round(best_gap, 3),
                    "peer_avg":      round(
                        peer_avg_gap, 3
                    ),
                    "uplift":        round(
                        best_gap /
                        max(peer_avg_gap, 0.01), 2
                    ),
                    "sample_size":   all_summaries[
                        best_game
                    ].get("review_count", 0),
                    "matched_topic": "整体情感",
                    "quote":         ""
                },
                "data_backed":    True,
                "confidence":     "High",
                "transferable_pattern":
                    "整体体验设计值得参考",
                "how_to_apply":
                    f"研究{best_game}的核心体验设计",
                "applicability":  "高",
                "why_valuable":   "",
                "pm_apply_tip":   ""
            })

    # ── 类型三：规避优势（严格门槛）──
    # 其他游戏 >= 2款有该投诉
    # 其他游戏平均投诉占比 >= 15%
    # 当前游戏该投诉占比 <= 5%
    # 当前游戏评论数 >= 100

    for game_name, summary in all_summaries.items():
        if summary.get("review_count", 0) < 100:
            continue

        my_complaints = {
            c["complaint_type"]: c["estimated_ratio"]
            for c in summary.get("complaints", [])
        }

        peer_stats = {}
        for other_game, other_sum in (
            all_summaries.items()
        ):
            if other_game == game_name:
                continue
            for c in other_sum.get("complaints", []):
                ctype = c["complaint_type"]
                ratio = float(c["estimated_ratio"])
                if ctype not in peer_stats:
                    peer_stats[ctype] = []
                peer_stats[ctype].append(ratio)

        for ctype, peer_ratios in peer_stats.items():
            if len(peer_ratios) < 2:
                continue
            peer_avg = (
                sum(peer_ratios) / len(peer_ratios)
            )
            if peer_avg < 0.15:
                continue
            my_ratio = my_complaints.get(ctype, 0.0)
            if my_ratio > 0.05:
                continue

            strengths[game_name].append({
                "strength":
                    f"有效规避「{ctype}」",
                "strength_type": "规避优势",
                "mechanism":
                    f"其他{len(peer_ratios)}款游戏"
                    f"平均{ctype}投诉占比"
                    f"{peer_avg*100:.0f}%，"
                    f"本游戏仅{my_ratio*100:.0f}%",
                "data_evidence": {
                    "source":
                        "rule_cross_game_avoidance",
                    "metric":
                        "complaint_ratio_comparison",
                    "current_value": round(my_ratio, 3),
                    "peer_avg":      round(peer_avg, 3),
                    "uplift":        round(
                        peer_avg /
                        max(my_ratio, 0.01), 1
                    ),
                    "sample_size":   summary.get(
                        "review_count", 0
                    ),
                    "matched_topic": ctype,
                    "quote":         ""
                },
                "data_backed":    True,
                "confidence":     "High",
                "transferable_pattern":
                    f"研究{game_name}如何规避{ctype}",
                "how_to_apply":
                    f"参考{game_name}在{ctype}方面"
                    f"的产品决策",
                "applicability":  "高",
                "why_valuable":   "",
                "pm_apply_tip":   ""
            })

    # ── 兜底规则 ──
    # 前三类均无结果时，取 ratio 最高的正面主题
    # confidence="Low"，data_backed 视数据情况定

    for game_name in all_summaries:
        if strengths[game_name]:
            continue  # 已有闪光点，不需要兜底

        summary = all_summaries[game_name]
        pos_tops = [
            t for t in summary.get("topics", [])
            if t.get("sentiment") == "positive"
        ]
        if not pos_tops:
            continue

        # 取 ratio 最高的正面主题
        best_topic = max(
            pos_tops,
            key=lambda t: float(t.get("ratio", 0))
        )
        ratio = float(best_topic.get("ratio", 0))
        review_count = summary.get("review_count", 0)
        sample_size = int(ratio * review_count)

        strengths[game_name].append({
            "strength":      best_topic.get(
                "topic_name", ""
            ),
            "strength_type": "基础优势",
            "mechanism":     best_topic.get(
                "core_demand", ""
            ),
            "data_evidence": {
                "source":        "rule_fallback",
                "metric":        "highest_positive_ratio",
                "current_value": round(ratio, 3),
                "peer_avg":      None,
                "uplift":        None,
                "sample_size":   sample_size,
                "matched_topic": best_topic.get(
                    "topic_name", ""
                ),
                "quote":         best_topic.get(
                    "quote", ""
                )
            },
            "data_backed":   (
                sample_size >= 10
            ),
            "confidence":    "Low",
            "transferable_pattern": best_topic.get(
                "core_demand", ""
            ),
            "how_to_apply":
                f"参考{game_name}的"
                f"{best_topic.get('topic_name', '')}体验",
            "applicability": "低",
            "why_valuable":  "",
            "pm_apply_tip":  ""
        })

    return strengths


def _llm_enrich_strengths(
    rule_strengths: dict
) -> dict:
    """
    LLM只补充两个解释性字段：
    why_valuable 和 pm_apply_tip
    不覆盖任何规则字段
    """
    enriched = {}

    for game_name, slist in rule_strengths.items():
        if not slist:
            enriched[game_name] = slist
            continue

        llm_input = [
            {
                "strength":      s["strength"],
                "strength_type": s["strength_type"],
                "mechanism":     s["mechanism"],
                "uplift":        s["data_evidence"].get(
                    "uplift"
                )
            }
            for s in slist
        ]

        prompt = f"""
以下是{game_name}通过数据分析得出的产品闪光点：
{json.dumps(llm_input, ensure_ascii=False)}

请为每个闪光点补充两个字段，输出JSON数组：
[
  {{
    "strength": "原闪光点名称（不要修改）",
    "why_valuable": "为什么对用户有价值（25字以内，中文）",
    "pm_apply_tip": "新游戏如何借鉴这个模式（25字以内，中文）"
  }}
]

规则：
- 不要修改 strength 字段的值
- 不要编造任何数字
- 只输出JSON，禁止输出其他文字
"""
        llm_result = call_llm_json(prompt)
        llm_map = {}
        if llm_result and isinstance(llm_result, list):
            for item in llm_result:
                name = item.get("strength", "")
                if name:
                    llm_map[name] = item

        enriched_list = []
        for s in slist:
            s_copy = dict(s)
            llm_item = llm_map.get(
                s["strength"], {}
            )
            # 只新增解释字段，不覆盖任何规则字段
            s_copy["why_valuable"] = llm_item.get(
                "why_valuable", ""
            )
            s_copy["pm_apply_tip"] = llm_item.get(
                "pm_apply_tip", ""
            )
            enriched_list.append(s_copy)

        enriched[game_name] = enriched_list

    return enriched


def _rule_common_problems(
    all_summaries: dict
) -> list:
    """
    规则引擎版共性问题（LLM Fallback）
    补充 avg_ratio 和 game_count 供后续模块复用
    """
    complaint_data = {}
    for game_name, summary in all_summaries.items():
        for c in summary.get("complaints", []):
            ctype = c.get("complaint_type", "")
            ratio = float(c.get("estimated_ratio", 0))
            if ctype:
                if ctype not in complaint_data:
                    complaint_data[ctype] = {
                        "games":  [],
                        "ratios": []
                    }
                complaint_data[ctype]["games"].append(
                    game_name
                )
                complaint_data[ctype]["ratios"].append(
                    ratio
                )

    common = []
    for ctype, data in complaint_data.items():
        games = data["games"]
        ratios = data["ratios"]
        if len(games) < 2:
            continue
        avg_ratio = sum(ratios) / len(ratios)
        game_count = len(games)
        common.append({
            "problem":        ctype,
            "description":
                f"{game_count}款游戏均有{ctype}投诉",
            "affected_games": games,
            "game_count":     game_count,
            "avg_ratio":      round(avg_ratio, 3),
            "severity":       (
                "High" if avg_ratio >= 0.3
                else "Medium" if avg_ratio >= 0.15
                else "Low"
            ),
            "evidence":
                f"出现于{game_count}款游戏，"
                f"平均投诉占比{avg_ratio*100:.0f}%"
        })

    return sorted(
        common,
        key=lambda x: x["avg_ratio"],
        reverse=True
    )


def cross_game_analysis(
    all_games_cache: dict
) -> dict:
    """
    跨游戏分析主入口
    执行顺序：
    1. 解析真实游戏名（不使用缓存键）
    2. 提取结构化摘要
    3. 规则引擎找闪光点（Primary）
    4. LLM补充解释字段（Enrich，失败降级）
    5. LLM找共性问题和机会点（失败用规则兜底）
    """
    # Step 1：解析真实游戏名
    resolved_cache = {}
    for cache_key, result in all_games_cache.items():
        real_name = _resolve_game_name(
            cache_key, result
        )
        resolved_cache[real_name] = result

    # Step 2：提取结构化摘要
    raw_summaries = {}
    for game_name, result in resolved_cache.items():
        r = normalize_result(result)
        raw_summaries[game_name] = (
            _extract_game_summary(game_name, r)
        )

    # Step 3：规则引擎闪光点
    rule_strengths = _rule_based_strengths(
        raw_summaries
    )

    # Step 4：LLM补充解释（失败降级）
    try:
        final_strengths = _llm_enrich_strengths(
            rule_strengths
        )
    except Exception:
        final_strengths = rule_strengths

    # Step 5：LLM共性问题和机会点
    summaries_json = json.dumps(
        raw_summaries, ensure_ascii=False, indent=2
    )
    cross_prompt = f"""
以下是{len(raw_summaries)}款游戏的结构化分析数据：
{summaries_json}

请输出JSON：
{{
  "common_problems": [
    {{
      "problem": "问题名称",
      "description": "具体描述",
      "affected_games": ["游戏A","游戏B"],
      "game_count": 2,
      "avg_ratio": 0.25,
      "severity": "High/Medium/Low",
      "evidence": "数据支撑"
    }}
  ],
  "opportunity_gaps": [
    {{
      "gap": "空白点描述",
      "evidence": "用户需求信号",
      "affected_games": ["游戏列表"],
      "potential": "High/Medium/Low"
    }}
  ]
}}

规则：
- common_problems：至少2款游戏同时存在才算共性
- opportunity_gaps：用户明确表达需要但所有游戏都没做好
- 只输出JSON，禁止输出解释文字
"""
    llm_result = call_llm_json(cross_prompt)

    if (llm_result
            and isinstance(llm_result, dict)
            and "common_problems" in llm_result):
        common_problems = llm_result.get(
            "common_problems", []
        )
        opportunity_gaps = llm_result.get(
            "opportunity_gaps", []
        )
    else:
        common_problems = _rule_common_problems(
            raw_summaries
        )
        opportunity_gaps = []

    return {
        "unique_strengths":  final_strengths,
        "common_problems":   common_problems,
        "opportunity_gaps":  opportunity_gaps,
        "raw_summaries":     raw_summaries
    }
