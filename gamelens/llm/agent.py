import json
# ============================================================
# Module: agent.py
# ============================================================
# 【业务作用】提供对话式分析助手，完成意图识别、工具规划、结果整合和产品经理可读回答
# 【上游】app.py 的 AI 助手和自动建议功能调用
# 【下游】依赖 game_finder、insights.engine、llm._base、utils.cache 等模块
# 【缺失影响】用户只能看固定面板，无法围绕当前分析结果进行追问和对比
# ============================================================

import re
from pathlib import Path
from typing import Any

import faiss

from config import EMBEDDING_DEPLOYMENT_NAME
from data.game_finder import search_games
from insights.engine import normalize_result, run_analysis_pipeline
from llm._base import _get_azure_client, call_llm, call_llm_json, log_validation
from utils.cache import get_or_fetch


TOOLS = [
    {
        "name": "fetch_and_analyze",
        "description": "抓取游戏评论并完成情感分析和主题分类",
        "params": ["app_id", "game_name"],
    },
    {
        "name": "discover_topics",
        "description": "用LLM归纳游戏的核心用户主题",
        "params": ["game_name"],
    },
    {
        "name": "abstract_problems",
        "description": "归纳游戏的Top3用户痛点",
        "params": ["game_name"],
    },
    {
        "name": "generate_decisions",
        "description": "生成产品优化建议",
        "params": ["game_name"],
    },
    {
        "name": "compare_games",
        "description": "对比多款游戏的核心指标",
        "params": ["game_names"],
    },
    {
        "name": "search_games",
        "description": "根据关键词搜索同类竞品游戏",
        "params": ["keyword"],
    },
]


def _retrieve_game_reviews(query: str, game_name: str, index_dir: str = "indices", k: int = 5) -> list[str]:
    try:
        index_path = Path(index_dir) / f"{game_name}.faiss"
        meta_path = Path(index_dir) / f"{game_name}_meta.json"
        if not index_path.exists() or not meta_path.exists():
            return []

        client = _get_azure_client()
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_NAME or "text-embedding-ada-002",
            input=str(query or "")[:8000],
        )
        query_embedding = response.data[0].embedding

        np = __import__("numpy")
        query_vector = np.array([query_embedding], dtype=np.float32)
        index = faiss.read_index(str(index_path))
        _, indices = index.search(query_vector, max(int(k or 5), 1))

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        results: list[str] = []
        for idx in indices[0]:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(meta):
                continue
            text = str((meta[idx_int] or {}).get("text", "") or "").strip()
            if text:
                results.append(text)
        return results
    except Exception:
        return []


def analyze_competitive_gap(
    question: str,
    target_game: str,
    competitor_game: str,
    target_reviews: list[str],
    competitor_reviews: list[str],
) -> dict:
    fallback = {
        "gap_summary": "分析暂时不可用，请检查LLM连接",
        "root_causes": [],
        "priority_score": 0,
        "error": True,
    }
    try:
        if not target_reviews and not competitor_reviews:
            return fallback

        result = call_llm_json(
            prompt=(
                f"分析问题：{question}\n\n"
                f"{target_game}的用户评论：\n" + "\n".join((target_reviews or [])[:5]) +
                f"\n\n{competitor_game}的用户评论：\n" + "\n".join((competitor_reviews or [])[:5]) +
                '\n\n返回格式：{"gap_summary": "差距总结", "root_causes": ["原因1", "原因2"], "priority_score": 1到5的整数}'
            ),
            system="你是游戏产品分析师。基于真实用户评论，用中文对比两款游戏在用户反馈上的差距。只返回JSON，不返回其他内容。",
        )

        if isinstance(result, list):
            result = result[0] if result and isinstance(result[0], dict) else {}
        if not isinstance(result, dict):
            result = {}

        result.setdefault("gap_summary", "")
        result.setdefault("root_causes", [])
        result.setdefault("priority_score", 0)
        return result
    except Exception:
        return fallback


INTENT_SYSTEM_PROMPT = """
你是一个任务分类器，只做分类不做分析。

判断用户问题属于哪一类：

comparison    → 涉及竞品对比、区别、哪个更好、
                优劣、适合场景对比
                示例：「和羊了个羊的区别」
                      「哪个更适合碎片时间」
                      「和同类游戏相比如何」

user_scenario → 涉及用户体验、使用场景、
                目标用户、用户心理
                示例：「用户为什么喜欢」
                      「什么场景下玩」

single_analysis → 单游戏的数据分析、问题诊断
                示例：「这款游戏最大的问题」
                      「用户评分为什么低」

suggestion    → 产品优化建议、如何改进
                示例：「给我3条改进建议」
                      「广告策略怎么优化」

只输出JSON，不输出其他任何内容：
{"intent": "comparison"}
"""


TOOL_PLANNING_RULES = """
工具选择规则（必须遵守）：

1. intent == "comparison" 时：
   - 必须调用 compare_games
   - game_names 必须包含用户问题里提到的游戏名
   - 若用户没有明确说竞品名，
     从 available_data 里取 all_games_cache 的键
   - 禁止不调用 compare_games 就回答对比问题

2. intent == "user_scenario" 时：
   - 优先调用 discover_topics
   - 数据重点：主题 + 原始评论
   - 不需要调用变现相关工具

3. intent == "single_analysis" 时：
   - 调用 abstract_problems
   - 数据重点：投诉 + 低分评论

4. intent == "suggestion" 时：
   - 调用 generate_decisions
   - 基于已有 pipeline_results 生成

5. 任何情况下：
   - 不要忽略用户问题里出现的游戏名称
   - 已有数据优先复用，不重复调用分析
"""


AGENT_SYSTEM_PROMPT = """
你是一名资深游戏产品经理，
同时具备数据分析（DA）和数据科学（DS）能力。

你的核心能力不是复述数据，而是基于数据做业务判断。

【思考步骤——每次回答前必须经历这四步】

Step 1：判断问题类型
  用户体验类 → 玩法/难度/节奏/沉浸感
  变现类     → 广告/付费/订阅/ROI
  用户分层类 → 新手/核心用户/流失用户
  竞品差异类 → 场景定位/用户群体/核心机制

Step 2：选择最相关的数据（不是全部用）
  用户体验类 → 优先用：用户主题 + 原始评论
  变现类     → 优先用：投诉数据 + 低分评论占比
  竞品差异类 → 优先用：多游戏对比 + 场景相关主题
  用户分层类 → 优先用：评分分布 + 情感分层数据

Step 3：做一层业务解释
  数据 → 用户行为
  用户行为 → 产品问题
  产品问题 → 业务影响（留存/变现/口碑）

Step 4：输出
  第一句：直接给结论（说清本质，不是说「根据数据」）
  中间：用最关键的1-2个数字 + 1条用户原话支撑
  最后：1-2条具体可执行的产品建议

【语言规范】
输出语言策略：
- 所有分析、结论、建议：100%中文
- 用户原话引用：可保留英文原文，
  但紧跟中文翻译，格式：
  「原话：'Too many ads'（广告太多了）」
- 数据指标名称：中文为主，
  英文缩写可保留括号说明，格式：
  「正面情感占比34%（Positive Sentiment）」
- 禁止在同一个句子里中英文混写
"""


def _to_json_text(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compute_positive_ratio(reviews: list) -> float:
    try:
        if not reviews:
            return 0.0
        positive_count = sum(
            1
            for review in reviews
            if str(getattr(review, "vader_label", "")).lower() == "positive"
        )
        return positive_count / len(reviews)
    except Exception:
        return 0.0


def _compute_avg_vader_score(reviews: list) -> float:
    try:
        if not reviews:
            return 0.0
        return sum(_safe_float(getattr(review, "vader_score", 0.0)) for review in reviews) / len(reviews)
    except Exception:
        return 0.0


def _normalize_llm_object(response: Any) -> dict:
    try:
        if isinstance(response, dict):
            return response
        if isinstance(response, list) and response and isinstance(response[0], dict):
            return response[0]
        return {}
    except Exception:
        return {}


def _extract_game_names(user_query: str, available_data: dict, pipeline_results: dict) -> list[str]:
    try:
        found = []
        current_game = str(
            available_data.get("game_name", "") or pipeline_results.get("game_name", "")
        ).strip()
        if current_game:
            found.append(current_game)

        all_cache = pipeline_results.get("all_games_cache", {}) or {}
        known_games = list(all_cache.keys())
        if current_game and current_game not in known_games:
            known_games.append(current_game)

        q = str(user_query or "").lower()
        for game_name in known_games:
            if game_name and game_name.lower() in q and game_name not in found:
                found.append(game_name)

        alias_map = {
            "羊了个羊": "羊了个羊",
            "vita mahjong": "Vita Mahjong",
            "vita麻将": "Vita Mahjong",
        }
        for alias, mapped in alias_map.items():
            if alias.lower() in q and mapped not in found:
                found.append(mapped)

        return found
    except Exception:
        return []


# 先判问题类型，是为了让 Agent 只读取最相关的数据，避免每个问题都把所有工具跑一遍。
def classify_intent(user_query: str) -> str:
    """
    识别用户问题的意图类型
    失败时返回 "single_analysis" 作为兜底
    """
    try:
        result = call_llm_json(
            prompt=(
                f"用户问题：{user_query}\n\n"
                '请只输出数组： [{"intent": "comparison"}]'
            ),
            system=INTENT_SYSTEM_PROMPT,
        )
        payload = _normalize_llm_object(result)
        intent = str(payload.get("intent", "") or "").strip()
        if intent in {"comparison", "user_scenario", "single_analysis", "suggestion"}:
            return intent
    except Exception as exc:
        log_validation("agent_intent", "failed", str(exc))

    q = str(user_query or "").lower()
    if any(w in q for w in ["区别", "对比", "相比", "哪个", "竞品", "vs", "更好", "更适合", "优劣", "差异"]):
        return "comparison"
    if any(w in q for w in ["场景", "用户", "体验", "为什么", "喜欢", "适合"]):
        return "user_scenario"
    if any(w in q for w in ["建议", "优化", "改进", "怎么改", "如何提升"]):
        return "suggestion"
    return "single_analysis"


# 先做摘要再交给模型，是为了控制上下文体积和成本，否则整份 pipeline 结果会很快变得又慢又贵。
def _build_data_summary(pipeline_results: dict) -> str:
    try:
        r = normalize_result(pipeline_results)
        reviews = r["reviews"]
        review_count = len(reviews)
        positive_pct = _compute_positive_ratio(reviews) * 100
        avg_score = _compute_avg_vader_score(reviews)

        insights_summary = ""
        for i, ins in enumerate(r["insights"][:3]):
            p = {0: "P0", 1: "P1"}.get(i, "P2")
            insights_summary += (
                f"\n  {p}: {getattr(ins, 'action', '')}"
                f"（影响{getattr(ins, 'evidence_count', 0)}条评论，"
                f"{_safe_float(getattr(ins, 'evidence_ratio', 0.0)) * 100:.0f}%低分评论）"
            )

        topics_summary = ""
        if r["topics"]:
            for topic in r["topics"][:3]:
                topics_summary += (
                    f"\n  - {topic.get('topic_name', '')}"
                    f"（{topic.get('sentiment', '')}）："
                    f"{topic.get('core_demand', '')}"
                )
        else:
            for name, stats in list((r["topic_stats"] or {}).items())[:3]:
                topics_summary += (
                    f"\n  - {name}："
                    f"{stats.get('count', 0)}条评论，"
                    f"均分{_safe_float(stats.get('avg_sentiment', 0.0)):.2f}"
                )

        complaints_summary = ""
        if r["complaints"]:
            for complaint in r["complaints"][:3]:
                complaints_summary += (
                    f"\n  - {complaint.get('complaint_type', '')}："
                    f"{complaint.get('core_demand', '')}"
                    f"（影响约{_safe_float(complaint.get('estimated_ratio', 0.0)) * 100:.0f}%用户）"
                )

        all_cache = pipeline_results.get("all_games_cache", {}) or {}
        compare_summary = ""
        if len(all_cache) > 1:
            for game_name, game_result in all_cache.items():
                gr = normalize_result(game_result)
                pos = _compute_positive_ratio(gr["reviews"]) * 100
                top_pain = getattr(gr["insights"][0], "action", "")[:30] if gr["insights"] else "无数据"
                compare_summary += (
                    f"\n  - {game_name}："
                    f"正面情感{pos:.1f}%，"
                    f"Top问题：{top_pain}"
                )

        summary = f"""
游戏：{pipeline_results.get('game_name', '未知')}
评论总数：{review_count}条（多国抓取）
正面情感：{positive_pct:.1f}%
平均情感分：{avg_score:.3f}

核心问题（按优先级）：{insights_summary if insights_summary else '暂无'}

用户主题：{topics_summary if topics_summary else '暂无'}

用户痛点：{complaints_summary if complaints_summary else '暂无'}

{f'竞品对比数据：{compare_summary}' if compare_summary else ''}
"""
        return summary.strip()
    except Exception as exc:
        log_validation("agent_data_summary", "failed", str(exc))
        return "暂无可用分析数据"


def _build_results_summary(results: dict, pipeline_results: dict) -> str:
    try:
        r = normalize_result(pipeline_results)
        lines = []

        lines.append("【用户体验相关数据】")
        by_rating = r.get("by_rating_stats", {}) or {}
        high = by_rating.get(5, {}) or {}
        low = by_rating.get(1, {}) or {}
        high_avg = _safe_float(high.get("avg_vader", high.get("avg_sentiment", 0.0)))
        low_avg = _safe_float(low.get("avg_vader", low.get("avg_sentiment", 0.0)))
        lines.append(
            f"  5星情感均分{high_avg:.2f} vs 1星情感均分{low_avg:.2f}"
            f"（差异越大说明体验分裂越严重）"
        )
        if r["topics"]:
            for topic in r["topics"][:3]:
                lines.append(
                    f"  · 主题「{topic.get('topic_name', '')}」"
                    f"（{topic.get('sentiment', '')}）："
                    f"{topic.get('core_demand', '')}，"
                    f"用户原话：'{topic.get('representative_review', '')[:60]}'"
                )

        lines.append("【变现相关数据】")
        negative_ratio = _safe_float(
            r["sentiment_stats"].get("negative_ratio", r["sentiment_stats"].get("neg_ratio", 0.0))
        )
        if negative_ratio > 1:
            negative_ratio = negative_ratio / 100.0
        lines.append(f"  负面情感占比{negative_ratio * 100:.1f}%，共{len(r['reviews'])}条评论")
        if r["complaints"]:
            for complaint in r["complaints"][:2]:
                lines.append(
                    f"  · {complaint.get('complaint_type', '')}："
                    f"影响约{_safe_float(complaint.get('estimated_ratio', 0.0)) * 100:.0f}%用户，"
                    f"原话：'{complaint.get('typical_quote', '')[:60]}'"
                )

        lines.append("【用户行为信号】")
        pos_ratio = _compute_positive_ratio(r["reviews"]) * 100
        avg_score = _compute_avg_vader_score(r["reviews"])
        lines.append(
            f"  正面情感{pos_ratio:.1f}%，"
            f"整体情感均分{avg_score:.3f}，"
            f"总评论{len(r['reviews'])}条"
        )
        if r["insights"]:
            top = r["insights"][0]
            lines.append(
                f"  当前最高优先级问题：{getattr(top, 'action', '')}"
                f"（影响{_safe_float(getattr(top, 'evidence_ratio', 0.0)) * 100:.0f}%低分评论）"
            )

        all_cache = pipeline_results.get("all_games_cache", {}) or {}
        if len(all_cache) > 1:
            lines.append("【竞品对比数据】")
            for game_name, game_result in all_cache.items():
                gr = normalize_result(game_result)
                pos = _compute_positive_ratio(gr["reviews"]) * 100
                top_theme = (
                    gr["topics"][0].get("topic_name", "")
                    if gr["topics"]
                    else (list((gr["topic_stats"] or {}).keys())[0] if gr["topic_stats"] else "无")
                )
                top_pain = getattr(gr["insights"][0], "action", "")[:40] if gr["insights"] else "无数据"
                lines.append(
                    f"  · {game_name}："
                    f"正面情感{pos:.1f}%，"
                    f"核心主题「{top_theme}」，"
                    f"P0问题：{top_pain}"
                )

        for tool_name, output in (results or {}).items():
            if output and tool_name not in {"fetch_and_analyze", "discover_topics", "abstract_problems"}:
                lines.append(f"【{tool_name}补充】{str(output)[:150]}")

        return "\n".join(lines)
    except Exception as exc:
        log_validation("agent_results_summary", "failed", str(exc))
        return "【用户行为信号】\n  暂无可用分析数据"


# 工具规划单独成层，可以让 Agent 先决定“看什么证据”，再去组织回答。
def plan_tools(user_query: str, intent: str, available_data: dict, pipeline_results: dict) -> list[dict]:
    try:
        available_data_summary = _build_data_summary(pipeline_results)
        step1_prompt = f"""
用户问题：{user_query}
问题类型：{intent}

当前已有数据：{available_data_summary}
可用工具：{_to_json_text(TOOLS)}

{TOOL_PLANNING_RULES}

按执行顺序输出需要调用的工具。
只输出JSON数组，不输出其他内容：
[{{"tool": "工具名", "params": {{...}}}}]
""".strip()

        tool_plan = call_llm_json(step1_prompt) or []
        if not isinstance(tool_plan, list):
            tool_plan = []

        mentioned_games = _extract_game_names(user_query, available_data, pipeline_results)
        all_cache = pipeline_results.get("all_games_cache", {}) or {}

        # 对比类问题强制补 compare_games，是为了避免模型只做口头比较却没有真正读取多游戏数据。
        if intent == "comparison":
            has_compare_tool = any(str(item.get("tool", "")).strip() == "compare_games" for item in tool_plan if isinstance(item, dict))
            compare_game_names = mentioned_games or list(all_cache.keys())
            if not compare_game_names:
                current_game = str(available_data.get("game_name", "") or pipeline_results.get("game_name", "")).strip()
                if current_game:
                    compare_game_names = [current_game]
            if has_compare_tool:
                normalized_plan = []
                for item in tool_plan:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("tool", "")).strip() == "compare_games":
                        params = dict(item.get("params", {}) or {})
                        params["game_names"] = compare_game_names
                        item["params"] = params
                    normalized_plan.append(item)
                tool_plan = normalized_plan
            else:
                tool_plan.insert(0, {"tool": "compare_games", "params": {"game_names": compare_game_names}})

        if intent == "user_scenario" and not any(
            str(item.get("tool", "")).strip() == "discover_topics"
            for item in tool_plan
            if isinstance(item, dict)
        ):
            tool_plan.insert(0, {"tool": "discover_topics", "params": {"game_name": pipeline_results.get("game_name", "")}})

        if intent == "single_analysis" and not any(
            str(item.get("tool", "")).strip() == "abstract_problems"
            for item in tool_plan
            if isinstance(item, dict)
        ):
            tool_plan.insert(0, {"tool": "abstract_problems", "params": {"game_name": pipeline_results.get("game_name", "")}})

        if intent == "suggestion" and not any(
            str(item.get("tool", "")).strip() == "generate_decisions"
            for item in tool_plan
            if isinstance(item, dict)
        ):
            tool_plan.insert(0, {"tool": "generate_decisions", "params": {"game_name": pipeline_results.get("game_name", "")}})

        log_validation("agent_tool_plan", "success", f"intent={intent}; steps={len(tool_plan)}")
        return [item for item in tool_plan if isinstance(item, dict)]
    except Exception as exc:
        log_validation("agent_tool_plan", "failed", str(exc))
        fallback_map = {
            "comparison": [{"tool": "compare_games", "params": {"game_names": _extract_game_names(user_query, available_data, pipeline_results)}}],
            "user_scenario": [{"tool": "discover_topics", "params": {"game_name": pipeline_results.get("game_name", "")}}],
            "single_analysis": [{"tool": "abstract_problems", "params": {"game_name": pipeline_results.get("game_name", "")}}],
            "suggestion": [{"tool": "generate_decisions", "params": {"game_name": pipeline_results.get("game_name", "")}}],
        }
        return fallback_map.get(intent, [])


def _compare_pipeline_games(game_names: Any, available_data: dict, pipeline_results: dict) -> list[dict]:
    try:
        normalized_names = []
        if isinstance(game_names, list):
            normalized_names = [str(name).strip() for name in game_names if str(name).strip()]
        elif isinstance(game_names, str) and game_names.strip():
            normalized_names = [game_names.strip()]

        all_cache = dict(pipeline_results.get("all_games_cache", {}) or {})
        current_game_name = str(pipeline_results.get("game_name", "") or available_data.get("game_name", "")).strip()
        if current_game_name and current_game_name not in all_cache:
            all_cache[current_game_name] = pipeline_results

        target_names = normalized_names or list(all_cache.keys())
        comparison = []
        for game_name in target_names:
            source_result = all_cache.get(game_name)
            if not source_result:
                continue
            r = normalize_result(source_result)
            comparison.append(
                {
                    "game_name": game_name,
                    "positive_ratio": round(_compute_positive_ratio(r["reviews"]) * 100, 1),
                    "negative_ratio": round(
                        _safe_float(
                            r["sentiment_stats"].get(
                                "negative_ratio",
                                r["sentiment_stats"].get("neg_ratio", 0.0),
                            )
                        ) * (100 if _safe_float(
                            r["sentiment_stats"].get(
                                "negative_ratio",
                                r["sentiment_stats"].get("neg_ratio", 0.0),
                            )
                        ) <= 1 else 1),
                        1,
                    ),
                    "avg_score": round(_compute_avg_vader_score(r["reviews"]), 3),
                    "review_count": len(r["reviews"]),
                    "top_theme": (
                        r["topics"][0].get("topic_name", "")
                        if r["topics"]
                        else (list((r["topic_stats"] or {}).keys())[0] if r["topic_stats"] else "无")
                    ),
                    "top_problem": getattr(r["insights"][0], "action", "") if r["insights"] else "无数据",
                }
            )
        return comparison
    except Exception as exc:
        log_validation("agent_compare_games", "failed", str(exc))
        return []


# 所有工具都走同一执行入口，便于统一处理参数兜底、错误日志和返回结构。
def _run_tool(tool_name: str, params: dict, available_data: dict, pipeline_results: dict) -> Any:
    try:
        normalized = normalize_result(pipeline_results)

        if tool_name == "fetch_and_analyze":
            if normalized["reviews"]:
                return pipeline_results
            app_id = str(params.get("app_id", "") or available_data.get("app_id", "")).strip()
            game_name = str(params.get("game_name", "") or available_data.get("game_name", "")).strip()
            if not app_id or not game_name:
                raise ValueError("fetch_and_analyze 缺少 app_id 或 game_name")
            reviews = get_or_fetch(app_id, game_name)
            result = run_analysis_pipeline(reviews, game_name)
            result["game_name"] = game_name
            return result

        if tool_name == "discover_topics":
            return normalized["topics"]

        if tool_name == "abstract_problems":
            return normalized["complaints"]

        if tool_name == "generate_decisions":
            if normalized["suggestions"]:
                return normalized["suggestions"]
            return [
                {
                    "problem": getattr(item, "rule_trigger", ""),
                    "action": getattr(item, "action", ""),
                    "expected_impact": getattr(item, "evidence", ""),
                    "impact_metric": getattr(item, "impact_metric", ""),
                    "priority": getattr(item, "priority", ""),
                }
                for item in normalized["insights"]
            ]

        if tool_name == "compare_games":
            return _compare_pipeline_games(params.get("game_names", []), available_data, pipeline_results)

        if tool_name == "search_games":
            return search_games(str(params.get("keyword", "") or "").strip())

        raise ValueError(f"未知工具: {tool_name}")
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def execute_tools(tool_plan: list[dict], available_data: dict, pipeline_results: dict) -> tuple[dict, list[dict]]:
    try:
        results: dict[str, Any] = {}
        tool_logs: list[dict] = []
        for item in tool_plan:
            tool_name = ""
            params: dict[str, Any] = {}
            try:
                tool_name = str((item or {}).get("tool", "")).strip()
                params = dict((item or {}).get("params", {}) or {})
                tool_log = {"tool": tool_name, "params": params, "status": "", "output": ""}
                output = _run_tool(tool_name, params, available_data, pipeline_results)
                results[tool_name] = output
                tool_log["status"] = "success"
                tool_log["output"] = _to_json_text(output)[:500]
                tool_logs.append(tool_log)
            except Exception as exc:
                tool_logs.append(
                    {
                        "tool": tool_name,
                        "params": params,
                        "status": "failed",
                        "output": str(exc),
                    }
                )
                log_validation("agent_tool_run", "failed", f"{tool_name}: {exc}")
        return results, tool_logs
    except Exception as exc:
        log_validation("agent_execute_tools", "failed", str(exc))
        return {}, []


# 这一步把技术输出翻译成产品经理语言，目的是让助手回答更像业务分析，而不是日志转储。
def rewrite_for_pm(results: dict) -> str:
    """
    把工具原始输出转译为PM语言
    英文→中文，技术术语→业务语言
    长句→简化
    """
    try:
        if not results:
            return ""

        rewrite_prompt = f"""
以下是游戏分析工具的原始输出：
{str(results)[:800]}

请将其转译为产品经理的语言：
- 全中文输出
- 去掉技术性表达，改为业务语言
- 每条建议不超过30字
- 保留核心数据和结论，去掉冗余描述

只输出转译后的内容，不输出原文。
"""
        rewritten = call_llm(rewrite_prompt)
        if rewritten:
            return rewritten.strip()
    except Exception as exc:
        log_validation("agent_rewrite_pm", "failed", str(exc))

    try:
        text = str(results)[:400]
        replacements = {
            "Implement": "建议",
            "implement": "建议",
            "Enforce": "优化",
            "enforce": "优化",
            "Improve": "优化",
            "improve": "优化",
            "Adjust": "调整",
            "adjust": "调整",
            "ad frequency cap": "广告频次限制",
            "session limits": "单次时长限制",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text
    except Exception:
        return ""


def _fallback_answer(pipeline_results: dict, intent: str) -> str:
    try:
        r = normalize_result(pipeline_results)
        positive_pct = _compute_positive_ratio(r["reviews"]) * 100
        top_theme = (
            r["topics"][0].get("topic_name", "")
            if r["topics"]
            else (list((r["topic_stats"] or {}).keys())[0] if r["topic_stats"] else "核心体验")
        )
        top_action = getattr(r["insights"][0], "action", "") if r["insights"] else "优先处理高频负面反馈"
        if intent == "comparison":
            return f"这款游戏的优势更偏向稳定可持续的体验，而不是强刺激。当前正面情感约{positive_pct:.1f}%，核心主题是「{top_theme}」，更适合做中轻度长线留存。建议继续围绕“{top_action}”优化关键摩擦点。"
        if intent == "suggestion":
            return f"最值得先做的是把资源集中到一条最影响口碑的问题上。当前正面情感约{positive_pct:.1f}%，建议先围绕“{top_action}”落地一条明确动作，再观察留存和差评变化。"
        return f"这款游戏当前最值得关注的是核心体验与用户预期之间的偏差。正面情感约{positive_pct:.1f}%，用户最常提到的主题是「{top_theme}」。建议优先处理“{top_action}”这类高频问题。"
    except Exception:
        return "这款游戏当前最值得讨论的是高频用户反馈背后的业务问题，建议先聚焦核心主题、主要痛点和最高优先级建议。"


# 最终回答统一收口，是为了不管调用了几种工具，输出都保持同一套业务表达风格。
def generate_answer(
    user_query: str,
    intent: str,
    results_summary: str,
    clean_results: str,
    pipeline_results: dict,
    rag_context: str = "",
) -> str:
    try:
        final_prompt = f"""
用户问题：{user_query}
问题类型：{intent}

【行业知识参考】
{rag_context if rag_context else "暂无相关行业知识"}

【当前游戏分析数据】
{results_summary}

【工具分析结果（PM语义化）】
{clean_results}

回答要求：
1. 第一句直接给结论。
2. 引用分析数据时说「用户反馈显示」。
3. 引用知识库时说「行业数据显示」。
4. 最后一行固定格式：
   「→ 建议查看：[Tab名称]」
   Tab仅可选：AI Overview / Data / Explore / Insights / Compare / Feedback / Assistant / Cross Strategy
5. 回答不超过200字。

只输出JSON数组，不输出其他内容：[{{"answer": "你的回答"}}]
""".strip()

        parsed_answer = call_llm_json(final_prompt, system=AGENT_SYSTEM_PROMPT) or []
        payload = _normalize_llm_object(parsed_answer)
        answer = str(payload.get("answer", "") or "").strip()
        if answer:
            log_validation("agent_final_answer", "success", f"intent={intent}")
            return answer
        log_validation("agent_final_answer", "failed", "answer json parse failed")
        return _fallback_answer(pipeline_results, intent)
    except Exception as exc:
        log_validation("agent_final_answer", "failed", str(exc))
        return _fallback_answer(pipeline_results, intent)


# 这是 AI 助手的总入口，负责把理解问题、调工具和生成回答串成一次完整交互。
def run_agent(user_query: str, available_data: dict, pipeline_results: dict) -> dict:
    try:
        user_query = str(user_query or "").strip()
        available_data = dict(available_data or {})
        pipeline_results = dict(pipeline_results or {})

        intent = classify_intent(user_query)
        tool_plan = plan_tools(user_query, intent, available_data, pipeline_results)
        results, tool_logs = execute_tools(tool_plan, available_data, pipeline_results)

        if not results:
            normalized = normalize_result(pipeline_results)
            results = {
                "topics": normalized["topics"],
                "complaints": normalized["complaints"],
                "suggestions": normalized["suggestions"],
            }

        clean_results = rewrite_for_pm(results)
        results_summary = _build_results_summary(results, pipeline_results)

        rag_context = ""
        try:
            from llm.rag import get_relevant_context

            rag_context = get_relevant_context(user_query)
        except Exception as rag_exc:
            log_validation("agent_rag", "failed", str(rag_exc))
            rag_context = ""

        answer = generate_answer(
            user_query,
            intent,
            results_summary,
            clean_results,
            pipeline_results,
            rag_context=rag_context,
        )

        return {
            "answer": answer,
            "intent": intent,
            "tools_called": [log["tool"] for log in tool_logs],
            "tool_logs": tool_logs,
            "results": results,
        }
    except Exception as exc:
        log_validation("run_agent", "failed", str(exc))
        return {
            "answer": "这款游戏当前最值得讨论的是高频用户反馈背后的业务问题，建议先查看核心主题、痛点和最高优先级建议。",
            "intent": "single_analysis",
            "tools_called": [],
            "tool_logs": [],
            "results": {},
        }
