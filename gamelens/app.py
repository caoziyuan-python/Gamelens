# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import time
# GameLens Streamlit entrypoint.
# Orchestrates review fetching, analysis, cache reuse, feedback, and Q&A.

import html
import re

import pandas as pd
import streamlit as st

from config import (
    FAST_MODE_COUNTRIES,
    FAST_MODE_PAGES_PER_COUNTRY,
    GAMES,
    THRESHOLDS,
    TOPIC_KEYWORDS,
)
from data.fetcher import clean_reviews, consume_last_fetch_summary, fetch_reviews
from data.game_finder import search_games
from insights.engine import normalize_result, run_analysis_pipeline
from insights.feedback import get_feedback_stats, record_feedback
from utils.cache import (
    build_game_summary_for_cross,
    get_or_fetch,
    list_saved_results,
    load_cross_result,
    load_game_result,
    load_reviews,
    save_cross_result,
    save_game_result,
    save_reviews,
)
from visualization.charts import (
    plot_keyword_cloud,
    plot_multi_game_comparison,
    plot_rating_distribution,
    plot_sentiment_by_rating,
    plot_sentiment_pie,
    plot_topic_heatmap,
)

st.set_page_config(page_title="GameLens", page_icon="🎮", layout="wide")

st.markdown("""
<style>
#root > div:nth-child(1) > div > div > div > div > section.main { background-color: #0d0f14 !important; }
.stApp > header { background-color: #0d0f14 !important; }
/* 全局背景 */
.stApp { background-color: #0d0f14; }
.main { background-color: #0d0f14 !important; }
.main .block-container { background-color: #0d0f14 !important; padding-top: 1.5rem; }

/* 侧边栏 */
section[data-testid="stSidebar"] { background-color: #111318 !important; border-right: 1px solid #1e2128 !important; }
section[data-testid="stSidebar"] > div:first-child { background-color: #111318 !important; }
[data-testid="stSidebar"] h1 { color: #00C896 !important; font-size: 20px !important; font-weight: 700 !important; letter-spacing: -0.3px !important; }
[data-testid="stSidebar"] .stCaption p { color: #8b8fa8 !important; font-size: 12px !important; }

/* 侧边栏按钮（开始分析/搜索竞品） */
[data-testid="stSidebar"] .stButton > button { background-color: #00C896 !important; color: #0d0f14 !important; border: none !important; width: 100% !important; padding: 10px 16px !important; font-size: 15px !important; font-weight: 600 !important; border-radius: 8px !important; }
[data-testid="stSidebar"] .stButton > button:hover { background-color: #00dba6 !important; transform: translateY(-1px) !important; }

/* 主要内容区域按钮 */
.main .stButton > button { background-color: transparent !important; border: 1px solid #2a2d3a !important; color: #cccccc !important; padding: 6px 14px !important; border-radius: 8px !important; font-size: 13px !important; }
.main .stButton > button:hover { border-color: #00C896 !important; color: #00C896 !important; }

/* 导出PRD按钮特殊样式 */
.main .stButton > button[kind="secondary"] { border-color: #185EA5 !important; color: #4da6ff !important; }

/* Tab导航 */
[data-baseweb="tab-list"] { background-color: transparent !important; border-bottom: 1px solid #1e2128 !important; gap: 4px !important; padding: 0 !important; }
[data-baseweb="tab"] { font-size: 14px !important; font-weight: 500 !important; color: #8b8fa8 !important; padding: 12px 20px !important; border-radius: 0 !important; border: none !important; background: transparent !important; }
[data-baseweb="tab"]:hover { color: #ffffff !important; }
[aria-selected="true"][data-baseweb="tab"] { color: #00C896 !important; border-bottom: 2px solid #00C896 !important; font-weight: 600 !important; }

/* 指标卡 */
[data-testid="metric-container"] { background-color: #161920 !important; border-radius: 12px !important; padding: 20px 24px !important; border: 1px solid #1e2128 !important; border-bottom: 3px solid #00C896 !important; }
[data-testid="stMetricLabel"] p { color: #8b8fa8 !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 32px !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }

/* Expander */
[data-testid="stExpander"] { background-color: #161920 !important; border: 1px solid #1e2128 !important; border-radius: 12px !important; margin-bottom: 8px !important; }
[data-testid="stExpander"] summary { color: #cccccc !important; font-weight: 500 !important; font-size: 14px !important; }
[data-testid="stExpander"] summary:hover { color: #00C896 !important; }

/* 下拉框 */
[data-baseweb="select"] > div { background-color: #1e2128 !important; border: 1px solid #2a2d3a !important; border-radius: 8px !important; color: #ffffff !important; }
[data-baseweb="select"] > div:hover { border-color: #00C896 !important; }
[data-baseweb="popover"] { background-color: #1e2128 !important; border: 1px solid #2a2d3a !important; border-radius: 8px !important; }
[role="option"] { color: #cccccc !important; }
[role="option"]:hover { background-color: #2a2d3a !important; color: #00C896 !important; }

/* 输入框 */
[data-baseweb="input"] > div { background-color: #1e2128 !important; border: 1px solid #2a2d3a !important; border-radius: 8px !important; }
[data-baseweb="input"] > div:focus-within { border-color: #00C896 !important; }
input { color: #ffffff !important; font-size: 14px !important; }
::placeholder { color: #8b8fa8 !important; }

/* 数据表格 */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #1e2128 !important; }
[data-testid="stDataFrame"] thead th { background-color: #1e2128 !important; color: #8b8fa8 !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; padding: 12px 16px !important; border-bottom: 1px solid #2a2d3a !important; }
[data-testid="stDataFrame"] tbody td { background-color: #161920 !important; color: #cccccc !important; padding: 12px 16px !important; border-bottom: 1px solid #1e2128 !important; font-size: 13px !important; }
[data-testid="stDataFrame"] tbody tr:hover td { background-color: #1e2128 !important; color: #ffffff !important; }

/* 话题热度进度条 */
[data-testid="stProgress"] > div { background-color: #1e2128 !important; border-radius: 4px !important; height: 6px !important; }
[data-testid="stProgress"] > div > div { background-color: #00C896 !important; border-radius: 4px !important; }

/* Status面板（Deep Dive Agent） */
[data-testid="stStatus"] { background-color: #161920 !important; border: 1px solid #1e2128 !important; border-left: 3px solid #00C896 !important; border-radius: 8px !important; }

/* Radio按钮 */
[data-testid="stRadio"] label { color: #cccccc !important; font-size: 14px !important; }
[data-testid="stRadio"] [aria-checked="true"] ~ div p { color: #00C896 !important; }

/* Checkbox */
[data-testid="stCheckbox"] label { color: #cccccc !important; font-size: 13px !important; }

/* 提示条 */
[data-testid="stAlert"] { border-radius: 8px !important; border: none !important; }
[data-testid="stAlert"][data-baseweb="notification"] { background-color: rgba(0, 200, 150, 0.1) !important; }

/* 分隔线 */
hr { border-color: #1e2128 !important; margin: 16px 0 !important; }

/* 滚动条 */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00C896; }

/* 图表容器背景 */
[data-testid="stPlotlyChart"] { background-color: #161920 !important; border-radius: 12px !important; padding: 8px !important; border: 1px solid #1e2128 !important; }

/* 成功/信息提示条颜色 */
.stSuccess { background-color: rgba(0,200,150,0.1) !important; border-left: 3px solid #00C896 !important; }
.stInfo { background-color: rgba(0,100,255,0.1) !important; }
.stWarning { background-color: rgba(255,140,0,0.1) !important; }
</style>
""", unsafe_allow_html=True)

for key, default in [
    ("result", None),
    ("current_game", None),
    ("current_reviews", []),
    ("all_games_cache", {}),
    ("cross_game_result", None),
    ("cross_game_keys", tuple()),
    ("cross_game_strategies", []),
    ("focus_cross_game_tab", False),
    ("chat_history", []),
    ("search_results", []),
    ("custom_games", {}),
    ("analyzed_games", []),
]:
    st.session_state.setdefault(key, default)


def _cache_display_name(cache_key: str, cache_result: dict) -> str:
    """Return a user-facing game name from cache key or cached payload."""
    try:
        return str((cache_result or {}).get("game_name") or cache_key.split("__l2_")[0] or cache_key)
    except Exception:
        return str(cache_key)


def _analysis_mode_priority(
    cache_result: dict,
    preferred_mode: str | None = None,
    preferred_run_level2: bool | None = None,
) -> int:
    """Rank cache candidates by preferred mode/L2, then by full-mode fallback."""
    analysis_mode = str((cache_result or {}).get("analysis_mode", "full")).lower()
    run_level2 = bool((cache_result or {}).get("run_level2", False))
    score = 1 if analysis_mode == "full" else 0
    if preferred_mode is not None and analysis_mode == str(preferred_mode).lower():
        score += 10
    if preferred_run_level2 is not None and run_level2 == bool(preferred_run_level2):
        score += 5
    return score


def _visible_analysis_cache(
    all_games_cache: dict,
    preferred_mode: str | None = None,
    preferred_run_level2: bool | None = None,
) -> dict:
    """Build a game->result view by choosing the best variant per display name."""
    visible: dict = {}
    for cache_key, cache_result in (all_games_cache or {}).items():
        display_name = _cache_display_name(cache_key, cache_result)
        chosen = visible.get(display_name)
        if chosen is None or _analysis_mode_priority(
            cache_result,
            preferred_mode=preferred_mode,
            preferred_run_level2=preferred_run_level2,
        ) >= _analysis_mode_priority(
            chosen,
            preferred_mode=preferred_mode,
            preferred_run_level2=preferred_run_level2,
        ):
            visible[display_name] = cache_result
    return visible


def _analysis_cache_key(
    game_name: str,
    run_level2: bool = False,
    analysis_mode: str = "full"
) -> str:
    """Stable cache key with game + L2 + analysis mode dimensions."""
    return f"{game_name}__l2_{int(bool(run_level2))}__mode_{analysis_mode}"


def _load_latest_review_cache(
    game_name: str,
    analysis_mode: str = "full"
) -> list:
    """Load latest local review cache and fallback to legacy default variant."""
    if analysis_mode == "fast":
        return load_reviews(game_name, cache_variant="fast") or load_reviews(game_name) or []
    return load_reviews(game_name, cache_variant="full") or load_reviews(game_name) or []


def _cached_analysis_is_stale(
    cached_result: dict | None,
    latest_reviews: list
) -> bool:
    """Check if cached analysis sample size mismatches current review sample."""
    if cached_result is None:
        return False
    cached_reviews = normalize_result(cached_result).get("reviews", []) or []
    return len(cached_reviews) != len(latest_reviews or [])


def _get_cached_analysis(
    game_name: str,
    run_level2: bool = False,
    analysis_mode: str = "full"
) -> dict | None:
    """Read session analysis cache with compatibility fallback for legacy keys."""
    cache_store = st.session_state.get("all_games_cache", {}) or {}
    primary_key = _analysis_cache_key(
        game_name,
        run_level2=run_level2,
        analysis_mode=analysis_mode,
    )
    if primary_key in cache_store:
        return cache_store.get(primary_key)

    if analysis_mode == "fast":
        full_key = _analysis_cache_key(
            game_name,
            run_level2=run_level2,
            analysis_mode="full",
        )
        if full_key in cache_store:
            return cache_store.get(full_key)

    legacy_result = cache_store.get(game_name)
    if legacy_result is None:
        return None
    if analysis_mode == "full" and str(legacy_result.get("analysis_mode", "full")).lower() == "fast":
        return None
    return legacy_result


def _store_analysis(
    game_name: str,
    result: dict,
    run_level2: bool = False,
    analysis_mode: str = "full",
    fetch_profile: str = "default"
) -> dict:
    """Persist analyzed result into session cache and invalidate cross-game cache."""
    analyzed = _with_enriched_stats(result or {})
    analyzed["game_name"] = game_name
    analyzed["analysis_mode"] = analysis_mode
    analyzed["fetch_profile"] = fetch_profile
    analyzed["run_level2"] = bool(run_level2)
    st.session_state["all_games_cache"][
        _analysis_cache_key(
            game_name,
            run_level2=run_level2,
            analysis_mode=analysis_mode,
        )
    ] = analyzed
    st.session_state["cross_game_keys"] = tuple()
    st.session_state["cross_game_strategies"] = []
    return analyzed


def _get_dominant_mode(all_games_cache: dict) -> dict:
    """Infer the dominant (analysis_mode, run_level2) pair from cached games."""
    from collections import Counter

    modes: list[tuple[str, bool]] = []
    for result in (all_games_cache or {}).values():
        meta = (result or {}).get("_cache_meta", {}) or {}
        mode = str(meta.get("analysis_mode", (result or {}).get("analysis_mode", "full"))).lower()
        run_level2 = bool(meta.get("run_level2", (result or {}).get("run_level2", False)))
        modes.append((mode, run_level2))

    if not modes:
        return {"mode": "full", "run_level2": False}
    mode, run_level2 = Counter(modes).most_common(1)[0][0]
    return {"mode": mode, "run_level2": run_level2}


def _fetch_reviews_force(
    app_id: str,
    game_name: str,
    countries: list | None = None,
    pages: int | None = None,
    cache_variant: str = "",
) -> list:
    """Force refetch reviews from source and overwrite local review cache."""
    raw_reviews = fetch_reviews(
        app_id=app_id,
        game_name=game_name,
        countries=countries,
        pages=pages,
    )
    cleaned_reviews = clean_reviews(raw_reviews)
    save_reviews(cleaned_reviews, game_name, cache_variant=cache_variant)
    return cleaned_reviews


def _get_reviews_with_refresh(
    app_id: str,
    game_name: str,
    force_refresh: bool = False,
    countries: list | None = None,
    pages: int | None = None,
    cache_variant: str = "",
    allow_default_cache_fallback: bool = False,
) -> list:
    """Fetch reviews with optional force-refresh while preserving cache behavior."""
    if force_refresh:
        return _fetch_reviews_force(
            app_id=app_id,
            game_name=game_name,
            countries=countries,
            pages=pages,
            cache_variant=cache_variant,
        )
    return get_or_fetch(
        app_id=app_id,
        game_name=game_name,
        countries=countries,
        pages=pages,
        cache_variant=cache_variant,
        allow_default_cache_fallback=allow_default_cache_fallback,
    )


def get_or_compute_cross_analysis(force_reload: bool = False) -> tuple[dict, list, dict]:
    """
    Unified cross-game entry with 3-layer cache:
    1) session_state
    2) disk cache via load_cross_result
    3) recompute and save
    """
    pref_mode = str(st.session_state.get("analysis_mode_radio", "full")).lower()
    pref_l2 = bool(st.session_state.get("run_level2_checkbox", False))
    all_cache = _visible_analysis_cache(
        st.session_state.get("all_games_cache", {}) or {},
        preferred_mode=pref_mode,
        preferred_run_level2=pref_l2,
    )
    cache_keys = tuple(sorted(all_cache.keys()))

    session_cross = st.session_state.get("cross_game_result")
    session_keys = st.session_state.get("cross_game_keys")
    if not force_reload and session_cross is not None and session_keys == cache_keys:
        return (
            session_cross,
            st.session_state.get("cross_game_strategies", []),
            {"from_cache": True, "layer": "session"},
        )

    game_summaries: dict = {}
    for cache_key, game_result in all_cache.items():
        gname = _cache_display_name(cache_key, game_result)
        reviews = normalize_result(game_result).get("reviews", []) or []
        mode = str((game_result or {}).get("analysis_mode", "full")).lower()
        run_level2 = bool((game_result or {}).get("run_level2", False))
        game_summaries[gname] = build_game_summary_for_cross(
            game_name=gname,
            result=game_result,
            reviews=reviews,
            analysis_mode=mode,
            run_level2=run_level2,
        )

    if not force_reload:
        cached_cross = load_cross_result(game_summaries)
        if cached_cross:
            cross = cached_cross.get("cross_result", {}) or {}
            strategies = cached_cross.get("strategies", []) or []
            st.session_state["cross_game_result"] = cross
            st.session_state["cross_game_strategies"] = strategies
            st.session_state["cross_game_keys"] = cache_keys
            return (
                cross,
                strategies,
                {
                    "from_cache": True,
                    "layer": "disk",
                    **(cached_cross.get("_cache_meta", {}) or {}),
                },
            )

    from insights.cross_game import cross_game_analysis

    cross = cross_game_analysis(all_cache)
    strategies = []
    try:
        from insights.strategy_ranker import generate_strategies  # type: ignore

        strategies = generate_strategies(cross)
    except ImportError:
        strategies = []
    except Exception as exc:
        print(f"[Warning] cross strategies generation failed: {exc}")
        strategies = []

    dominant_mode = _get_dominant_mode(st.session_state.get("all_games_cache", {}) or {})
    save_cross_result(
        game_summaries=game_summaries,
        cross_result=cross,
        strategies=strategies,
        analysis_mode=dominant_mode["mode"],
        run_level2=dominant_mode["run_level2"],
    )
    st.session_state["cross_game_result"] = cross
    st.session_state["cross_game_strategies"] = strategies
    st.session_state["cross_game_keys"] = cache_keys
    return cross, strategies, {"from_cache": False}


def _with_enriched_stats(result: dict) -> dict:
    try:
        normalized = normalize_result(result)
        stats = dict(normalized.get("sentiment_stats", {}) or {})
        reviews = normalized.get("reviews", []) or []
        total = len(reviews)

        if total > 0:
            # Recompute ratios defensively for UI consistency.
            pos_cnt = sum(1 for r in reviews if str(getattr(r, "vader_label", "")).lower() == "positive")
            neu_cnt = sum(1 for r in reviews if str(getattr(r, "vader_label", "")).lower() == "neutral")
            neg_cnt = sum(1 for r in reviews if str(getattr(r, "vader_label", "")).lower() == "negative")
            positive_ratio = pos_cnt / total
            neutral_ratio = neu_cnt / total
            negative_ratio = neg_cnt / total
            agree_cnt = sum(
                1
                for r in reviews
                if str(getattr(r, "vader_label", "")).lower() == str(getattr(r, "textblob_label", "")).lower()
            )
            agreement_rate = agree_cnt / total
        else:
            positive_ratio = 0.0
            neutral_ratio = 0.0
            negative_ratio = 0.0
            agreement_rate = 0.0

        stats["positive_ratio"] = float(positive_ratio)
        stats["neutral_ratio"] = float(neutral_ratio)
        stats["neg_ratio"] = float(stats.get("neg_ratio", negative_ratio) or negative_ratio)
        stats["negative_ratio"] = float(negative_ratio)
        stats["agreement_rate"] = float(agreement_rate)
        stats["low_agreement_warning"] = (
            float(stats.get("agreement_rate", 0.0)) < float(THRESHOLDS.get("low_agreement_warning", 0.70))
        )

        merged = dict(result or {})
        merged["sentiment_stats"] = stats
        merged.setdefault("error_details", normalized.get("error_details", {}))
        merged.setdefault("topic_cards", normalized.get("topic_cards", []))
        merged.setdefault("topic_source", normalized.get("topic_source", "rule_fallback"))
        return merged
    except Exception:
        return result or {}


def render_skeleton(lines: int = 3):
    bars = ""
    widths = ["100%", "85%", "70%", "90%", "60%"]
    for i in range(lines):
        w = widths[i % len(widths)]
        bars += f"""
        <div style="
            height: 14px;
            width: {w};
            background: linear-gradient(
                90deg,
                rgba(255,255,255,0.06) 25%,
                rgba(255,255,255,0.12) 50%,
                rgba(255,255,255,0.06) 75%
            );
            background-size: 200% 100%;
            border-radius: 4px;
            margin-bottom: 10px;
            animation: skeleton-wave 1.5s infinite;
        "></div>
        """

    st.markdown(
        f"""
        <style>
        @keyframes skeleton-wave {{
            0%   {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        </style>
        <div style="padding: 16px; border-radius: 8px; background: rgba(255,255,255,0.03);">
            {bars}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence_reviews(
    reviews: list,
    insight,
    max_count: int = 3,
):
    """Render representative evidence reviews for one insight."""
    reviews = reviews or []
    if not reviews:
        st.caption("暂无匹配的证据评论")
        return

    ranked = sorted(
        reviews,
        key=lambda r: (
            int(getattr(r, "rating", 0) or 0),
            float(getattr(r, "vader_score", 0.0) or 0.0),
        ),
    )
    picked = ranked[:max_count]
    if not picked:
        st.caption("暂无匹配的证据评论")
        return

    for review in picked:
        text = str(getattr(review, "text", "") or "").strip()
        rating = int(getattr(review, "rating", 0) or 0)
        country = str(getattr(review, "country", "") or "").upper()
        vader = float(getattr(review, "vader_score", 0.0) or 0.0)
        if len(text) > 140:
            text = text[:140] + "..."
        st.markdown(
            f'- "{text}"  \n'
            f"  *{rating} | {country} | VADER {vader:.2f}"
        )


def render_conclusion_banner(
    top_action: str,
    llm_available: bool,
    confidence: str,
    review_count: int,
    game_name: str,
):
    """Render a compact top conclusion banner."""
    source_label = "AI + 规则" if llm_available else "规则引擎"
    st.markdown(
        f"### {game_name} 核心结论\n"
        f"- 优先动作：**{top_action}**\n"
        f"- 评论样本：{review_count} 条\n"
        f"- 生成来源：{source_label}\n"
        f"- 置信度：{confidence}"
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
<div style="
    background:#161920;
    border-radius:12px;
    padding:20px 24px;
    border:1px solid #1e2128;
    border-bottom:3px solid #00C896;
    min-height:110px;
    box-sizing:border-box;
">
    <div style="
        color:#8b8fa8;
        font-size:11px;
        text-transform:uppercase;
        letter-spacing:0.8px;
        font-weight:500;
        margin-bottom:10px;
    ">{label}</div>
    <div style="
        color:#ffffff;
        font-size:32px;
        font-weight:700;
        letter-spacing:-0.5px;
        line-height:1.1;
    ">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_insight_card(ins, rank: int, game_name: str, prefix: str):
    """Render one insight card with feedback controls and PRD export."""
    p_label = {0: "P0", 1: "P1"}.get(rank, "P2")
    priority = p_label
    title = str(getattr(ins, "action", "") or "").strip()
    description = str(getattr(ins, "evidence", "") or "").strip()
    priority_colors = {"P0": "#FF4D4D", "P1": "#FF8C00", "P2": "#F5C518"}
    priority_backgrounds = {
        "P0": "rgba(255,77,77,0.15)",
        "P1": "rgba(255,140,0,0.15)",
        "P2": "rgba(245,197,24,0.15)",
    }
    border_color = priority_colors.get(priority, "#444")
    tag_background = priority_backgrounds.get(priority, "rgba(255,255,255,0.08)")
    st.markdown(
        f"""
<div style="
    background:#161920;
    border-radius:12px;
    border-left:4px solid {border_color};
    padding:20px 24px;
    margin-bottom:12px;
    border:1px solid #1e2128;
    border-left:4px solid {border_color};
    display:block;
    width:100%;
    box-sizing:border-box;
    overflow:hidden;
">
    <span style="background:{tag_background};color:{border_color};
        padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">
        {priority}
    </span>
    <h4 style="color:#ffffff;margin:12px 0 6px;font-size:16px;font-weight:600;">{title}</h4>
    <p style="color:#8b8fa8;font-size:14px;line-height:1.6;margin:0;">{description}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    evidence_count = int(getattr(ins, "evidence_count", 0) or 0)
    impact_metric = str(getattr(ins, "impact_metric", "") or "").strip()
    confidence = str(getattr(ins, "confidence", "") or "").strip()
    summary_bits = [f"影响评论 {evidence_count} 条"] if evidence_count > 0 else []
    if impact_metric:
        summary_bits.append(f"关注指标：{impact_metric}")
    if confidence:
        summary_bits.append(f"置信度：{confidence}")
    if summary_bits:
        st.caption(" | ".join(summary_bits))

    evidence_text = str(getattr(ins, "evidence", "") or "").strip()
    if evidence_text:
        st.caption(evidence_text)

    b1, b2, b3, b4 = st.columns([1, 1, 2, 2])
    with b1:
        if st.button("有帮助", key=f"{prefix}_h_{ins.insight_id}"):
            record_feedback(ins.insight_id, game_name, ins.source, ins.priority, "useful")
            st.toast("反馈已保存", icon="OK")
    with b2:
        if st.button("没帮助", key=f"{prefix}_nh_{ins.insight_id}"):
            record_feedback(ins.insight_id, game_name, ins.source, ins.priority, "not_useful")
            st.toast("反馈已保存", icon="INFO")

    prd_md = (
        f"## 产品需求 [{p_label}]\n\n"
        f"**问题**: {getattr(ins, 'evidence', '')}\n\n"
        f"**动作**: {getattr(ins, 'action', '')}\n\n"
        f"**指标**: {getattr(ins, 'impact_metric', '')}\n\n"
        f"**优先级**: {p_label}\n"
    )
    with b3:
        st.download_button(
            "导出 PRD",
            data=prd_md,
            file_name=f"prd_{p_label}_{ins.insight_id}.md",
            mime="text/markdown",
            key=f"{prefix}_dl_{ins.insight_id}",
        )
    with b4:
        if st.button("复制到飞书", key=f"{prefix}_lark_{ins.insight_id}"):
            st.code(prd_md, language="markdown")
            st.caption("可直接复制到你的文档工具中")

    with st.expander("查看证据", expanded=False):
        render_evidence_reviews(
            reviews=st.session_state.get("current_reviews", []),
            insight=ins,
            max_count=3,
        )


def render_ai_suggestion(answer: str, llm_available: bool):
    """Render compact AI suggestion box."""
    badge = "AI生成" if llm_available else "规则生成"
    st.info(f"[{badge}] {answer}")


def _top_pain_label(gr: dict) -> str:
    """Best-effort top pain label for compare table."""
    reviews = gr.get("reviews", []) or []
    insights = gr.get("insights", []) or []
    if insights:
        return str(getattr(insights[0], "action", "") or "暂无明确核心痛点")
    if len(reviews) < 50:
        return "样本不足"

    topic_stats = gr.get("topic_stats", {}) or {}
    topic_action_map = {
        "Ads": "规则判断：优先优化广告触发逻辑",
        "Gameplay": "规则判断：优先优化玩法公平性",
        "Monetization": "规则判断：优先优化价格与价值感知",
        "UX_Issues": "规则判断：优先修复稳定性与体验问题",
        "Positive": "规则判断：优先强化现有优势",
    }
    for topic_name, stats in topic_stats.items():
        if bool((stats or {}).get("is_high_priority", False)):
            return topic_action_map.get(str(topic_name), f"规则判断：优先处理 {topic_name}")

    llm_available = bool(gr.get("llm_available", False))
    if not llm_available:
        return "当前为规则兜底模式"
    return "暂无明确核心痛点"


# Merge built-in and user-added games into one selector source.
all_games = {**GAMES, **st.session_state["custom_games"]}

with st.sidebar:
    st.title("GameLens 🎮")
    st.caption("竞品情报分析平台")
    st.divider()

    selected_game = st.selectbox("选择游戏", options=list(all_games.keys()))
    analysis_mode = st.radio(
        "分析模式",
        options=["full", "fast"],
        format_func=lambda x: "完整模式（含AI）" if x == "full" else "快速模式（规则引擎）",
        index=0,
        key="analysis_mode_radio",
    )
    run_level2 = st.session_state.get("run_level2_checkbox", False)
    analyze_btn = st.button("开始分析", type="primary", use_container_width=True)

    st.divider()
    st.subheader("竞品搜索")
    search_keyword = st.text_input("关键词", placeholder="例如：mahjong puzzle casual")
    search_btn = st.button("搜索竞品", use_container_width=True)

    if search_btn and search_keyword:
        with st.spinner("正在搜索竞品..."):
            st.session_state["search_results"] = search_games(search_keyword)
        if not st.session_state["search_results"]:
            st.warning("没有找到相关竞品，请尝试更换关键词。")

    for game in st.session_state["search_results"]:
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.caption(f"* {game['rating']} | {game['category']}")
            st.write(game["name"])
        with col_btn:
            if st.button("添加", key=f"add_{game['app_id']}"):
                st.session_state["custom_games"][game["name"]] = {"app_id": game["app_id"]}
                st.success(f"已添加竞品：{game['name']}")
                st.rerun()

    selected_cache_key = _analysis_cache_key(
        selected_game,
        run_level2=bool(run_level2),
        analysis_mode=analysis_mode,
    )

    with st.expander("⚙️ 高级设置", expanded=False):
        run_level2 = st.checkbox("启用 Level 2 验证", value=bool(run_level2), key="run_level2_checkbox")

        col_force1, col_force2 = st.columns(2)
        with col_force1:
            if st.button("重新分析", help="保留评论缓存，仅重新执行分析", use_container_width=True):
                st.session_state[f"force_reanalyze_{selected_cache_key}"] = True
                st.rerun()
        with col_force2:
            if st.button("刷新评论", help="重新抓取评论并重新分析", use_container_width=True):
                st.session_state[f"force_reviews_{selected_cache_key}"] = True
                st.session_state[f"force_reanalyze_{selected_cache_key}"] = True
                st.rerun()

        batch_btn = st.button(
            "批量分析全部游戏",
            use_container_width=True,
            help="分析全部游戏，并自动准备竞品对比结果",
        )
        batch_enable_llm = st.checkbox(
            "批量启用AI（较慢）",
            value=False,
            help="开启后批量按完整模式运行并写入完整缓存",
        )
        st.caption("批量模式开启 AI 时速度会明显变慢。" if batch_enable_llm else "批量模式默认使用规则引擎，速度更快。")

        st.markdown("---")
        saved = list_saved_results()
        st.caption(f"缓存结果：{len(saved)}")
        if st.button("刷新缓存列表", key="refresh_saved"):
            st.rerun()

        for idx, s in enumerate(saved[:6]):
            mode_saved = str(s.get("analysis_mode", "full")).lower()
            run_level2_saved = bool(s.get("run_level2", False))
            mode_label = "完整模式" if mode_saved == "full" else "快速模式"
            l2_label = "L2 开启" if run_level2_saved else "L2 关闭"
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.caption(
                    f"**{s.get('game_name', '')}**  \n"
                    f"{int(s.get('review_count', 0))} 条评论 | {mode_label} | {l2_label}  \n"
                    f"{s.get('created_at', '')}"
                )
            with col_btn:
                load_key = f"load_{s.get('game_name', '')}_{mode_saved}_{int(run_level2_saved)}_{idx}"
                if st.button("加载", key=load_key):
                    all_map = {**GAMES, **st.session_state.get("custom_games", {})}
                    game_name_saved = str(s.get("game_name", ""))
                    game_info = all_map.get(game_name_saved) or {}
                    app_id_saved = str(game_info.get("app_id", "") or "")
                    if app_id_saved:
                        cache_variant = "fast" if mode_saved == "fast" else "full"
                        reviews = _get_reviews_with_refresh(
                            app_id=app_id_saved,
                            game_name=game_name_saved,
                            force_refresh=False,
                            cache_variant=cache_variant,
                            allow_default_cache_fallback=(mode_saved == "fast"),
                        )
                        loaded = load_game_result(
                            game_name=game_name_saved,
                            reviews=reviews,
                            analysis_mode=mode_saved,
                            run_level2=run_level2_saved,
                        )
                    else:
                        loaded = load_game_result(
                            game_name=game_name_saved,
                            reviews=[],
                            analysis_mode=mode_saved,
                            run_level2=run_level2_saved,
                            policy="relaxed",
                        )
                        reviews = normalize_result(loaded or {}).get("reviews", []) if loaded else []
                        if loaded:
                            st.caption("⚠️ 未找到该游戏 App ID，已直接从缓存加载，评论无法刷新。")
                    if loaded:
                        loaded_key = _analysis_cache_key(
                            game_name_saved,
                            run_level2=run_level2_saved,
                            analysis_mode=mode_saved,
                        )
                        st.session_state.setdefault("all_games_cache", {})[loaded_key] = loaded
                        st.session_state["result"] = loaded
                        st.session_state["current_game"] = game_name_saved
                        st.session_state["current_reviews"] = reviews or []
                        st.success(f"已加载 {game_name_saved}（{mode_label}，{l2_label}）")
                        st.rerun()
                    else:
                        st.warning(f"{game_name_saved} 的缓存已失效，请重新分析。")

        st.markdown("---")
        st.caption("学习反馈")
        feedback_stats = get_feedback_stats(st.session_state.get("current_game", "") or selected_game)
        if feedback_stats.get("total", 0) == 0:
            st.info("暂无学习反馈数据。")
        else:
            col1, col2 = st.columns(2)
            col1.metric("反馈总数", int(feedback_stats.get("total", 0)))
            col2.metric("有帮助占比", f"{float(feedback_stats.get('useful_rate', 0.0)) * 100:.1f}%")

        st.markdown("---")
        st.caption("知识库管理")
        if st.button("重建知识库索引", help="重新读取 knowledge/ 目录并生成向量索引", use_container_width=True):
            from llm.rag import build_index

            with st.spinner("正在构建知识库索引..."):
                build_res = build_index()
            if int(build_res.get("count", 0)) > 0:
                st.success(f"索引构建完成：{build_res['count']} 个知识块")
            else:
                st.error("索引构建失败，请检查 knowledge/ 目录与 Embedding 配置。")

        if st.button("生成评论知识库", help="把当前已分析游戏的评论摘要加入知识库", use_container_width=True):
            all_cache = st.session_state.get("all_games_cache", {}) or {}
            if not all_cache:
                st.warning("请先分析至少一款游戏。")
            else:
                from llm.rag import build_index, build_review_knowledge

                with st.spinner("正在生成评论知识库并重建索引..."):
                    build_review_knowledge(all_cache)
                    build_res = build_index()
                st.success(
                    f"已将 {len(all_cache)} 款游戏评论加入知识库，当前索引块数：{int(build_res.get('count', 0))}"
                )

if "batch_btn" not in locals():
    batch_btn = False
if "batch_enable_llm" not in locals():
    batch_enable_llm = False

# Single-game analysis with 3-layer cache.
if analyze_btn:
    app_id = all_games[selected_game]["app_id"]
    cache_key = _analysis_cache_key(
        selected_game,
        run_level2=run_level2,
        analysis_mode=analysis_mode,
    )
    force_reanalyze_flag = bool(st.session_state.get(f"force_reanalyze_{cache_key}", False))
    force_reviews_flag = bool(st.session_state.get(f"force_reviews_{cache_key}", False))
    session_cache = st.session_state.get("all_games_cache", {}) or {}

    if cache_key in session_cache and not force_reanalyze_flag and not force_reviews_flag:
        result = dict(session_cache[cache_key] or {})
        reviews = normalize_result(result).get("reviews", []) or _load_latest_review_cache(
            selected_game,
            analysis_mode=analysis_mode,
        )
        st.caption("已命中会话缓存。")
    else:
        with st.spinner(f"正在抓取 {selected_game} 的评论..."):
            reviews = _get_reviews_with_refresh(
                app_id=app_id,
                game_name=selected_game,
                force_refresh=force_reviews_flag,
                cache_variant=analysis_mode,
            )
            fetch_summary = consume_last_fetch_summary()
            if fetch_summary.get("message"):
                st.caption(str(fetch_summary.get("message", "")))

        cached = load_game_result(
            game_name=selected_game,
            reviews=reviews,
            analysis_mode=analysis_mode,
            run_level2=run_level2,
            policy="force" if force_reanalyze_flag else "default",
        )

        if cached:
            result = dict(cached)
            cache_meta = cached.get("_cache_meta", {}) or {}
            st.success(
                f"已加载磁盘缓存：{selected_game}（{int(cache_meta.get('review_count', len(reviews)))} 条评论，"
                f"{'完整模式' if analysis_mode == 'full' else '快速模式'}，{'L2 开启' if run_level2 else 'L2 关闭'}）"
            )
        else:
            step_label = "AI分析中" if analysis_mode != "fast" else "规则分析中"
            with st.spinner(f"正在分析 {selected_game}（{step_label}）..."):
                result = run_analysis_pipeline(
                    reviews,
                    selected_game,
                    run_level2=run_level2,
                    enable_llm=(analysis_mode != "fast"),
                )
            result["game_name"] = selected_game
            result["analysis_mode"] = analysis_mode
            result["run_level2"] = bool(run_level2)
            result["fetch_profile"] = "default"
            save_game_result(
                game_name=selected_game,
                result=result,
                reviews=reviews,
                analysis_mode=analysis_mode,
                run_level2=run_level2,
            )
            st.success(f"{selected_game} 分析完成并已保存。")

    result = _with_enriched_stats(result or {})
    result["game_name"] = selected_game
    result["analysis_mode"] = analysis_mode
    result["run_level2"] = bool(run_level2)
    result["fetch_profile"] = str(result.get("fetch_profile", "default"))

    st.session_state.setdefault("all_games_cache", {})[cache_key] = result
    st.session_state["current_reviews"] = reviews
    st.session_state["result"] = result
    st.session_state["current_game"] = selected_game
    st.session_state["cross_game_keys"] = tuple()
    st.session_state["cross_game_strategies"] = []

    for flag in (f"force_reanalyze_{cache_key}", f"force_reviews_{cache_key}"):
        if flag in st.session_state:
            del st.session_state[flag]

if batch_btn:
    batch_games = {**GAMES}
    if "custom_games" in st.session_state:
        batch_games.update(st.session_state["custom_games"])

    total = len(batch_games)
    progress = st.progress(0, text="正在准备批量分析...")
    status = st.empty()

    st.session_state.setdefault("all_games_cache", {})

    failed = []
    skipped = 0
    last_ready_result = None
    last_ready_game = None
    last_ready_reviews = []

    current_mode = "full" if batch_enable_llm else "fast"
    batch_run_level2 = False

    for i, (game_name, game_info) in enumerate(batch_games.items()):
        cache_key = _analysis_cache_key(
            game_name,
            run_level2=batch_run_level2,
            analysis_mode=current_mode,
        )

        if cache_key in st.session_state.get("all_games_cache", {}):
            skipped += 1
            progress.progress((i + 1) / total, text=f"已命中会话缓存：{game_name}")
            cached_session = st.session_state["all_games_cache"].get(cache_key)
            last_ready_result = _with_enriched_stats(cached_session or {})
            last_ready_game = game_name
            last_ready_reviews = normalize_result(last_ready_result).get("reviews", []) or []
            continue

        try:
            fetch_countries = None if batch_enable_llm else FAST_MODE_COUNTRIES
            fetch_pages = None if batch_enable_llm else FAST_MODE_PAGES_PER_COUNTRY
            cache_variant = current_mode

            status.info(f"正在抓取 {game_name} 的评论...")
            reviews = _get_reviews_with_refresh(
                app_id=game_info["app_id"],
                game_name=game_name,
                force_refresh=False,
                countries=fetch_countries,
                pages=fetch_pages,
                cache_variant=cache_variant,
                allow_default_cache_fallback=(current_mode == "fast"),
            )
            fetch_summary = consume_last_fetch_summary()
            if fetch_summary.get("message"):
                status.info(f"抓取提示：{fetch_summary['message']}")

            cached_disk = load_game_result(
                game_name=game_name,
                reviews=reviews,
                analysis_mode=current_mode,
                run_level2=batch_run_level2,
            )
            if cached_disk:
                st.session_state["all_games_cache"][cache_key] = cached_disk
                skipped += 1
                review_count = int((cached_disk.get("_cache_meta", {}) or {}).get("review_count", len(reviews)))
                progress.progress((i + 1) / total, text=f"已命中磁盘缓存：{game_name}（{review_count} 条评论）")
                last_ready_result = _with_enriched_stats(cached_disk)
                last_ready_game = game_name
                last_ready_reviews = reviews
                continue

            status.info(f"正在分析 {game_name}...")
            analyzed = run_analysis_pipeline(
                reviews,
                game_name,
                run_level2=batch_run_level2,
                enable_llm=batch_enable_llm,
            )
            analyzed["game_name"] = game_name
            analyzed["analysis_mode"] = current_mode
            analyzed["run_level2"] = bool(batch_run_level2)
            analyzed["fetch_profile"] = "default" if batch_enable_llm else "fast"

            save_game_result(
                game_name=game_name,
                result=analyzed,
                reviews=reviews,
                analysis_mode=current_mode,
                run_level2=batch_run_level2,
            )

            analyzed = _with_enriched_stats(analyzed)
            st.session_state["all_games_cache"][cache_key] = analyzed
            progress.progress((i + 1) / total, text=f"分析完成：{game_name}")

            last_ready_result = analyzed
            last_ready_game = game_name
            last_ready_reviews = reviews
        except Exception as exc:
            failed.append(game_name)
            progress.progress((i + 1) / total, text=f"分析失败：{game_name}")
            status.warning(f"{game_name} 分析失败：{exc}")

    progress.empty()
    status.empty()

    success_count = total - len(failed)
    if last_ready_result is not None and last_ready_game is not None:
        ready_result = dict(last_ready_result)
        ready_result["all_games_cache"] = dict(st.session_state["all_games_cache"])
        st.session_state["result"] = ready_result
        st.session_state["current_game"] = last_ready_game
        st.session_state["current_reviews"] = last_ready_reviews

    st.session_state["cross_game_keys"] = tuple()
    st.session_state["cross_game_strategies"] = []

    if success_count >= 2:
        st.success(f"批量分析完成：成功 {success_count}/{total}，复用缓存 {skipped} 个。")
    elif success_count == 1:
        st.warning("仅有一款游戏分析成功，至少需要两款游戏才能进行竞品对比。")
    else:
        st.error("批量分析全部失败，请检查网络或配置。")

    if failed:
        st.caption(f"失败游戏：{' / '.join(failed)}")

result = _with_enriched_stats(st.session_state["result"] or {})
game_name = st.session_state["current_game"] or selected_game

visible_cache_for_state = _visible_analysis_cache(
    st.session_state.get("all_games_cache", {}),
    preferred_mode=str(st.session_state.get("analysis_mode_radio", "full")).lower(),
    preferred_run_level2=bool(st.session_state.get("run_level2_checkbox", False)),
)
st.session_state["analyzed_games"] = list(visible_cache_for_state.keys())

st.caption(f"当前分析游戏：{game_name}")

if st.session_state["result"] is None:
    st.info("请在左侧选择游戏并点击“开始分析”。")
    st.stop()

current_analysis_mode = str((st.session_state["result"] or {}).get("analysis_mode", "full")).lower()
current_run_level2 = bool((st.session_state["result"] or {}).get("run_level2", False))
latest_visible_reviews = _load_latest_review_cache(game_name, analysis_mode=current_analysis_mode)
if latest_visible_reviews and _cached_analysis_is_stale(st.session_state["result"], latest_visible_reviews):
    with st.spinner(f"检测到 {game_name} 的评论缓存已更新，正在同步最新分析..."):
        refreshed_result = run_analysis_pipeline(
            latest_visible_reviews,
            game_name,
            run_level2=current_run_level2,
            enable_llm=(current_analysis_mode != "fast"),
        )
        refreshed_result["game_name"] = game_name
        refreshed_result["analysis_mode"] = current_analysis_mode
        refreshed_result["run_level2"] = bool(current_run_level2)
        refreshed_result["fetch_profile"] = str((st.session_state["result"] or {}).get("fetch_profile", "default"))
        save_game_result(
            game_name=game_name,
            result=refreshed_result,
            reviews=latest_visible_reviews,
            analysis_mode=current_analysis_mode,
            run_level2=current_run_level2,
        )
        refreshed_result = _with_enriched_stats(refreshed_result)
        refreshed_key = _analysis_cache_key(
            game_name,
            run_level2=current_run_level2,
            analysis_mode=current_analysis_mode,
        )
        st.session_state.setdefault("all_games_cache", {})[refreshed_key] = refreshed_result
        refreshed_result["all_games_cache"] = dict(st.session_state["all_games_cache"])
        st.session_state["result"] = refreshed_result
        st.session_state["current_reviews"] = latest_visible_reviews
        result = _with_enriched_stats(refreshed_result)
        st.info(f"已同步最新评论缓存，共 {len(latest_visible_reviews)} 条评论。")

tab1, tab2, tab3 = st.tabs(["📊 竞品雷达", "💬 用户声音", "🎯 产品决策"])

with tab1:
    r = normalize_result(result)
    reviews = r.get("reviews", []) or []
    insights = r.get("insights", []) or []
    stats = r.get("sentiment_stats", {}) or {}
    llm_available = bool(r.get("llm_available", False))

    top_insight = insights[0] if insights else None
    top_action = str(getattr(top_insight, "action", "") or "暂无明确优先动作")
    confidence = str(getattr(top_insight, "confidence", "") or "中")
    render_conclusion_banner(
        top_action=top_action,
        llm_available=llm_available,
        confidence=confidence,
        review_count=len(reviews),
        game_name=game_name,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("评论总数", str(len(reviews)))
    with col2:
        render_metric_card("正向情感占比", f"{float(stats.get('positive_ratio', 0.0)) * 100:.1f}%")
    with col3:
        render_metric_card("模型一致性", f"{float(stats.get('agreement_rate', 0.0)) * 100:.1f}%")

    if insights:
        st.markdown("### 核心洞察")
        for idx, ins in enumerate(insights[:3]):
            render_insight_card(ins, rank=idx, game_name=game_name, prefix=f"radar_{idx}")
    else:
        st.info("当前还没有可展示的核心洞察。")

    st.markdown("### 跨游戏对比")
    all_cache = _visible_analysis_cache(
        st.session_state.get("all_games_cache", {}),
        preferred_mode=str(st.session_state.get("analysis_mode_radio", "full")).lower(),
        preferred_run_level2=bool(st.session_state.get("run_level2_checkbox", False)),
    )
    if all_cache:
        compare_rows = []
        compare_stats = {}
        for gkey, gresult in all_cache.items():
            gname = _cache_display_name(gkey, gresult)
            gr = normalize_result(_with_enriched_stats(gresult))
            game_reviews = gr.get("reviews", []) or []
            game_stats = gr.get("sentiment_stats", {}) or {}
            avg_rating = (
                sum(int(getattr(rv, "rating", 0) or 0) for rv in game_reviews) / len(game_reviews)
                if game_reviews else 0.0
            )
            compare_rows.append(
                {
                    "游戏": gname,
                    "评论数": len(game_reviews),
                    "平均评分": f"{avg_rating:.2f}",
                    "正向情感占比": f"{float(game_stats.get('positive_ratio', 0.0)) * 100:.1f}%",
                    "核心问题": _top_pain_label(gr),
                }
            )
            compare_stats[gname] = {
                "avg_rating": avg_rating,
                "positive_ratio": float(game_stats.get("positive_ratio", 0.0)),
            }

        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)
        try:
            st.plotly_chart(plot_multi_game_comparison(compare_stats), use_container_width=True)
        except Exception:
            st.info("暂时无法显示跨游戏对比图表。")
    else:
        st.info("暂无可对比的缓存游戏，请先分析至少一款游戏。")

    st.markdown("### 竞品机会点")
    if len(all_cache) < 2:
        st.info("请先分析至少两款游戏，即可查看竞品机会点。")
    else:
        with st.spinner("正在计算跨游戏策略..."):
            cross_result, strategies, cross_meta = get_or_compute_cross_analysis(
                force_reload=bool(st.session_state.get("force_cross_reload", False))
            )
        if "force_cross_reload" in st.session_state:
            del st.session_state["force_cross_reload"]

        if cross_meta.get("from_cache"):
            if cross_meta.get("layer") == "session":
                st.caption("已命中会话级跨游戏缓存。")
            else:
                created_at = str(cross_meta.get("created_at", ""))[:16]
                game_count = len(cross_meta.get("game_names", []) or [])
                st.caption(f"已命中磁盘缓存 | {created_at} | {game_count} 款游戏")
        else:
            st.caption("已重新计算并写入缓存。")

        strengths = cross_result.get("unique_strengths", {}) or {}
        common_problems = cross_result.get("common_problems", []) or []
        opportunity_gaps = cross_result.get("opportunity_gaps", []) or []

        if strengths:
            st.markdown("**差异化优势**")
            for game, slist in strengths.items():
                st.markdown(f"**{game}**")
                for sv in slist:
                    st.write(f"- {sv.get('strength', '')}：{sv.get('mechanism', '')}")

        if common_problems:
            st.markdown("**共性问题**")
            for item in common_problems:
                st.write(f"- {item.get('problem', '')}：{item.get('description', '')}")

        if opportunity_gaps:
            st.markdown("**机会空白**")
            for item in opportunity_gaps:
                st.write(f"- {item.get('gap', '')}：{item.get('evidence', '')}")

        if not strengths and not common_problems and not opportunity_gaps:
            st.info("当前没有足够的跨游戏机会点信息。")

with tab2:
    r = normalize_result(result)
    reviews = r.get("reviews", []) or []
    sentiment_stats = r.get("sentiment_stats", {}) or {}
    by_rating_stats = sentiment_stats.get("by_rating", {}) or {}
    topic_stats = r.get("topic_stats", {}) or {}

    st.subheader("用户情绪与主题分布")
    c1, c2 = st.columns(2)
    with c1:
        try:
            st.plotly_chart(plot_rating_distribution(reviews), use_container_width=True)
        except Exception:
            st.info("暂时无法显示评分分布图。")
    with c2:
        try:
            st.plotly_chart(plot_sentiment_pie(sentiment_stats), use_container_width=True)
        except Exception:
            st.info("暂时无法显示情绪占比图。")

    c3, c4 = st.columns(2)
    with c3:
        try:
            st.plotly_chart(plot_sentiment_by_rating(by_rating_stats), use_container_width=True)
        except Exception:
            st.info("暂时无法显示评分-情绪关联图。")
    with c4:
        try:
            st.plotly_chart(plot_topic_heatmap(topic_stats, reviews), use_container_width=True)
        except Exception:
            st.info("暂时无法显示主题热力图。")

    topics = r.get("topics", []) or []
    st.markdown("### 用户主题")
    if topics:
        for topic in topics[:8]:
            st.write(f"- {topic.get('topic_name', '未知主题')}（{topic.get('sentiment', 'mixed')}）")
    else:
        st.info("当前没有可展示的主题数据。")

    st.markdown("### 原始评论")
    if not reviews:
        st.info("当前没有原始评论数据。")
    else:
        review_rows = []
        for review in reviews[:50]:
            review_rows.append(
                {
                    "评分": int(getattr(review, "rating", 0) or 0),
                    "国家": str(getattr(review, "country", "") or "").upper(),
                    "情绪": str(getattr(review, "vader_label", "") or ""),
                    "评论": str(getattr(review, "text", "") or "")[:180],
                }
            )
        st.dataframe(pd.DataFrame(review_rows), use_container_width=True)

with tab3:
    st.subheader("产品建议")
    insights = normalize_result(result).get("insights", []) or []
    if insights:
        for idx, ins in enumerate(insights[:10]):
            render_insight_card(ins, rank=idx, game_name=game_name, prefix=f"decision_{idx}")
    else:
        st.info("当前没有可展示的产品建议。")

    st.markdown("### AI 助手")
    user_input = st.chat_input("请输入你想追问的问题...")
    if user_input:
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                from llm.agent import run_agent

                pipeline_payload = dict(result or {})
                pipeline_payload["all_games_cache"] = {
                    _cache_display_name(cache_key, cache_result): cache_result
                    for cache_key, cache_result in _visible_analysis_cache(
                        st.session_state.get("all_games_cache", {}),
                        preferred_mode=str(st.session_state.get("analysis_mode_radio", "full")).lower(),
                        preferred_run_level2=bool(st.session_state.get("run_level2_checkbox", False)),
                    ).items()
                }
                agent_result = run_agent(
                    user_query=user_input,
                    available_data={
                        "game_name": game_name,
                        "app_id": all_games.get(game_name, {}).get("app_id", ""),
                    },
                    pipeline_results=pipeline_payload,
                )
            st.write(agent_result.get("answer", ""))

    st.subheader("🔍 深度竞品分析")
    st.caption("输入你的问题，Agent将检索真实用户评论并对比竞品差距")
    question = st.text_input("你的问题", placeholder="例如：为什么用户在匹配机制上的抱怨比竞品更多？")

    available_competitors = [
        g for g in st.session_state.get("analyzed_games", [])
        if g != game_name and Path(f"indices/{g}.faiss").exists()
    ]

    if not available_competitors:
        st.info("请先分析一款竞品游戏，即可启用深度竞品分析功能")
    else:
        competitor = st.selectbox("对比竞品", available_competitors)
        if st.button("开始分析", key="deep_dive_btn") and question:
            from llm.agent import analyze_competitive_gap, _retrieve_game_reviews

            with st.status("Agent 正在分析...", expanded=True) as status:
                st.write("🔍 第一步：提取问题关键词...")
                keywords = " ".join([w for w in question.replace("？", "").replace("?", "").split() if len(w) > 1])
                st.write(f"关键词：{keywords}")

                st.write(f"📚 第二步：检索「{game_name}」相关评论...")
                target_reviews = _retrieve_game_reviews(keywords, game_name)
                st.write(f"找到 {len(target_reviews)} 条相关评论")

                st.write(f"📚 第三步：检索「{competitor}」相关评论...")
                comp_reviews = _retrieve_game_reviews(keywords, competitor)
                st.write(f"找到 {len(comp_reviews)} 条相关评论")

                st.write("🧠 第四步：对比分析，生成差距报告...")
                result_gap = analyze_competitive_gap(question, game_name, competitor, target_reviews, comp_reviews)
                status.update(label="✅ 分析完成", state="complete")

            if not result_gap.get("error"):
                st.info(result_gap["gap_summary"])
                if result_gap["root_causes"]:
                    st.markdown("**核心原因：**")
                    for c in result_gap["root_causes"]:
                        st.markdown(f"- {c}")
                st.metric("优先级评分", f"{result_gap['priority_score']}/5")
            else:
                st.warning(result_gap["gap_summary"])
