# -*- coding: utf-8 -*-
# Cache utilities for GameLens:
# - Reviews CSV cache (persistent)
# - Game result JSON cache (persistent)
# - Cross-game result JSON cache (persistent)
# - Cache index for fast lookup

import ast
import hashlib
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data.schema import InsightSchema, ReviewSchema

CACHE_DIR = "cache"
REVIEWS_CACHE_DIR = "cache/reviews"
GAME_CACHE_DIR = "cache/game"
CROSS_CACHE_DIR = "cache/cross"
CACHE_INDEX_FILE = "cache/cache_index.json"
CACHE_VERSION = {
    "game": "v2",
    "cross": "v2",
}
REVIEW_CHANGE_THRESHOLD = 0.05


def _ensure_cache_dirs() -> None:
    Path(CACHE_DIR).mkdir(exist_ok=True)
    Path(REVIEWS_CACHE_DIR).mkdir(exist_ok=True)
    Path(GAME_CACHE_DIR).mkdir(exist_ok=True)
    Path(CROSS_CACHE_DIR).mkdir(exist_ok=True)


def _cache_file_path(game_name: str, cache_variant: str = "") -> str:
    safe_variant = str(cache_variant or "").strip().replace(" ", "_")
    suffix = f"__{safe_variant}" if safe_variant else ""
    return os.path.join(REVIEWS_CACHE_DIR, f"{game_name}_reviews{suffix}.csv")


def _legacy_review_cache_file_path(game_name: str, cache_variant: str = "") -> str:
    safe_variant = str(cache_variant or "").strip().replace(" ", "_")
    suffix = f"__{safe_variant}" if safe_variant else ""
    return os.path.join(CACHE_DIR, f"{game_name}_reviews{suffix}.csv")


def _game_result_path(
    game_name: str,
    analysis_mode: str = "full",
    run_level2: bool = False,
) -> str:
    safe_name = game_name.replace(" ", "_")
    safe_mode = str(analysis_mode or "full").strip().lower()
    l2_tag = "l2on" if run_level2 else "l2off"
    return os.path.join(GAME_CACHE_DIR, f"{safe_name}_{safe_mode}_{l2_tag}_result.json")


def _cross_result_path(combo_id: str) -> str:
    return os.path.join(CROSS_CACHE_DIR, f"cross_game_{combo_id}.json")


def _safe_literal_list(raw_value):
    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    if isinstance(raw_value, list):
        return raw_value
    return []


def _safe_asdict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)


def _calc_review_fingerprint(reviews: list) -> str:
    """
    基于评论内容计算指纹。
    """
    if not reviews:
        return "empty"
    content = "".join(
        f"{r.review_id}{(r.text or '')[:50]}"
        for r in sorted(reviews, key=lambda x: x.review_id)
    )
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _calc_combo_fingerprint(game_summaries: dict) -> str:
    """
    对游戏组合计算稳定指纹。
    """
    parts = []
    for name in sorted(game_summaries.keys()):
        summary = game_summaries[name] or {}
        parts.append(
            f"{name}"
            f"{summary.get('review_fingerprint', '')}"
            f"{summary.get('analysis_mode', 'full')}"
            f"{summary.get('run_level2', False)}"
        )
    content = "|".join(parts)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _load_cache_index() -> dict:
    """
    读取缓存索引。
    """
    index_path = Path(CACHE_INDEX_FILE)
    if not index_path.exists():
        return {"game": {}, "cross": {}}
    try:
        with open(index_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {"game": {}, "cross": {}}
        data.setdefault("game", {})
        data.setdefault("cross", {})
        return data
    except Exception:
        return {"game": {}, "cross": {}}


def _update_cache_index(entry_type: str, key: str, meta: dict) -> None:
    """
    更新缓存索引文件。
    """
    index = _load_cache_index()
    if entry_type not in index:
        index[entry_type] = {}
    index[entry_type][key] = meta

    try:
        _ensure_cache_dirs()
        with open(CACHE_INDEX_FILE, "w", encoding="utf-8") as file:
            json.dump(index, file, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"⚠️ 索引更新失败：{exc}")


def _migrate_legacy_review_cache(game_name: str, cache_variant: str = "") -> str | None:
    new_path = _cache_file_path(game_name, cache_variant=cache_variant)
    legacy_path = _legacy_review_cache_file_path(game_name, cache_variant=cache_variant)
    if os.path.exists(new_path):
        return new_path
    if not os.path.exists(legacy_path):
        return None

    try:
        _ensure_cache_dirs()
        shutil.move(legacy_path, new_path)
        print(f"✅ 已迁移评论缓存：{legacy_path} -> {new_path}")
        return new_path
    except Exception as exc:
        print(f"⚠️ 评论缓存迁移失败：{exc}")
        return legacy_path


def save_reviews(
    reviews: List[ReviewSchema],
    game_name: str,
    cache_variant: str = ""
) -> None:
    """
    将评论保存到 reviews 分层目录。
    """
    try:
        _ensure_cache_dirs()
        cache_path = _cache_file_path(game_name, cache_variant=cache_variant)
        rows = [review.__dict__ for review in reviews]
        pd.DataFrame(rows).to_csv(cache_path, index=False)
    except Exception as exc:
        print(f"[Warning] 保存缓存失败: {exc}")


def load_reviews(
    game_name: str,
    cache_variant: str = ""
) -> Optional[List[ReviewSchema]]:
    """
    从新的 reviews 目录读取评论缓存，并兼容旧路径迁移。
    """
    cache_path = _cache_file_path(game_name, cache_variant=cache_variant)
    if not os.path.exists(cache_path):
        migrated_path = _migrate_legacy_review_cache(game_name, cache_variant=cache_variant)
        if migrated_path:
            cache_path = migrated_path
    if not os.path.exists(cache_path):
        return None

    try:
        df = pd.read_csv(cache_path)
        reviews: List[ReviewSchema] = []

        for _, row in df.iterrows():
            rule_topics = _safe_literal_list(row.get("rule_topics", "[]"))
            llm_topics = _safe_literal_list(row.get("llm_topics", "[]"))

            agreement_raw = row.get("agreement", False)
            if isinstance(agreement_raw, str):
                agreement = agreement_raw.strip().lower() in {"true", "1", "yes"}
            else:
                agreement = bool(agreement_raw)

            reviews.append(
                ReviewSchema(
                    review_id=str(row.get("review_id", "")),
                    game_name=str(row.get("game_name", game_name)),
                    country=str(row.get("country", "")),
                    rating=int(row.get("rating", 0)) if pd.notna(row.get("rating", 0)) else 0,
                    text=str(row.get("text", "")),
                    date=str(row.get("date", "")),
                    vader_score=float(row.get("vader_score", 0.0)) if pd.notna(row.get("vader_score", 0.0)) else 0.0,
                    vader_label=str(row.get("vader_label", "Neutral")),
                    textblob_score=float(row.get("textblob_score", 0.0)) if pd.notna(row.get("textblob_score", 0.0)) else 0.0,
                    textblob_label=str(row.get("textblob_label", "Neutral")),
                    rule_topics=rule_topics,
                    llm_topics=llm_topics,
                    agreement=agreement,
                )
            )

        print(f"从缓存读取 {len(reviews)} 条评论")
        return reviews
    except Exception as exc:
        print(f"[Warning] 读取缓存失败: {exc}")
        return None


def get_or_fetch(
    app_id: str,
    game_name: str,
    countries: list = None,
    pages: int = None,
    cache_variant: str = "",
    allow_default_cache_fallback: bool = False
) -> List[ReviewSchema]:
    """
    缓存优先的评论获取逻辑。
    """
    cached_reviews = load_reviews(game_name, cache_variant=cache_variant)
    if cached_reviews is None and (cache_variant or allow_default_cache_fallback):
        cached_reviews = load_reviews(game_name)
    if cached_reviews is not None:
        return cached_reviews

    from data.fetcher import clean_reviews, fetch_reviews

    raw_reviews = fetch_reviews(
        app_id=app_id,
        game_name=game_name,
        countries=countries,
        pages=pages,
    )
    cleaned_reviews = clean_reviews(raw_reviews)
    save_reviews(cleaned_reviews, game_name, cache_variant=cache_variant)
    return cleaned_reviews


def save_game_result(
    game_name: str,
    result: dict,
    reviews: list,
    analysis_mode: str = "full",
    run_level2: bool = False
) -> None:
    """
    保存单游戏分析结果。
    """
    _ensure_cache_dirs()
    file_path = _game_result_path(
        game_name=game_name,
        analysis_mode=analysis_mode,
        run_level2=run_level2,
    )

    serializable = {}
    for key, value in (result or {}).items():
        if key in {"reviews", "insights"}:
            try:
                serializable[key] = [_safe_asdict(item) for item in (value or [])]
            except Exception:
                serializable[key] = []
        else:
            try:
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                serializable[key] = str(value)

    now = datetime.now()
    payload = {
        "cache_version": CACHE_VERSION["game"],
        "game_name": game_name,
        "analysis_mode": analysis_mode,
        "run_level2": run_level2,
        "review_count": len(reviews or []),
        "review_fingerprint": _calc_review_fingerprint(reviews or []),
        "created_at": now.isoformat(),
        "created_at_epoch": now.timestamp(),
        "result": serializable,
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    _update_cache_index(
        entry_type="game",
        key=f"{game_name}__{analysis_mode}__{'l2on' if run_level2 else 'l2off'}",
        meta={
            "game_name": game_name,
            "review_count": len(reviews or []),
            "analysis_mode": analysis_mode,
            "run_level2": run_level2,
            "created_at": now.isoformat(),
            "epoch": now.timestamp(),
        },
    )
    print(f"✅ 已保存分析结果：{game_name}")


def load_game_result(
    game_name: str,
    reviews: list,
    analysis_mode: str = "full",
    run_level2: bool = False,
    force_reload: bool = False,
    policy: str = "default",
) -> dict | None:
    """
    读取单游戏分析结果，并按策略决定是否复用缓存。
    """
    if force_reload:
        policy = "force"
    if policy == "force":
        return None

    file_path = _game_result_path(
        game_name=game_name,
        analysis_mode=analysis_mode,
        run_level2=run_level2,
    )
    if not Path(file_path).exists():
        legacy_path = os.path.join(CACHE_DIR, f"{game_name.replace(' ', '_')}_result.json")
        if Path(legacy_path).exists():
            file_path = legacy_path
        else:
            return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return None

    if payload.get("cache_version") != CACHE_VERSION["game"]:
        print(f"⚠️ {game_name} 缓存版本不匹配，将重新计算")
        return None

    if payload.get("analysis_mode") != analysis_mode:
        print(
            f"⚠️ {game_name} 分析模式不匹配（缓存:{payload.get('analysis_mode')} vs 当前:{analysis_mode}），将重新计算"
        )
        return None

    if bool(payload.get("run_level2", False)) != bool(run_level2):
        print(f"⚠️ {game_name} Level2 配置不匹配，将重新计算")
        return None

    if policy != "relaxed":
        current_fp = _calc_review_fingerprint(reviews or [])
        cached_fp = payload.get("review_fingerprint", "")
        cached_count = int(payload.get("review_count", 0) or 0)
        current_count = len(reviews or [])

        if current_fp == cached_fp:
            pass
        elif cached_count > 0:
            change_ratio = abs(current_count - cached_count) / cached_count
            if change_ratio < REVIEW_CHANGE_THRESHOLD:
                print(
                    f"ℹ️ {game_name}：评论小幅变化（{cached_count}→{current_count}条，{change_ratio * 100:.1f}%），复用缓存"
                )
            else:
                print(
                    f"⚠️ {game_name}：评论变化超过阈值（{cached_count}→{current_count}条，{change_ratio * 100:.1f}%），重新计算"
                )
                return None
        else:
            return None

    data = dict(payload.get("result", {}) or {})
    try:
        if data.get("reviews"):
            data["reviews"] = [ReviewSchema(**item) for item in data["reviews"]]
        if data.get("insights"):
            data["insights"] = [InsightSchema(**item) for item in data["insights"]]
    except Exception as exc:
        print(f"⚠️ 反序列化失败：{exc}，将重新计算")
        return None

    data["_cache_meta"] = {
        "from_cache": True,
        "created_at": payload.get("created_at", ""),
        "review_count": payload.get("review_count", 0),
        "analysis_mode": payload.get("analysis_mode", ""),
        "run_level2": payload.get("run_level2", False),
        "review_fingerprint": payload.get("review_fingerprint", ""),
    }

    print(
        f"✅ 命中分析缓存：{game_name} "
        f"({payload.get('review_count')}条评论 · {payload.get('analysis_mode')} · "
        f"{'L2开' if payload.get('run_level2') else 'L2关'})"
    )
    return data


def build_game_summary_for_cross(
    game_name: str,
    result: dict,
    reviews: list,
    analysis_mode: str = "full",
    run_level2: bool = False
) -> dict:
    """
    为跨游戏分析构建标准化摘要。
    """
    return {
        "game_name": game_name,
        "review_fingerprint": _calc_review_fingerprint(reviews or []),
        "review_count": len(reviews or []),
        "analysis_mode": analysis_mode,
        "run_level2": run_level2,
    }


def save_cross_result(
    game_summaries: dict,
    cross_result: dict,
    strategies: list | None = None,
    analysis_mode: str = "full",
    run_level2: bool = False
) -> str:
    """
    保存跨游戏组合分析结果。
    """
    _ensure_cache_dirs()
    combo_id = _calc_combo_fingerprint(game_summaries)
    file_path = _cross_result_path(combo_id)
    now = datetime.now()

    serialized_strategies = []
    for item in (strategies or []):
        serialized_strategies.append(_safe_asdict(item))

    payload = {
        "cache_version": CACHE_VERSION["cross"],
        "combo_id": combo_id,
        "game_names": sorted(game_summaries.keys()),
        "combo_fingerprint": combo_id,
        "analysis_mode": analysis_mode,
        "run_level2": run_level2,
        "created_at": now.isoformat(),
        "created_at_epoch": now.timestamp(),
        "cross_result": cross_result,
        "strategies": serialized_strategies,
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    _update_cache_index(
        entry_type="cross",
        key=combo_id,
        meta={
            "game_names": sorted(game_summaries.keys()),
            "created_at": now.isoformat(),
            "epoch": now.timestamp(),
        },
    )
    print(f"✅ 已保存跨游戏缓存：{combo_id}")
    return combo_id


def load_cross_result(
    game_summaries: dict,
    force_reload: bool = False,
    policy: str = "default",
) -> dict | None:
    """
    读取跨游戏组合分析结果。
    """
    if force_reload:
        policy = "force"
    if policy == "force":
        return None

    combo_fp = _calc_combo_fingerprint(game_summaries)
    file_path = _cross_result_path(combo_fp)
    if not Path(file_path).exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return None

    if payload.get("cache_version") != CACHE_VERSION["cross"]:
        print("⚠️ 跨游戏缓存版本不匹配，将重新计算")
        return None

    if policy != "relaxed" and payload.get("combo_fingerprint") != combo_fp:
        print("⚠️ 跨游戏组合已变化，将重新计算")
        return None

    strategies_raw = payload.get("strategies", []) or []
    try:
        from insights.strategy_ranker import Strategy  # type: ignore

        strategies = [Strategy(**item) for item in strategies_raw]
    except Exception:
        strategies = strategies_raw

    print(
        f"✅ 命中跨游戏缓存：{combo_fp} "
        f"({len(payload.get('game_names', []))}款游戏 · {payload.get('created_at', '')[:16]})"
    )
    return {
        "cross_result": payload.get("cross_result", {}),
        "strategies": strategies,
        "_cache_meta": {
            "from_cache": True,
            "created_at": payload.get("created_at", ""),
            "game_names": payload.get("game_names", []),
            "combo_id": combo_fp,
        },
    }


def list_saved_results() -> list:
    """
    列出所有已保存的单游戏分析结果，优先读取索引。
    """
    index = _load_cache_index()
    game_index = index.get("game", {})

    if game_index:
        results = []
        for game_key, meta in game_index.items():
            results.append(
                {
                    "game_name": meta.get("game_name", str(game_key).split("__")[0]),
                    "review_count": meta.get("review_count", 0),
                    "analysis_mode": meta.get("analysis_mode", "full"),
                    "run_level2": meta.get("run_level2", False),
                    "created_at": str(meta.get("created_at", ""))[:16],
                    "epoch": meta.get("epoch", 0),
                }
            )
        return sorted(results, key=lambda item: item["epoch"], reverse=True)

    results = []
    cache_dir = Path(GAME_CACHE_DIR)
    if not cache_dir.exists():
        cache_dir = Path(CACHE_DIR)

    for file in cache_dir.glob("*_result.json"):
        try:
            with open(file, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if payload.get("cache_version") not in CACHE_VERSION.values():
                continue
            results.append(
                {
                    "game_name": payload.get("game_name", ""),
                    "review_count": payload.get("review_count", 0),
                    "analysis_mode": payload.get("analysis_mode", "full"),
                    "run_level2": payload.get("run_level2", False),
                    "created_at": str(payload.get("created_at", ""))[:16],
                    "epoch": payload.get("created_at_epoch", 0),
                }
            )
        except Exception:
            continue

    return sorted(results, key=lambda item: item["epoch"], reverse=True)
