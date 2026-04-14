# ============================================================
# Module: game_finder.py
# ============================================================
# 【业务作用】根据关键词搜索竞品游戏，并补充基础商店信息供用户加入对比池
# 【上游】app.py 侧边栏竞品搜索功能调用
# 【下游】依赖 iTunes Search / Lookup API
# 【缺失影响】用户只能分析内置游戏，无法快速扩展到自定义竞品
# ============================================================

from typing import Optional

import requests
import streamlit as st


ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"
REQUEST_TIMEOUT_SECONDS = 10


# 这个函数把商店返回的原始字段统一整理成页面可直接消费的结构，
# 避免界面层被第三方接口字段名绑死。
def _normalize_game_item(item: dict) -> Optional[dict]:
    try:
        if not isinstance(item, dict):
            return None

        genre = str(item.get("primaryGenreName", "") or "")
        # 只保留 Game 类目，是为了把搜索结果聚焦到真正可比的竞品，
        # 否则工具类、壁纸类等软件也可能混进来，干扰产品判断。
        if "game" not in genre.lower():
            return None

        track_id = item.get("trackId")
        if track_id is None:
            return None

        try:
            rating_value = float(item.get("averageUserRating", 0.0) or 0.0)
        except Exception:
            rating_value = 0.0

        return {
            "app_id": str(track_id),
            "name": str(item.get("trackName", "") or ""),
            "developer": str(item.get("artistName", "") or ""),
            "rating": rating_value,
            "category": genre,
            "icon_url": str(item.get("artworkUrl60", "") or ""),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
# 搜索结果缓存 1 小时，是为了让产品侧反复尝试关键词时保持响应快，
# 同时避免每次键入都打到苹果接口。
def search_games(keyword: str, limit: int = 10) -> list[dict]:
    try:
        keyword = str(keyword or "").strip()
        if not keyword:
            return []

        # 把搜索上限压到 25 以内，是为了维持结果可读性；
        # 竞品探索要的是“挑几个可对比对象”，不是把应用商店整个翻出来。
        safe_limit = max(1, min(int(limit or 10), 25))
        response = requests.get(
            ITUNES_SEARCH_URL,
            params={
                "term": keyword,
                "entity": "software",
                "country": "us",
                "limit": safe_limit,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("results", [])

        results: list[dict] = []
        for item in items:
            normalized = _normalize_game_item(item)
            if normalized is not None:
                results.append(normalized)
        return results
    except Exception:
        return []


# Lookup 和 Search 分开保留，是为了兼顾两种场景：
# 一个是“按关键词找新游戏”，一个是“已知 app_id 后补全信息”。
def get_game_info(app_id: str) -> Optional[dict]:
    try:
        app_id = str(app_id or "").strip()
        if not app_id:
            return None

        response = requests.get(
            ITUNES_LOOKUP_URL,
            params={"id": app_id, "country": "us"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("results", [])
        if not items:
            return None
        return _normalize_game_item(items[0])
    except Exception:
        return None


if __name__ == "__main__":
    results = search_games("mahjong puzzle")
    print(f"搜索结果：{len(results)} 款游戏")
    for r in results[:3]:
        print(f"  {r['name']} | {r['developer']} | ⭐{r['rating']}")
