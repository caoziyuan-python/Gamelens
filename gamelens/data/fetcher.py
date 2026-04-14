# ============================================================
# Module: fetcher.py
# ============================================================
# 【业务作用】从 App Store 多国家接口抓取评论并做基础清洗
# 【上游】utils.cache.get_or_fetch 在缓存未命中时调用
# 【下游】依赖 config 抓取参数和 data.schema.ReviewSchema
# 【缺失影响】系统将失去实时数据入口，只能依赖历史缓存，无法分析新评论
# ============================================================

import hashlib
import json
import os
import sys
import time
from typing import Dict, List

import requests

try:
    from config import COUNTRIES, MAX_RETRY, PAGES_PER_COUNTRY, REQUEST_SLEEP
except ModuleNotFoundError:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from config import COUNTRIES, MAX_RETRY, PAGES_PER_COUNTRY, REQUEST_SLEEP

from data.schema import ReviewSchema

_LAST_FETCH_SUMMARY: Dict[str, object] = {}


def consume_last_fetch_summary() -> dict:
    global _LAST_FETCH_SUMMARY
    summary = dict(_LAST_FETCH_SUMMARY)
    _LAST_FETCH_SUMMARY = {}
    return summary


# 这个函数承担“把外部评论源稳定拉回本地”的职责。
# 没有它，后续情感分析和洞察模块就失去了输入数据，整条链路都会空转。
def fetch_reviews(
    app_id: str,
    game_name: str,
    countries: list = None,
    pages: int = None
) -> List[ReviewSchema]:
    """
    多国评论抓取。

    处理逻辑：
    - 循环 countries，每国抓 pages 页。
    - 使用 iTunes RSS API 拉取评论数据。
    - 每次请求后 sleep(REQUEST_SLEEP)。
    - 超时请求重试 MAX_RETRY 次，其余异常打印 warning 后跳过当前页。
    - 返回空 entry 或 JSON 解析失败时打印 warning 并跳过当前页。
    - entry[0] 为 app 元信息，不作为评论，解析从 entry[1:] 开始。
    - 使用 hashlib.md5(f"{text}{country}".encode()).hexdigest() 去重。

    参数：
    - app_id: App Store 应用 ID。
    - game_name: 游戏名称，用于结果标识。
    - countries: 可选国家列表，默认使用 config.COUNTRIES。
    - pages: 每个国家抓取页数，默认使用 config.PAGES_PER_COUNTRY。

    返回：
    - List[ReviewSchema]: 去重后的原始评论列表（未做长度清洗）。
    """
    target_countries = countries or COUNTRIES
    target_pages = pages or PAGES_PER_COUNTRY

    global _LAST_FETCH_SUMMARY
    _LAST_FETCH_SUMMARY = {}

    all_reviews: List[ReviewSchema] = []
    seen_review_ids = set()
    empty_entry_pages: Dict[str, List[int]] = {}

    for country in target_countries:
        for page in range(1, target_pages + 1):
            if page == 1:
                url = (
                    f"https://itunes.apple.com/{country}/rss/customerreviews/"
                    f"id={app_id}/sortby=mostrecent/json"
                )
            else:
                url = (
                    f"https://itunes.apple.com/{country}/rss/customerreviews/"
                    f"page={page}/id={app_id}/sortby=mostrecent/json"
                )

            payload = None
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    payload = response.json()
                    break
                except requests.exceptions.Timeout:
                    if attempt < MAX_RETRY:
                        print(
                            f"[Warning] 请求超时，重试中 "
                            f"({attempt}/{MAX_RETRY}) country={country}, page={page}"
                        )
                    else:
                        print(
                            f"[Warning] 请求超时，已跳过该页 "
                            f"country={country}, page={page}"
                        )
                except (json.JSONDecodeError, ValueError):
                    print(
                        f"[Warning] JSON解析失败，已跳过该页 "
                        f"country={country}, page={page}"
                    )
                    break
                except requests.RequestException as exc:
                    print(
                        f"[Warning] 请求失败({exc})，已跳过该页 "
                        f"country={country}, page={page}"
                    )
                    break
                finally:
                    # 每次请求后都 sleep，是为了把“接口稳定拿到数据”放在第一优先级，
                    # 否则瞬时并发过高更容易触发限流，最终整体抓取反而更慢。
                    time.sleep(REQUEST_SLEEP)

            if payload is None:
                continue

            entries = payload.get("feed", {}).get("entry", [])
            if not entries:
                empty_entry_pages.setdefault(country, []).append(page)
                break

            # RSS 返回的第一条是应用元信息而不是用户评论，
            # 如果不跳过，后续会把商店元数据误当成一条评论写入分析结果。
            safe_entries = entries[1:] if entries and len(entries) > 1 else (entries or [])
            for entry in safe_entries:
                if not isinstance(entry, dict):
                    continue
                content_obj = entry.get("content", {}) if isinstance(entry, dict) else {}
                text = content_obj.get("label", "") if isinstance(content_obj, dict) else ""
                text = (text or "").strip()

                # 用评论文本 + 国家生成 ID，是为了在多国抓取时做稳定去重；
                # 否则同一条评论在不同页或重试场景下会被重复计入样本。
                review_id = hashlib.md5(f"{text}{country}".encode()).hexdigest()
                if review_id in seen_review_ids:
                    continue
                seen_review_ids.add(review_id)

                rating_obj = entry.get("im:rating", {}) if isinstance(entry, dict) else {}
                rating_raw = rating_obj.get("label", 0) if isinstance(rating_obj, dict) else 0
                try:
                    rating = int(rating_raw)
                except (TypeError, ValueError):
                    rating = 0

                updated_obj = entry.get("updated", {}) if isinstance(entry, dict) else {}
                date = updated_obj.get("label", "") if isinstance(updated_obj, dict) else ""

                all_reviews.append(
                    ReviewSchema(
                        review_id=review_id,
                        game_name=game_name,
                        country=country,
                        rating=rating,
                        text=text,
                        date=str(date or ""),
                    )
                )

    if empty_entry_pages:
        summary_chunks = []
        total_empty_pages = 0
        for country, pages_hit in empty_entry_pages.items():
            unique_pages = sorted(set(int(page) for page in pages_hit))
            total_empty_pages += len(unique_pages)
            page_label = "/".join(str(page) for page in unique_pages)
            summary_chunks.append(f"{country.upper()}(page {page_label})")
        message = "部分国家高页数无评论，已自动跳过：" + " / ".join(summary_chunks)
        _LAST_FETCH_SUMMARY = {
            "message": message,
            "empty_page_countries": {
                country: sorted(set(int(page) for page in pages_hit))
                for country, pages_hit in empty_entry_pages.items()
            },
            "empty_page_count": total_empty_pages,
        }
        print(f"[Info] {message}")
    else:
        _LAST_FETCH_SUMMARY = {
            "message": "",
            "empty_page_countries": {},
            "empty_page_count": 0,
        }

    return all_reviews


# 这个清洗步骤看似简单，但它在业务上负责把“无意义噪声”挡在分析链路外面，
# 否则模型和规则会把空文本、无效短句也当成真实用户声音。
def clean_reviews(reviews: List[ReviewSchema]) -> List[ReviewSchema]:
    """
    数据清洗。

    处理逻辑：
    - 过滤 text 为空或 len(text.strip()) < 5 的评论。
    - 打印清洗前后数量。

    参数：
    - reviews: 原始评论列表。

    返回：
    - List[ReviewSchema]: 清洗后的评论列表。
    """
    cleaned = []
    for review in reviews:
        text = (review.text or "").strip()
        if len(text) >= 5:
            cleaned.append(review)

    print(f"清洗前: {len(reviews)} 条，清洗后: {len(cleaned)} 条")
    return cleaned


if __name__ == "__main__":
    from utils.cache import get_or_fetch

    reviews = get_or_fetch(
        app_id="1617391485",
        game_name="Block Blast",
        countries=["us"],
        pages=1
    )
    print(f"获取到 {len(reviews)} 条评论")
    if reviews:
        print(f"示例：{reviews[0]}")
