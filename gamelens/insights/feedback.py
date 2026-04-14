# ============================================================
# Module: feedback.py
# ============================================================
# 【业务作用】记录用户对 Insight 的反馈，并生成学习闭环统计
# 【上游】app.py 的 Helpful / Not Helpful 按钮调用
# 【下游】依赖本地 logs/feedback.csv
# 【缺失影响】系统无法知道哪些建议真正有用，后续规则和模型优化会失去反馈依据
# ============================================================

import csv
import os
from datetime import datetime
from typing import Optional

FEEDBACK_FILE = "logs/feedback.csv"
FEEDBACK_FIELDS = ["timestamp", "insight_id", "game_name", "source", "priority", "feedback"]


# 这个函数解决的是“用户觉得建议有没有帮助”无法被系统记住的问题。
# 没有反馈沉淀，洞察页面就只能展示结果，无法形成后续优化闭环。
def record_feedback(
    insight_id: str,
    game_name: str,
    source: str,
    priority: str,
    feedback: str
) -> None:
    """
    记录用户反馈到 logs/feedback.csv。

    处理逻辑：
    - feedback 仅接受 "useful" 或 "not_useful"，其他值抛 ValueError。
    - 自动创建 logs/ 目录与 CSV 表头。
    - 以追加模式写入，不覆盖历史记录。

    参数：
    - insight_id: Insight 唯一标识。
    - game_name: 游戏名称。
    - source: Insight 来源（llm/rule_fallback）。
    - priority: Insight 优先级。
    - feedback: 反馈值，useful 或 not_useful。
    """
    value = str(feedback).strip().lower()
    if value not in {"useful", "not_useful"}:
        raise ValueError("feedback 只接受 'useful' 或 'not_useful'")

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    need_header = not os.path.exists(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=FEEDBACK_FIELDS)
        if need_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "insight_id": insight_id,
                "game_name": game_name,
                "source": source,
                "priority": priority,
                "feedback": value,
            }
        )


# 统计逻辑单独抽出来，是为了让页面层只负责展示，
# 不必关心 CSV 结构、过滤条件和来源聚合这些实现细节。
def get_feedback_stats(game_name: Optional[str] = None) -> dict:
    """
    统计反馈结果。

    处理逻辑：
    - game_name 不为 None 时仅统计该游戏。
    - 文件不存在时返回空统计。
    - 输出总体统计与分来源统计（llm/rule_fallback）。

    参数：
    - game_name: 可选游戏名过滤条件。

    返回：
    - dict: 反馈统计结构。
    """
    def build_bucket() -> dict:
        return {"useful": 0, "not_useful": 0, "useful_rate": 0.0}

    result = {
        "total": 0,
        "useful": 0,
        "not_useful": 0,
        "useful_rate": 0.0,
        "by_source": {
            "llm": build_bucket(),
            "rule_fallback": build_bucket(),
        },
    }

    if not os.path.exists(FEEDBACK_FILE):
        return result

    try:
        with open(FEEDBACK_FILE, "r", newline="", encoding="utf-8-sig") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if game_name is not None and row.get("game_name") != game_name:
                    continue

                fb = str(row.get("feedback", "")).strip().lower()
                if fb not in {"useful", "not_useful"}:
                    continue

                src = str(row.get("source", "")).strip()
                if src not in result["by_source"]:
                    result["by_source"][src] = build_bucket()

                result["total"] += 1
                result[fb] += 1
                result["by_source"][src][fb] += 1

        if result["total"] > 0:
            result["useful_rate"] = result["useful"] / result["total"]

        for src, bucket in result["by_source"].items():
            src_total = bucket["useful"] + bucket["not_useful"]
            bucket["useful_rate"] = (bucket["useful"] / src_total) if src_total > 0 else 0.0

        return result
    except Exception as exc:
        print(f"[Warning] 反馈统计失败: {exc}")
        return result
