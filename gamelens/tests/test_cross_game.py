# ============================================================
# Module: test_cross_game.py
# ============================================================
# 【业务作用】验证跨游戏分析里的游戏名解析、规则闪光点兜底、LLM enrich 职责边界
# 【上游】开发者或 CI 执行测试时运行
# 【下游】依赖 insights.cross_game
# 【缺失影响】跨游戏分析很容易在后续迭代中悄悄破坏数据支撑或覆盖规则字段
# ============================================================

import unittest
from unittest.mock import patch

from insights.cross_game import (
    _llm_enrich_strengths,
    _resolve_game_name,
    cross_game_analysis,
)


def _build_game_result(
    game_name: str,
    positive_ratio: float,
    negative_ratio: float,
    reviews_count: int,
    topics: list,
    complaints: list,
) -> dict:
    return {
        "game_name": game_name,
        "reviews": [{} for _ in range(reviews_count)],
        "llm_topics": topics,
        "topic_cards": topics,
        "topic_stats": {
            topic["topic_name"]: {"ratio": topic.get("ratio", 0.0)}
            for topic in topics
        },
        "complaints": complaints,
        "sentiment_stats": {
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "avg_score": 4.2,
        },
        "insights": [],
        "validation_results": {},
        "llm_available": True,
    }


# 这组测试的目标是守住跨游戏分析的三个关键承诺：
# 名称解析正确、LLM 挂了仍有闪光点、LLM enrich 不会篡改规则字段。
class CrossGameTests(unittest.TestCase):
    def test_resolve_game_name_prefers_result_name(self):
        self.assertEqual(
            _resolve_game_name("Demo__l2_0", {"game_name": "Real Demo"}),
            "Real Demo",
        )
        self.assertEqual(
            _resolve_game_name("Demo__l2_0", {}),
            "Demo",
        )

    @patch("insights.cross_game.call_llm_json", return_value=None)
    def test_cross_game_analysis_has_strengths_when_llm_unavailable(self, _mock_llm):
        all_games_cache = {
            "Game A__l2_0": _build_game_result(
                game_name="Game A",
                positive_ratio=0.62,
                negative_ratio=0.12,
                reviews_count=120,
                topics=[
                    {
                        "topic_name": "Core Delight",
                        "sentiment": "positive",
                        "core_demand": "用户喜欢轻松解压的核心体验",
                        "representative_review": "Very relaxing and satisfying to play.",
                        "ratio": 0.30,
                    }
                ],
                complaints=[
                    {
                        "complaint_type": "Ads",
                        "estimated_ratio": 0.03,
                        "typical_quote": "Too many ads.",
                    }
                ],
            ),
            "Game B__l2_0": _build_game_result(
                game_name="Game B",
                positive_ratio=0.44,
                negative_ratio=0.21,
                reviews_count=130,
                topics=[
                    {
                        "topic_name": "Visual Polish",
                        "sentiment": "positive",
                        "core_demand": "用户认可画面和反馈设计",
                        "representative_review": "Looks polished and feels smooth.",
                        "ratio": 0.18,
                    }
                ],
                complaints=[
                    {
                        "complaint_type": "Ads",
                        "estimated_ratio": 0.22,
                        "typical_quote": "Way too many forced ads.",
                    },
                    {
                        "complaint_type": "Difficulty",
                        "estimated_ratio": 0.20,
                        "typical_quote": "Levels are unfair.",
                    }
                ],
            ),
            "Game C__l2_0": _build_game_result(
                game_name="Game C",
                positive_ratio=0.40,
                negative_ratio=0.18,
                reviews_count=140,
                topics=[
                    {
                        "topic_name": "Relaxing Session",
                        "sentiment": "positive",
                        "core_demand": "用户在碎片时间也能轻松游玩",
                        "representative_review": "Great for short relaxing sessions.",
                        "ratio": 0.16,
                    }
                ],
                complaints=[
                    {
                        "complaint_type": "Ads",
                        "estimated_ratio": 0.18,
                        "typical_quote": "Ads interrupt the flow.",
                    },
                    {
                        "complaint_type": "Difficulty",
                        "estimated_ratio": 0.17,
                        "typical_quote": "Difficulty spikes too hard.",
                    }
                ],
            ),
        }

        result = cross_game_analysis(all_games_cache)

        self.assertEqual(set(result["raw_summaries"].keys()), {"Game A", "Game B", "Game C"})
        for game_name in ("Game A", "Game B", "Game C"):
            self.assertGreaterEqual(len(result["unique_strengths"].get(game_name, [])), 1)

        avoidance_strengths = [
            item for item in result["unique_strengths"]["Game A"]
            if item.get("strength_type") == "规避优势"
        ]
        self.assertTrue(avoidance_strengths)
        self.assertIn("avg_ratio", result["common_problems"][0])
        self.assertIn("game_count", result["common_problems"][0])

    @patch(
        "insights.cross_game.call_llm_json",
        return_value=[{"strength": "Core Delight", "why_valuable": "更放松", "pm_apply_tip": "强化反馈节奏"}],
    )
    def test_llm_enrich_does_not_override_rule_fields(self, _mock_llm):
        rule_strengths = {
            "Game A": [
                {
                    "strength": "Core Delight",
                    "strength_type": "局部优势",
                    "mechanism": "用户喜欢轻松解压的核心体验",
                    "data_evidence": {
                        "source": "rule_cross_game_exact_topic",
                        "metric": "topic_positive_ratio",
                        "current_value": 0.30,
                        "peer_avg": 0.15,
                        "uplift": 2.0,
                        "sample_size": 36,
                        "matched_topic": "Core Delight",
                        "quote": "Very relaxing and satisfying to play.",
                    },
                    "data_backed": True,
                    "confidence": "High",
                    "transferable_pattern": "轻松解压体验",
                    "how_to_apply": "参考Game A在Core Delight上的产品设计（uplift=2.0x）",
                    "applicability": "高",
                    "why_valuable": "",
                    "pm_apply_tip": "",
                }
            ]
        }

        enriched = _llm_enrich_strengths(rule_strengths)
        item = enriched["Game A"][0]

        self.assertEqual(item["data_backed"], True)
        self.assertEqual(item["confidence"], "High")
        self.assertEqual(item["strength_type"], "局部优势")
        self.assertEqual(
            item["how_to_apply"],
            "参考Game A在Core Delight上的产品设计（uplift=2.0x）",
        )
        self.assertEqual(item["why_valuable"], "更放松")
        self.assertEqual(item["pm_apply_tip"], "强化反馈节奏")


if __name__ == "__main__":
    unittest.main()
