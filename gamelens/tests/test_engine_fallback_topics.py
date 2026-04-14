# ============================================================
# Module: test_engine_fallback_topics.py
# ============================================================
# 【业务作用】验证规则主题 Fallback 在没有 LLM 主题时仍能输出可展示卡片
# 【上游】开发者或 CI 执行测试时运行
# 【下游】依赖 insights.engine.build_fallback_topics
# 【缺失影响】LLM 失败场景可能悄悄退化，直到页面空白时才被发现
# ============================================================

import unittest

from data.schema import ReviewSchema
from insights.engine import build_fallback_topics


# 这组测试守住的是“规则兜底必须可用”这条产品底线。
class BuildFallbackTopicsTests(unittest.TestCase):
    def test_builds_topic_cards_from_rule_stats(self):
        reviews = [
            ReviewSchema(
                review_id="1",
                game_name="Demo",
                country="us",
                rating=1,
                text="Too many ads and redirects after every level.",
                date="2026-01-01",
                vader_score=-0.7,
                vader_label="Negative",
                textblob_score=-0.4,
                textblob_label="Negative",
                rule_topics=["Ads"],
            ),
            ReviewSchema(
                review_id="2",
                game_name="Demo",
                country="us",
                rating=5,
                text="Fun and relaxing gameplay overall.",
                date="2026-01-02",
                vader_score=0.8,
                vader_label="Positive",
                textblob_score=0.5,
                textblob_label="Positive",
                rule_topics=["Positive"],
            ),
        ]

        topic_stats = {
            "Ads": {"count": 1, "ratio": 0.5, "avg_sentiment": -0.7, "low_star_ratio": 1.0},
            "Positive": {"count": 1, "ratio": 0.5, "avg_sentiment": 0.8, "low_star_ratio": 0.0},
        }

        cards = build_fallback_topics(reviews, topic_stats)

        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["topic_source"], "rule_fallback")
        self.assertEqual(cards[0]["sentiment"], "negative")
        self.assertIn("ads", " ".join(cards[0]["keywords"]).lower())


if __name__ == "__main__":
    unittest.main()
