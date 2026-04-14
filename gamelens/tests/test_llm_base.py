# ============================================================
# Module: test_llm_base.py
# ============================================================
# 【业务作用】验证底层 JSON 解析器能从常见脏输出里恢复结构化数组
# 【上游】开发者或 CI 执行测试时运行
# 【下游】依赖 llm._base.parse_json_response
# 【缺失影响】模型输出稍带说明文字时，整条 LLM 链路都可能被误判失败
# ============================================================

import unittest

from llm._base import parse_json_response


# 这组测试守住的是“格式噪声不应轻易毁掉分析结果”的底线。
class ParseJsonResponseTests(unittest.TestCase):
    def test_parses_plain_json_array(self):
        parsed = parse_json_response('[{"topic_name":"Ads"}]')
        self.assertEqual(parsed, [{"topic_name": "Ads"}])

    def test_parses_fenced_json_array(self):
        parsed = parse_json_response('```json\n[{"topic_name":"Ads"}]\n```')
        self.assertEqual(parsed, [{"topic_name": "Ads"}])

    def test_parses_nested_object_payload(self):
        parsed = parse_json_response(
            'Here is the result:\n{"payload":{"results":[{"complaint_type":"Ads"}]}}\nThanks'
        )
        self.assertEqual(parsed, [{"complaint_type": "Ads"}])

    def test_parses_python_like_list_with_booleans(self):
        parsed = parse_json_response(
            "Summary: [{'problem': 'Ads', 'ok': true}, {'problem': 'UX', 'ok': false}]"
        )
        self.assertEqual(
            parsed,
            [{"problem": "Ads", "ok": True}, {"problem": "UX", "ok": False}],
        )


if __name__ == "__main__":
    unittest.main()
