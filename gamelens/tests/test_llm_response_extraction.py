# ============================================================
# Module: test_llm_response_extraction.py
# ============================================================
# 【业务作用】验证响应抽取和空响应报错格式化逻辑
# 【上游】开发者或 CI 执行测试时运行
# 【下游】依赖 llm._base 的响应提取能力
# 【缺失影响】SDK 响应结构变化时，模型调用可能表面成功却拿不到正文
# ============================================================

import unittest
from types import SimpleNamespace

from llm._base import _extract_response_text, _format_llm_error


# 使用轻量假对象代替真实请求，是为了让基础测试保持稳定、快速和可离线执行。
class FakeResponse:
    def __init__(self, content=None, finish_reason="stop", refusal=None, content_filter_results=None):
        self.choices = [
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal, tool_calls=None),
            )
        ]
        self._content_filter_results = content_filter_results

    def model_dump(self):
        return {
            "choices": [
                {
                    "message": {"content": self.choices[0].message.content},
                    "content_filter_results": self._content_filter_results,
                }
            ]
        }


# 这组测试守住的是“模型回了内容，系统能不能正确拿到”这件最基础的事。
class ResponseExtractionTests(unittest.TestCase):
    def test_extracts_plain_string_content(self):
        text, debug = _extract_response_text(FakeResponse(content="hello"))
        self.assertEqual(text, "hello")
        self.assertIn("finish_reason", debug)

    def test_extracts_list_content_parts(self):
        content = [{"type": "output_text", "text": "hello"}, {"type": "output_text", "text": "world"}]
        text, _ = _extract_response_text(FakeResponse(content=content))
        self.assertEqual(text, "hello\nworld")

    def test_formats_empty_response_with_details(self):
        _, debug = _extract_response_text(
            FakeResponse(content=None, finish_reason="content_filter", content_filter_results={"hate": {"filtered": False}})
        )
        msg = _format_llm_error("azure", f"empty_response | {debug}")
        self.assertIn("empty_response", msg)
        self.assertIn("content_filter", msg)


if __name__ == "__main__":
    unittest.main()
