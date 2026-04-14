# ============================================================
# Module: export.py
# ============================================================
# 【业务作用】预留导出能力，未来可将评论和洞察输出为可分享文档
# 【上游】页面按钮或外部脚本后续可调用
# 【下游】依赖 data.schema 中的评论和洞察结构
# 【缺失影响】当前主流程不受影响，但后续分享、存档、汇报能力会分散实现
# ============================================================

from typing import List
from data.schema import InsightSchema, ReviewSchema


# 保留这个接口，是为了把“结果分析”和“结果分发”解耦，
# 避免以后做 CSV 导出时把文件写入逻辑塞回页面层。
def export_reviews_csv(reviews: List[ReviewSchema], output_path: str) -> str:
    """导出评论明细为 CSV 文件。"""
    pass


# 洞察导出单独预留 Markdown 版本，是因为产品汇报更常见的是文档流转，
# 而不是直接传播原始 JSON 或表格。
def export_insights_md(insights: List[InsightSchema], output_path: str) -> str:
    """导出洞察报告为 Markdown 文件。"""
    pass
