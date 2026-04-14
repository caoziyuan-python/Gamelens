# -*- coding: utf-8 -*-
# llm/rag.py
# RAG knowledge retrieval for GameLens.

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import AzureOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AZURE_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    EMBEDDING_DEPLOYMENT_NAME,
    RAG_EMBEDDING_PROVIDER,
    RAG_LOCAL_EMBED_MODEL,
    RAG_INDEX_PATH,
    RAG_KNOWLEDGE_DIR,
    RAG_MAX_CONTEXT,
    RAG_META_PATH,
    RAG_SIM_THRESHOLD,
    RAG_TOP_K,
)

QUERY_EXPANSION = {
    "广告": "广告 变现 插屏 激励式 频率 rewarded",
    "留存": "留存 流失 次日 D7 D30 retention churn",
    "评分": "评分 差评 App Store rating review",
    "付费": "付费 内购 订阅 变现 IAP purchase",
    "麻将": "麻将 mahjong 用户画像 场景 solitaire",
    "竞品": "竞品 对比 市场 行业 benchmark competitor",
    "难度": "难度 关卡 曲线 挫败 difficulty level",
    "用户": "用户 玩家 画像 场景 行为 user player",
}

_LOCAL_EMBEDDER = None
_LOCAL_EMBEDDER_READY = True
_AZURE_EMBEDDING_READY = True


def _get_embedding_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )


def _get_local_embedder():
    global _LOCAL_EMBEDDER, _LOCAL_EMBEDDER_READY
    if not _LOCAL_EMBEDDER_READY:
        return None
    if _LOCAL_EMBEDDER is not None:
        return _LOCAL_EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer

        _LOCAL_EMBEDDER = SentenceTransformer(RAG_LOCAL_EMBED_MODEL)
        return _LOCAL_EMBEDDER
    except Exception as exc:
        _LOCAL_EMBEDDER_READY = False
        print(f"[WARN] 本地Embedding模型加载失败：{exc}")
        return None


def _embedding_via_local(text: str) -> Optional[list[float]]:
    model = _get_local_embedder()
    if model is None:
        return None
    try:
        vec = model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()
    except Exception as exc:
        print(f"[WARN] 本地Embedding生成失败：{exc}")
        return None


def _embedding_via_azure(text: str) -> Optional[list[float]]:
    global _AZURE_EMBEDDING_READY
    if not _AZURE_EMBEDDING_READY:
        return None
    if not EMBEDDING_DEPLOYMENT_NAME:
        _AZURE_EMBEDDING_READY = False
        return None
    try:
        client = _get_embedding_client()
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_NAME,
            input=text[:8000],
        )
        return response.data[0].embedding
    except Exception as exc:
        _AZURE_EMBEDDING_READY = False
        print(f"[WARN] Azure Embedding生成失败：{exc}")
        return None


def _get_embedding(text: str) -> Optional[list[float]]:
    provider = str(RAG_EMBEDDING_PROVIDER or "local").lower()
    if provider == "local":
        return _embedding_via_local(text)
    if provider == "azure":
        vec = _embedding_via_azure(text)
        return vec if vec else _embedding_via_local(text)
    # auto/unknown
    vec = _embedding_via_local(text)
    if vec:
        return vec
    return _embedding_via_azure(text)


def expand_query(query: str) -> str:
    expanded = query
    for keyword, expansion in QUERY_EXPANSION.items():
        if keyword in query:
            expanded += " " + expansion
    return expanded


def _chunk_by_heading(text: str, source: str) -> list[dict]:
    chunks: list[dict] = []
    lines = text.split("\n")
    current_h1 = ""
    current_h2 = ""
    current_content: list[str] = []

    def flush_chunk() -> None:
        content = "\n".join(current_content).strip()
        if content and len(content) > 20:
            header = ""
            if current_h1:
                header += f"主题：{current_h1}\n"
            if current_h2:
                header += f"子主题：{current_h2}\n"
            chunks.append(
                {
                    "text": header + content,
                    "source": source,
                    "h1": current_h1,
                    "h2": current_h2,
                }
            )

    for line in lines:
        if line.startswith("## "):
            flush_chunk()
            current_h1 = line[3:].strip()
            current_h2 = ""
            current_content = []
        elif line.startswith("### "):
            flush_chunk()
            current_h2 = line[4:].strip()
            current_content = []
        else:
            if line.strip():
                current_content.append(line)

    flush_chunk()
    return chunks


def _chunk_with_overlap(text: str, source: str, size: int = 400, overlap: int = 80) -> list[dict]:
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            last_period = text.rfind("。", start, end)
            if last_period > start + size // 2:
                end = last_period + 1
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:
            chunks.append(
                {
                    "text": chunk,
                    "source": source,
                    "h1": "",
                    "h2": "",
                }
            )
        start += size - overlap
    return chunks


def smart_chunk(text: str, source: str) -> list[dict]:
    has_headings = "## " in text or "### " in text
    if has_headings:
        chunks = _chunk_by_heading(text, source)
        final_chunks: list[dict] = []
        for chunk in chunks:
            if len(chunk["text"]) > 600:
                final_chunks.extend(_chunk_with_overlap(chunk["text"], chunk["source"]))
            else:
                final_chunks.append(chunk)
        return final_chunks
    return _chunk_with_overlap(text, source)


def _save_meta(chunks: list[dict]) -> None:
    os.makedirs("cache", exist_ok=True)
    payload = {
        "built_at": datetime.now().isoformat(),
        "embedding_provider": RAG_EMBEDDING_PROVIDER,
        "local_embedding_model": RAG_LOCAL_EMBED_MODEL,
        "embedding_deployment": EMBEDDING_DEPLOYMENT_NAME,
        "sim_threshold": RAG_SIM_THRESHOLD,
        "top_k": RAG_TOP_K,
        "chunks": chunks,
    }
    with open(RAG_META_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_index() -> dict:
    knowledge_dir = Path(RAG_KNOWLEDGE_DIR)
    if not knowledge_dir.exists():
        print(f"[WARN] {RAG_KNOWLEDGE_DIR}/ 目录不存在")
        return {"chunks": [], "count": 0}

    all_chunks: list[dict] = []
    for md_file in knowledge_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            chunks = smart_chunk(content, md_file.name)
            all_chunks.extend(chunks)
            print(f"[OK] {md_file.name}：{len(chunks)}个块")
        except Exception as exc:
            print(f"[WARN] 读取{md_file.name}失败：{exc}")

    if not all_chunks:
        print("[WARN] 没有找到知识文件")
        return {"chunks": [], "count": 0}

    embeddings: list[list[float]] = []
    valid_chunks: list[dict] = []
    print(f"正在为{len(all_chunks)}个块生成向量...")
    for idx, chunk in enumerate(all_chunks):
        emb = _get_embedding(chunk["text"])
        if emb:
            embeddings.append(emb)
            valid_chunks.append(chunk)
        if (idx + 1) % 10 == 0:
            print(f"  进度：{idx + 1}/{len(all_chunks)}")

    if not embeddings:
        print("[WARN] 向量生成全部失败")
        return {"chunks": [], "count": 0}

    vectors = np.array(embeddings, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8

    _save_meta(valid_chunks)

    try:
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        faiss.write_index(index, RAG_INDEX_PATH)
        print(f"[OK] RAG索引构建完成（FAISS）：{len(valid_chunks)}个块")
    except Exception as exc:
        np.savez("cache/rag_index_fallback.npz", vectors=vectors)
        print(f"[WARN] FAISS不可用，已降级为numpy索引：{exc}")

    return {"chunks": valid_chunks, "count": len(valid_chunks)}


def index_game_reviews(game_name: str, reviews: list[str], index_dir: str = "indices") -> bool:
    try:
        target_dir = Path(index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        embeddings: list[list[float]] = []
        meta: list[dict] = []
        for review in (reviews or []):
            original_text = str(review or "").strip()
            if not original_text:
                continue
            document = f"[{game_name}] {original_text}"
            embedding = _get_embedding(document)
            if not embedding:
                continue
            embeddings.append(embedding)
            meta.append({"game": game_name, "text": original_text})

        if not embeddings:
            return False

        vectors = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(target_dir / f"{game_name}.faiss"))
        (target_dir / f"{game_name}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        print(f"[RAG] {game_name}索引构建失败: {e}")
        return False


def load_index():
    if not Path(RAG_META_PATH).exists():
        build_index()
    if not Path(RAG_META_PATH).exists():
        return None, None, None

    try:
        payload = json.loads(Path(RAG_META_PATH).read_text(encoding="utf-8"))
        chunks = payload.get("chunks", []) or []
    except Exception as exc:
        print(f"[WARN] 读取元数据失败：{exc}")
        return None, None, None

    try:
        if not Path(RAG_INDEX_PATH).exists():
            build_index()
        if Path(RAG_INDEX_PATH).exists():
            return "faiss", faiss.read_index(RAG_INDEX_PATH), chunks
    except Exception:
        pass

    fallback = Path("cache/rag_index_fallback.npz")
    if fallback.exists():
        try:
            data = np.load(fallback)
            return "numpy", data["vectors"], chunks
        except Exception as exc:
            print(f"[WARN] 读取numpy索引失败：{exc}")

    return None, None, None


def retrieve(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    try:
        backend, index_obj, meta = load_index()
        if backend is None or index_obj is None or not meta:
            return []

        query_emb = _get_embedding(expand_query(query))
        if not query_emb:
            return []

        query_vec = np.array([query_emb], dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8

        raw_k = max(top_k * 3, top_k)

        results: list[dict] = []
        if backend == "faiss":
            scores, indices = index_obj.search(query_vec, raw_k)
            pairs = zip(scores[0], indices[0])
        else:
            vectors = index_obj
            sims = np.dot(vectors, query_vec[0])
            idxs = np.argsort(sims)[-raw_k:][::-1]
            pairs = ((float(sims[idx]), int(idx)) for idx in idxs)

        for score, idx in pairs:
            if idx < 0 or idx >= len(meta):
                continue
            if float(score) < RAG_SIM_THRESHOLD:
                continue
            item = meta[idx]
            results.append(
                {
                    "text": item.get("text", ""),
                    "source": item.get("source", ""),
                    "h1": item.get("h1", ""),
                    "h2": item.get("h2", ""),
                    "similarity": round(float(score), 3),
                }
            )
            if len(results) >= top_k:
                break

        return results
    except Exception as exc:
        print(f"[WARN] RAG检索失败：{exc}")
        return []


def get_relevant_context(query: str) -> str:
    try:
        chunks = retrieve(query)
        if not chunks:
            return ""

        parts: list[str] = []
        total = 0
        for chunk in chunks:
            text = chunk["text"]
            if total + len(text) > RAG_MAX_CONTEXT:
                break
            label = chunk.get("h2") or chunk.get("h1") or chunk.get("source") or "knowledge"
            parts.append(f"[{label}]\n{text}")
            total += len(text)

        return "\n\n".join(parts)
    except Exception as exc:
        print(f"[WARN] get_relevant_context失败：{exc}")
        return ""


def build_review_knowledge(all_games_cache: dict) -> None:
    try:
        from insights.engine import normalize_result

        content = "# 游戏用户评论知识库\n\n（基于真实App Store评论分析生成）\n\n"
        for game_name, result in (all_games_cache or {}).items():
            r = normalize_result(result)
            reviews = r.get("reviews", []) or []
            stats = r.get("sentiment_stats", {}) or {}
            pos = float(stats.get("positive_ratio", 0.0)) * 100
            neg = float(stats.get("negative_ratio", stats.get("neg_ratio", 0.0)))
            if neg <= 1:
                neg *= 100

            content += f"## {game_name}\n\n"
            content += f"正面情感：{pos:.1f}%，负面情感：{neg:.1f}%，评论总数：{len(reviews)}\n\n"

            topics = r.get("topics", []) or []
            if topics:
                content += "### 用户核心主题\n"
                for topic in topics[:3]:
                    rep = str(topic.get("representative_review", "") or "")[:80]
                    content += (
                        f"- {topic.get('topic_name', '')}（{topic.get('sentiment', '')}）：{topic.get('core_demand', '')}\n"
                        f"  代表评论：「{rep}」\n"
                    )
                content += "\n"

            complaints = r.get("complaints", []) or []
            if complaints:
                content += "### 用户核心投诉\n"
                for complaint in complaints[:3]:
                    ratio = float(complaint.get("estimated_ratio", 0.0)) * 100
                    quote = str(complaint.get("typical_quote", "") or "")[:80]
                    content += (
                        f"- {complaint.get('complaint_type', '')}：{complaint.get('core_demand', '')}（影响约{ratio:.0f}%用户）\n"
                        f"  用户原话：「{quote}」\n"
                    )
                content += "\n"

        os.makedirs(RAG_KNOWLEDGE_DIR, exist_ok=True)
        Path(f"{RAG_KNOWLEDGE_DIR}/reviews_summary.md").write_text(content, encoding="utf-8")
        print("[OK] 评论知识库已生成：reviews_summary.md")
    except Exception as exc:
        print(f"[WARN] 评论知识库生成失败：{exc}")


if __name__ == "__main__":
    print("=== 测试RAG系统 ===")
    res = build_index()
    print(f"构建结果：{res.get('count', 0)}个块")
    for q in ["广告频率怎么优化", "麻将游戏用户画像", "留存率行业基准"]:
        items = retrieve(q)
        print(f"「{q}」-> {len(items)}个结果")
        if items:
            print(f"最相关：{items[0].get('h2') or items[0].get('h1')}（相似度{items[0].get('similarity')}）")
    context = get_relevant_context("激励式广告最佳实践")
    print(f"上下文长度：{len(context)}")
