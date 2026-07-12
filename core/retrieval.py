"""
retrieval.py — BM25 index, hybrid dense+sparse retrieval (RRF fusion),
cross-encoder reranker.
Corresponds to notebook Cell 3.
"""

from .config import re, BM25Okapi, requests, HF_TOKEN
from .store import collection, emb_model, Q_PREFIX

# ---------- BM25 ----------

def bm25_tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())

data = collection.get(include=["documents"])
chunk_ids, docs = data["ids"], data["documents"]
doc_lookup = dict(zip(chunk_ids, docs))

if docs:
    bm25 = BM25Okapi([bm25_tok(d) for d in docs])
    print(f"BM25 built: {len(docs)} chunks")
else:
    bm25 = None
    print("BM25 skipped: corpus empty (will rebuild after ingestion)")

def rebuild_bm25():
    global chunk_ids, docs, doc_lookup, bm25
    data = collection.get(include=["documents"])
    chunk_ids, docs = data["ids"], data["documents"]
    doc_lookup = dict(zip(chunk_ids, docs))
    bm25 = BM25Okapi([bm25_tok(d) for d in docs])
    print(f"BM25 rebuilt: {len(docs)} chunks")

# ---------- Hybrid Retrieval (Dense + BM25 + RRF) ----------

def hybrid_retrieve(q, top_k=5, k_each=20):
    qv = emb_model.encode(Q_PREFIX + q, normalize_embeddings=True)
    dense = collection.query(
        query_embeddings=[qv.tolist()], n_results=k_each)["ids"][0]
    scores = bm25.get_scores(bm25_tok(q))
    sparse = [chunk_ids[i] for i in
              sorted(range(len(scores)), key=lambda i: -scores[i])[:k_each]]
    K, fused = 60, {}
    for lst in (dense, sparse):
        for r, cid in enumerate(lst):
            fused[cid] = fused.get(cid, 0) + 1/(K + r + 1)
    return sorted(fused, key=fused.get, reverse=True)[:top_k]

# ---------- Reranker ----------

# HF Inference API wrapper — same interface as CrossEncoder so every
# reranker.predict() call site in retrieval.py and agents.py stays unchanged.
class HFReranker:
    """
    Wraps the HF Inference API Text Ranking endpoint for
    BAAI/bge-reranker-base. Returns a list[float] of relevance scores
    identical in meaning to CrossEncoder.predict() output.
    """
    _MODEL = "BAAI/bge-reranker-base"
    _URL   = f"https://api-inference.huggingface.co/models/{_MODEL}"

    def __init__(self):
        self._headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    def predict(self, pairs, **kwargs):
        """
        pairs: list of [query, text] or (query, text) tuples
        Returns: list[float] — one score per pair, same as CrossEncoder.predict()
        """
        import time as _time, numpy as _np
        scores = []
        for query, text in pairs:
            for attempt in range(2):
                resp = requests.post(
                    self._URL,
                    headers=self._headers,
                    json={"inputs": {"source_sentence": query,
                                     "sentences": [text]},
                          "options": {"wait_for_model": True}},
                    timeout=30,
                )
                if resp.status_code == 503 and attempt == 0:
                    _time.sleep(20)
                    continue
                resp.raise_for_status()
                result = resp.json()
                # API returns list[float] with one score per sentence
                score = result[0] if isinstance(result, list) else result
                scores.append(float(score))
                break
        return scores


reranker = HFReranker()
print("reranker ready ✓ (HF Inference API)")

print("retrieval stack ready ✓")