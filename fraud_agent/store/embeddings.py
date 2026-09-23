"""384-d embeddings for TigerVector. Uses sentence-transformers/all-MiniLM-L6-v2 when installed, otherwise a
deterministic hashed bag-of-words projection so the pipeline still runs without the model download."""
from __future__ import annotations

import hashlib
import math
import re

DIM = 384
_model = None


def embed(text: str) -> list[float]:
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            _model = False
    if _model:
        return [float(x) for x in _model.encode(text or '', normalize_embeddings=True)]
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9$.]+", (text or '').lower()):
        for n in (tok, tok[:5]):  # word + prefix stem so 'device'/'devices' collide
            h = int(hashlib.md5(n.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 20) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]
