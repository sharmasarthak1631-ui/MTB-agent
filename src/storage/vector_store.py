from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class VectorRecord:
    record_id: str
    namespace: str
    text_summary: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    record_id: str
    namespace: str
    cosine_similarity: float
    hnsw_layer_hops: int
    text_summary: str
    metadata: Dict[str, Any]


class DomainEmbeddingEncoder:
    ONCOLOGY_CONCEPTS = [
        "braf", "v600e", "mek", "egfr", "l858r", "t790m", "kras", "g12c",
        "msh2", "mlh1", "dmmr", "msi-h", "microsatellite", "tils", "lymphocyte",
        "pd-1", "pd-l1", "pembrolizumab", "nivolumab", "immunotherapy", "tmb",
        "brca1", "brca2", "palb2", "hrd", "parp", "olaparib", "niraparib",
        "pik3ca", "h1047r", "erbb2", "her2", "trastuzumab", "cetuximab",
        "encorafenib", "dabrafenib", "trametinib", "sotorasib", "adagrasib",
        "osimertinib", "amivantamab", "colorectal", "adenocarcinoma", "nsclc",
        "lung", "melanoma", "ovarian", "breast", "pancreatic", "metastatic",
        "refractory", "phase", "renal", "egfr_clearance", "neuropathy", "ecog",
        "recist", "progression", "resistance", "vus", "splice", "frameshift",
        "necrosis", "cellularity"
    ]

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str, boost_terms: Optional[List[str]] = None) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        lowered = text.lower()
        for idx, concept in enumerate(self.ONCOLOGY_CONCEPTS[: self.dim]):
            if concept.lower() in lowered:
                vec[idx] += 2.5
        if boost_terms:
            for term in boost_terms:
                t_low = term.lower()
                for idx, concept in enumerate(self.ONCOLOGY_CONCEPTS[: self.dim]):
                    if concept.lower() in t_low or t_low in concept.lower():
                        vec[idx] += 3.0
        tokens = [tok.strip(".,;:()[]") for tok in lowered.split() if len(tok) > 2]
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
            dim_idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[dim_idx] += sign * 0.35

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec


class PgVectorHNSWIndex:
    def __init__(self, dim: int = 64, m: int = 16, ef_construction: int = 100, ef_search: int = 50):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.encoder = DomainEmbeddingEncoder(dim=dim)
        self._records: Dict[str, VectorRecord] = {}
        self._layers: List[Dict[str, List[str]]] = [{}, {}, {}]
        self._entry_point: Optional[str] = None

    def get_pgvector_ddl(self) -> str:
        lines = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            "CREATE TABLE IF NOT EXISTS mtb_embeddings (",
            "    record_id VARCHAR(64) PRIMARY KEY,",
            "    namespace VARCHAR(32) NOT NULL,",
            "    text_summary TEXT NOT NULL,",
            "    metadata JSONB NOT NULL,",
            f"    embedding vector({self.dim}) NOT NULL",
            ");",
            "CREATE INDEX IF NOT EXISTS idx_mtb_embeddings_hnsw",
            "ON mtb_embeddings USING hnsw (embedding vector_cosine_ops)",
            f"WITH (m = {self.m}, ef_construction = {self.ef_construction});",
        ]
        return "\n".join(lines)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def upsert(
        self,
        record_id: str,
        namespace: str,
        text_summary: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        boost_terms: Optional[List[str]] = None,
    ) -> VectorRecord:
        if embedding is None:
            embedding = self.encoder.encode(text_summary, boost_terms=boost_terms)
        rec = VectorRecord(
            record_id=record_id,
            namespace=namespace,
            text_summary=text_summary,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._records[record_id] = rec

        h_val = int(hashlib.md5(record_id.encode("utf-8")).hexdigest()[:6], 16)
        max_layer = 2 if h_val % 9 == 0 else (1 if h_val % 3 == 0 else 0)

        for l_idx in range(max_layer + 1):
            layer_graph = self._layers[l_idx]
            if record_id not in layer_graph:
                layer_graph[record_id] = []
            candidates = [
                (other_id, self._cosine_sim(embedding, self._records[other_id].embedding))
                for other_id in layer_graph.keys()
                if other_id != record_id
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            neighbors = [cid for cid, _ in candidates[: self.m]]
            layer_graph[record_id] = neighbors
            for nb in neighbors:
                if record_id not in layer_graph[nb] and len(layer_graph[nb]) < self.m * 2:
                    layer_graph[nb].append(record_id)

        if self._entry_point is None or max_layer == 2:
            self._entry_point = record_id
        return rec

    def search(
        self,
        query_text: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
        boost_terms: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        q_vec = self.encoder.encode(query_text, boost_terms=boost_terms)
        return self.search_by_vector(q_vec, namespace=namespace, top_k=top_k)

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        namespace: Optional[str] = None,
        top_k: int = 5,
    ) -> List[VectorSearchResult]:
        if not self._records:
            return []

        visited = set()
        current_id = self._entry_point or next(iter(self._records.keys()))
        hops = 0

        for l_idx in range(len(self._layers) - 1, -1, -1):
            layer_graph = self._layers[l_idx]
            if current_id not in layer_graph and layer_graph:
                current_id = next(iter(layer_graph.keys()))
            improved = True
            while improved and current_id in layer_graph:
                improved = False
                curr_sim = self._cosine_sim(query_vec, self._records[current_id].embedding)
                for nb_id in layer_graph.get(current_id, []):
                    hops += 1
                    visited.add(nb_id)
                    nb_sim = self._cosine_sim(query_vec, self._records[nb_id].embedding)
                    if nb_sim > curr_sim + 1e-6:
                        current_id = nb_id
                        curr_sim = nb_sim
                        improved = True

        scored: List[VectorSearchResult] = []
        for rid, rec in self._records.items():
            if namespace and rec.namespace != namespace:
                continue
            sim = self._cosine_sim(query_vec, rec.embedding)
            scored.append(
                VectorSearchResult(
                    record_id=rid,
                    namespace=rec.namespace,
                    cosine_similarity=round(max(0.0, sim), 4),
                    hnsw_layer_hops=max(1, hops),
                    text_summary=rec.text_summary,
                    metadata=rec.metadata,
                )
            )
        scored.sort(key=lambda r: r.cosine_similarity, reverse=True)
        return scored[:top_k]
