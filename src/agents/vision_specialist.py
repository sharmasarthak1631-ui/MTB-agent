from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import numpy as np


@dataclass
class WSIPatchMetrics:
    patch_id: str
    magnification: str
    x_coord: int
    y_coord: int
    tumor_cellularity: float
    til_density_per_mm2: float
    msi_morphology_prob: float
    hrd_morphology_prob: float
    necrosis_ratio: float
    feature_vector: List[float] = field(default_factory=list)


@dataclass
class HistopathologyReport:
    slide_id: str
    backbone_model: str
    magnification_levels: List[str]
    total_patches_analyzed: int
    mean_tumor_cellularity_pct: float
    til_infiltration_score: str
    mean_til_density_mm2: float
    necrosis_area_pct: float
    msi_morphology_probability: float
    hrd_phenotype_probability: float
    morphological_phenotypes: List[str]
    pooled_slide_embedding: np.ndarray
    top_attention_patches: List[WSIPatchMetrics]

    def to_summary_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("pooled_slide_embedding", None)
        for p in d["top_attention_patches"]:
            p.pop("feature_vector", None)
        return d


class HistopathologyVisionSpecialist:
    def __init__(self, backbone_model: str = "UNI (ViT-L/16)", embedding_dim: int = 64):
        self.backbone_model = backbone_model
        self.embedding_dim = embedding_dim

    def analyze_slide(self, slide_payload: bytes, filename: str = "slide.svs") -> HistopathologyReport:
        meta: Dict[str, Any] = {}
        try:
            text = slide_payload.decode("utf-8").strip()
            if text.startswith("{"):
                meta = json.loads(text)
        except Exception:
            meta = {}

        slide_id = meta.get("slide_id", filename.replace(".svs", "").replace(".dicom", ""))
        base_cellularity = float(meta.get("tumor_cellularity_pct", 74.0))
        base_til = float(meta.get("til_density_mm2", 680.0))
        base_msi = float(meta.get("msi_morphology_prob", 0.84))
        base_hrd = float(meta.get("hrd_morphology_prob", 0.18))
        base_necrosis = float(meta.get("necrosis_pct", 11.5))
        phenotypes = meta.get(
            "morphological_phenotypes",
            [
                "Crohn-like peritumoral lymphoid aggregates",
                "High intratumoral lymphocyte infiltration (TIL-rich)",
                "Focal mucinous / medullary architectural pattern",
            ],
        )

        num_patches = int(meta.get("total_patches", 16))
        patches: List[WSIPatchMetrics] = []
        patch_vectors: List[np.ndarray] = []

        for idx in range(num_patches):
            seed_str = f"{slide_id}_patch_{idx}_{self.backbone_model}"
            h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(h)

            mag = "40x" if idx % 2 == 0 else "20x"
            cell_p = float(np.clip(base_cellularity / 100.0 + rng.normal(0, 0.06), 0.15, 0.98))
            til_p = float(np.clip(base_til + rng.normal(0, 55.0), 40.0, 1500.0))
            msi_p = float(np.clip(base_msi + rng.normal(0, 0.05), 0.02, 0.99))
            hrd_p = float(np.clip(base_hrd + rng.normal(0, 0.05), 0.02, 0.99))
            nec_p = float(np.clip(base_necrosis / 100.0 + rng.normal(0, 0.03), 0.01, 0.60))

            vec = rng.normal(0, 0.08, size=self.embedding_dim).astype(np.float32)
            vec[0] = cell_p
            vec[1] = min(1.0, til_p / 1000.0)
            vec[2] = msi_p
            vec[3] = hrd_p
            vec[4] = nec_p
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm

            patch_vectors.append(vec)
            patches.append(
                WSIPatchMetrics(
                    patch_id=f"P-{idx+1:03d}",
                    magnification=mag,
                    x_coord=(idx % 4) * 2048,
                    y_coord=(idx // 4) * 2048,
                    tumor_cellularity=round(cell_p * 100.0, 1),
                    til_density_per_mm2=round(til_p, 1),
                    msi_morphology_prob=round(msi_p, 3),
                    hrd_morphology_prob=round(hrd_p, 3),
                    necrosis_ratio=round(nec_p * 100.0, 1),
                    feature_vector=vec.tolist(),
                )
            )

        pooled = np.mean(np.stack(patch_vectors, axis=0), axis=0)
        norm_p = float(np.linalg.norm(pooled))
        if norm_p > 1e-8:
            pooled = pooled / norm_p

        mean_til = float(np.mean([p.til_density_per_mm2 for p in patches]))
        if mean_til >= 500.0:
            til_label = "High (Brisk)"
        elif mean_til >= 200.0:
            til_label = "Moderate"
        else:
            til_label = "Low (Sparse)"

        patches_sorted = sorted(
            patches,
            key=lambda p: (p.msi_morphology_prob + p.hrd_morphology_prob + p.til_density_per_mm2 / 1000.0),
            reverse=True,
        )

        return HistopathologyReport(
            slide_id=slide_id,
            backbone_model=self.backbone_model,
            magnification_levels=["20x", "40x"],
            total_patches_analyzed=num_patches,
            mean_tumor_cellularity_pct=round(float(np.mean([p.tumor_cellularity for p in patches])), 1),
            til_infiltration_score=til_label,
            mean_til_density_mm2=round(mean_til, 1),
            necrosis_area_pct=round(float(np.mean([p.necrosis_ratio for p in patches])), 1),
            msi_morphology_probability=round(float(np.mean([p.msi_morphology_prob for p in patches])), 3),
            hrd_phenotype_probability=round(float(np.mean([p.hrd_morphology_prob for p in patches])), 3),
            morphological_phenotypes=phenotypes,
            pooled_slide_embedding=pooled,
            top_attention_patches=patches_sorted[:6],
        )
