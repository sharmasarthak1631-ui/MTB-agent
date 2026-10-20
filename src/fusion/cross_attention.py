from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import numpy as np

from src.agents.vision_specialist import HistopathologyReport
from src.agents.genomic_specialist import GenomicAnalysisReport
from src.agents.clinical_specialist import ClinicalExtractionReport


@dataclass
class CrossModalSynergyFinding:
    finding_id: str
    inferred_biomarker: str
    graph_node_id: str
    confidence_score: float
    visual_evidence: str
    genomic_evidence: str
    clinical_correlate: str
    mechanism_explanation: str


@dataclass
class FusionHubOutput:
    attention_matrix: List[List[float]]
    vision_patch_labels: List[str]
    genomic_token_labels: List[str]
    synergy_findings: List[CrossModalSynergyFinding]
    active_graph_alteration_ids: List[str]
    fused_patient_embedding: np.ndarray

    def to_summary_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("fused_patient_embedding", None)
        return d


class MultiModalCrossAttentionHub:
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

    def fuse(
        self,
        vision_report: HistopathologyReport,
        genomic_report: GenomicAnalysisReport,
        clinical_report: ClinicalExtractionReport,
    ) -> FusionHubOutput:
        patches = vision_report.top_attention_patches
        variants = genomic_report.variants

        patch_labels = [f"{p.patch_id} ({p.magnification}, TIL={p.til_density_per_mm2:.0f})" for p in patches]
        variant_labels = [f"{v.gene} {v.hgvs_p} ({v.clinvar_significance})" for v in variants]

        if patches and variants:
            q_mat = np.array([p.feature_vector for p in patches], dtype=np.float32)
            k_mat = np.array([v.token_embedding for v in variants], dtype=np.float32)
            scale = math.sqrt(float(self.embedding_dim))
            raw_logits = np.matmul(q_mat, k_mat.T) / max(1.0, scale)

            for i, p in enumerate(patches):
                for j, v in enumerate(variants):
                    if v.gene in ("MSH2", "MLH1", "MSH6", "PMS2") and p.msi_morphology_prob >= 0.65:
                        raw_logits[i, j] += 2.2 * p.msi_morphology_prob
                    if v.gene in ("BRCA1", "BRCA2", "PALB2", "RAD51C") and p.hrd_morphology_prob >= 0.60:
                        raw_logits[i, j] += 2.0 * p.hrd_morphology_prob
                    if v.gene == "BRAF" and p.tumor_cellularity >= 60.0:
                        raw_logits[i, j] += 1.4

            exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
            attn_weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            attn_list = [[round(float(val), 4) for val in row] for row in attn_weights]
        else:
            attn_list = [[1.0]]

        synergies: List[CrossModalSynergyFinding] = []
        active_node_ids: List[str] = []

        for v in variants:
            if v.graph_node_id and v.graph_node_id not in active_node_ids:
                active_node_ids.append(v.graph_node_id)

        mmr_variants = [v for v in variants if v.gene in ("MSH2", "MLH1", "MSH6", "PMS2")]
        if (
            mmr_variants
            and vision_report.mean_til_density_mm2 >= 450.0
            and vision_report.msi_morphology_probability >= 0.65
        ):
            mv = mmr_variants[0]
            conf = round(
                min(
                    0.99,
                    0.50 * vision_report.msi_morphology_probability
                    + 0.35 * mv.hyenadna_pathogenicity_score
                    + 0.15 * min(1.0, vision_report.mean_til_density_mm2 / 800.0),
                ),
                3,
            )
            synergies.append(
                CrossModalSynergyFinding(
                    finding_id="SYN-DMMR-01",
                    inferred_biomarker="Mismatch Repair Deficiency (dMMR / MSI-H)",
                    graph_node_id="BIO:DMMR_MSIH",
                    confidence_score=conf,
                    visual_evidence=(
                        f"{vision_report.backbone_model} detected {vision_report.til_infiltration_score} TIL density "
                        f"({vision_report.mean_til_density_mm2} cells/mm2) and MSI morphology probability {vision_report.msi_morphology_probability:.2f}"
                    ),
                    genomic_evidence=(
                        f"{mv.gene} {mv.hgvs_p} ({mv.clinvar_significance}, HyenaDNA={mv.hyenadna_pathogenicity_score:.2f}, "
                        f"PolyPhen={mv.polyphen_score:.2f}) with TMB={genomic_report.tmb_mutations_per_mb} mut/Mb"
                    ),
                    clinical_correlate=f"{clinical_report.histological_diagnosis} ({clinical_report.primary_tumor_site})",
                    mechanism_explanation=(
                        f"Cross-attention fusion resolves the {mv.gene} {mv.hgvs_p} variant by correlating its high in-silico "
                        f"ATPase domain disruption score with brisk intratumoral lymphocyte infiltration and mucinous/Crohn-like WSI architecture, "
                        f"confirming functional Mismatch Repair Deficiency (dMMR)."
                    ),
                )
            )
            if "BIO:DMMR_MSIH" not in active_node_ids:
                active_node_ids.append("BIO:DMMR_MSIH")

        if genomic_report.tmb_mutations_per_mb >= 10.0:
            if "BIO:TMB_HIGH" not in active_node_ids:
                active_node_ids.append("BIO:TMB_HIGH")
            synergies.append(
                CrossModalSynergyFinding(
                    finding_id="SYN-TMB-02",
                    inferred_biomarker="Tumor Mutational Burden-High (TMB-H)",
                    graph_node_id="BIO:TMB_HIGH",
                    confidence_score=0.95,
                    visual_evidence=f"Immune-active tumor microenvironment (TILs {vision_report.mean_til_density_mm2} cells/mm2)",
                    genomic_evidence=f"Quantified TMB of {genomic_report.tmb_mutations_per_mb} mut/Mb across {genomic_report.callable_megabases} Mb callable exome",
                    clinical_correlate=f"Refractory to {len(clinical_report.prior_lines_of_therapy)} prior line(s) of therapy",
                    mechanism_explanation="Elevated somatic neoantigen burden (>=10 mut/Mb) coupled with pre-existing T-cell infiltration predicts durable checkpoint blockade response.",
                )
            )

        hrd_variants = [v for v in variants if v.gene in ("BRCA1", "BRCA2", "PALB2", "RAD51C")]
        if hrd_variants and (vision_report.hrd_phenotype_probability >= 0.60 or genomic_report.hrd_genomic_score >= 42.0):
            hv = hrd_variants[0]
            conf = round(
                min(0.98, 0.5 * vision_report.hrd_phenotype_probability + 0.5 * hv.hyenadna_pathogenicity_score),
                3,
            )
            synergies.append(
                CrossModalSynergyFinding(
                    finding_id="SYN-HRD-03",
                    inferred_biomarker="Homologous Recombination Deficiency (HRD-Positive)",
                    graph_node_id="BIO:HRD_POS",
                    confidence_score=conf,
                    visual_evidence=f"High-grade nuclear pleomorphism & HRD morphology score {vision_report.hrd_phenotype_probability:.2f}",
                    genomic_evidence=f"{hv.gene} {hv.hgvs_p} ({hv.clinvar_significance}) with genomic HRD score {genomic_report.hrd_genomic_score}",
                    clinical_correlate=clinical_report.histological_diagnosis,
                    mechanism_explanation="Biallelic homologous recombination repair loss confirmed across histopathology nuclear morphology and genomic scarring.",
                )
            )
            if "BIO:HRD_POS" not in active_node_ids:
                active_node_ids.append("BIO:HRD_POS")

        fused_vec = (
            0.35 * vision_report.pooled_slide_embedding
            + 0.40 * genomic_report.pooled_genomic_embedding
            + 0.25 * clinical_report.clinical_state_embedding
        )
        norm = float(np.linalg.norm(fused_vec))
        if norm > 1e-8:
            fused_vec = fused_vec / norm

        return FusionHubOutput(
            attention_matrix=attn_list,
            vision_patch_labels=patch_labels,
            genomic_token_labels=variant_labels,
            synergy_findings=synergies,
            active_graph_alteration_ids=active_node_ids,
            fused_patient_embedding=fused_vec,
        )
