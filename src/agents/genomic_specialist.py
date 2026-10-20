from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class AnnotatedVariant:
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    hgvs_p: str
    hgvs_c: str
    consequence: str
    variant_type: str
    vaf: float
    read_depth: int
    clinvar_significance: str
    sift_score: float
    polyphen_score: float
    hyenadna_pathogenicity_score: float
    resolved_classification: str
    graph_node_id: Optional[str]
    token_embedding: List[float] = field(default_factory=list)


@dataclass
class GenomicAnalysisReport:
    sample_id: str
    genomic_model: str
    callable_megabases: float
    total_somatic_variants: int
    tmb_mutations_per_mb: float
    tmb_status: str
    msi_genomic_status: str
    hrd_genomic_score: float
    variants: List[AnnotatedVariant]
    unresolved_vus_list: List[AnnotatedVariant]
    pooled_genomic_embedding: np.ndarray

    def to_summary_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("pooled_genomic_embedding", None)
        for v in d["variants"]:
            v.pop("token_embedding", None)
        for v in d["unresolved_vus_list"]:
            v.pop("token_embedding", None)
        return d


class GenomicVariantSpecialist:
    KNOWN_GRAPH_NODE_MAP = {
        ("BRAF", "p.V600E"): "MUT:BRAF_V600E",
        ("MSH2", "p.A636P"): "MUT:MSH2_A636P",
        ("EGFR", "p.L858R"): "MUT:EGFR_L858R",
        ("EGFR", "p.T790M"): "MUT:EGFR_T790M",
        ("KRAS", "p.G12C"): "MUT:KRAS_G12C",
        ("BRCA2", "p.E1953*"): "MUT:BRCA2_E1953X",
        ("PIK3CA", "p.H1047R"): "MUT:PIK3CA_H1047R",
    }

    def __init__(self, genomic_model: str = "HyenaDNA-1M", embedding_dim: int = 64):
        self.genomic_model = genomic_model
        self.embedding_dim = embedding_dim

    def analyze_vcf(self, vcf_bytes: bytes, sample_id: str = "NGS-SAMPLE-01") -> GenomicAnalysisReport:
        text = vcf_bytes.decode("utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        callable_mb = 32.5
        somatic_coding_count = 0
        msi_status = "MSS / Ambiguous"
        hrd_score = 18.0
        variants: List[AnnotatedVariant] = []
        vus_list: List[AnnotatedVariant] = []
        embeddings: List[np.ndarray] = []

        for ln in lines:
            if ln.startswith("##callable_mb="):
                try:
                    callable_mb = float(ln.split("=")[1])
                except ValueError:
                    pass
                continue
            if ln.startswith("##background_somatic_mutations="):
                try:
                    somatic_coding_count += int(ln.split("=")[1])
                except ValueError:
                    pass
                continue
            if ln.startswith("##msi_genomic_status="):
                msi_status = ln.split("=", 1)[1].strip()
                continue
            if ln.startswith("##hrd_genomic_score="):
                try:
                    hrd_score = float(ln.split("=")[1])
                except ValueError:
                    pass
                continue
            if ln.startswith("#"):
                continue

            parts = ln.split("\t")
            if len(parts) < 8:
                parts = ln.split()
            if len(parts) < 8:
                continue

            chrom, pos_s, _vid, ref, alt, _qual, _flt, info_str = parts[:8]
            pos = int(pos_s)
            info_map: Dict[str, str] = {}
            for item in info_str.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    info_map[k.strip()] = v.strip()

            gene = info_map.get("GENE", "UNKNOWN")
            hgvs_p = info_map.get("HGVSp", "p.?")
            hgvs_c = info_map.get("HGVSc", "c.?")
            consequence = info_map.get("CSQ", "missense_variant")
            vaf = float(info_map.get("AF", "0.35"))
            dp = int(info_map.get("DP", "450"))
            clinvar = info_map.get("CLINVAR", "VUS")
            sift = float(info_map.get("SIFT", "0.02"))
            polyphen = float(info_map.get("POLYPHEN", "0.91"))
            vtype = info_map.get("VTYPE", "SNV")

            seed = f"{gene}_{hgvs_p}_{ref}_{alt}_{self.genomic_model}"
            h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(h)
            hyena_score = float(
                np.clip(0.55 * polyphen + 0.35 * (1.0 - sift) + float(rng.uniform(0.02, 0.10)), 0.01, 0.99)
            )
            if "HYENA" in info_map:
                hyena_score = float(info_map["HYENA"])

            if clinvar in ("Pathogenic", "Likely Pathogenic"):
                resolved = clinvar
            elif clinvar == "VUS" and hyena_score >= 0.78:
                resolved = "VUS - High In-Silico Deleteriousness (Candidate for Multi-Modal Fusion)"
            else:
                resolved = clinvar

            node_id = self.KNOWN_GRAPH_NODE_MAP.get((gene, hgvs_p))

            vec = rng.normal(0, 0.08, size=self.embedding_dim).astype(np.float32)
            vec[0] = vaf
            vec[1] = polyphen
            vec[2] = 1.0 - sift
            vec[3] = hyena_score
            vec[4] = 1.0 if gene in ("MSH2", "MLH1") else 0.0
            vec[5] = 1.0 if gene in ("BRCA1", "BRCA2", "PALB2") else 0.0
            vec[6] = 1.0 if gene == "BRAF" else 0.0
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm

            embeddings.append(vec)
            somatic_coding_count += 1

            avar = AnnotatedVariant(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                gene=gene,
                hgvs_p=hgvs_p,
                hgvs_c=hgvs_c,
                consequence=consequence,
                variant_type=vtype,
                vaf=round(vaf, 3),
                read_depth=dp,
                clinvar_significance=clinvar,
                sift_score=round(sift, 3),
                polyphen_score=round(polyphen, 3),
                hyenadna_pathogenicity_score=round(hyena_score, 3),
                resolved_classification=resolved,
                graph_node_id=node_id,
                token_embedding=vec.tolist(),
            )
            variants.append(avar)
            if clinvar == "VUS":
                vus_list.append(avar)

        tmb = round(somatic_coding_count / max(1.0, callable_mb), 2)
        tmb_status = "TMB-High (>=10 mut/Mb)" if tmb >= 10.0 else "TMB-Low (<10 mut/Mb)"

        if embeddings:
            pooled = np.mean(np.stack(embeddings, axis=0), axis=0)
            norm = float(np.linalg.norm(pooled))
            if norm > 1e-8:
                pooled = pooled / norm
        else:
            pooled = np.zeros(self.embedding_dim, dtype=np.float32)
            pooled[0] = 1.0

        return GenomicAnalysisReport(
            sample_id=sample_id,
            genomic_model=self.genomic_model,
            callable_megabases=callable_mb,
            total_somatic_variants=somatic_coding_count,
            tmb_mutations_per_mb=tmb,
            tmb_status=tmb_status,
            msi_genomic_status=msi_status,
            hrd_genomic_score=hrd_score,
            variants=variants,
            unresolved_vus_list=vus_list,
            pooled_genomic_embedding=pooled,
        )
