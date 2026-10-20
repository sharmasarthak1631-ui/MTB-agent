from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import numpy as np


@dataclass
class PriorTherapyLine:
    line_number: int
    regimen: str
    start_date: str
    end_date: str
    best_recist_response: str
    discontinuation_reason: str


@dataclass
class OrganFunctionLabs:
    egfr_ml_min: float
    serum_creatinine_mg_dl: float
    total_bilirubin_mg_dl: float
    ast_u_l: float
    alt_u_l: float
    anc_cells_ul: int
    platelets_k_ul: int
    hemoglobin_g_dl: float
    lvef_pct: float
    qtc_ms: int


@dataclass
class ClinicalExtractionReport:
    patient_id: str
    patient_age: int
    sex: str
    primary_tumor_site: str
    histological_diagnosis: str
    clinical_stage: str
    clinical_llm_backbone: str
    ecog_performance_status: int
    most_recent_recist: str
    prior_lines_of_therapy: List[PriorTherapyLine]
    labs: OrganFunctionLabs
    active_comorbidities_and_toxicities: List[str]
    extracted_clinical_narrative: str
    clinical_state_embedding: np.ndarray

    def to_summary_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("clinical_state_embedding", None)
        return d


class ClinicalReasoningSpecialist:
    def __init__(self, llm_backbone: str = "BioMistral-7B-Clinical", embedding_dim: int = 64):
        self.llm_backbone = llm_backbone
        self.embedding_dim = embedding_dim

    def extract_ehr(self, ehr_payload: bytes) -> ClinicalExtractionReport:
        text = ehr_payload.decode("utf-8", errors="replace").strip()
        data: Dict[str, Any] = {}
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except Exception:
                data = {}

        patient_id = data.get("patient_id", self._regex_extract(text, r"Patient ID:\s*([A-Za-z0-9\-]+)", "PT-MTB-001"))
        age = int(data.get("age", self._regex_extract(text, r"Age:\s*(\d+)", "62")))
        sex = data.get("sex", self._regex_extract(text, r"Sex:\s*([A-Za-z]+)", "Female"))
        tumor_site = data.get("primary_tumor_site", self._regex_extract(text, r"Primary Site:\s*([^\n]+)", "Right Colon"))
        diagnosis = data.get(
            "histological_diagnosis",
            self._regex_extract(text, r"Diagnosis:\s*([^\n]+)", "Colorectal Adenocarcinoma"),
        )
        stage = data.get("clinical_stage", self._regex_extract(text, r"Stage:\s*([^\n]+)", "Stage IVB (IVA/IVB Metastatic)"))
        ecog = int(data.get("ecog_performance_status", self._regex_extract(text, r"ECOG(?:\s*PS)?:\s*(\d)", "1")))
        recist = data.get("most_recent_recist", self._regex_extract(text, r"RECIST(?:\s*v1\.1)?:\s*([A-Z]{2})", "PD"))

        labs_dict = data.get("labs", {})
        egfr = float(labs_dict.get("egfr_ml_min", self._regex_extract(text, r"eGFR:\s*([\d\.]+)", "38.0")))
        creat = float(labs_dict.get("serum_creatinine_mg_dl", 1.65))
        bili = float(labs_dict.get("total_bilirubin_mg_dl", 0.8))
        ast = float(labs_dict.get("ast_u_l", 28.0))
        alt = float(labs_dict.get("alt_u_l", 31.0))
        anc = int(labs_dict.get("anc_cells_ul", 4200))
        plt = int(labs_dict.get("platelets_k_ul", 195))
        hgb = float(labs_dict.get("hemoglobin_g_dl", 11.4))
        lvef = float(labs_dict.get("lvef_pct", 58.0))
        qtc = int(labs_dict.get("qtc_ms", 432))

        labs = OrganFunctionLabs(
            egfr_ml_min=egfr,
            serum_creatinine_mg_dl=creat,
            total_bilirubin_mg_dl=bili,
            ast_u_l=ast,
            alt_u_l=alt,
            anc_cells_ul=anc,
            platelets_k_ul=plt,
            hemoglobin_g_dl=hgb,
            lvef_pct=lvef,
            qtc_ms=qtc,
        )

        prior_lines_raw = data.get("prior_lines_of_therapy", [])
        prior_lines: List[PriorTherapyLine] = []
        for idx, pl in enumerate(prior_lines_raw):
            prior_lines.append(
                PriorTherapyLine(
                    line_number=int(pl.get("line_number", idx + 1)),
                    regimen=pl.get("regimen", "FOLFOX + Bevacizumab"),
                    start_date=pl.get("start_date", "2025-11"),
                    end_date=pl.get("end_date", "2026-06"),
                    best_recist_response=pl.get("best_recist_response", "SD"),
                    discontinuation_reason=pl.get("discontinuation_reason", "Radiographic progression (RECIST PD)"),
                )
            )
        if not prior_lines:
            prior_lines.append(
                PriorTherapyLine(
                    line_number=1,
                    regimen="mFOLFOX6 + Bevacizumab",
                    start_date="2025-10",
                    end_date="2026-06",
                    best_recist_response="PR",
                    discontinuation_reason="Radiographic hepatic progression (RECIST PD)",
                )
            )

        comorbidities = data.get(
            "active_comorbidities_and_toxicities",
            ["Stage 3b Chronic Kidney Disease (eGFR 38 mL/min)", "Grade 1 Oxaliplatin-Induced Sensory Neuropathy"],
        )
        narrative = data.get(
            "clinical_note_summary",
            text if not text.startswith("{") else f"{age}yo {sex} with {stage} {diagnosis} ({tumor_site}), ECOG {ecog}, latest RECIST {recist}, eGFR {egfr} mL/min.",
        )

        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        vec[0] = float(ecog) / 4.0
        vec[1] = min(1.0, egfr / 100.0)
        vec[2] = 1.0 if recist == "PD" else 0.5
        vec[3] = float(len(prior_lines)) / 5.0
        vec[4] = 1.0 if egfr < 50.0 else 0.0
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm

        return ClinicalExtractionReport(
            patient_id=patient_id,
            patient_age=age,
            sex=sex,
            primary_tumor_site=tumor_site,
            histological_diagnosis=diagnosis,
            clinical_stage=stage,
            clinical_llm_backbone=self.llm_backbone,
            ecog_performance_status=ecog,
            most_recent_recist=recist,
            prior_lines_of_therapy=prior_lines,
            labs=labs,
            active_comorbidities_and_toxicities=comorbidities,
            extracted_clinical_narrative=narrative,
            clinical_state_embedding=vec,
        )

    @staticmethod
    def _regex_extract(text: str, pattern: str, default: str) -> str:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default
