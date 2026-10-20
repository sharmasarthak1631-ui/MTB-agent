from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.coordinator import (
    FederatedMTBCoordinator,
    MTBPipelineExecutionResult,
    SAMPLE_CLINICAL_CASES,
)


def apply_theme_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }

        h1 {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.15rem !important;
            color: #0f172a;
        }

        h2, h3 {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
            color: #1e293b;
        }

        .patient-banner {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }

        .patient-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .patient-subtitle {
            font-size: 0.9rem;
            color: #475569;
            margin-bottom: 0.75rem;
        }

        .pill-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .pill {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
        }

        .pill-target {
            background-color: #eff6ff;
            border-color: #bfdbfe;
            color: #1d4ed8;
        }

        .pill-warning {
            background-color: #fffbeb;
            border-color: #fde68a;
            color: #92400e;
        }

        .pill-success {
            background-color: #f0fdf4;
            border-color: #bbf7d0;
            color: #166534;
        }

        .action-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }

        .action-card-warning {
            background-color: #fffaf0;
            border: 1px solid #fed7aa;
            border-left: 4px solid #ea580c;
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.75rem;
        }

        .card-heading {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }

        .card-subheading {
            font-size: 0.82rem;
            color: #64748b;
            margin-bottom: 0.4rem;
        }

        .card-detail {
            font-size: 0.88rem;
            color: #334155;
            line-height: 1.4;
        }

        .match-badge {
            float: right;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background-color: #dbeafe;
            color: #1e40af;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: #0f172a;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.04em;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_case_inputs(
    selected_case_key: str,
    input_mode: str,
    uploaded_wsi: Any,
    uploaded_vcf: Any,
    uploaded_ehr: Any,
    override_egfr: float,
    override_ecog: int,
) -> Dict[str, Any]:
    base_case = SAMPLE_CLINICAL_CASES[selected_case_key]

    if input_mode == "Upload Custom Files" and (uploaded_wsi or uploaded_vcf or uploaded_ehr):
        wsi_bytes = uploaded_wsi.read() if uploaded_wsi else base_case["wsi_payload"].encode("utf-8")
        vcf_bytes = uploaded_vcf.read() if uploaded_vcf else base_case["vcf_payload"].encode("utf-8")
        ehr_bytes = uploaded_ehr.read() if uploaded_ehr else base_case["ehr_payload"].encode("utf-8")
        wsi_name = uploaded_wsi.name if uploaded_wsi else base_case["wsi_filename"]
        vcf_name = uploaded_vcf.name if uploaded_vcf else base_case["vcf_filename"]
        ehr_name = uploaded_ehr.name if uploaded_ehr else base_case["ehr_filename"]
    else:
        wsi_bytes = base_case["wsi_payload"].encode("utf-8")
        vcf_bytes = base_case["vcf_payload"].encode("utf-8")
        ehr_bytes = base_case["ehr_payload"].encode("utf-8")
        wsi_name = base_case["wsi_filename"]
        vcf_name = base_case["vcf_filename"]
        ehr_name = base_case["ehr_filename"]

    try:
        ehr_obj = json.loads(ehr_bytes.decode("utf-8"))
        ehr_obj["ecog_performance_status"] = int(override_ecog)
        if "labs" not in ehr_obj:
            ehr_obj["labs"] = {}
        ehr_obj["labs"]["egfr_ml_min"] = float(override_egfr)
        ehr_bytes = json.dumps(ehr_obj, indent=2).encode("utf-8")
    except Exception:
        pass

    return {
        "patient_id": base_case["patient_id"],
        "wsi_bytes": wsi_bytes,
        "vcf_bytes": vcf_bytes,
        "ehr_bytes": ehr_bytes,
        "wsi_filename": wsi_name,
        "vcf_filename": vcf_name,
        "ehr_filename": ehr_name,
    }


def render_app() -> None:
    st.set_page_config(
        page_title="Molecular Tumor Board Copilot",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme_styles()

    st.title("Molecular Tumor Board Copilot")
    st.caption("AI-Assisted Precision Oncology Decision Support")

    with st.sidebar:
        st.subheader("Select Patient Case")
        case_keys = list(SAMPLE_CLINICAL_CASES.keys())
        selected_case_key = st.selectbox("Cohort Profile", options=case_keys, index=0)
        default_ehr = json.loads(SAMPLE_CLINICAL_CASES[selected_case_key]["ehr_payload"])
        default_egfr = float(default_ehr.get("labs", {}).get("egfr_ml_min", 38.0))
        default_ecog = int(default_ehr.get("ecog_performance_status", 1))

        input_mode = st.radio("Source Mode", options=["Cohort Example", "Upload Custom Files"], horizontal=True)
        uploaded_wsi = None
        uploaded_vcf = None
        uploaded_ehr = None
        if input_mode == "Upload Custom Files":
            uploaded_wsi = st.file_uploader("Histology Slides (.svs, .json)", type=["svs", "dicom", "json"])
            uploaded_vcf = st.file_uploader("Genomic Variants (.vcf)", type=["vcf", "txt"])
            uploaded_ehr = st.file_uploader("EHR Summary (.json)", type=["json", "txt"])

        st.subheader("Simulate Clinical Changes")
        override_egfr = st.slider(
            "Kidney Function (eGFR mL/min)",
            min_value=10.0,
            max_value=110.0,
            value=default_egfr,
            step=1.0,
            help="Adjusting eGFR tests real-time trial pruning and drug safety checks.",
        )
        override_ecog = st.slider(
            "Performance Status (ECOG 0-4)",
            min_value=0,
            max_value=4,
            value=default_ecog,
            step=1,
            help="0 is fully active; 4 is bedbound.",
        )

        with st.expander("AI Model Backbones", expanded=False):
            vision_backbone = st.selectbox("Vision Encoder", ["UNI (ViT-L/16)", "Prov-GigaPath", "CONCH"], index=0)
            genomic_backbone = st.selectbox("Genomic Model", ["HyenaDNA-1M", "DNABERT-2"], index=0)
            clinical_backbone = st.selectbox("Clinical LLM", ["BioMistral-7B-Clinical", "Llama-3-Med-70B"], index=0)

        st.button("Update Analysis", type="primary", use_container_width=True)

    coordinator = FederatedMTBCoordinator(
        vision_backbone=vision_backbone,
        genomic_backbone=genomic_backbone,
        clinical_backbone=clinical_backbone,
    )
    inputs = build_case_inputs(
        selected_case_key=selected_case_key,
        input_mode=input_mode,
        uploaded_wsi=uploaded_wsi,
        uploaded_vcf=uploaded_vcf,
        uploaded_ehr=uploaded_ehr,
        override_egfr=override_egfr,
        override_ecog=override_ecog,
    )
    result: MTBPipelineExecutionResult = coordinator.execute_patient_workflow(**inputs)

    renal_status = "Normal Renal Clearance" if result.clinical_report.labs.egfr_ml_min >= 60 else "Moderate Renal Impairment"
    target_names = [b["Biomarker / Alteration"].split(" ")[0] for b in result.dossier.actionable_biomarker_summary[:3]]
    targets_str = ", ".join(target_names) if target_names else "Standard Profile"

    st.markdown(
        f"""
        <div class="patient-banner">
            <div class="patient-title">Patient {result.clinical_report.patient_id} • {result.clinical_report.patient_age}y {result.clinical_report.sex}</div>
            <div class="patient-subtitle">{result.clinical_report.clinical_stage} {result.clinical_report.histological_diagnosis} ({result.clinical_report.primary_tumor_site})</div>
            <div class="pill-container">
                <span class="pill pill-target">Targets: {targets_str}</span>
                <span class="pill">Response: {result.clinical_report.most_recent_recist} (Progression)</span>
                <span class="pill {'pill-warning' if result.clinical_report.labs.egfr_ml_min < 50 else 'pill-success'}">eGFR {result.clinical_report.labs.egfr_ml_min} mL/min ({renal_status})</span>
                <span class="pill">ECOG PS: {result.clinical_report.ecog_performance_status}</span>
                <span class="pill pill-success">Safety Gate: {('VERIFIED' if result.dossier.verification_passed else 'ATTENTION')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_action, tab_trials, tab_evidence, tab_fusion, tab_system = st.tabs(
        [
            "Treatment Plan",
            "Clinical Trials",
            "Patient Evidence",
            "AI Multi-Modal Fusion",
            "Verification & Audit",
        ]
    )

    with tab_action:
        col_rec, col_res = st.columns([3, 2])

        with col_rec:
            st.subheader("Recommended Therapies")
            if result.dossier.verified_therapy_recommendations:
                for r in result.dossier.verified_therapy_recommendations:
                    ev_badge = r["Evidence"].split(" ")[0]
                    dose_guidance = r["Dosing & Organ Guidance"].split("|")[-1].strip()
                    st.markdown(
                        f"""
                        <div class="action-card">
                            <span class="match-badge">{ev_badge}</span>
                            <div class="card-heading">{r['Regimen']}</div>
                            <div class="card-subheading">Target: <b>{r['Driver']}</b> • {r['Class']}</div>
                            <div class="card-detail">{dose_guidance}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No targeted therapies currently qualify under current kidney clearance parameters.")

        with col_res:
            st.subheader("Therapies to Avoid")
            if result.dossier.resistance_contraindication_warnings:
                for w in result.dossier.resistance_contraindication_warnings:
                    reason = w["Mechanism of Resistance"].split(".")[0]
                    st.markdown(
                        f"""
                        <div class="action-card-warning">
                            <div class="card-heading">Avoid {w['Avoided Regimen']}</div>
                            <div class="card-subheading">Resistance Driver: <b>{w['Driver']}</b></div>
                            <div class="card-detail">{reason}.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No active resistance flags detected.")

        st.subheader("Identified Actionable Biomarkers")
        clean_bio = [
            {
                "Biomarker": b["Biomarker / Alteration"],
                "Origin": b["Modality"],
                "Finding": b["Metric / VAF"],
                "Clinical Impact": b["Classification"].split(".")[0],
            }
            for b in result.dossier.actionable_biomarker_summary
        ]
        st.dataframe(pd.DataFrame(clean_bio), use_container_width=True, hide_index=True)

        with st.expander("Full Clinical Narrative", expanded=False):
            st.write(result.dossier.executive_clinical_summary)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Treatment Summary (Markdown)",
                data=result.dossier.markdown_export,
                file_name=f"{result.patient_id}_mtb_dossier.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "Export Full Record (JSON)",
                data=json.dumps(result.dossier.to_summary_dict(), indent=2),
                file_name=f"{result.patient_id}_mtb_dossier.json",
                mime="application/json",
                use_container_width=True,
            )

    with tab_trials:
        m_t1, m_t2, m_t3 = st.columns(3)
        m_t1.metric("Eligible Trials", len(result.graphrag_output.eligible_trials))
        m_t2.metric("Excluded Trials", len(result.graphrag_output.pruned_trials), help="Excluded automatically by safety rules (e.g. kidney function)")
        m_t3.metric("Top Semantic Match", f"{result.graphrag_output.eligible_trials[0].vector_similarity_score * 100:.0f}%" if result.graphrag_output.eligible_trials else "N/A")

        st.subheader("Recommended Eligible Trials")
        if result.dossier.eligible_clinical_trials:
            for t in result.dossier.eligible_clinical_trials:
                sim_pct = int(t["Vector Sim"] * 100)
                st.markdown(
                    f"""
                    <div class="action-card">
                        <span class="match-badge">{sim_pct}% Match</span>
                        <div class="card-heading">{t['NCT ID']} • {t['Phase']}</div>
                        <div class="card-subheading"><b>{t['Regimen']}</b> • Target: {t['Driver']}</div>
                        <div class="card-detail">{t['Title']}</div>
                        <div style="margin-top: 0.35rem; font-size: 0.8rem; color: #166534;">Patient qualifies: {t['Renal Gate']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No matching clinical trials meet current patient eligibility.")

        if result.dossier.pruned_clinical_trials:
            with st.expander(f"Excluded Trials ({len(result.dossier.pruned_clinical_trials)}) - Ineligible due to safety criteria", expanded=False):
                for p in result.dossier.pruned_clinical_trials:
                    st.markdown(
                        f"""
                        <div class="action-card-warning">
                            <div class="card-heading">{p['NCT ID']} ({p['Phase']}) - {p['Regimen']}</div>
                            <div class="card-detail" style="color: #991b1b;"><b>Reason Excluded:</b> {p['Exclusion Reason']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with st.expander("Semantic Search Scores (pgvector HNSW)", expanded=False):
            vec_df = pd.DataFrame(
                [
                    {
                        "Trial ID": r.record_id,
                        "Similarity Score": f"{r.cosine_similarity * 100:.1f}%",
                        "Graph Hops": r.hnsw_layer_hops,
                        "Overview": r.text_summary,
                    }
                    for r in result.graphrag_output.vector_candidates
                ]
            )
            st.dataframe(vec_df, use_container_width=True, hide_index=True)

    with tab_evidence:
        col_wsi, col_dna, col_ehr = st.columns(3)

        with col_wsi:
            st.subheader("Tumor Histology (WSI)")
            st.markdown(
                f"""
                <div class="action-card" style="border-left-color: #0284c7;">
                    <div class="card-heading">{result.vision_report.backbone_model}</div>
                    <div class="card-detail">
                        - <b>Tumor Cellularity:</b> {result.vision_report.mean_tumor_cellularity_pct}%<br>
                        - <b>TIL Infiltration:</b> {result.vision_report.til_infiltration_score}<br>
                        - <b>TIL Density:</b> {result.vision_report.mean_til_density_mm2} cells/mm2<br>
                        - <b>MSI Morphology:</b> {result.vision_report.msi_morphology_probability:.2f}<br>
                        - <b>Tiles Scanned:</b> {result.vision_report.total_patches_analyzed} tiles (20x/40x)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_dna:
            st.subheader("DNA Genomics (VCF)")
            st.markdown(
                f"""
                <div class="action-card" style="border-left-color: #059669;">
                    <div class="card-heading">{result.genomic_report.genomic_model}</div>
                    <div class="card-detail">
                        - <b>TMB:</b> {result.genomic_report.tmb_mutations_per_mb} mut/Mb<br>
                        - <b>TMB Category:</b> {result.genomic_report.tmb_status.split(' ')[0]}<br>
                        - <b>Somatic Count:</b> {result.genomic_report.total_somatic_variants}<br>
                        - <b>Exome Coverage:</b> {result.genomic_report.callable_megabases} Mb<br>
                        - <b>MSI Genomic:</b> {result.genomic_report.msi_genomic_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_ehr:
            st.subheader("Clinical History (EHR)")
            st.markdown(
                f"""
                <div class="action-card" style="border-left-color: #7c3aed;">
                    <div class="card-heading">{result.clinical_report.clinical_llm_backbone}</div>
                    <div class="card-detail">
                        - <b>Prior Therapies:</b> {len(result.clinical_report.prior_lines_of_therapy)} line(s)<br>
                        - <b>Latest RECIST:</b> {result.clinical_report.most_recent_recist} (Progression)<br>
                        - <b>eGFR:</b> {result.clinical_report.labs.egfr_ml_min} mL/min<br>
                        - <b>Creatinine:</b> {result.clinical_report.labs.serum_creatinine_mg_dl} mg/dL<br>
                        - <b>LVEF:</b> {result.clinical_report.labs.lvef_pct}% | QTc: {result.clinical_report.labs.qtc_ms} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Detected Somatic Alterations")
        var_clean = [
            {
                "Gene": v.gene,
                "Variant": v.hgvs_p,
                "Allele Frequency (VAF)": f"{v.vaf * 100:.1f}%",
                "Coverage Depth": f"{v.read_depth}x",
                "ClinVar": v.clinvar_significance,
                "Predicted Harm": f"{v.hyenadna_pathogenicity_score:.2f}",
                "Resolved Status": v.resolved_classification.split("-")[0].strip(),
            }
            for v in result.genomic_report.variants
        ]
        st.dataframe(pd.DataFrame(var_clean), use_container_width=True, hide_index=True)

    with tab_fusion:
        st.subheader("How Cross-Modal Fusion Works")
        st.caption("When genomics identifies an uncertain variant (VUS), the AI checks pathology images for tissue-level markers to verify if the mutation is active.")

        if result.fusion_output.synergy_findings:
            for s in result.fusion_output.synergy_findings:
                st.markdown(
                    f"""
                    <div class="action-card" style="border-left-color: #0d9488;">
                        <span class="match-badge">{int(s.confidence_score * 100)}% Confidence</span>
                        <div class="card-heading">Synergy Found: {s.inferred_biomarker}</div>
                        <div class="card-detail" style="margin-top: 0.4rem;">
                            <b>Pathology Finding:</b> {s.visual_evidence}<br>
                            <b>Genomic Finding:</b> {s.genomic_evidence}<br>
                            <b>Clinical Conclusion:</b> {s.mechanism_explanation}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Visual vs Genomic Attention Weights")
        attn_df = pd.DataFrame(
            result.fusion_output.attention_matrix,
            index=result.fusion_output.vision_patch_labels,
            columns=result.fusion_output.genomic_token_labels,
        )
        st.dataframe(attn_df, use_container_width=True)

    with tab_system:
        st.subheader("Clinical Safety Guardrail Checklist")
        guard_df = pd.DataFrame(
            [
                {
                    "Guardrail": a.rule_name,
                    "Category": a.category,
                    "Verdict": "PASS" if a.passed else "FLAGGED",
                    "Details": a.verification_detail,
                }
                for a in result.dossier.assertions
            ]
        )
        st.dataframe(guard_df, use_container_width=True, hide_index=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            with st.expander("S3 / MinIO Chunk Manifests", expanded=False):
                s3_df = pd.DataFrame(
                    [
                        {
                            "Modality": m.modality,
                            "File": m.filename,
                            "Size (Bytes)": m.total_bytes,
                            "Chunks": len(m.chunks),
                            "SHA-256": m.merkle_root_sha256[:12] + "...",
                        }
                        for m in result.manifests
                    ]
                )
                st.dataframe(s3_df, use_container_width=True, hide_index=True)

        with col_m2:
            with st.expander("Kafka Modality Topic Partitions", expanded=False):
                kf_df = pd.DataFrame(
                    [
                        {
                            "Topic": k.topic,
                            "Partition": k.partition,
                            "Offset": k.offset,
                            "Key": k.key,
                        }
                        for k in result.kafka_messages
                    ]
                )
                st.dataframe(kf_df, use_container_width=True, hide_index=True)

        with st.expander("Temporal State Machine Execution Trace", expanded=False):
            wf_df = pd.DataFrame(
                [
                    {
                        "Pipeline Step": w.step_name,
                        "State": w.state,
                        "Queue": w.worker_queue,
                        "Duration (ms)": w.duration_ms,
                    }
                    for w in result.workflow_logs
                ]
            )
            st.dataframe(wf_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_app()
