from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ingestion.pipeline import (
    MinIOChunkedStorageEngine,
    KafkaStreamBroker,
    TemporalWorkflowExecutor,
    WorkflowState,
    PayloadManifest,
    KafkaMessageEnvelope,
    WorkflowStepLog,
)
from src.storage.vector_store import PgVectorHNSWIndex
from src.storage.knowledge_graph import BiomedicalKnowledgeHypergraph
from src.agents.vision_specialist import HistopathologyVisionSpecialist, HistopathologyReport
from src.agents.genomic_specialist import GenomicVariantSpecialist, GenomicAnalysisReport
from src.agents.clinical_specialist import ClinicalReasoningSpecialist, ClinicalExtractionReport
from src.fusion.cross_attention import MultiModalCrossAttentionHub, FusionHubOutput
from src.graphrag.retriever import HybridGraphRAGRetriever, GraphRAGRetrievalOutput
from src.verification.guardrails import DualLayerVerificationEngine, VerifiedMTBDossier


@dataclass
class MTBPipelineExecutionResult:
    patient_id: str
    manifests: List[PayloadManifest]
    kafka_messages: List[KafkaMessageEnvelope]
    workflow_logs: List[WorkflowStepLog]
    vision_report: HistopathologyReport
    genomic_report: GenomicAnalysisReport
    clinical_report: ClinicalExtractionReport
    fusion_output: FusionHubOutput
    graphrag_output: GraphRAGRetrievalOutput
    dossier: VerifiedMTBDossier


SAMPLE_CLINICAL_CASES: Dict[str, Dict[str, Any]] = {
    "CASE-01: Metastatic Colorectal Cancer (BRAF V600E + MSH2 VUS + CKD eGFR 38)": {
        "patient_id": "PT-CRC-4092",
        "wsi_filename": "TCGA-CRC-4092-DX1.svs",
        "vcf_filename": "PT-CRC-4092-somatic.vcf",
        "ehr_filename": "PT-CRC-4092-ehr.json",
        "wsi_payload": json.dumps(
            {
                "slide_id": "TCGA-CRC-4092-DX1",
                "tumor_cellularity_pct": 76.5,
                "til_density_mm2": 720.0,
                "msi_morphology_prob": 0.88,
                "hrd_morphology_prob": 0.14,
                "necrosis_pct": 12.0,
                "total_patches": 16,
                "morphological_phenotypes": [
                    "Brisk tumor-infiltrating lymphocytes (720 cells/mm2)",
                    "Crohn-like peritumoral lymphoid reaction",
                    "Mucinous and poorly differentiated medullary architecture",
                ],
            },
            indent=2,
        ),
        "vcf_payload": (
            "##fileformat=VCFv4.2\n"
            "##callable_mb=32.0\n"
            "##background_somatic_mutations=460\n"
            "##msi_genomic_status=Borderline / Indeterminate\n"
            "##hrd_genomic_score=14.0\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr7\t140753336\trs113488022\tA\tT\t999\tPASS\tGENE=BRAF;HGVSp=p.V600E;HGVSc=c.1799T>A;CSQ=missense_variant;VTYPE=SNV;AF=0.42;DP=610;CLINVAR=Pathogenic;SIFT=0.00;POLYPHEN=0.99;HYENA=0.98\n"
            "chr2\t47641560\trs63750019\tG\tC\t850\tPASS\tGENE=MSH2;HGVSp=p.A636P;HGVSc=c.1906G>C;CSQ=missense_variant;VTYPE=SNV;AF=0.39;DP=480;CLINVAR=VUS;SIFT=0.01;POLYPHEN=0.95;HYENA=0.91\n"
        ),
        "ehr_payload": json.dumps(
            {
                "patient_id": "PT-CRC-4092",
                "age": 64,
                "sex": "Female",
                "primary_tumor_site": "Ascending Right Colon",
                "histological_diagnosis": "Colorectal Adenocarcinoma",
                "clinical_stage": "Stage IVB (Hepatic and Peritoneal Metastases)",
                "ecog_performance_status": 1,
                "most_recent_recist": "PD",
                "labs": {
                    "egfr_ml_min": 38.0,
                    "serum_creatinine_mg_dl": 1.68,
                    "total_bilirubin_mg_dl": 0.7,
                    "ast_u_l": 29.0,
                    "alt_u_l": 33.0,
                    "anc_cells_ul": 4300,
                    "platelets_k_ul": 210,
                    "hemoglobin_g_dl": 11.2,
                    "lvef_pct": 60.0,
                    "qtc_ms": 428,
                },
                "prior_lines_of_therapy": [
                    {
                        "line_number": 1,
                        "regimen": "mFOLFOX6 + Bevacizumab",
                        "start_date": "2025-09",
                        "end_date": "2026-05",
                        "best_recist_response": "PR",
                        "discontinuation_reason": "Radiographic hepatic progression (RECIST v1.1 PD)",
                    }
                ],
                "active_comorbidities_and_toxicities": [
                    "Stage 3b Chronic Kidney Disease (eGFR 38.0 mL/min/1.73m2)",
                    "Controlled Essential Hypertension",
                ],
                "clinical_note_summary": (
                    "64-year-old female with metastatic right-sided colorectal adenocarcinoma progressing on first-line "
                    "mFOLFOX6 + Bevacizumab (RECIST PD). Labs notable for chronic renal impairment with eGFR 38.0 mL/min "
                    "(creatinine 1.68 mg/dL) and ECOG performance status 1."
                ),
            },
            indent=2,
        ),
    },
    "CASE-02: Advanced NSCLC (EGFR L858R + Acquired T790M + Renal Impairment eGFR 42)": {
        "patient_id": "PT-NSCLC-8104",
        "wsi_filename": "TCGA-LUAD-8104-DX1.dicom",
        "vcf_filename": "PT-NSCLC-8104-ctdna.vcf",
        "ehr_filename": "PT-NSCLC-8104-ehr.json",
        "wsi_payload": json.dumps(
            {
                "slide_id": "TCGA-LUAD-8104-DX1",
                "tumor_cellularity_pct": 68.0,
                "til_density_mm2": 240.0,
                "msi_morphology_prob": 0.08,
                "hrd_morphology_prob": 0.12,
                "necrosis_pct": 8.5,
                "total_patches": 16,
                "morphological_phenotypes": [
                    "Acinar and micropapillary lung adenocarcinoma pattern",
                    "Moderate stromal desmoplasia",
                    "Low-to-moderate peritumoral lymphocytic infiltrate",
                ],
            },
            indent=2,
        ),
        "vcf_payload": (
            "##fileformat=VCFv4.2\n"
            "##callable_mb=30.0\n"
            "##background_somatic_mutations=110\n"
            "##msi_genomic_status=MSS\n"
            "##hrd_genomic_score=11.0\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr7\t55191822\trs121434568\tT\tG\t999\tPASS\tGENE=EGFR;HGVSp=p.L858R;HGVSc=c.2573T>G;CSQ=missense_variant;VTYPE=SNV;AF=0.34;DP=820;CLINVAR=Pathogenic;SIFT=0.00;POLYPHEN=0.99;HYENA=0.97\n"
            "chr7\t55181378\trs121434569\tC\tT\t920\tPASS\tGENE=EGFR;HGVSp=p.T790M;HGVSc=c.2369C>T;CSQ=missense_variant;VTYPE=SNV;AF=0.19;DP=790;CLINVAR=Pathogenic;SIFT=0.00;POLYPHEN=0.98;HYENA=0.96\n"
        ),
        "ehr_payload": json.dumps(
            {
                "patient_id": "PT-NSCLC-8104",
                "age": 69,
                "sex": "Male",
                "primary_tumor_site": "Right Upper Lobe Lung",
                "histological_diagnosis": "Non-Small Cell Lung Cancer (Adenocarcinoma)",
                "clinical_stage": "Stage IVA (Pleural and Contralateral Pulmonary Nodules)",
                "ecog_performance_status": 1,
                "most_recent_recist": "PD",
                "labs": {
                    "egfr_ml_min": 42.0,
                    "serum_creatinine_mg_dl": 1.54,
                    "total_bilirubin_mg_dl": 0.6,
                    "ast_u_l": 24.0,
                    "alt_u_l": 26.0,
                    "anc_cells_ul": 3900,
                    "platelets_k_ul": 230,
                    "hemoglobin_g_dl": 12.1,
                    "lvef_pct": 56.0,
                    "qtc_ms": 440,
                },
                "prior_lines_of_therapy": [
                    {
                        "line_number": 1,
                        "regimen": "Erlotinib 150 mg QD (1st-Gen EGFR TKI)",
                        "start_date": "2025-04",
                        "end_date": "2026-08",
                        "best_recist_response": "PR",
                        "discontinuation_reason": "Acquired resistance with new pleural lesions (RECIST PD)",
                    }
                ],
                "active_comorbidities_and_toxicities": [
                    "Moderate Renal Insufficiency (eGFR 42.0 mL/min)",
                    "Prior Grade 1 Acneiform Rash on Erlotinib",
                ],
                "clinical_note_summary": (
                    "69-year-old male with Stage IVA EGFR-mutant lung adenocarcinoma progressing after 16 months on first-line "
                    "Erlotinib. Repeat biopsy and ctDNA NGS confirm acquired EGFR p.T790M gatekeeper mutation alongside baseline "
                    "EGFR p.L858R. eGFR is 42.0 mL/min, ECOG 1."
                ),
            },
            indent=2,
        ),
    },
    "CASE-03: High-Grade Serous Ovarian Cancer (BRCA2 Truncating + HRD Phenotype + Neuropathy)": {
        "patient_id": "PT-OV-2291",
        "wsi_filename": "TCGA-OV-2291-DX1.svs",
        "vcf_filename": "PT-OV-2291-somatic.vcf",
        "ehr_filename": "PT-OV-2291-ehr.json",
        "wsi_payload": json.dumps(
            {
                "slide_id": "TCGA-OV-2291-DX1",
                "tumor_cellularity_pct": 81.0,
                "til_density_mm2": 490.0,
                "msi_morphology_prob": 0.11,
                "hrd_morphology_prob": 0.89,
                "necrosis_pct": 16.0,
                "total_patches": 16,
                "morphological_phenotypes": [
                    "Solid, pseudoendometrioid, and transitional (SET) HRD architecture",
                    "Marked nuclear atypia and frequent atypical mitotic figures",
                    "Moderate tumor-infiltrating lymphocytes",
                ],
            },
            indent=2,
        ),
        "vcf_payload": (
            "##fileformat=VCFv4.2\n"
            "##callable_mb=32.0\n"
            "##background_somatic_mutations=195\n"
            "##msi_genomic_status=MSS\n"
            "##hrd_genomic_score=64.0\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr13\t32340301\trs80358815\tG\tT\t999\tPASS\tGENE=BRCA2;HGVSp=p.E1953*;HGVSc=c.5857G>T;CSQ=stop_gained;VTYPE=SNV;AF=0.58;DP=540;CLINVAR=Pathogenic;SIFT=0.00;POLYPHEN=0.99;HYENA=0.99\n"
            "chr3\t179234297\trs121913279\tA\tG\t890\tPASS\tGENE=PIK3CA;HGVSp=p.H1047R;HGVSc=c.3140A>G;CSQ=missense_variant;VTYPE=SNV;AF=0.29;DP=490;CLINVAR=Pathogenic;SIFT=0.01;POLYPHEN=0.97;HYENA=0.94\n"
        ),
        "ehr_payload": json.dumps(
            {
                "patient_id": "PT-OV-2291",
                "age": 58,
                "sex": "Female",
                "primary_tumor_site": "Bilateral Ovary / Peritoneum",
                "histological_diagnosis": "Ovarian High-Grade Serous Carcinoma",
                "clinical_stage": "Stage IIIC",
                "ecog_performance_status": 1,
                "most_recent_recist": "PR",
                "labs": {
                    "egfr_ml_min": 44.0,
                    "serum_creatinine_mg_dl": 1.42,
                    "total_bilirubin_mg_dl": 0.5,
                    "ast_u_l": 22.0,
                    "alt_u_l": 25.0,
                    "anc_cells_ul": 3600,
                    "platelets_k_ul": 180,
                    "hemoglobin_g_dl": 10.9,
                    "lvef_pct": 62.0,
                    "qtc_ms": 420,
                },
                "prior_lines_of_therapy": [
                    {
                        "line_number": 1,
                        "regimen": "Carboplatin + Paclitaxel (6 Cycles)",
                        "start_date": "2025-11",
                        "end_date": "2026-04",
                        "best_recist_response": "PR",
                        "discontinuation_reason": "Completed induction; persistent Grade 2 Peripheral Neuropathy",
                    }
                ],
                "active_comorbidities_and_toxicities": [
                    "Grade 2 Peripheral Neuropathy (Taxane-induced)",
                    "Mild-to-Moderate Renal Impairment (eGFR 44.0 mL/min)",
                ],
                "clinical_note_summary": (
                    "58-year-old female with Stage IIIC high-grade serous ovarian carcinoma following cytoreductive surgery "
                    "and 6 cycles of Carboplatin + Paclitaxel. Residual Grade 2 peripheral neuropathy and eGFR 44.0 mL/min."
                ),
            },
            indent=2,
        ),
    },
}


class FederatedMTBCoordinator:
    def __init__(
        self,
        vision_backbone: str = "UNI (ViT-L/16)",
        genomic_backbone: str = "HyenaDNA-1M",
        clinical_backbone: str = "BioMistral-7B-Clinical",
    ):
        self.storage_engine = MinIOChunkedStorageEngine()
        self.kafka_broker = KafkaStreamBroker()
        self.vector_index = PgVectorHNSWIndex()
        self.knowledge_graph = BiomedicalKnowledgeHypergraph()
        self.vision_specialist = HistopathologyVisionSpecialist(backbone_model=vision_backbone)
        self.genomic_specialist = GenomicVariantSpecialist(genomic_model=genomic_backbone)
        self.clinical_specialist = ClinicalReasoningSpecialist(llm_backbone=clinical_backbone)
        self.fusion_hub = MultiModalCrossAttentionHub()
        self.graphrag_retriever = HybridGraphRAGRetriever(
            vector_index=self.vector_index,
            knowledge_graph=self.knowledge_graph,
        )
        self.verification_engine = DualLayerVerificationEngine()

    def execute_patient_workflow(
        self,
        patient_id: str,
        wsi_bytes: bytes,
        vcf_bytes: bytes,
        ehr_bytes: bytes,
        wsi_filename: str = "slide.svs",
        vcf_filename: str = "variants.vcf",
        ehr_filename: str = "clinical_notes.json",
    ) -> MTBPipelineExecutionResult:
        workflow = TemporalWorkflowExecutor()
        manifests: List[PayloadManifest] = []
        kafka_msgs: List[KafkaMessageEnvelope] = []

        def _ingest_s3() -> None:
            m_wsi = self.storage_engine.upload_chunked_payload(patient_id, "dicom", wsi_filename, wsi_bytes)
            m_vcf = self.storage_engine.upload_chunked_payload(patient_id, "vcf", vcf_filename, vcf_bytes)
            m_ehr = self.storage_engine.upload_chunked_payload(patient_id, "ehr", ehr_filename, ehr_bytes)
            manifests.extend([m_wsi, m_vcf, m_ehr])

        workflow.run_step(
            step_name="Presigned S3/MinIO Chunked Upload & SHA-256 Merkle Hashing",
            target_state=WorkflowState.INGESTING_S3,
            worker_queue="minio-ingest-io",
            fn=_ingest_s3,
            compensate_desc="Purge uncommitted multipart S3 chunks",
        )

        def _publish_kafka() -> None:
            for m in manifests:
                env = self.kafka_broker.publish_manifest(m)
                kafka_msgs.append(env)

        workflow.run_step(
            step_name="Apache Kafka Modality Topic Partitioning (dicom/vcf/ehr-stream)",
            target_state=WorkflowState.KAFKA_QUEUED,
            worker_queue="kafka-broker-partitioner",
            fn=_publish_kafka,
            compensate_desc="Publish tombstone offset markers to modality topics",
        )

        vision_report: HistopathologyReport = workflow.run_step(
            step_name=f"Histopathology Vision Specialist ({self.vision_specialist.backbone_model})",
            target_state=WorkflowState.PROCESSING_VISION,
            worker_queue="celery-gpu-vision",
            fn=lambda: self.vision_specialist.analyze_slide(
                self.storage_engine.verify_and_retrieve(manifests[0].manifest_id),
                filename=wsi_filename,
            ),
            compensate_desc="Release GPU tile buffers and drop temporary patch vectors",
        )

        genomic_report: GenomicAnalysisReport = workflow.run_step(
            step_name=f"Genomic Variant Specialist ({self.genomic_specialist.genomic_model} + VEP/ClinVar)",
            target_state=WorkflowState.PROCESSING_GENOMICS,
            worker_queue="celery-gpu-genomics",
            fn=lambda: self.genomic_specialist.analyze_vcf(
                self.storage_engine.verify_and_retrieve(manifests[1].manifest_id),
                sample_id=f"{patient_id}-NGS",
            ),
            compensate_desc="Invalidate variant annotation cache entries",
        )

        clinical_report: ClinicalExtractionReport = workflow.run_step(
            step_name=f"Clinical Reasoning Specialist ({self.clinical_specialist.llm_backbone})",
            target_state=WorkflowState.PROCESSING_CLINICAL,
            worker_queue="celery-gpu-clinical-llm",
            fn=lambda: self.clinical_specialist.extract_ehr(
                self.storage_engine.verify_and_retrieve(manifests[2].manifest_id)
            ),
            compensate_desc="Discard transient clinical state extraction graph",
        )

        fusion_output: FusionHubOutput = workflow.run_step(
            step_name="Multi-Modal Cross-Attention Fusion Hub & VUS Resolution",
            target_state=WorkflowState.CROSS_ATTENTION_FUSION,
            worker_queue="celery-gpu-fusion",
            fn=lambda: self.fusion_hub.fuse(vision_report, genomic_report, clinical_report),
            compensate_desc="Reset cross-modal attention hypergraph edges",
        )

        self.vector_index.upsert(
            record_id=f"WSI:{vision_report.slide_id}",
            namespace="wsi_patch",
            text_summary=f"WSI {vision_report.slide_id} TILs {vision_report.mean_til_density_mm2}",
            embedding=vision_report.pooled_slide_embedding,
        )
        self.vector_index.upsert(
            record_id=f"VCF:{genomic_report.sample_id}",
            namespace="genomic_variant",
            text_summary=f"VCF {genomic_report.sample_id} TMB {genomic_report.tmb_mutations_per_mb}",
            embedding=genomic_report.pooled_genomic_embedding,
        )

        gene_terms = [v.gene for v in genomic_report.variants] + [v.hgvs_p for v in genomic_report.variants]
        graphrag_output: GraphRAGRetrievalOutput = workflow.run_step(
            step_name="Hybrid GraphRAG (pgvector HNSW + Neo4j Cypher Multi-Hop + Hard Exclusion Pruning)",
            target_state=WorkflowState.GRAPHRAG_TRAVERSAL,
            worker_queue="graphrag-traversal-workers",
            fn=lambda: self.graphrag_retriever.retrieve_and_prune(
                clinical_report=clinical_report,
                fusion_output=fusion_output,
                gene_boost_terms=gene_terms,
            ),
            compensate_desc="Release Neo4j read transaction snapshot",
        )

        dossier: VerifiedMTBDossier = workflow.run_step(
            step_name="Dual-Layer Verification & Guardrail Assertion Engine",
            target_state=WorkflowState.GUARDRAIL_VERIFICATION,
            worker_queue="guardrail-verifier",
            fn=lambda: self.verification_engine.verify_and_compile_dossier(
                vision_report=vision_report,
                genomic_report=genomic_report,
                clinical_report=clinical_report,
                fusion_output=fusion_output,
                graphrag_output=graphrag_output,
            ),
            compensate_desc="Quarantine unverified MTB dossier output",
        )

        workflow.current_state = WorkflowState.COMPLETED
        return MTBPipelineExecutionResult(
            patient_id=clinical_report.patient_id,
            manifests=manifests,
            kafka_messages=kafka_msgs,
            workflow_logs=workflow.execution_log,
            vision_report=vision_report,
            genomic_report=genomic_report,
            clinical_report=clinical_report,
            fusion_output=fusion_output,
            graphrag_output=graphrag_output,
            dossier=dossier,
        )
