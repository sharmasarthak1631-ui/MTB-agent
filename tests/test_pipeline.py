from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.coordinator import FederatedMTBCoordinator, SAMPLE_CLINICAL_CASES
from src.ingestion.pipeline import MinIOChunkedStorageEngine


class TestMTBCopilotPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = FederatedMTBCoordinator()

    def test_s3_chunked_manifest_and_merkle_verification(self) -> None:
        engine = MinIOChunkedStorageEngine(chunk_size_bytes=64)
        payload = b"##fileformat=VCFv4.2\n" * 20
        manifest = engine.upload_chunked_payload("PT-TEST-01", "vcf", "sample.vcf", payload)
        self.assertGreater(len(manifest.chunks), 3)
        retrieved = engine.verify_and_retrieve(manifest.manifest_id)
        self.assertEqual(retrieved, payload)

    def test_case_01_crc_braf_msh2_vus_fusion_and_renal_pruning(self) -> None:
        case = SAMPLE_CLINICAL_CASES["CASE-01: Metastatic Colorectal Cancer (BRAF V600E + MSH2 VUS + CKD eGFR 38)"]
        res = self.coordinator.execute_patient_workflow(
            patient_id=case["patient_id"],
            wsi_bytes=case["wsi_payload"].encode("utf-8"),
            vcf_bytes=case["vcf_payload"].encode("utf-8"),
            ehr_bytes=case["ehr_payload"].encode("utf-8"),
            wsi_filename=case["wsi_filename"],
            vcf_filename=case["vcf_filename"],
            ehr_filename=case["ehr_filename"],
        )
        inferred_ids = [s.graph_node_id for s in res.fusion_output.synergy_findings]
        self.assertIn("BIO:DMMR_MSIH", inferred_ids)
        self.assertIn("BIO:TMB_HIGH", inferred_ids)

        pruned_ncts = [t.nct_id for t in res.graphrag_output.pruned_trials]
        eligible_ncts = [t.nct_id for t in res.graphrag_output.eligible_trials]
        self.assertIn("NCT03668431", pruned_ncts)
        self.assertIn("NCT04294160", eligible_ncts)
        self.assertIn("NCT02563002", eligible_ncts)

        resistance_drugs = [r.therapy_name for r in res.graphrag_output.resistance_alerts]
        self.assertTrue(any("Cetuximab" in d for d in resistance_drugs))
        self.assertTrue(res.dossier.verification_passed)

    def test_case_02_nsclc_egfr_t790m(self) -> None:
        case = SAMPLE_CLINICAL_CASES["CASE-02: Advanced NSCLC (EGFR L858R + Acquired T790M + Renal Impairment eGFR 42)"]
        res = self.coordinator.execute_patient_workflow(
            patient_id=case["patient_id"],
            wsi_bytes=case["wsi_payload"].encode("utf-8"),
            vcf_bytes=case["vcf_payload"].encode("utf-8"),
            ehr_bytes=case["ehr_payload"].encode("utf-8"),
        )
        eligible_ncts = [t.nct_id for t in res.graphrag_output.eligible_trials]
        pruned_ncts = [t.nct_id for t in res.graphrag_output.pruned_trials]
        self.assertIn("NCT05388669", eligible_ncts)
        self.assertIn("NCT04487080", pruned_ncts)
        self.assertTrue(res.dossier.verification_passed)

    def test_case_03_ovarian_brca2_hrd_neuropathy_pruning(self) -> None:
        case = SAMPLE_CLINICAL_CASES["CASE-03: High-Grade Serous Ovarian Cancer (BRCA2 Truncating + HRD Phenotype + Neuropathy)"]
        res = self.coordinator.execute_patient_workflow(
            patient_id=case["patient_id"],
            wsi_bytes=case["wsi_payload"].encode("utf-8"),
            vcf_bytes=case["vcf_payload"].encode("utf-8"),
            ehr_bytes=case["ehr_payload"].encode("utf-8"),
        )
        eligible_ncts = [t.nct_id for t in res.graphrag_output.eligible_trials]
        pruned_ncts = [t.nct_id for t in res.graphrag_output.pruned_trials]
        self.assertIn("NCT03742895", eligible_ncts)
        self.assertIn("NCT04171700", pruned_ncts)
        self.assertTrue(res.dossier.verification_passed)


if __name__ == "__main__":
    unittest.main()
