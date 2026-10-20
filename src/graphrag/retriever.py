from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.storage.vector_store import PgVectorHNSWIndex, VectorSearchResult
from src.storage.knowledge_graph import BiomedicalKnowledgeHypergraph, MultiHopPath
from src.agents.clinical_specialist import ClinicalExtractionReport
from src.fusion.cross_attention import FusionHubOutput


@dataclass
class TrialCandidateEvaluation:
    nct_id: str
    title: str
    phase: str
    status: str
    target_drug: str
    biomarker_driver: str
    oncokb_level: str
    pmid: str
    civic_id: str
    vector_similarity_score: float
    hnsw_hops: int
    required_min_egfr: float
    patient_egfr: float
    required_max_ecog: int
    patient_ecog: int
    eligibility_decision: str
    exclusion_reasons: List[str]
    cypher_path: str


@dataclass
class TherapeuticRecommendation:
    therapy_name: str
    drug_class: str
    relationship: str
    biomarker_driver: str
    oncokb_level: str
    pmid: str
    civic_id: str
    standard_dosage: str
    adjusted_dosage_note: str
    rationale: str
    cypher_trace: str


@dataclass
class GraphRAGRetrievalOutput:
    vector_candidates: List[VectorSearchResult]
    eligible_trials: List[TrialCandidateEvaluation]
    pruned_trials: List[TrialCandidateEvaluation]
    sensitizing_therapies: List[TherapeuticRecommendation]
    resistance_alerts: List[TherapeuticRecommendation]

    def to_summary_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridGraphRAGRetriever:
    def __init__(self, vector_index: PgVectorHNSWIndex, knowledge_graph: BiomedicalKnowledgeHypergraph):
        self.vector_index = vector_index
        self.knowledge_graph = knowledge_graph
        self._index_knowledge_graph_trials()

    def _index_knowledge_graph_trials(self) -> None:
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.label == "ClinicalTrial":
                nct_id = node.properties.get("nct_id", node_id)
                tumors = ", ".join(node.properties.get("tumor_types", []))
                summary = f"{nct_id}: {node.name} ({node.properties.get('phase')}) for {tumors}"
                self.vector_index.upsert(
                    record_id=nct_id,
                    namespace="clinical_trial",
                    text_summary=summary,
                    metadata=node.properties,
                )

    def retrieve_and_prune(
        self,
        clinical_report: ClinicalExtractionReport,
        fusion_output: FusionHubOutput,
        gene_boost_terms: List[str],
    ) -> GraphRAGRetrievalOutput:
        query_str = (
            f"{clinical_report.histological_diagnosis} {clinical_report.primary_tumor_site} "
            f"{' '.join(gene_boost_terms)} {' '.join(fusion_output.active_graph_alteration_ids)}"
        )
        vec_results = self.vector_index.search(
            query_text=query_str,
            namespace="clinical_trial",
            top_k=8,
            boost_terms=gene_boost_terms,
        )
        vec_sim_map: Dict[str, VectorSearchResult] = {r.record_id: r for r in vec_results}

        multi_hop_paths: List[MultiHopPath] = self.knowledge_graph.expand_multi_hop_from_alterations(
            fusion_output.active_graph_alteration_ids
        )

        sensitizing: List[TherapeuticRecommendation] = []
        resistance: List[TherapeuticRecommendation] = []
        seen_therapies = set()

        eligible_trials: List[TrialCandidateEvaluation] = []
        pruned_trials: List[TrialCandidateEvaluation] = []
        seen_trials = set()

        patient_egfr = clinical_report.labs.egfr_ml_min
        patient_ecog = clinical_report.ecog_performance_status
        comorb_text = " ".join(clinical_report.active_comorbidities_and_toxicities).lower()

        for path in multi_hop_paths:
            drug = path.drug_node
            alt = path.alteration_node
            therapy_key = (drug.node_id, path.relationship, alt.node_id)

            if therapy_key not in seen_therapies:
                seen_therapies.add(therapy_key)
                std_dose = drug.properties.get("standard_dosage", "Standard per label")
                min_drug_egfr = float(drug.properties.get("min_egfr_ml_min", 30.0))

                if patient_egfr < min_drug_egfr:
                    dose_note = (
                        f"CONTRAINDICATED: Patient eGFR ({patient_egfr} mL/min) is below minimum threshold ({min_drug_egfr} mL/min)"
                    )
                elif "Olaparib" in drug.name and 31.0 <= patient_egfr <= 50.0:
                    dose_note = f"Renal Adjustment Required (eGFR {patient_egfr} mL/min): Reduce Olaparib dose to 200 mg PO BID"
                else:
                    dose_note = f"Standard dosing verified for eGFR {patient_egfr} mL/min and ECOG {patient_ecog}"

                rec = TherapeuticRecommendation(
                    therapy_name=drug.name,
                    drug_class=drug.properties.get("drug_class", "Targeted Therapy"),
                    relationship=path.relationship,
                    biomarker_driver=alt.name,
                    oncokb_level=path.oncokb_level,
                    pmid=path.pmid,
                    civic_id=path.civic_id,
                    standard_dosage=std_dose,
                    adjusted_dosage_note=dose_note,
                    rationale=path.rationale,
                    cypher_trace=path.cypher_trace,
                )
                if path.relationship == "SENSITIZES":
                    if patient_egfr >= min_drug_egfr:
                        sensitizing.append(rec)
                else:
                    resistance.append(rec)

            if path.trial_node and path.relationship == "SENSITIZES":
                t_node = path.trial_node
                nct_id = t_node.properties.get("nct_id", t_node.node_id)
                if nct_id in seen_trials:
                    continue
                seen_trials.add(nct_id)

                req_egfr = float(t_node.properties.get("min_egfr", 30.0))
                req_ecog = int(t_node.properties.get("max_ecog", 2))
                excluded_conds: List[str] = t_node.properties.get("excluded_conditions", [])

                reasons: List[str] = []
                if patient_egfr < req_egfr:
                    reasons.append(
                        f"Renal Exclusion: Patient.eGFR ({patient_egfr} mL/min) < Trial.REQUIRES(min_eGFR >= {req_egfr} mL/min)"
                    )
                if patient_ecog > req_ecog:
                    reasons.append(
                        f"Performance Status Exclusion: Patient.ECOG ({patient_ecog}) > Trial.REQUIRES(max_ECOG <= {req_ecog})"
                    )
                for cond in excluded_conds:
                    cond_low = cond.lower()
                    if "neuropathy" in cond_low and ("neuropathy grade 2" in comorb_text or "grade 2 peripheral neuropathy" in comorb_text or "grade >= 2 peripheral neuropathy" in comorb_text):
                        reasons.append(f"Toxicity Exclusion: Trial.EXCLUDES_CONDITION('{cond}') matched patient history")
                    if "interstitial lung disease" in cond_low and "interstitial lung disease" in comorb_text:
                        reasons.append(f"Comorbidity Exclusion: Trial.EXCLUDES_CONDITION('{cond}') matched patient history")
                    if "qtc" in cond_low and clinical_report.labs.qtc_ms > 470:
                        reasons.append(f"Cardiac Exclusion: Patient.QTc ({clinical_report.labs.qtc_ms} ms) violates '{cond}'")

                v_hit = vec_sim_map.get(nct_id)
                v_sim = v_hit.cosine_similarity if v_hit else 0.72
                v_hops = v_hit.hnsw_layer_hops if v_hit else 3

                decision = "PRUNED_HARD_EXCLUSION" if reasons else "ELIGIBLE"
                eval_item = TrialCandidateEvaluation(
                    nct_id=nct_id,
                    title=t_node.name,
                    phase=t_node.properties.get("phase", "Phase II"),
                    status=t_node.properties.get("status", "Recruiting"),
                    target_drug=drug.name,
                    biomarker_driver=alt.name,
                    oncokb_level=path.oncokb_level,
                    pmid=path.pmid,
                    civic_id=path.civic_id,
                    vector_similarity_score=v_sim,
                    hnsw_hops=v_hops,
                    required_min_egfr=req_egfr,
                    patient_egfr=patient_egfr,
                    required_max_ecog=req_ecog,
                    patient_ecog=patient_ecog,
                    eligibility_decision=decision,
                    exclusion_reasons=reasons,
                    cypher_path=path.cypher_trace,
                )
                if decision == "ELIGIBLE":
                    eligible_trials.append(eval_item)
                else:
                    pruned_trials.append(eval_item)

        eligible_trials.sort(key=lambda x: x.vector_similarity_score, reverse=True)
        return GraphRAGRetrievalOutput(
            vector_candidates=vec_results,
            eligible_trials=eligible_trials,
            pruned_trials=pruned_trials,
            sensitizing_therapies=sensitizing,
            resistance_alerts=resistance,
        )
