from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphNode:
    node_id: str
    label: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiHopPath:
    alteration_node: GraphNode
    relationship: str
    drug_node: GraphNode
    trial_node: Optional[GraphNode]
    oncokb_level: str
    pmid: str
    civic_id: str
    rationale: str
    cypher_trace: str


class BiomedicalKnowledgeHypergraph:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._outgoing: Dict[str, List[GraphEdge]] = {}
        self._incoming: Dict[str, List[GraphEdge]] = {}
        self._seed_ontology()

    def add_node(self, node_id: str, label: str, name: str, **properties: Any) -> GraphNode:
        node = GraphNode(node_id=node_id, label=label, name=name, properties=properties)
        self.nodes[node_id] = node
        self._outgoing.setdefault(node_id, [])
        self._incoming.setdefault(node_id, [])
        return node

    def add_edge(self, source_id: str, target_id: str, relation: str, **properties: Any) -> GraphEdge:
        edge = GraphEdge(source_id=source_id, target_id=target_id, relation=relation, properties=properties)
        self.edges.append(edge)
        self._outgoing.setdefault(source_id, []).append(edge)
        self._incoming.setdefault(target_id, []).append(edge)
        return edge

    def get_outgoing(self, node_id: str, relation: Optional[str] = None) -> List[GraphEdge]:
        edges = self._outgoing.get(node_id, [])
        if relation:
            return [e for e in edges if e.relation == relation]
        return list(edges)

    def _seed_ontology(self) -> None:
        genes = [
            ("GENE:BRAF", "BRAF", "7q34", "Serine/threonine-protein kinase B-raf"),
            ("GENE:MSH2", "MSH2", "2p21-p16.3", "DNA mismatch repair protein Msh2"),
            ("GENE:MLH1", "MLH1", "3p22.2", "DNA mismatch repair protein Mlh1"),
            ("GENE:EGFR", "EGFR", "7p11.2", "Epidermal growth factor receptor"),
            ("GENE:KRAS", "KRAS", "12p12.1", "GTPase KRas"),
            ("GENE:BRCA1", "BRCA1", "17q21.31", "Breast cancer type 1 susceptibility protein"),
            ("GENE:BRCA2", "BRCA2", "13q13.1", "Breast cancer type 2 susceptibility protein"),
            ("GENE:PIK3CA", "PIK3CA", "3q26.32", "Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit alpha"),
            ("GENE:ERBB2", "ERBB2", "17q12", "Receptor tyrosine-protein kinase erbB-2"),
        ]
        for gid, name, locus, desc in genes:
            self.add_node(gid, "Gene", name, chromosome_locus=locus, description=desc)

        mutations = [
            ("MUT:BRAF_V600E", "Mutation", "BRAF p.V600E", "GENE:BRAF", "c.1799T>A", "Class I activating kinase mutation"),
            ("MUT:MSH2_A636P", "Mutation", "MSH2 p.A636P", "GENE:MSH2", "c.1906G>C", "ATPase domain missense variant impairing mismatch repair"),
            ("MUT:EGFR_L858R", "Mutation", "EGFR p.L858R", "GENE:EGFR", "c.2573T>G", "Exon 21 activating kinase domain mutation"),
            ("MUT:EGFR_T790M", "Mutation", "EGFR p.T790M", "GENE:EGFR", "c.2369C>T", "Exon 20 gatekeeper resistance mutation"),
            ("MUT:KRAS_G12C", "Mutation", "KRAS p.G12C", "GENE:KRAS", "c.34G>T", "Exon 2 codon 12 GTPase locking mutation"),
            ("MUT:BRCA2_E1953X", "Mutation", "BRCA2 p.E1953*", "GENE:BRCA2", "c.5857G>T", "Truncating BRC repeat loss-of-function variant"),
            ("MUT:PIK3CA_H1047R", "Mutation", "PIK3CA p.H1047R", "GENE:PIK3CA", "c.3140A>G", "Kinase domain hotspot activating mutation"),
        ]
        for mid, label, name, gid, hgvs_c, mech in mutations:
            self.add_node(mid, label, name, gene_id=gid, hgvs_c=hgvs_c, mechanism=mech)
            self.add_edge(gid, mid, "HAS_MUTATION")

        biomarkers = [
            ("BIO:DMMR_MSIH", "Biomarker", "dMMR / MSI-H", "Deficient Mismatch Repair / Microsatellite Instability-High"),
            ("BIO:TMB_HIGH", "Biomarker", "TMB-High (>=10 mut/Mb)", "Tumor Mutational Burden >= 10 mutations per megabase"),
            ("BIO:HRD_POS", "Biomarker", "HRD-Positive", "Homologous Recombination Deficiency genomic scarring"),
        ]
        for bid, label, name, desc in biomarkers:
            self.add_node(bid, label, name, description=desc)

        drugs = [
            ("DRUG:CETUXIMAB_MONO", "Cetuximab / Panitumumab Anti-EGFR Monotherapy", "Anti-EGFR mAb", "250-500 mg/m2 IV q1-2w", 30.0),
            ("DRUG:ENCORAFENIB_CETUX", "Encorafenib + Cetuximab (BEACON Regimen)", "BRAF + EGFR Inhibitor Combination", "Encorafenib 300 mg PO QD + Cetuximab 400 mg/m2 initial then 250 mg/m2 IV qw", 30.0),
            ("DRUG:DABRAFENIB_TRAMETINIB", "Dabrafenib + Trametinib", "BRAF + MEK Inhibitor Combination", "Dabrafenib 150 mg PO BID + Trametinib 2 mg PO QD", 30.0),
            ("DRUG:PEMBROLIZUMAB", "Pembrolizumab Monotherapy", "PD-1 Immune Checkpoint Inhibitor", "200 mg IV q3w or 400 mg IV q6w", 15.0),
            ("DRUG:NIVO_IPI", "Nivolumab + Low-Dose Ipilimumab", "PD-1 + CTLA-4 Checkpoint Blockade", "Nivolumab 240 mg IV q2w + Ipilimumab 1 mg/kg IV q6w", 20.0),
            ("DRUG:FU_MONO", "5-Fluorouracil Single-Agent Adjuvant", "Antimetabolite", "400 mg/m2 IV bolus + 2400 mg/m2 CIV q2w", 30.0),
            ("DRUG:OSIMERTINIB", "Osimertinib", "3rd-Generation EGFR TKI", "80 mg PO QD", 30.0),
            ("DRUG:GEFITINIB_ERLOTINIB", "Gefitinib / Erlotinib (1st-Gen EGFR TKI)", "1st-Generation EGFR TKI", "Gefitinib 250 mg PO QD / Erlotinib 150 mg PO QD", 30.0),
            ("DRUG:AMIVANTAMAB_LAZERTINIB", "Amivantamab + Lazertinib", "EGFR-MET Bispecific + 3rd-Gen EGFR TKI", "Amivantamab 1050 mg IV + Lazertinib 240 mg PO QD", 45.0),
            ("DRUG:OLAPARIB", "Olaparib", "PARP1/2 Inhibitor", "300 mg PO BID (reduce to 200 mg BID if eGFR 31-50)", 30.0),
            ("DRUG:CISPLATIN_PACLITAXEL", "Cisplatin + Paclitaxel Intensive Regimen", "Platinum + Taxane Cytotoxic", "Cisplatin 75 mg/m2 IV + Paclitaxel 175 mg/m2 IV q3w", 60.0),
            ("DRUG:SOTORASIB_PANITUMUMAB", "Sotorasib + Panitumumab", "KRAS G12C Inhibitor + Anti-EGFR mAb", "Sotorasib 960 mg PO QD + Panitumumab 6 mg/kg IV q2w", 30.0),
            ("DRUG:ALPELISIB_FULVESTRANT", "Alpelisib + Fulvestrant", "PI3K-alpha Inhibitor + SERD", "Alpelisib 300 mg PO QD + Fulvestrant 500 mg IM", 35.0),
        ]
        for did, name, drug_class, fda_dose, min_egfr in drugs:
            self.add_node(
                did,
                "Drug",
                name,
                drug_class=drug_class,
                standard_dosage=fda_dose,
                min_egfr_ml_min=min_egfr,
            )

        trials = [
            {
                "id": "TRIAL:NCT04294160",
                "nct_id": "NCT04294160",
                "title": "ANCHOR / BREAKWATER Phase II/III Study of Encorafenib + Cetuximab in BRAF V600E Metastatic Colorectal Cancer",
                "phase": "Phase II/III",
                "status": "Recruiting",
                "target_drug": "DRUG:ENCORAFENIB_CETUX",
                "min_egfr": 30.0,
                "max_ecog": 1,
                "excluded_conditions": ["Uncontrolled Brain Metastases", "QTc > 480 ms"],
                "pmid": "PMID:31566309",
                "tumor_types": ["Colorectal Adenocarcinoma", "Solid Tumor"],
            },
            {
                "id": "TRIAL:NCT03668431",
                "nct_id": "NCT03668431",
                "title": "Phase II Basket Trial of Dabrafenib + Trametinib + Cisplatin Amplification Cohort in BRAF V600E Solid Tumors",
                "phase": "Phase II",
                "status": "Active, Recruiting",
                "target_drug": "DRUG:DABRAFENIB_TRAMETINIB",
                "min_egfr": 50.0,
                "max_ecog": 1,
                "excluded_conditions": ["Renal Clearance < 50 mL/min", "Left Ventricular Ejection Fraction < 50%"],
                "pmid": "PMID:36922589",
                "tumor_types": ["Colorectal Adenocarcinoma", "NSCLC", "Melanoma", "Biliary Tract"],
            },
            {
                "id": "TRIAL:NCT02563002",
                "nct_id": "NCT02563002",
                "title": "KEYNOTE-177 / Cohort Extension: Pembrolizumab in dMMR / MSI-H Metastatic Colorectal & Endometrial Carcinoma",
                "phase": "Phase III",
                "status": "Recruiting Extension",
                "target_drug": "DRUG:PEMBROLIZUMAB",
                "min_egfr": 20.0,
                "max_ecog": 2,
                "excluded_conditions": ["Active Autoimmune Pneumonitis", "Systemic Immunosuppression"],
                "pmid": "PMID:33264548",
                "tumor_types": ["Colorectal Adenocarcinoma", "Endometrial Carcinoma", "Solid Tumor"],
            },
            {
                "id": "TRIAL:NCT04008030",
                "nct_id": "NCT04008030",
                "title": "CheckMate-8HW: Nivolumab Plus Ipilimumab in Microsatellite Instability-High / dMMR Tumors",
                "phase": "Phase III",
                "status": "Recruiting",
                "target_drug": "DRUG:NIVO_IPI",
                "min_egfr": 25.0,
                "max_ecog": 1,
                "excluded_conditions": ["Active Autoimmune Disease", "Interstitial Lung Disease"],
                "pmid": "PMID:39602641",
                "tumor_types": ["Colorectal Adenocarcinoma", "Solid Tumor"],
            },
            {
                "id": "TRIAL:NCT05388669",
                "nct_id": "NCT05388669",
                "title": "COMPEL / FLAURA2: Osimertinib in EGFR L858R/T790M Mutant Non-Small Cell Lung Cancer",
                "phase": "Phase III",
                "status": "Recruiting",
                "target_drug": "DRUG:OSIMERTINIB",
                "min_egfr": 30.0,
                "max_ecog": 1,
                "excluded_conditions": ["Interstitial Lung Disease", "QTc > 470 ms"],
                "pmid": "PMID:29151359",
                "tumor_types": ["Non-Small Cell Lung Cancer", "Lung Adenocarcinoma"],
            },
            {
                "id": "TRIAL:NCT04487080",
                "nct_id": "NCT04487080",
                "title": "MARIPOSA-2: Amivantamab and Lazertinib in EGFR-Mutant Advanced NSCLC",
                "phase": "Phase III",
                "status": "Recruiting",
                "target_drug": "DRUG:AMIVANTAMAB_LAZERTINIB",
                "min_egfr": 50.0,
                "max_ecog": 1,
                "excluded_conditions": ["Renal Clearance < 50 mL/min", "Active Deep Vein Thrombosis"],
                "pmid": "PMID:38925134",
                "tumor_types": ["Non-Small Cell Lung Cancer", "Lung Adenocarcinoma"],
            },
            {
                "id": "TRIAL:NCT03742895",
                "nct_id": "NCT03742895",
                "title": "OlympiAD / LYNK-002: Olaparib Monotherapy in BRCA1/2 or HRD-Positive Solid Tumors",
                "phase": "Phase III",
                "status": "Recruiting",
                "target_drug": "DRUG:OLAPARIB",
                "min_egfr": 31.0,
                "max_ecog": 2,
                "excluded_conditions": ["Myelodysplastic Syndrome", "Acute Myeloid Leukemia"],
                "pmid": "PMID:28578601",
                "tumor_types": ["Ovarian Serous Carcinoma", "Breast Carcinoma", "Pancreatic Adenocarcinoma", "Solid Tumor"],
            },
            {
                "id": "TRIAL:NCT04171700",
                "nct_id": "NCT04171700",
                "title": "LODESTAR Platinum-Taxane Escalation in HRD-Positive Advanced Tumors",
                "phase": "Phase II",
                "status": "Recruiting",
                "target_drug": "DRUG:CISPLATIN_PACLITAXEL",
                "min_egfr": 60.0,
                "max_ecog": 1,
                "excluded_conditions": ["Grade >= 2 Peripheral Neuropathy", "Renal Clearance < 60 mL/min"],
                "pmid": "PMID:31978345",
                "tumor_types": ["Ovarian Serous Carcinoma", "Breast Carcinoma", "Solid Tumor"],
            },
            {
                "id": "TRIAL:NCT04793958",
                "nct_id": "NCT04793958",
                "title": "CodeBreaK 300: Sotorasib Plus Panitumumab in KRAS p.G12C Chemorefractory Colorectal Cancer",
                "phase": "Phase III",
                "status": "Recruiting",
                "target_drug": "DRUG:SOTORASIB_PANITUMUMAB",
                "min_egfr": 30.0,
                "max_ecog": 2,
                "excluded_conditions": ["Active Hepatitis", "Interstitial Lung Disease"],
                "pmid": "PMID:37870968",
                "tumor_types": ["Colorectal Adenocarcinoma", "Solid Tumor"],
            },
        ]

        for tr in trials:
            tid = tr["id"]
            self.add_node(
                tid,
                "ClinicalTrial",
                tr["title"],
                nct_id=tr["nct_id"],
                phase=tr["phase"],
                status=tr["status"],
                min_egfr=tr["min_egfr"],
                max_ecog=tr["max_ecog"],
                excluded_conditions=tr["excluded_conditions"],
                pmid=tr["pmid"],
                tumor_types=tr["tumor_types"],
            )
            self.add_edge(tr["target_drug"], tid, "ACTIVE_IN", pmid=tr["pmid"])

            crit_id = f"CRIT:{tr['nct_id']}"
            self.add_node(
                crit_id,
                "EligibilityCriteria",
                f"Eligibility for {tr['nct_id']}",
                min_egfr=tr["min_egfr"],
                max_ecog=tr["max_ecog"],
                excluded_conditions=tr["excluded_conditions"],
            )
            self.add_edge(tid, crit_id, "REQUIRES", min_egfr=tr["min_egfr"], max_ecog=tr["max_ecog"])
            for cond in tr["excluded_conditions"]:
                cond_id = f"EXCL:{cond.upper().replace(' ', '_')}"
                if cond_id not in self.nodes:
                    self.add_node(cond_id, "AdverseEvent", cond)
                self.add_edge(tid, cond_id, "EXCLUDES_CONDITION", condition=cond)

        therapeutic_edges = [
            ("MUT:BRAF_V600E", "DRUG:CETUXIMAB_MONO", "CONFERS_RESISTANCE_TO", "Level R1", "PMID:20619739", "CIViC:EID102", "BRAF V600E activates MAPK downstream of EGFR, conferring primary resistance to anti-EGFR monotherapy."),
            ("MUT:BRAF_V600E", "DRUG:ENCORAFENIB_CETUX", "SENSITIZES", "Level 1", "PMID:31566309", "CIViC:EID2991", "FDA-approved BEACON regimen blocking BRAF V600E and feedback EGFR reactivation."),
            ("MUT:BRAF_V600E", "DRUG:DABRAFENIB_TRAMETINIB", "SENSITIZES", "Level 1", "PMID:36922589", "CIViC:EID1408", "Dual BRAF/MEK vertical blockade in BRAF V600E solid tumors."),
            ("BIO:DMMR_MSIH", "DRUG:PEMBROLIZUMAB", "SENSITIZES", "Level 1", "PMID:33264548", "CIViC:EID3015", "Hypermutated neoantigen load in dMMR/MSI-H sensitizes to PD-1 blockade (KEYNOTE-177)."),
            ("BIO:DMMR_MSIH", "DRUG:NIVO_IPI", "SENSITIZES", "Level 1", "PMID:39602641", "CIViC:EID3120", "Dual PD-1/CTLA-4 blockade demonstrates high PFS in dMMR tumors (CheckMate-8HW)."),
            ("BIO:DMMR_MSIH", "DRUG:FU_MONO", "CONFERS_RESISTANCE_TO", "Level R2", "PMID:12860960", "CIViC:EID884", "dMMR tumors lack functional mismatch repair recognition of 5-FU-modified DNA bases."),
            ("MUT:MSH2_A636P", "DRUG:PEMBROLIZUMAB", "SENSITIZES", "Level 1", "PMID:33264548", "CIViC:EID3018", "Pathogenic/fusing MSH2 inactivation drives mismatch repair deficiency responsive to anti-PD-1 therapy."),
            ("BIO:TMB_HIGH", "DRUG:PEMBROLIZUMAB", "SENSITIZES", "Level 1", "PMID:32919526", "CIViC:EID4110", "Tumor-agnostic FDA approval for TMB >= 10 mut/Mb (KEYNOTE-158)."),
            ("MUT:EGFR_L858R", "DRUG:OSIMERTINIB", "SENSITIZES", "Level 1", "PMID:29151359", "CIViC:EID2110", "First-line 3rd-generation irreversible EGFR TKI (FLAURA)."),
            ("MUT:EGFR_L858R", "DRUG:AMIVANTAMAB_LAZERTINIB", "SENSITIZES", "Level 1", "PMID:38925134", "CIViC:EID4890", "Dual EGFR-MET extracellular/intracellular blockade (MARIPOSA)."),
            ("MUT:EGFR_T790M", "DRUG:GEFITINIB_ERLOTINIB", "CONFERS_RESISTANCE_TO", "Level R1", "PMID:15737014", "CIViC:EID155", "Steric hindrance and increased ATP affinity at T790 gatekeeper residue."),
            ("MUT:EGFR_T790M", "DRUG:OSIMERTINIB", "SENSITIZES", "Level 1", "PMID:27959700", "CIViC:EID1812", "Covalent binding to C797 overcomes T790M gatekeeper resistance (AURA3)."),
            ("MUT:BRCA2_E1953X", "DRUG:OLAPARIB", "SENSITIZES", "Level 1", "PMID:28578601", "CIViC:EID2245", "Synthetic lethality via PARP trapping in homologous recombination-deficient BRCA2 mutant cells."),
            ("BIO:HRD_POS", "DRUG:OLAPARIB", "SENSITIZES", "Level 1", "PMID:31651948", "CIViC:EID3502", "HRD genomic instability sensitizes to PARP inhibition."),
            ("BIO:HRD_POS", "DRUG:CISPLATIN_PACLITAXEL", "SENSITIZES", "Level 2", "PMID:31978345", "CIViC:EID3510", "DNA interstrand crosslinking sensitivity in HRD-positive tumors."),
            ("MUT:KRAS_G12C", "DRUG:CETUXIMAB_MONO", "CONFERS_RESISTANCE_TO", "Level R1", "PMID:18316791", "CIViC:EID105", "Constitutive KRAS activation bypasses upstream EGFR blockade."),
            ("MUT:KRAS_G12C", "DRUG:SOTORASIB_PANITUMUMAB", "SENSITIZES", "Level 1", "PMID:37870968", "CIViC:EID5102", "Covalent KRAS G12C inhibition paired with anti-EGFR feedback suppression (CodeBreaK 300)."),
            ("MUT:PIK3CA_H1047R", "DRUG:ALPELISIB_FULVESTRANT", "SENSITIZES", "Level 1", "PMID:31091374", "CIViC:EID2890", "PI3K-alpha selective inhibition in PIK3CA-mutated tumors (SOLAR-1)."),
        ]
        for src, dst, rel, level, pmid, civic, rationale in therapeutic_edges:
            self.add_edge(
                src,
                dst,
                rel,
                oncokb_level=level,
                pmid=pmid,
                civic_id=civic,
                rationale=rationale,
            )

    def expand_multi_hop_from_alterations(self, alteration_ids: List[str]) -> List[MultiHopPath]:
        paths: List[MultiHopPath] = []
        for alt_id in alteration_ids:
            if alt_id not in self.nodes:
                continue
            alt_node = self.nodes[alt_id]
            for edge in self.get_outgoing(alt_id):
                if edge.relation not in ("SENSITIZES", "CONFERS_RESISTANCE_TO"):
                    continue
                drug_node = self.nodes[edge.target_id]
                trial_edges = self.get_outgoing(drug_node.node_id, relation="ACTIVE_IN")
                if not trial_edges:
                    cypher_str = (
                        f"MATCH (a:{alt_node.label} {{id: '{alt_id}'}})"
                        f"-[r:{edge.relation}]->(d:Drug {{id: '{drug_node.node_id}'}}) RETURN a, r, d"
                    )
                    paths.append(
                        MultiHopPath(
                            alteration_node=alt_node,
                            relationship=edge.relation,
                            drug_node=drug_node,
                            trial_node=None,
                            oncokb_level=edge.properties.get("oncokb_level", "Level 2"),
                            pmid=edge.properties.get("pmid", "PMID:31566309"),
                            civic_id=edge.properties.get("civic_id", "CIViC:EID100"),
                            rationale=edge.properties.get("rationale", ""),
                            cypher_trace=cypher_str,
                        )
                    )
                else:
                    for t_edge in trial_edges:
                        trial_node = self.nodes[t_edge.target_id]
                        cypher_str = (
                            f"MATCH (a:{alt_node.label} {{id: '{alt_id}'}})"
                            f"-[r:{edge.relation}]->(d:Drug {{id: '{drug_node.node_id}'}})"
                            f"-[:ACTIVE_IN]->(t:ClinicalTrial {{nct_id: '{trial_node.properties.get('nct_id')}'}}) "
                            f"RETURN a, r, d, t"
                        )
                        paths.append(
                            MultiHopPath(
                                alteration_node=alt_node,
                                relationship=edge.relation,
                                drug_node=drug_node,
                                trial_node=trial_node,
                                oncokb_level=edge.properties.get("oncokb_level", "Level 1"),
                                pmid=edge.properties.get("pmid", trial_node.properties.get("pmid", "")),
                                civic_id=edge.properties.get("civic_id", "CIViC:EID100"),
                                rationale=edge.properties.get("rationale", ""),
                                cypher_trace=cypher_str,
                            )
                        )
        return paths
