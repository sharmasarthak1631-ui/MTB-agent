# Molecular Tumor Board (MTB) Multi-Model Agentic Copilot

A federated multi-agent precision oncology decision support system that synthesizes histopathology whole-slide imaging (`.svs`/`.dicom`), somatic genomic variant files (`.vcf`), and unstructured Electronic Health Records (`.json`/`.txt`) through Cross-Attention Fusion, Hybrid GraphRAG (pgvector HNSW + Neo4j Cypher multi-hop traversal), and Deterministic Guardrail Verification.

## Architecture & Module Map

- `app.py`: Clean, minimalist Streamlit interface with zero emoji clutter. Provides real-time patient state controls (`eGFR`, `ECOG`), model backbone selectors, cross-attention matrix inspection, GraphRAG trial pruning traces, and one-click MTB dossier export.
- `src/ingestion/pipeline.py`:
  - `MinIOChunkedStorageEngine`: Splits large binary payloads into deterministic S3 chunks with SHA-256 part hashes and Merkle root verification so raw binaries bypass message brokers.
  - `KafkaStreamBroker`: Partitions lightweight payload manifests across `dicom-stream`, `vcf-stream`, and `ehr-stream`.
  - `TemporalWorkflowExecutor`: State-machine workflow coordinator with deterministic rollback compensation stack.
- `src/agents/`:
  - `vision_specialist.py`: Simulates `OpenSlide`/`cuCIM` 20x/40x magnification tiling and foundation pathology encoders (`UNI (ViT-L/16)`, `Prov-GigaPath`, `CONCH`), quantifying tumor cellularity, TIL density, necrosis, and MSI/HRD morphology probabilities.
  - `genomic_specialist.py`: Parses VCF 4.2 somatic variants, integrates `HyenaDNA-1M` / `DNABERT-2` sequence pathogenicity scoring with `VEP`, `ClinVar`, `SIFT`, and `PolyPhen-2`, flags VUS candidates, and calculates Tumor Mutational Burden (TMB).
  - `clinical_specialist.py`: Extracts longitudinal therapy timelines, ECOG performance status, RECIST v1.1 evaluations, organ function labs (`eGFR`, creatinine, LVEF, QTc), and active toxicities using `BioMistral-7B-Clinical` / `Llama-3-Med-70B`.
- `src/fusion/cross_attention.py`:
  - `MultiModalCrossAttentionHub`: Computes scaled dot-product cross-attention across WSI patch vectors and genomic variant tokens, resolving ambiguous VUS alterations (e.g., high TIL density + *MSH2* `p.A636P` VUS -> actionable Mismatch Repair Deficiency `dMMR / MSI-H`).
- `src/storage/` & `src/graphrag/`:
  - `vector_store.py`: Multi-layer HNSW (`M=16`, `ef_construction=100`) vector index with PostgreSQL `pgvector` schema generation.
  - `knowledge_graph.py`: Biomedical Knowledge Hypergraph modeling `Gene`, `Mutation`, `Biomarker`, `Drug`, `ClinicalTrial`, `EligibilityCriteria`, and `AdverseEvent` nodes with `SENSITIZES`, `CONFERS_RESISTANCE_TO`, `ACTIVE_IN`, `REQUIRES`, and `EXCLUDES_CONDITION` edges.
  - `retriever.py`: Executes hybrid vector filtering, multi-hop Cypher expansion, and deterministic hard exclusion pruning (e.g., pruning trials where `Patient.eGFR < Trial.REQUIRES(min_eGFR)` or where grade 2+ neuropathy is excluded).
- `src/verification/guardrails.py`:
  - `DualLayerVerificationEngine`: Programmatic assertion engine verifying mandatory `PMID`, `OncoKB Level`, `CIViC ID`, and 8-digit `NCT` citations, renal/ECOG safety gates, resistance non-contradiction, and FDA label dosing.

## Quickstart

### 1. Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Run Unit & Integration Tests

```bash
python3 tests/test_pipeline.py
```

### 3. Run via Docker Compose

```bash
docker compose up --build
```
