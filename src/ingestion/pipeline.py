from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class WorkflowState(str, Enum):
    PENDING = "PENDING"
    INGESTING_S3 = "INGESTING_S3"
    KAFKA_QUEUED = "KAFKA_QUEUED"
    PROCESSING_VISION = "PROCESSING_VISION"
    PROCESSING_GENOMICS = "PROCESSING_GENOMICS"
    PROCESSING_CLINICAL = "PROCESSING_CLINICAL"
    CROSS_ATTENTION_FUSION = "CROSS_ATTENTION_FUSION"
    GRAPHRAG_TRAVERSAL = "GRAPHRAG_TRAVERSAL"
    GUARDRAIL_VERIFICATION = "GUARDRAIL_VERIFICATION"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


@dataclass
class S3ChunkDescriptor:
    part_number: int
    byte_offset: int
    byte_length: int
    sha256_hash: str
    object_uri: str


@dataclass
class PayloadManifest:
    manifest_id: str
    patient_id: str
    modality: str
    filename: str
    total_bytes: int
    merkle_root_sha256: str
    presigned_endpoint: str
    bucket: str
    object_key: str
    chunks: List[S3ChunkDescriptor] = field(default_factory=list)
    created_at_epoch: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KafkaMessageEnvelope:
    topic: str
    partition: int
    offset: int
    key: str
    manifest_id: str
    patient_id: str
    modality: str
    merkle_root_sha256: str
    s3_uri: str
    timestamp_ms: int


@dataclass
class WorkflowStepLog:
    step_name: str
    state: str
    worker_queue: str
    duration_ms: float
    details: str
    compensating_action: str


class MinIOChunkedStorageEngine:
    def __init__(self, endpoint: str = "https://minio.mtb-cluster.internal:9000", bucket: str = "mtb-raw-modalities", chunk_size_bytes: int = 4096):
        self.endpoint = endpoint
        self.bucket = bucket
        self.chunk_size_bytes = chunk_size_bytes
        self._object_store: Dict[str, bytes] = {}
        self._manifests: Dict[str, PayloadManifest] = {}

    def upload_chunked_payload(
        self,
        patient_id: str,
        modality: str,
        filename: str,
        raw_bytes: bytes,
    ) -> PayloadManifest:
        manifest_id = f"mnf-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{patient_id}-{modality}-{filename}').hex[:12]}"
        object_key = f"{patient_id}/{modality}/{filename}"
        chunks: List[S3ChunkDescriptor] = []
        chunk_hashes: List[str] = []

        total_len = len(raw_bytes)
        if total_len == 0:
            raw_bytes = b"{}"
            total_len = len(raw_bytes)

        part_num = 1
        for offset in range(0, total_len, self.chunk_size_bytes):
            slice_bytes = raw_bytes[offset : offset + self.chunk_size_bytes]
            chunk_hash = hashlib.sha256(slice_bytes).hexdigest()
            chunk_hashes.append(chunk_hash)
            chunk_uri = f"s3://{self.bucket}/{object_key}?partNumber={part_num}"
            chunks.append(
                S3ChunkDescriptor(
                    part_number=part_num,
                    byte_offset=offset,
                    byte_length=len(slice_bytes),
                    sha256_hash=chunk_hash,
                    object_uri=chunk_uri,
                )
            )
            part_num += 1

        combined_hashes = "".join(chunk_hashes).encode("utf-8")
        merkle_root = hashlib.sha256(combined_hashes).hexdigest()
        presigned_url = f"{self.endpoint}/{self.bucket}/{object_key}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-SignedHeaders=host&manifest={manifest_id}"

        manifest = PayloadManifest(
            manifest_id=manifest_id,
            patient_id=patient_id,
            modality=modality,
            filename=filename,
            total_bytes=total_len,
            merkle_root_sha256=merkle_root,
            presigned_endpoint=presigned_url,
            bucket=self.bucket,
            object_key=object_key,
            chunks=chunks,
        )
        self._object_store[object_key] = raw_bytes
        self._manifests[manifest_id] = manifest
        return manifest

    def verify_and_retrieve(self, manifest_id: str) -> bytes:
        manifest = self._manifests.get(manifest_id)
        if not manifest:
            raise KeyError(f"Manifest {manifest_id} not found in object store.")
        raw_bytes = self._object_store[manifest.object_key]
        recomputed_hashes = []
        for ch in manifest.chunks:
            segment = raw_bytes[ch.byte_offset : ch.byte_offset + ch.byte_length]
            h = hashlib.sha256(segment).hexdigest()
            if h != ch.sha256_hash:
                raise ValueError(f"Cryptographic hash mismatch at chunk {ch.part_number}")
            recomputed_hashes.append(h)
        merkle_check = hashlib.sha256("".join(recomputed_hashes).encode("utf-8")).hexdigest()
        if merkle_check != manifest.merkle_root_sha256:
            raise ValueError("Merkle root verification failed for S3 payload.")
        return raw_bytes


class KafkaStreamBroker:
    TOPICS = {
        "dicom": "dicom-stream",
        "vcf": "vcf-stream",
        "ehr": "ehr-stream",
    }

    def __init__(self, num_partitions: int = 8):
        self.num_partitions = num_partitions
        self._topic_logs: Dict[str, List[KafkaMessageEnvelope]] = {
            t: [] for t in self.TOPICS.values()
        }

    def publish_manifest(self, manifest: PayloadManifest) -> KafkaMessageEnvelope:
        topic = self.TOPICS.get(manifest.modality, "ehr-stream")
        partition = int(hashlib.md5(manifest.patient_id.encode("utf-8")).hexdigest(), 16) % self.num_partitions
        offset = len(self._topic_logs[topic])
        envelope = KafkaMessageEnvelope(
            topic=topic,
            partition=partition,
            offset=offset,
            key=manifest.patient_id,
            manifest_id=manifest.manifest_id,
            patient_id=manifest.patient_id,
            modality=manifest.modality,
            merkle_root_sha256=manifest.merkle_root_sha256,
            s3_uri=f"s3://{manifest.bucket}/{manifest.object_key}",
            timestamp_ms=int(time.time() * 1000),
        )
        self._topic_logs[topic].append(envelope)
        return envelope

    def get_topic_messages(self, topic: str) -> List[KafkaMessageEnvelope]:
        return list(self._topic_logs.get(topic, []))


class TemporalWorkflowExecutor:
    def __init__(self):
        self.current_state: WorkflowState = WorkflowState.PENDING
        self.execution_log: List[WorkflowStepLog] = []
        self._compensation_stack: List[Callable[[], None]] = []

    def run_step(
        self,
        step_name: str,
        target_state: WorkflowState,
        worker_queue: str,
        fn: Callable[[], Any],
        compensate_desc: str,
        compensate_fn: Optional[Callable[[], None]] = None,
    ) -> Any:
        t0 = time.perf_counter()
        self.current_state = target_state
        try:
            result = fn()
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self.execution_log.append(
                WorkflowStepLog(
                    step_name=step_name,
                    state=target_state.value,
                    worker_queue=worker_queue,
                    duration_ms=elapsed_ms,
                    details="Committed deterministically",
                    compensating_action=compensate_desc,
                )
            )
            if compensate_fn:
                self._compensation_stack.append(compensate_fn)
            return result
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self.execution_log.append(
                WorkflowStepLog(
                    step_name=step_name,
                    state=WorkflowState.FAILED.value,
                    worker_queue=worker_queue,
                    duration_ms=elapsed_ms,
                    details=f"Exception: {exc}",
                    compensating_action=compensate_desc,
                )
            )
            self.rollback()
            raise

    def rollback(self) -> None:
        while self._compensation_stack:
            comp = self._compensation_stack.pop()
            try:
                comp()
            except Exception:
                pass
        self.current_state = WorkflowState.ROLLED_BACK
