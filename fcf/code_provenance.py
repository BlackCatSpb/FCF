"""
CodeProvenance — отслеживание происхождения латентных кодов.

Каждый код хранит цепочку: из какого домена, через какой оператор
State Algebra, после скольких итераций KCA он был создан.
Позволяет аудит и обратимость эволюции знаний.
"""

import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ProvenanceRecord:
    code_id: str
    domain_id: str
    level: str
    created_via: str
    parent_codes: List[str] = field(default_factory=list)
    kca_iterations: int = 0
    final_srg: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "code_id": self.code_id,
            "domain_id": self.domain_id,
            "level": self.level,
            "created_via": self.created_via,
            "parent_codes": self.parent_codes,
            "kca_iterations": self.kca_iterations,
            "final_srg": self.final_srg,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class CodeProvenance:
    def __init__(self):
        self.records: Dict[str, ProvenanceRecord] = {}
        self._lineage: Dict[str, List[str]] = {}

    def record(
        self,
        code_id: str,
        domain_id: str,
        level: str,
        created_via: str = "direct",
        parent_codes: List[str] = None,
        kca_iterations: int = 0,
        final_srg: float = 0.0,
        **metadata,
    ):
        rec = ProvenanceRecord(
            code_id=code_id,
            domain_id=domain_id,
            level=level,
            created_via=created_via,
            parent_codes=parent_codes or [],
            kca_iterations=kca_iterations,
            final_srg=final_srg,
            metadata=metadata,
        )
        self.records[code_id] = rec

        for parent in rec.parent_codes:
            if parent not in self._lineage:
                self._lineage[parent] = []
            self._lineage[parent].append(code_id)

    def get_lineage(self, code_id: str) -> List[str]:
        result = []
        visited = set()

        def _trace(cid):
            if cid in visited:
                return
            visited.add(cid)
            for child in self._lineage.get(cid, []):
                result.append(child)
                _trace(child)

        _trace(code_id)
        return result

    def can_reconstruct(self, code_id: str) -> bool:
        if code_id not in self.records:
            return False
        rec = self.records[code_id]
        if rec.created_via == "direct":
            return True
        return all(p in self.records for p in rec.parent_codes)

    def summary(self, code_id: str) -> str:
        if code_id not in self.records:
            return f"Code {code_id}: no record"
        rec = self.records[code_id]
        return (
            f"Code {code_id}: via={rec.created_via}, "
            f"domain={rec.domain_id}, level={rec.level}, "
            f"parents={len(rec.parent_codes)}, kca={rec.kca_iterations}, "
            f"srg={rec.final_srg:.3f}"
        )
