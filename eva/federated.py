"""
FederatedFabric + Collaborative SRG.

Позволяет экземплярам FCF обмениваться латентными кодами
без раскрытия исходных данных через федеративный протокол.
"""

import numpy as np
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FederatedUpdate:
    """Сообщение между экземплярами FCF."""
    instance_id: str
    domain_centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    srg_stats: Dict[str, float] = field(default_factory=dict)
    code_count: int = 0
    timestamp: float = 0.0

    def serialize(self) -> bytes:
        data = {
            "instance_id": self.instance_id,
            "domain_centroids": {
                k: v.tolist() for k, v in self.domain_centroids.items()
            },
            "srg_stats": self.srg_stats,
            "code_count": self.code_count,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "FederatedUpdate":
        obj = json.loads(data.decode("utf-8"))
        return cls(
            instance_id=obj["instance_id"],
            domain_centroids={
                k: np.array(v, dtype=np.float32)
                for k, v in obj["domain_centroids"].items()
            },
            srg_stats=obj["srg_stats"],
            code_count=obj["code_count"],
            timestamp=obj["timestamp"],
        )


class CollaborativeSRG:
    """
    Федеративное обучение SRG: несколько экземпляров обмениваются только
    градиентами и центроидами, не раскрывая данные пользователей.
    """

    def __init__(self, instance_id: str = "default"):
        self.instance_id = instance_id
        self.local_srg_stats: Dict[str, List[float]] = {}
        self.federated_updates: List[FederatedUpdate] = []

    def record(self, code_id: str, score: float):
        if code_id not in self.local_srg_stats:
            self.local_srg_stats[code_id] = []
        self.local_srg_stats[code_id].append(score)
        if len(self.local_srg_stats[code_id]) > 100:
            self.local_srg_stats[code_id].pop(0)

    def get_local_stats(self) -> Dict[str, float]:
        return {
            cid: float(np.mean(scores))
            for cid, scores in self.local_srg_stats.items()
            if scores
        }

    def receive_update(self, update: FederatedUpdate):
        self.federated_updates.append(update)
        if len(self.federated_updates) > 100:
            self.federated_updates.pop(0)
        logger.debug(
            f"[Federated] Update from {update.instance_id}: "
            f"{len(update.domain_centroids)} domains, "
            f"{update.code_count} codes"
        )

    def merge(self, gmm, alpha: float = 0.1):
        """Влить федеративные центроиды в локальный GMM."""
        for update in self.federated_updates[-10:]:
            for domain_id, centroid in update.domain_centroids.items():
                if domain_id in gmm.domains:
                    gmm.domains[domain_id].update(centroid, alpha)
                else:
                    gmm.add_or_update(centroid)


class FederatedFabric:
    """Федеративный протокол FCF."""

    def __init__(self, instance_id: str = "default"):
        self.instance_id = instance_id
        self.peers: Dict[str, float] = {}
        self.outgoing: List[FederatedUpdate] = []

    def register_peer(self, peer_id: str):
        self.peers[peer_id] = 0.0

    def prepare_update(self, gmm, srg_stats: Dict[str, float]) -> FederatedUpdate:
        import time
        centroids = {}
        for level, level_gmm in gmm.gmms.items() if hasattr(gmm, 'gmms') else {}:
            for did, dom in level_gmm.domains.items():
                centroids[f"{level}_{did}"] = dom.centroid.copy()

        return FederatedUpdate(
            instance_id=self.instance_id,
            domain_centroids=centroids,
            srg_stats=srg_stats,
            code_count=sum(len(d) for d in centroids.values()),
            timestamp=time.time(),
        )

    def broadcast(self, update: FederatedUpdate) -> bytes:
        self.outgoing.append(update)
        return update.serialize()

    def receive(self, data: bytes):
        update = FederatedUpdate.deserialize(data)
        self.peers[update.instance_id] = update.timestamp
        return update
