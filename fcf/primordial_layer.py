"""
PrimordialLayer — фундаментальный вычислительный элемент ЕВА.

Объединяет:
- Embedding (токены → векторы)
- TransformerBlock (Causal Self-Attention + SwiGLU FFN + RMSNorm)
- LM Head (векторы → логиты, weight tied с embedding)
- StateStorage (FAISS-хранилище успешных состояний)
- SemanticRelevanceGate (самооценка)
- EthicsFilter (этический фильтр)
- MetaMemory (мета-память)
- GrowthController (контроллер роста)
- CuriosityLoop (генерация уточняющих вопросов)

Это единственный слой на старте системы. Все последующие слои
создаются как его копии с пустыми хранилищами.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger

from .config import FCFConfig, load_config
from .transformer import TransformerBlock
from .state_storage import StateStorage
from .srg import SemanticRelevanceGate
from .meta_memory import MetaMemory
from .growth_controller import GrowthController
from .curiosity_loop import CuriosityLoop
from .domain_registry import DomainRegistry
from .lora_adapter import LoRAAdapter


class PrimordialLayer(nn.Module):

    def __init__(self, config: FCFConfig = None):
        super().__init__()
        self.config = config or load_config()

        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.d_model
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.transformer = TransformerBlock(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            ff_mult=self.config.ff_mult,
            max_seq_len=self.config.max_seq_len,
        )

        self.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embedding.weight

        self.state_storage = StateStorage(
            dim=self.config.d_model,
            max_snapshots=self.config.max_snapshots,
        )

        self.srg = SemanticRelevanceGate(
            w_sim=self.config.srg.w_sim,
            w_ent=self.config.srg.w_ent,
            w_eth=self.config.srg.w_eth,
            ethics_threshold=self.config.srg.ethics_threshold,
        )

        self.meta = MetaMemory(history_size=100)
        self.growth = GrowthController(
            width_threshold=self.config.growth.width_threshold,
            depth_threshold=self.config.growth.depth_threshold,
            gradient_threshold=self.config.growth.gradient_threshold,
            patience=self.config.growth.patience,
        )
        self.curiosity = CuriosityLoop(
            threshold=self.config.curiosity.threshold,
        )

        self.domain_registry = DomainRegistry()
        self._active_adapter: Optional[LoRAAdapter] = None
        self._adapter_backup: Dict[str, torch.Tensor] = {}

        self._device = torch.device("cpu")
        self.layer_idx: int = 0

        self._eval_context_vector: Optional[np.ndarray] = None
        self._eval_response_vector: Optional[np.ndarray] = None
        self._eval_logits: Optional[np.ndarray] = None
        self._eval_text: str = ""

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward_transformer(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        return self.transformer(x, attention_mask)

    def forward_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    @torch.no_grad()
    def _encode(self, tokenizer, text: str, device=None) -> torch.Tensor:
        encoding = tokenizer.encode(text)
        ids = encoding.ids if hasattr(encoding, 'ids') else encoding
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        elif isinstance(ids, torch.Tensor):
            ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
        else:
            ids = torch.tensor([list(ids)], dtype=torch.long)
        if device is not None:
            ids = ids.to(device)
        return ids

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            if seq_len > self.config.max_seq_len:
                break

            x = self.embed(input_ids)
            hidden = self.transformer(x)
            logits = self.lm_head(hidden)
            logits_last = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                topk_vals, topk_idx = torch.topk(
                    logits_last, min(top_k, logits_last.shape[-1])
                )
                mask = torch.full_like(logits_last, float("-inf"))
                mask.scatter_(-1, topk_idx, topk_vals)
                logits_last = mask

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                cutoff = (cumsum > top_p).float()
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = 0.0
                sorted_logits[cutoff.bool()] = float("-inf")
                logits_last = sorted_logits.gather(-1, sorted_idx.argsort(-1))

            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids

    def get_context_vector(self, input_ids: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = self.embed(input_ids)
            hidden = self.transformer(x)
            last_hidden = hidden[:, -1, :]
        return last_hidden.squeeze(0).cpu().numpy()

    def evaluate_response(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        response_text: str = "",
    ) -> dict:
        c_query = self.get_context_vector(query_ids)
        c_response = self.get_context_vector(response_ids)

        self.eval()
        with torch.no_grad():
            x = self.embed(response_ids)
            hidden = self.transformer(x)
            logits = self.lm_head(hidden)
            last_logits = logits[:, -1, :].squeeze(0).cpu().numpy()

        result = self.srg.evaluate_full(
            c_query=c_query,
            c_response=c_response,
            logits=last_logits,
            response_text=response_text,
        )

        self._eval_context_vector = c_query
        self._eval_response_vector = c_response
        self._eval_logits = last_logits
        self._eval_text = response_text

        self.meta.record(result["confidence"])

        return result

    def save_snapshot_if_confident(self, domain: str = "general") -> bool:
        if self._eval_context_vector is None:
            return False

        if not self.meta.confidence_history:
            return False

        confidence = self.meta.confidence_history[-1]
        if confidence < self.config.srg.snapshot_confidence_threshold:
            return False

        K = np.zeros(
            (self.config.num_heads, self.config.head_dim), dtype=np.float32
        )
        V = np.zeros(
            (self.config.num_heads, self.config.head_dim), dtype=np.float32
        )

        idx = self.state_storage.add(
            c=self._eval_context_vector,
            K=K,
            V=V,
            confidence=confidence,
            domain=domain,
        )
        logger.debug(f"[Snapshot] Сохранён слепок #{idx}, confidence={confidence:.3f}")
        return True

    def process_query(
        self,
        query: str,
        tokenizer,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        domain: str = "general",
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        query_ids = self._encode(tokenizer, query, device)

        c_query = self.get_context_vector(query_ids)

        snapshot_idx = self.state_storage.search(c_query, threshold=0.95)
        if snapshot_idx >= 0:
            logger.debug(f"[Query] Точное совпадение в хранилище (idx={snapshot_idx})")

        matched_domain = self.domain_registry.find_best(c_query)
        if matched_domain:
            logger.info(f"[Domain] Найден домен: {matched_domain}")
            rule = self.domain_registry.get_rule(matched_domain)
            if rule and os.path.exists(rule.adapter_path):
                try:
                    adapter = LoRAAdapter.load(rule.adapter_path)
                    self._adapter_backup = adapter.apply_to_layer(
                        self.transformer.attention
                    )
                    ffn_backup = adapter.apply_to_layer(self.transformer.ffn)
                    self._adapter_backup.update(ffn_backup)
                    self._active_adapter = adapter
                    logger.debug(f"[Domain] Адаптер {matched_domain} применён")
                except Exception as e:
                    logger.warning(f"[Domain] Ошибка загрузки адаптера: {e}")

        response_ids = self.generate(
            query_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        response_text = tokenizer.decode(
            response_ids[0].tolist(), skip_special_tokens=True
        )

        if self._active_adapter is not None:
            self._active_adapter.remove_from_layer(
                self.transformer.attention, self._adapter_backup
            )
            self._active_adapter.remove_from_layer(
                self.transformer.ffn, self._adapter_backup
            )
            self._active_adapter = None
            self._adapter_backup = {}

        eval_result = self.evaluate_response(
            query_ids=query_ids,
            response_ids=response_ids,
            response_text=response_text,
        )

        if matched_domain:
            rule = self.domain_registry.get_rule(matched_domain)
            if rule:
                rule.record_confidence(eval_result["confidence"])
                rule.update_centroid(c_query)

        self.save_snapshot_if_confident(domain=matched_domain or domain)

        snapshot_idx = self.state_storage.search(
            self._eval_context_vector,
            threshold=0.7,
        )

        if eval_result["confidence"] < 0.6:
            if self.curiosity.should_ask(eval_result["confidence"]):
                question = self.curiosity.generate_clarification(
                    layer=self,
                    tokenizer=tokenizer,
                    original_query=query,
                    generated_answer=response_text,
                )
                self.curiosity.add_pending_clarification(
                    query=query,
                    answer=response_text,
                    question=question,
                )
                self.meta.last_clarification = question
            else:
                question = None
        else:
            question = None
            self.curiosity.reset()

        gradient_norm = self._compute_gradient_norm()
        growth_signal = self.growth.evaluate(
            self.meta, gradient_norm=gradient_norm
        )

        return {
            "response": response_text,
            "confidence": eval_result["confidence"],
            "similarity": eval_result["similarity"],
            "entropy_score": eval_result["entropy_score"],
            "ethics_score": eval_result["ethics_score"],
            "axiom_scores": eval_result["axiom_scores"],
            "snapshot_idx": snapshot_idx if snapshot_idx >= 0 else None,
            "clarification_question": question,
            "growth_signal": growth_signal,
            "timestamp": time.time(),
        }

    def _compute_gradient_norm(self) -> float:
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return total_norm ** 0.5

    def to(self, device):
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        return super().to(device)

    @property
    def device(self):
        return self._device

    def summary(self) -> str:
        return (
            f"PrimordialLayer("
            f"d_model={self.config.d_model}, "
            f"num_heads={self.config.num_heads}, "
            f"snapshots={len(self.state_storage)}, "
            f"avg_confidence={self.meta.average_confidence():.3f}, "
            f"usage={self.meta.usage_count}"
            f")"
        )

    def __repr__(self) -> str:
        return self.summary()
