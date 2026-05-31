"""
PotentialTrainer — обучение через накопление потенциалов связей.

НЕ кросс-энтропия next-token prediction.
Вместо этого:
1. Подаём последовательность символов
2. Трансформер генерирует attention matrix
3. PotentialField обновляется: связи с высоким attention усиливаются
4. SemanticClosureChecker верифицирует логичность сборки
5. AssemblyState сохраняется как "опыт"

Цикл:
  for text in data:
    symbols = tokenize(text)
    attention = transformer(symbols)
    state = build_assembly(symbols, attention)
    if closure_check(state):
      save_state(state)
      strengthen_potentials(state)
    else:
      weaken_potentials(state)  # наказываем за нелогичную сборку
"""

import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
from loguru import logger

from .char_vocab import CharacterVocab
from .potential_field import PotentialField
from .potential_dynamics import PotentialDynamics
from .assembly_graph import AssemblyState, AssemblyEdge
from .assembly_grammar import AssemblyGrammar
from .semantic_closure import SemanticClosureChecker
from .continuation_propagator import ContinuationPropagator
from .logic_bridge import LogicBridge
from .assembly_validator import AssemblyValidator
from .assembly_explorer import AssemblyExplorer
from .sleep_mode_symbolic import SleepModeSymbolic
from .topological_field import TopologicalField
from .natural_clusterer import NaturalClusterer
from .geodesic_navigator import GeodesicNavigator, TangentSpace
from .curvature_analyzer import CurvatureAnalyzer
from .contradiction_filter import SymbolicContradictionFilter, ContradictionType
from .concept_miner import SymbolicConceptMiner


class PotentialTrainer:
    """
    Обучает модель не предсказывать токены, а накапливать
    потенциалы связей между символами.

    Парадигма: опыт > инструкция > логика, а не loss → градиент → обновление.
    """

    def __init__(
        self,
        layer,                           # PrimordialLayer с трансформером
        char_vocab: CharacterVocab,
        embed_dim: int = 256,
        checkpoint_dir: str = None,
        max_states: int = 50000,
    ):
        self.layer = layer
        self.char_vocab = char_vocab
        self.embed_dim = embed_dim
        self.checkpoint_dir = checkpoint_dir or "checkpoints/symbolic"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Core symbolic components
        self.potential_field = PotentialField(
            vocab_size=char_vocab.vocab_size,
            embed_dim=embed_dim,
        )
        self.dynamics = PotentialDynamics(
            vocab_size=char_vocab.vocab_size,
            embed_dim=embed_dim,
        )
        self.semantic_checker = SemanticClosureChecker()
        self.propagator = ContinuationPropagator(
            potential_field=self.potential_field,
            semantic_checker=self.semantic_checker,
            embed_dim=embed_dim,
        )
        self.grammar = AssemblyGrammar(
            potential_field=self.potential_field,
            vocab_size=char_vocab.vocab_size,
            embed_dim=embed_dim,
        )
        # Топология
        self.topological_field = TopologicalField(
            potential_field=self.potential_field,
            coord_dim=embed_dim,
        )
        # Двойной движок
        self.contradiction_filter = SymbolicContradictionFilter(
            potential_field=self.potential_field,
            topological_field=self.topological_field,
        )
        self.logic_bridge = LogicBridge(
            potential_field=self.potential_field,
            vocab_size=char_vocab.vocab_size,
        )
        self.validator = AssemblyValidator(
            potential_field=self.potential_field,
            semantic_checker=self.semantic_checker,
            logic_bridge=self.logic_bridge,
            grammar=self.grammar,
        )
        self.explorer = AssemblyExplorer(
            potential_field=self.potential_field,
            grammar=self.grammar,
            validator=self.validator,
        )
        self.tangent_space = TangentSpace(
            potential_field=self.potential_field,
            topological_field=self.topological_field,
        )
        self.geodesic_nav = GeodesicNavigator(
            potential_field=self.potential_field,
            topological_field=self.topological_field,
            tangent_space=self.tangent_space,
        )
        self.concept_miner = SymbolicConceptMiner(
            potential_field=self.potential_field,
            topological_field=self.topological_field,
            contradiction_filter=self.contradiction_filter,
            grammar=self.grammar,
            logic_bridge=self.logic_bridge,
            geodesic_navigator=self.geodesic_nav,
        )
        self.curvature_analyzer = CurvatureAnalyzer(
            potential_field=self.potential_field,
            topological_field=self.topological_field,
        )
        self.clusterer = NaturalClusterer(
            potential_field=self.potential_field,
            topological_field=self.topological_field,
        )
        self.sleep_mode = SleepModeSymbolic(
            potential_dynamics=self.dynamics,
            grammar=self.grammar,
            validator=self.validator,
            explorer=self.explorer,
            logic_bridge=self.logic_bridge,
        )

        # State storage
        self.states: List[AssemblyState] = []
        self.max_states = max_states
        self.total_assemblies: int = 0
        self.valid_assemblies: int = 0
        self.invalid_assemblies: int = 0

        # Training stats
        self.step: int = 0
        self.best_coherence: float = 0.0
        self.coherence_history: deque = deque(maxlen=100)
        self.avg_potential_strength_history: deque = deque(maxlen=100)

    def build_assembly(
        self,
        symbol_ids: List[int],
        attention_matrix: Optional[np.ndarray] = None,
    ) -> AssemblyState:
        """
        Строит инструкцию сборки из символьной последовательности и attention.

        Если attention_matrix не передана — генерируем через трансформер.
        """
        state = AssemblyState(
            symbols=symbol_ids,
            timestamp=time.time(),
        )

        n = len(symbol_ids)

        # Если attention не предоставлен, генерируем через forward pass
        if attention_matrix is None:
            device = next(self.layer.parameters()).device
            inp = torch.tensor([symbol_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                self.layer.eval()
                x = self.layer.embed(inp)
                hidden = self.layer.forward_transformer(x)
                # Извлекаем attention из последнего прохода
                attn = self.layer.transformer.attention.last_attention
                if attn is not None:
                    # [B, H, L, L] → [L, L]
                    attention_matrix = attn[0].mean(dim=0).cpu().numpy()
                    attention_matrix = attention_matrix[:n, :n]
                else:
                    attention_matrix = np.eye(n)

        state.attention_matrix = attention_matrix

        # Строим граф: добавляем связи между символами с высоким вниманием
        for i in range(n):
            for j in range(i + 1, n):
                attn_w = float(attention_matrix[i, j])
                if attn_w < 0.01:
                    continue

                # Аффинность из PotentialField
                if i < self.potential_field.vocab_size and j < self.potential_field.vocab_size:
                    affinity = float(self.potential_field.affinity[symbol_ids[i], symbol_ids[j]])
                else:
                    affinity = 0.5

                # Семантическая близость
                pi = self.propagator.compute_symbol_potential(symbol_ids[i])
                pj = self.propagator.compute_symbol_potential(symbol_ids[j])
                sem_score = float(np.dot(pi, pj) / (np.linalg.norm(pi) * np.linalg.norm(pj) + 1e-8))

                edge = AssemblyEdge(
                    src_idx=i,
                    dst_idx=j,
                    attention_weight=attn_w,
                    affinity=affinity,
                    semantic_score=sem_score,
                )
                state.edges.append(edge)

        # Проверяем связность
        symbol_potentials, seq_potential = self.propagator.propagate_sequence(
            symbol_ids, attention_matrix
        )
        passed, score, components = self.semantic_checker.full_check(
            symbol_potentials, attention_matrix, seq_potential
        )
        state.coherence_score = score
        state.confidence = score

        return state

    def train_on_text(
        self,
        text: str,
        max_len: int = 256,
    ) -> AssemblyState:
        """
        Обрабатывает один текст: строит сборку, проверяет, обновляет потенциалы.
        """
        symbol_ids = self.char_vocab.encode(text)[:max_len]

        # Шаг 1: прямой проход для получения attention
        device = next(self.layer.parameters()).device
        inp = torch.tensor([symbol_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            self.layer.eval()
            x = self.layer.embed(inp)
            hidden = self.layer.forward_transformer(x)
            attn = self.layer.transformer.attention.last_attention

        attention_matrix = None
        if attn is not None:
            n = len(symbol_ids)
            attention_matrix = attn[0].mean(dim=0).cpu().numpy()[:n, :n]

        # Шаг 2: построить сборку
        state = self.build_assembly(symbol_ids, attention_matrix)

        # Шаг 3: ДУАЛЬНЫЙ ДВИЖОК
        if state.coherence_score > 0.5:
            # Валидная сборка — усиливаем связи
            if attention_matrix is not None and attn is not None:
                self.potential_field.strengthen_batch(
                    inp, attn, confidence=state.coherence_score
                )
                self.dynamics.reinforce_sequence(symbol_ids, confidence=state.coherence_score)

            self.valid_assemblies += 1
            self.semantic_checker.add_valid_potential(
                self.propagator.propagate_sequence(symbol_ids, attention_matrix)[1]
            )

            # Запись в sleep buffer
            self.sleep_mode.record_assembly(symbol_ids)

        else:
            # Нелогичная сборка → ищем противоречия
            self.potential_field.weaken_all(factor=0.999)
            self.invalid_assemblies += 1

            # Детектируем и запоминаем противоречия
            contras = self.contradiction_filter.detect_structural_contradictions(symbol_ids)

            # Семантические противоречия: каждые 100 шагов
            if self.step % 100 == 0:
                for si in symbol_ids[:5]:
                    self.contradiction_filter.detect_semantic_contradictions(si)

        # Шаг 4: каждые 500 шагов — поиск концептов в свободном пространстве
        if self.step % 500 == 0 and self.step > 0:
            concepts = self.concept_miner.search_free_space(max_tries=5)
            if concepts:
                logger.info(
                    f"[PotentialTrainer] Обнаружено концептов: {len(concepts)}"
                )

        # Шаг 5: каждые 1000 шагов — обновить топологию
        if self.step % 1000 == 0 and self.step > 0:
            self.topological_field.update_after_learning()
            self.clusterer.cluster_by_density()
            self.clusterer.discover_hierarchy()
            logger.info(
                f"[PotentialTrainer] Топология обновлена. "
                f"Противоречий: {len(self.contradiction_filter.forbidden)}, "
                f"Концептов: {len(self.concept_miner.concepts)}"
            )

        # Шаг 6: sleep mode каждые 2000 шагов
        if self.step % 2000 == 0 and self.step > 0 and self.sleep_mode.should_sleep():
            self.sleep_mode.run_sleep_cycle()

        # Шаг 7: сохранить состояние
        self.states.append(state)
        if len(self.states) > self.max_states:
            self.states = self.states[-self.max_states:]

        self.total_assemblies += 1
        self.step += 1
        self.coherence_history.append(state.coherence_score)

        midx = min(len(symbol_ids), self.potential_field.vocab_size - 1)
        avg_pot = float(self.potential_field.affinity[symbol_ids[:midx]].mean())
        self.avg_potential_strength_history.append(avg_pot)

        return state

    def train_on_file(
        self,
        text_file: str,
        max_steps: int = 50000,
        max_len: int = 256,
        log_interval: int = 100,
        save_interval: int = 5000,
    ) -> Dict[str, Any]:
        """
        Обучает на текстовом файле: читает строки, строит сборки, накапливает потенциалы.
        """
        logger.info(f"[PotentialTrainer] Файл: {text_file}, макс. шагов: {max_steps}")

        if not os.path.exists(text_file):
            logger.error(f"Файл не найден: {text_file}")
            return {"error": "file_not_found"}

        start_time = time.time()
        lines_processed = 0

        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.step >= max_steps:
                    break

                line = line.strip()
                if not line or len(line) < 5:
                    continue

                # Фильтр: только кириллица (для качества)
                cyr = sum(1 for c in line if 0x0400 <= ord(c) <= 0x04FF)
                if cyr < len(line) * 0.5:
                    continue

                try:
                    state = self.train_on_text(line, max_len=max_len)
                    lines_processed += 1

                    if self.step % log_interval == 0 and self.step > 0:
                        elapsed = time.time() - start_time
                        avg_coh = np.mean(list(self.coherence_history)[-100:]) if self.coherence_history else 0
                        avg_pot = np.mean(list(self.avg_potential_strength_history)[-100:]) if self.avg_potential_strength_history else 0

                        logger.info(
                            f"[PotentialTrainer] step={self.step} "
                            f"| coherence={avg_coh:.3f} "
                            f"| avg_potential={avg_pot:.4f} "
                            f"| valid={self.valid_assemblies}/{self.total_assemblies} "
                            f"| lines_s={lines_processed / max(elapsed, 1):.0f}/s"
                        )

                    if self.step % save_interval == 0:
                        self.save()

                except Exception as e:
                    logger.debug(f"[PotentialTrainer] Ошибка строки: {e}")

        elapsed = time.time() - start_time
        self.save(final=True)

        stats = {
            "steps": self.step,
            "lines_processed": lines_processed,
            "elapsed": elapsed,
            "valid_assemblies": self.valid_assemblies,
            "invalid_assemblies": self.invalid_assemblies,
            "total_states": len(self.states),
            "best_coherence": self.best_coherence,
            "avg_potential_strength": float(self.potential_field.affinity.mean()),
        }
        logger.info(f"[PotentialTrainer] Завершено: {stats}")
        return stats

    def train_on_batch(
        self,
        texts: List[str],
        max_len: int = 256,
    ) -> List[AssemblyState]:
        """Батчевая обработка: N текстов → один forward pass. 10-50x быстрее на GPU."""
        device = next(self.layer.parameters()).device
        B = len(texts)
        if B == 0:
            return []

        all_ids = []
        for text in texts:
            ids = self.char_vocab.encode(text)[:max_len]
            all_ids.append(ids)

        max_L = max(len(ids) for ids in all_ids)
        batch_ids = torch.full((B, max_L), self.char_vocab.PAD_IDX, dtype=torch.long, device=device)
        for i, ids in enumerate(all_ids):
            L = len(ids)
            batch_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)

        with torch.no_grad():
            self.layer.eval()
            x = self.layer.embed(batch_ids)
            hidden = self.layer.forward_transformer(x)
            attn = self.layer.transformer.attention.last_attention

        states = []
        for i in range(B):
            L = len(all_ids[i])
            if L < 2:
                continue

            symbol_ids = all_ids[i]
            attention_matrix = attn[i].mean(dim=0).cpu().numpy()[:L, :L] if attn is not None else np.eye(L)

            state = self.build_assembly(symbol_ids, attention_matrix)
            states.append(state)

            if state.coherence_score > 0.5:
                if attn is not None:
                    self.potential_field.strengthen_batch(
                        batch_ids[i:i+1, :L],
                        attn[i:i+1, :, :L, :L],
                        confidence=state.coherence_score,
                    )
                    self.dynamics.reinforce_sequence(symbol_ids, confidence=state.coherence_score)
                self.valid_assemblies += 1
                sp, sq = self.propagator.propagate_sequence(symbol_ids, attention_matrix)
                self.semantic_checker.add_valid_potential(sq)
                self.sleep_mode.record_assembly(symbol_ids)
            else:
                self.potential_field.weaken_all(factor=0.999)
                self.invalid_assemblies += 1
                self.contradiction_filter.detect_structural_contradictions(symbol_ids)

            self.states.append(state)
            self.total_assemblies += 1
            self.step += 1
            self.coherence_history.append(state.coherence_score)

        if self.step % 500 == 0 and self.step > 0:
            self.concept_miner.search_free_space(max_tries=3)
        if self.step % 1000 == 0 and self.step > 0:
            self.topological_field.update_after_learning()
            self.clusterer.cluster_by_density()
        if self.step % 2000 == 0 and self.step > 0 and self.sleep_mode.should_sleep():
            self.sleep_mode.run_sleep_cycle()

        return states

    def train_on_file_batched(
        self,
        text_file: str,
        max_steps: int = 50000,
        batch_size: int = 16,
        max_len: int = 256,
        log_interval: int = 100,
        save_interval: int = 5000,
    ) -> Dict:
        """Батчевое обучение на файле для GPU."""
        logger.info(f"[Batch] Файл: {text_file}, batch={batch_size}")

        start_time = time.time()
        batch_buffer = []

        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.step >= max_steps:
                    break
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                cyr = sum(1 for c in line if 0x0400 <= ord(c) <= 0x04FF)
                if cyr < len(line) * 0.5:
                    continue
                batch_buffer.append(line)

                if len(batch_buffer) >= batch_size:
                    self.train_on_batch(batch_buffer, max_len=max_len)
                    batch_buffer = []

                    if self.step % log_interval == 0 and self.step > 0:
                        elapsed = time.time() - start_time
                        lps = self.step / max(elapsed, 0.01)
                        logger.info(
                            f"step={self.step} | lps={lps:.0f}/s | "
                            f"valid={self.valid_assemblies}/{self.total_assemblies} | "
                            f"forbidden={len(self.contradiction_filter.forbidden)}"
                        )
                    if self.step % save_interval == 0:
                        self.save()

            if batch_buffer:
                self.train_on_batch(batch_buffer, max_len=max_len)

        self.save(final=True)
        stats = {"steps": self.step, "elapsed": time.time() - start_time}
        return stats

    def train_on_npy(self, npy_file: str, max_steps: int = 50000,
                      batch_size: int = 128, block_size: int = 128,
                      log_interval: int = 500, save_interval: int = 5000) -> Dict:
        """GPU-оптимизированное обучение на пред-токенизированном .npy."""
        import numpy as np, torch, time

        device = next(self.layer.parameters()).device
        all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
        total = len(all_ids)
        logger.info(f"[NPY] {total:,} токенов, batch={batch_size}")

        pos, start_time = 0, time.time()
        while self.step < max_steps and pos + block_size + 2 < total:
            batch_ids_list, lengths = [], []
            for _ in range(batch_size):
                if pos + block_size + 2 > total:
                    pos = 0
                end = min(pos + block_size, total)
                chunk = all_ids[pos:end]
                sep_pos = np.where((chunk == 0) | (chunk == 3))[0]
                if len(sep_pos) > 0 and sep_pos[0] < block_size // 2:
                    end = pos + sep_pos[0] + 1
                    chunk = all_ids[pos:end]
                ids = [int(x) for x in chunk if int(x) >= 0][:block_size]
                batch_ids_list.append(ids)
                lengths.append(len(ids))
                pos = min(pos + max(len(ids), 32), total)

            max_len = max(lengths)
            bt = torch.full((batch_size, max_len), self.char_vocab.PAD_IDX, dtype=torch.long, device=device)
            for i, ids in enumerate(batch_ids_list):
                bt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            with torch.no_grad():
                self.layer.eval()
                x = self.layer.embed(bt)
                self.layer.forward_transformer(x)
                attn = self.layer.transformer.attention.last_attention

            for i in range(batch_size):
                L = lengths[i]
                if L < 4: self.step += 1; continue
                ids = batch_ids_list[i][:L]
                am = attn[i].mean(dim=0).cpu().numpy()[:L, :L] if attn is not None else np.eye(L)
                state = self.build_assembly(ids, am)
                if state.coherence_score > 0.5:
                    if attn is not None:
                        self.potential_field.strengthen_batch(bt[i:i+1, :L], attn[i:i+1, :, :L, :L], confidence=state.coherence_score)
                    self.valid_assemblies += 1; self.sleep_mode.record_assembly(ids)
                else:
                    self.invalid_assemblies += 1
                self.states.append(state); self.total_assemblies += 1; self.step += 1

            if self.step % 500 == 0 and self.step > 0: self.concept_miner.search_free_space(max_tries=3)
            if self.step % 1000 == 0 and self.step > 0:
                self.topological_field.update_after_learning(); self.clusterer.cluster_by_density()
            if self.step % log_interval == 0 and self.step > 0:
                elapsed = time.time() - start_time
                logger.info(f"step={self.step} | lps={self.step/max(elapsed,0.01):.0f}/s | pot={float(self.potential_field.affinity.mean()):.4f}")
            if self.step % save_interval == 0: self.save()

        self.save(final=True)
        return {"steps": self.step, "elapsed": time.time() - start_time}

    def save(self, final: bool = False):
        path = os.path.join(
            self.checkpoint_dir,
            f"step_{self.step:06d}" if not final else "final",
        )
        os.makedirs(path, exist_ok=True)

        torch.save(self.potential_field.state_dict(), os.path.join(path, "potential_field.pt"))
        torch.save(self.layer.state_dict(), os.path.join(path, "weights.pt"))

        # Сохраняем грамматику (паттерны)
        grammar_data = {
            "digrams": {ph: {"symbols": p.symbol_indices, "coherence": p.coherence_score, "usage": p.usage_count}
                        for ph, p in self.grammar.patterns.get(0, {}).items()},
            "ngrams": {ph: {"symbols": p.symbol_indices, "coherence": p.coherence_score, "usage": p.usage_count}
                       for ph, p in self.grammar.patterns.get(1, {}).items()},
        }
        with open(os.path.join(path, "grammar.json"), "w") as f:
            json.dump(grammar_data, f)

        states_data = [s.to_dict() for s in self.states[-1000:]]
        with open(os.path.join(path, "states.json"), "w") as f:
            json.dump(states_data, f, indent=2)

        status = {
            "step": self.step,
            "valid_assemblies": self.valid_assemblies,
            "invalid_assemblies": self.invalid_assemblies,
            "total_states": len(self.states),
            "best_coherence": self.best_coherence,
            "avg_potential": float(self.potential_field.affinity.mean()),
            "timestamp": time.time(),
        }
        with open(os.path.join(path, "status.json"), "w") as f:
            json.dump(status, f, indent=2)

        logger.info(f"[Save] step={self.step} states={len(self.states)} avg_pot={float(self.potential_field.affinity.mean()):.4f}")

    def summary(self) -> str:
        return (
            f"PotentialTrainer: steps={self.step}, "
            f"valid={self.valid_assemblies}/{self.total_assemblies}, "
            f"states={len(self.states)}, "
            f"forbidden={len(self.contradiction_filter.forbidden)}, "
            f"concepts={len(self.concept_miner.concepts)}, "
            f"domains={len(self.clusterer.domains)}"
        )
