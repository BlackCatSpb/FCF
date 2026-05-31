"""
AutoCurriculum — автономный цикл самообучения без участия пользователя.

Архитектура:
1. AutoCuriosity — детектирует пробелы в знаниях через SRG confidence
2. QueryGenerator — генерирует поисковые запросы из контекстных векторов
3. WebSearch — ищет релевантную информацию через Wikipedia API
4. KnowledgeExtractor — извлекает факты из результатов поиска
5. AutoTrainer — дообучает модель на новых данных (lazy-learn)
6. KnowledgeTracker — отслеживает что уже изучено, избегает повторов

Цикл:
  while active:
    if gap_detected:
      query = generate_targeted_query(c_vec, confidence)
      articles = search_wikipedia(query)
      facts = extract_knowledge(articles)
      if novel(facts):
        add_to_buffer(facts)
        train_on_buffer()
"""

import os
import re
import json
import time
import hashlib
import threading
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from urllib.parse import quote_plus
from urllib.request import urlopen, Request

import numpy as np
from loguru import logger


class KnowledgeTracker:
    """Отслеживает что модель уже изучила, избегает повторов."""

    def __init__(self, max_entries: int = 10000):
        self.learned_hashes: set = set()
        self.learned_topics: deque = deque(maxlen=max_entries)
        self.query_history: deque = deque(maxlen=500)
        self.topic_confidence: Dict[str, float] = {}

    def is_learned(self, text: str) -> bool:
        h = hashlib.md5(text[:200].encode()).hexdigest()
        return h in self.learned_hashes

    def mark_learned(self, text: str, topic: str = ""):
        h = hashlib.md5(text[:200].encode()).hexdigest()
        self.learned_hashes.add(h)
        if topic:
            self.learned_topics.append(topic)
        if len(self.learned_hashes) > 50000:
            self.learned_hashes = set(list(self.learned_hashes)[-25000:])

    def was_queried(self, query: str) -> bool:
        q_hash = hashlib.md5(query.encode()).hexdigest()
        return q_hash in {hashlib.md5(q.encode()).hexdigest() for q in self.query_history}

    def record_query(self, query: str):
        self.query_history.append(query)

    def update_topic_confidence(self, topic: str, confidence: float):
        if topic in self.topic_confidence:
            self.topic_confidence[topic] = 0.7 * self.topic_confidence[topic] + 0.3 * confidence
        else:
            self.topic_confidence[topic] = confidence

    def get_weak_topics(self, threshold: float = 0.6) -> List[str]:
        return [t for t, c in self.topic_confidence.items() if c < threshold]


class WikipediaSearch:
    """Коннектор к Wikipedia API для поиска релевантных статей."""

    API_URL = "https://ru.wikipedia.org/w/api.php"

    def __init__(self, language: str = "ru"):
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Поиск статей на Wikipedia."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit,
            "srprop": "snippet|titlesnippet",
        }
        url = self.base_url + "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())

        try:
            req = Request(url, headers={"User-Agent": "EVA-AutoCurriculum/1.0"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return data.get("query", {}).get("search", [])
        except Exception as e:
            logger.debug(f"[WikiSearch] Ошибка поиска: {e}")
            return []

    def get_article_text(self, page_title: str, max_chars: int = 5000) -> str:
        """Получение текста статьи Wikipedia."""
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "titles": page_title,
            "format": "json",
            "exchars": max_chars,
        }
        url = self.base_url + "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())

        try:
            req = Request(url, headers={"User-Agent": "EVA-AutoCurriculum/1.0"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                return page.get("extract", "")
        except Exception as e:
            logger.debug(f"[WikiSearch] Ошибка статьи: {e}")
        return ""


class QueryGenerator:
    """Генерирует целевые поисковые запросы из контекстного вектора."""

    def __init__(self, layer, tokenizer):
        self.layer = layer
        self.tokenizer = tokenizer

        self._query_prefixes = [
            "Дай определение термину",
            "Объясни понятие",
            "Что такое",
            "Расскажи о",
            "Как работает",
            "История возникновения",
            "Основные принципы",
        ]

        self._fallback_topics = [
            "природа вселенная",
            "история человечества",
            "наука физика",
            "математика теория",
            "технологии компьютеры",
            "искусство культура",
            "философия этика",
            "биология эволюция",
            "химия элементы",
            "астрономия космос",
            "география континенты",
            "экономика рынок",
            "психология сознание",
            "литература поэзия",
            "музыка гармония",
        ]

    def generate_from_context(self, c_vec: np.ndarray, confidence: float) -> str:
        """Генерирует поисковый запрос. Использует модель или fallback-темы."""
        try:
            import torch

            query_prompts = [
                "Необходимо найти информацию по теме:",
                "Требуется изучить вопрос о следующем:",
                "Нужна статья на тему:",
                "Следует разобраться в понятии:",
            ]

            best_query = ""

            for prefix in query_prompts:
                try:
                    enc = self.tokenizer.encode(prefix)
                    ids = enc.ids if hasattr(enc, 'ids') else enc
                    inp = torch.tensor([ids], dtype=torch.long)
                    if torch.cuda.is_available() and next(self.layer.parameters()).is_cuda:
                        inp = inp.cuda()

                    out = self.layer.generate(inp, max_new_tokens=24, temperature=0.85, top_p=0.88, top_k=35)
                    resp = self.tokenizer.decode(out[0].tolist())

                    continuation = resp[len(prefix):].strip()

                    if len(continuation) > 5 and not self._is_noise(continuation):
                        continuation = re.sub(r'[^\w\s\-.,;:!?()\"\'«»а-яА-ЯёЁ]', '', continuation)
                        continuation = re.sub(r'\s+', ' ', continuation).strip()
                        if len(continuation) > 5:
                            best_query = continuation
                            break
                except Exception:
                    continue

            if not best_query or len(best_query) < 5:
                import random
                idx = hash(float(confidence) * 1000 + float(np.mean(c_vec))) % len(self._fallback_topics)
                best_query = random.Random(idx).choice(self._fallback_topics)

            best_query = best_query[:120]
            logger.info(f"[QueryGen] Запрос: {best_query}")
            return best_query

        except Exception as e:
            logger.debug(f"[QueryGen] Ошибка: {e}")
            import random
            return random.choice(self._fallback_topics)

    def _is_noise(self, text: str) -> bool:
        """Проверяет является ли текст шумом (повторы, мусор)."""
        if len(text) < 5:
            return True
        words = text.split()
        if len(words) < 2:
            return True
        unique = len(set(w.lower() for w in words))
        ratio = unique / max(len(words), 1)
        if ratio < 0.3:
            return True
        cyr = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
        if cyr < len(text) * 0.3:
            return True
        return False


class KnowledgeExtractor:
    """Извлекает факты из результатов поиска для обучения."""

    def __init__(self, min_text_length: int = 100):
        self.min_text_length = min_text_length

    def extract_facts(self, articles: List[Dict[str, str]], wiki: WikipediaSearch) -> List[str]:
        """Извлекает факты из статей Wikipedia."""
        facts = []

        for article in articles:
            title = article.get("title", "")
            snippet = article.get("snippet", "")

            snippet_clean = re.sub(r'<[^>]+>', '', snippet)
            snippet_clean = re.sub(r'\s+', ' ', snippet_clean).strip()

            if len(snippet_clean) > self.min_text_length:
                facts.append(f"{title}: {snippet_clean}")

            if len(facts) >= 10:
                break

            if title:
                full_text = wiki.get_article_text(title, max_chars=3000)
                if full_text and len(full_text) > self.min_text_length:
                    sentences = re.split(r'[.!?]+', full_text)
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > self.min_text_length * 2:
                            facts.append(sent)
                        if len(facts) >= 20:
                            break

        return facts

    def extract_keywords(self, text: str) -> List[str]:
        """Извлекает ключевые понятия из текста."""
        words = re.findall(r'[А-Яа-яA-Za-z]{3,}', text)
        word_freq = {}
        for w in words:
            w = w.lower()
            if len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [w for w, _ in top_words]


class AutoCurriculum:
    """
    Автономный цикл самообучения.

    Мониторит SRG confidence, детектирует пробелы знаний, ищет
    информацию в Wikipedia и дообучает модель без участия пользователя.
    """

    def __init__(
        self,
        layer,
        tokenizer,
        checkpoint_dir: str = None,
        curiosity_threshold: float = 0.6,
        min_gap_count: int = 5,
        search_interval: float = 120.0,
        train_interval: float = 300.0,
        max_buffer_size: int = 200,
        train_steps: int = 50,
    ):
        self.layer = layer
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir or os.path.dirname(__file__)

        self.tracker = KnowledgeTracker()
        self.wiki = WikipediaSearch(language="ru")
        self.query_gen = QueryGenerator(layer, tokenizer)
        self.extractor = KnowledgeExtractor()

        self.curiosity_threshold = curiosity_threshold
        self.min_gap_count = min_gap_count
        self.search_interval = search_interval
        self.train_interval = train_interval
        self.max_buffer_size = max_buffer_size
        self.train_steps = train_steps

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.knowledge_buffer: List[str] = []
        self.gap_counter: int = 0
        self.last_gap_context: Optional[np.ndarray] = None
        self.last_gap_confidence: float = 0.0
        self.gap_topic: str = ""

        self.total_facts_added: int = 0
        self.total_searches: int = 0
        self.total_trainings: int = 0

        self._curriculum_optimizer = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[AutoCurriculum] Автономное дообучение запущено")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[AutoCurriculum] Остановлено")

    def _loop(self):
        last_search_time = 0
        last_train_time = 0

        while self._running:
            try:
                now = time.time()

                if self.gap_counter >= self.min_gap_count and (now - last_search_time) >= self.search_interval:
                    self._handle_knowledge_gap()
                    last_search_time = now

                if (now - last_train_time) >= self.train_interval and self.knowledge_buffer:
                    self._train_on_buffer()
                    last_train_time = now

                time.sleep(10)

            except Exception as e:
                logger.warning(f"[AutoCurriculum] Ошибка цикла: {e}")
                time.sleep(30)

    def record_confidence(self, c_vec: np.ndarray, confidence: float, response_text: str = ""):
        """Записывает результат оценки SRG. Если confidence низкая — накапливает gap."""
        if confidence < self.curiosity_threshold:
            self.gap_counter += 1
            self.last_gap_context = c_vec
            self.last_gap_confidence = confidence
            if response_text:
                self.gap_topic = response_text[:100]
            if self.gap_counter % 3 == 0:
                logger.debug(
                    f"[AutoCurriculum] Пробел знаний #{self.gap_counter}: "
                    f"confidence={confidence:.3f}, topic={self.gap_topic[:50]}"
                )
        else:
            self.gap_counter = max(0, self.gap_counter - 1)

    def _handle_knowledge_gap(self):
        """Обрабатывает накопленный пробел знаний."""
        logger.info(
            f"[AutoCurriculum] Обработка пробела: "
            f"gap_count={self.gap_counter}, confidence={self.last_gap_confidence:.3f}"
        )

        if self.last_gap_context is None:
            self.gap_counter = 0
            return

        query = self.query_gen.generate_from_context(self.last_gap_context, self.last_gap_confidence)

        if self.tracker.was_queried(query):
            logger.debug(f"[AutoCurriculum] Запрос уже был: {query}")
            self.gap_counter = max(0, self.gap_counter - 2)
            return

        self.tracker.record_query(query)

        logger.info(f"[AutoCurriculum] Поиск Wikipedia: {query}")
        articles = self.wiki.search(query, limit=5)
        self.total_searches += 1

        if not articles:
            logger.info("[AutoCurriculum] Ничего не найдено")
            self.gap_counter = 0
            return

        facts = self.extractor.extract_facts(articles, self.wiki)
        logger.info(f"[AutoCurriculum] Найдено фактов: {len(facts)}")

        new_facts = []
        for fact in facts:
            if not self.tracker.is_learned(fact):
                new_facts.append(fact)
                self.tracker.mark_learned(fact, query)

        if new_facts:
            self.knowledge_buffer.extend(new_facts)
            self.total_facts_added += len(new_facts)

            while len(self.knowledge_buffer) > self.max_buffer_size:
                self.knowledge_buffer.pop(0)

            logger.info(f"[AutoCurriculum] Новых фактов: {len(new_facts)}, буфер: {len(self.knowledge_buffer)}")

        self.tracker.update_topic_confidence(query, self.last_gap_confidence)
        self.gap_counter = 0

    def _train_on_buffer(self):
        """Дообучает модель на накопленных фактах."""
        if not self.knowledge_buffer:
            return

        logger.info(f"[AutoCurriculum] Дообучение на {len(self.knowledge_buffer)} фактах")

        import torch
        import torch.nn.functional as F

        device = next(self.layer.parameters()).device
        self.layer.train()

        if self._curriculum_optimizer is None:
            self._curriculum_optimizer = torch.optim.AdamW(
                self.layer.parameters(),
                lr=5e-6,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )

        optimizer = self._curriculum_optimizer

        blocks = []
        for fact in self.knowledge_buffer[-100:]:
            try:
                encoding = self.tokenizer.encode(fact)
                ids = encoding.ids if hasattr(encoding, 'ids') else encoding
                ids = ids[:128]
                while len(ids) < 128:
                    ids.append(3)
                blocks.append((
                    torch.tensor([ids[:-1]], dtype=torch.long).to(device),
                    torch.tensor([ids[1:]], dtype=torch.long).to(device),
                ))
            except Exception:
                continue
            if len(blocks) >= 16:
                break

        if not blocks:
            self.layer.eval()
            return

        total_loss = 0.0
        for step in range(self.train_steps):
            optimizer.zero_grad()

            step_loss = 0.0
            for input_ids, labels in blocks:
                x = self.layer.embed(input_ids)
                hidden = self.layer.forward_transformer(x)
                logits = self.layer.forward_logits(hidden)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=3,
                )
                step_loss = step_loss + loss

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.layer.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += step_loss.item()

        self.layer.eval()
        self.total_trainings += 1

        avg_loss = total_loss / self.train_steps
        logger.info(
            f"[AutoCurriculum] Дообучение завершено: "
            f"steps={self.train_steps}, loss={avg_loss:.4f}"
        )

        self.knowledge_buffer = self.knowledge_buffer[-50:]

    def summary(self) -> Dict[str, Any]:
        return {
            "searches": self.total_searches,
            "facts_added": self.total_facts_added,
            "trainings": self.total_trainings,
            "buffer_size": len(self.knowledge_buffer),
            "gap_counter": self.gap_counter,
            "weak_topics": self.tracker.get_weak_topics(),
            "running": self._running,
        }
