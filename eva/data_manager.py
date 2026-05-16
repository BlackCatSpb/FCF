"""
DataManager — загрузка и подготовка данных для обучения.

Источники:
- Wikipedia (русская, через datasets)
- Saiga (инструктивный датасет)
- RuBQ (структурированные знания Wikidata)
- ConceptNet (семантические связи)
"""

import os
import json
from typing import Iterator, Optional, Dict, Any, List
from loguru import logger


class DataManager:

    @staticmethod
    def load_wikipedia(
        split: str = "train",
        streaming: bool = True,
        language: str = "ru",
    ) -> Optional[Iterator[Dict[str, Any]]]:
        try:
            from datasets import load_dataset
            import os
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

            dataset = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{language}",
                split=split,
                streaming=streaming,
            )
            logger.info(
                f"[DataManager] Wikipedia загружена: "
                f"wikimedia/wikipedia 20231101.{language}"
            )
            return dataset

        except ImportError:
            logger.error("[DataManager] `datasets` не установлен. pip install datasets")
            return None
        except Exception as e:
            logger.error(f"[DataManager] Ошибка загрузки Wikipedia: {e}")
            return None

    @staticmethod
    def load_wikipedia_blocks(
        block_size: int = 512,
        split: str = "train",
        streaming: bool = True,
    ) -> Optional[Iterator[Dict[str, Any]]]:
        dataset = DataManager.load_wikipedia(split=split, streaming=streaming)
        if dataset is None:
            return None

        def block_iterator():
            for item in dataset:
                text = item.get("text", "")
                if not text:
                    continue

                words = text.split()
                for i in range(0, len(words), block_size):
                    block = " ".join(words[i : i + block_size])
                    if len(block.strip()) > 50:
                        yield {"text": block}

        return block_iterator()

    @staticmethod
    def load_saiga() -> Optional[Iterator[Dict[str, Any]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                "IlyaGusev/saiga2_70b_lora",
                split="train",
                streaming=True,
            )
            logger.info("[DataManager] Saiga загружен")
            return dataset
        except ImportError:
            logger.error("[DataManager] `datasets` не установлен.")
            return None
        except Exception as e:
            logger.error(f"[DataManager] Ошибка загрузки Saiga: {e}")
            return None

    @staticmethod
    def load_saiga_pairs() -> Optional[Iterator[Dict[str, str]]]:
        dataset = DataManager.load_saiga()
        if dataset is None:
            return None

        def pair_iterator():
            for item in dataset:
                instruction = item.get("instruction", "")
                output = item.get("output", "")
                if instruction and output:
                    yield {"instruction": instruction, "output": output}

        return pair_iterator()

    @staticmethod
    def load_texts_from_file(path: str) -> Optional[Iterator[str]]:
        if not os.path.exists(path):
            logger.error(f"[DataManager] Файл не найден: {path}")
            return None

        def file_iterator():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

        return file_iterator()

    @staticmethod
    def load_rubq(path: str) -> list:
        if not os.path.exists(path):
            logger.warning(f"[DataManager] RuBQ файл не найден: {path}")
            return []

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    @staticmethod
    def load_rus_dataset(streaming: bool = True) -> Optional[Iterator[str]]:
        """danneyankeee/rus — огромный датасет русских текстов."""
        try:
            from datasets import load_dataset
            ds = load_dataset("danneyankeee/rus", split="train", streaming=streaming, trust_remote_code=True)
            logger.info("[DataManager] danneyankeee/rus загружен")
            def text_iter():
                for item in ds:
                    text = item.get("text", "") or item.get("content", "") or ""
                    if len(text) > 100:
                        yield text
            return text_iter()
        except Exception as e:
            logger.warning(f"[DataManager] rus dataset: {e}")
            return None

    @staticmethod
    def load_ru_instruct() -> Optional[List[Dict[str, str]]]:
        """d0rj/ru-instruct — 754K инструкций на русском."""
        try:
            from datasets import load_dataset
            ds = load_dataset("d0rj/ru-instruct", split="train", streaming=True)
            logger.info("[DataManager] d0rj/ru-instruct загружен")
            pairs = []
            for item in ds:
                inst = item.get("instruction", "") or item.get("input", "")
                out = item.get("output", "") or item.get("response", "")
                if inst and out:
                    pairs.append({"instruction": inst, "output": out})
                if len(pairs) >= 10000:
                    break
            return pairs
        except Exception as e:
            logger.warning(f"[DataManager] ru-instruct: {e}")
            return []

    @staticmethod
    def load_conversations() -> Optional[Iterator[str]]:
        """inkoziev/Conversations — 9M диалогов на русском."""
        try:
            from datasets import load_dataset
            ds = load_dataset("inkoziev/Conversations", split="train", streaming=True, trust_remote_code=True)
            logger.info("[DataManager] inkoziev/Conversations загружен")
            def text_iter():
                for item in ds:
                    text = item.get("text", "") or str(item)
                    if len(text) > 50:
                        yield text
            return text_iter()
        except Exception as e:
            logger.warning(f"[DataManager] Conversations: {e}")
            return None

    @staticmethod
    def load_conceptnet(
        db_path: str,
        language: str = "ru",
        max_facts: int = 100000,
    ) -> Optional[List[Dict[str, str]]]:
        try:
            from conceptnet_lite import connect
            connect(db_path)
            from conceptnet_lite.db import Edge
            logger.info(f"[DataManager] ConceptNet подключена: {db_path}")

            facts = []
            seen = set()
            lang_prefix = f"/c/{language}/"
            count = 0

            edges = Edge.select().iterator()

            for edge in edges:
                count += 1
                try:
                    start_uri = edge.start.uri
                    end_uri = edge.end.uri

                    if lang_prefix not in start_uri and lang_prefix not in end_uri:
                        continue

                    rel = edge.relation.name
                    weight = edge.etc.get('weight', 1.0) if edge.etc else 1.0

                    if weight < 0.5:
                        continue

                    rel_name = rel.replace("/r/", "")
                    start_name = start_uri.split("/")[-1].replace("_", " ")
                    end_name = end_uri.split("/")[-1].replace("_", " ")

                    key = f"{start_name}|{rel_name}|{end_name}"
                    if key in seen:
                        continue
                    seen.add(key)

                    facts.append({
                        "concept": start_name,
                        "relation": rel_name,
                        "target": end_name,
                        "weight": weight,
                    })

                    if len(facts) >= max_facts:
                        break

                except Exception:
                    continue

            logger.info(f"[DataManager] ConceptNet: {len(facts)} фактов из {count} рёбер ({language})")
            return facts

        except ImportError:
            logger.warning("[DataManager] conceptnet-lite не установлен")
            return None
        except Exception as e:
            logger.error(f"[DataManager] Ошибка ConceptNet: {e}")
            return None

    @staticmethod
    def load_conceptnet_as_texts(
        db_path: str,
        language: str = "ru",
    ) -> Optional[Iterator[str]]:
        """Загружает ConceptNet факты в виде текстовых строк для обучения языку."""
        facts = DataManager.load_conceptnet(db_path, language=language)
        if not facts:
            return None

        def text_iterator():
            for fact in facts:
                text = f"{fact['concept']} {fact['relation']} {fact['target']}"
                yield text

        return text_iterator()

    @staticmethod
    def format_chat_template(
        instruction: str,
        output: str = "",
        system_prompt: str = "",
    ) -> str:
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{instruction}<|im_end|>")
        if output:
            parts.append(f"<|im_start|>assistant\n{output}<|im_end|>")
        else:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
