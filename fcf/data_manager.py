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

            dataset = None
            errors = []

            for dataset_id in [
                f"wikipedia",  # старый формат
                f"wikimedia/wikipedia",  # новый формат
            ]:
                for year in ["20220301", "20231101"]:
                    try:
                        dataset = load_dataset(
                            dataset_id,
                            f"{year}.{language}",
                            split=split,
                            streaming=streaming,
                            trust_remote_code=True,
                        )
                        logger.info(
                            f"[DataManager] Wikipedia загружена: {dataset_id} "
                            f"({year}.{language})"
                        )
                        return dataset
                    except Exception as e:
                        errors.append(f"{dataset_id}/{year}.{language}: {e}")
                        continue

            logger.warning(
                f"[DataManager] Wikipedia не загружена ни через один источник. "
                f"Ошибки: {'; '.join(errors[-3:])}"
            )
            return None

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

        logger.info(f"[DataManager] RuBQ загружен: {len(data)} записей")
        return data

    @staticmethod
    def load_conceptnet(
        db_path: str,
        language: str = "ru",
    ) -> Optional[List[Dict[str, str]]]:
        """Загружает факты из локальной ConceptNet SQLite базы."""
        import sqlite3

        if not os.path.exists(db_path):
            logger.error(f"[DataManager] ConceptNet база не найдена: {db_path}")
            return None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT relation, start, end, weight
                FROM edges
                WHERE (
                    start LIKE ? || '/c/%'
                    OR end LIKE ? || '/c/%'
                )
                AND weight > 1.0
                LIMIT 100000
                """,
                (language, language),
            )

            rows = cursor.fetchall()
            conn.close()

            data = []
            for relation, start, end, weight in rows:
                start_label = start.split("/")[-1].replace("_", " ")
                end_label = end.split("/")[-1].replace("_", " ")
                rel_label = relation.split("/")[-1].replace("_", " ")

                data.append({
                    "concept": start_label,
                    "relation": rel_label,
                    "target": end_label,
                    "weight": weight,
                })

            logger.info(
                f"[DataManager] ConceptNet загружен: {len(data)} фактов (ru)"
            )
            return data

        except Exception as e:
            logger.error(f"[DataManager] Ошибка загрузки ConceptNet: {e}")
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
