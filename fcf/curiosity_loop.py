"""
CuriosityLoop — генерация уточняющих вопросов при систематической неуверенности.

Принцип работы:
1. Счётчик неуверенных ответов инкрементируется при confidence < порога
2. При достижении counter >= threshold система генерирует уточняющий вопрос
3. Вопрос генерируется через прямой проход того же слоя (авторегрессия)
4. Сгенерированный вопрос передаётся пользователю или сохраняется в буфер

Важно: никаких жёстко заданных строк — модель использует свои обучаемые матрицы.
"""

import torch
from typing import Optional
from loguru import logger


class CuriosityLoop:

    def __init__(self, threshold: int = 10):
        self.counter: int = 0
        self.threshold: int = threshold
        self._pending_clarifications: list = []

    def should_ask(self, confidence: float) -> bool:
        if confidence < 0.6:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.threshold

    def generate_clarification(
        self,
        layer: "PrimordialLayer",
        tokenizer,
        original_query: str,
        generated_answer: str,
    ) -> str:
        prompt = (
            "<|im_start|>system\n"
            "Ты задаёшь уточняющий вопрос, потому что не уверен в ответе. "
            "Сформулируй ровно один чёткий вопрос.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Исходный запрос: {original_query}\n"
            f"Мой неуверенный ответ: {generated_answer}\n"
            f"Задай уточняющий вопрос.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        device = next(layer.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            output_ids = layer.generate(
                input_ids,
                max_new_tokens=64,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )

        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in prompt:
            prompt_end = prompt.rfind(assistant_marker) + len(assistant_marker)
            question = full_output[prompt_end:].strip()
        else:
            question = full_output[len(prompt):].strip()

        question = self._clean_question(question)
        logger.info(f"[Curiosity] Сгенерирован уточняющий вопрос: {question}")
        return question

    def _clean_question(self, text: str) -> str:
        for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
            text = text.replace(marker, "")

        text = text.strip()

        if not text.endswith("?"):
            texts = text.split(".")
            for t in reversed(texts):
                t = t.strip()
                if t and "?" in t:
                    text = t
                    break
            else:
                text = texts[-1].strip() if texts else text

        first_q = text.find("?")
        if first_q > 0:
            text = text[: first_q + 1]

        return text

    def reset(self):
        self.counter = 0

    def add_pending_clarification(self, query: str, answer: str, question: str):
        self._pending_clarifications.append({
            "query": query,
            "answer": answer,
            "question": question,
        })

    def get_pending_clarifications(self) -> list:
        pending = list(self._pending_clarifications)
        self._pending_clarifications.clear()
        return pending
