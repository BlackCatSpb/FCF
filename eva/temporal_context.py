"""
TemporalContextCompression — сжатие истории диалога в латентный код.

Сжимает последовательность реплик в один компактный код z_dialog
с помощью лёгкой RNN. Позволяет хранить контекст всей беседы
вместо последовательности всех реплик.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from loguru import logger


class DialogCompressor(nn.Module):
    """
    Сжимает последовательность реплик → один латентный код z_dialog.
    Использует bidirectional GRU + attention pooling.
    """

    def __init__(self, d_model: int = 2560, hidden: int = 256, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.GRU(d_model, hidden, 1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden * 2, 1)
        self.proj = nn.Linear(hidden * 2, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        B, T, D = embeddings.shape
        out, _ = self.encoder(embeddings)
        weights = torch.softmax(self.attention(out).squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        return self.proj(context)


class TemporalContextCompressor:
    """
    Сжимает историю диалога в латентный код.
    Каждая новая реплика обновляет код через EMA.
    """

    def __init__(self, d_model: int = 2560):
        self.compressor = DialogCompressor(d_model)
        self._dialog_code: np.ndarray = np.zeros(
            self.compressor.proj.out_features, dtype=np.float32
        )
        self._turn_count = 0

    def update(self, layer, turn_text: str, tokenizer):
        encoding = tokenizer.encode(turn_text)
        ids = encoding.ids if hasattr(encoding, "ids") else encoding
        input_ids = torch.tensor([ids[:64]], dtype=torch.long)

        device = next(layer.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            emb = layer.embed(input_ids).unsqueeze(0)

        z_turn = self.compressor(emb).squeeze(0).cpu().numpy()

        self._turn_count += 1
        alpha = min(0.5, 1.0 / max(self._turn_count, 1))
        self._dialog_code = (
            (1 - alpha) * self._dialog_code + alpha * z_turn
        )

    def get_code(self) -> np.ndarray:
        return self._dialog_code.copy()

    def reset(self):
        self._dialog_code = np.zeros_like(self._dialog_code)
        self._turn_count = 0
