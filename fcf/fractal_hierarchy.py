"""
FractalHierarchy — фрактальная иерархия латентных кодов.
Организует коды в 4 уровня: символ → слово → предложение → текст.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class HierarchyConfig:
    k_sym: int = 64
    k_word: int = 96
    k_sent: int = 128
    k_text: int = 128
    d_head: int = 32
    num_iterations: int = 5


class SymbolEncoder(nn.Module):
    """Кодирует embedding-векторы в символьные коды z^(sym)."""

    def __init__(self, d_model: int = 2560, k_sym: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, k_sym, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class WordAggregator(nn.Module):
    """Агрегирует символы → слово: z^(word) = Σ α_i · z_i^(sym) с self-referencing attention."""

    def __init__(self, k_sym: int = 64, k_word: int = 96, d_head: int = 32):
        super().__init__()
        self.k_sym = k_sym
        self.k_word = k_word
        self.d_head = d_head
        self.up_proj = nn.Linear(k_sym, k_word, bias=False)
        self.decoder_Q = nn.Sequential(
            nn.Linear(k_word, 64), nn.SiLU(), nn.Linear(64, d_head * k_sym)
        )
        self.decoder_K = nn.Sequential(
            nn.Linear(k_word, 64), nn.SiLU(), nn.Linear(64, d_head * k_sym)
        )
        nn.init.xavier_uniform_(self.up_proj.weight)
        for m in [self.decoder_Q, self.decoder_K]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def aggregate(self, z_word: torch.Tensor, z_sym: torch.Tensor) -> torch.Tensor:
        B, N, _ = z_sym.shape
        W_Q = self.decoder_Q(z_word).view(B, self.d_head, self.k_sym)
        W_K = self.decoder_K(z_word).view(B, self.d_head, self.k_sym)
        q = torch.bmm(z_sym.mean(1, True), W_Q.transpose(1, 2))
        k = torch.bmm(z_sym, W_K.transpose(1, 2))
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / np.sqrt(self.k_sym)
        alpha = torch.softmax(scores, dim=-1)
        z_sym_weighted = torch.bmm(alpha.unsqueeze(1), z_sym).squeeze(1)
        return self.up_proj(z_sym_weighted)

    def refine(self, z_sym: torch.Tensor, n_iter: int = 5, rho: float = 0.85) -> torch.Tensor:
        z = self.up_proj(z_sym.mean(1))
        for _ in range(n_iter):
            z_new = self.aggregate(z, z_sym)
            z = rho * z + (1 - rho) * z_new
            if torch.norm(z_new - z).item() < 1e-4:
                break
        return z


class SentenceAggregator(nn.Module):
    """Агрегирует слова → предложение: z^(sent) с causal self-referencing attention."""

    def __init__(self, k_word: int = 96, k_sent: int = 128, d_head: int = 32):
        super().__init__()
        self.k_word = k_word
        self.k_sent = k_sent
        self.d_head = d_head
        self.up_proj = nn.Linear(k_word, k_sent, bias=False)
        self.decoder_Q = nn.Sequential(
            nn.Linear(k_sent, 64), nn.SiLU(), nn.Linear(64, d_head * k_word)
        )
        self.decoder_K = nn.Sequential(
            nn.Linear(k_sent, 64), nn.SiLU(), nn.Linear(64, d_head * k_word)
        )
        nn.init.xavier_uniform_(self.up_proj.weight)
        for m in [self.decoder_Q, self.decoder_K]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def aggregate(self, z_sent: torch.Tensor, z_words: torch.Tensor) -> torch.Tensor:
        B, N, _ = z_words.shape
        W_Q = self.decoder_Q(z_sent).view(B, self.d_head, self.k_word)
        W_K = self.decoder_K(z_sent).view(B, self.d_head, self.k_word)
        q = torch.bmm(z_words.mean(1, True), W_Q.transpose(1, 2))
        k = torch.bmm(z_words, W_K.transpose(1, 2))
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / np.sqrt(self.k_word)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        causal = torch.triu(torch.ones(scores.shape[1], scores.shape[1], device=scores.device), 1).bool()
        scores = scores.masked_fill(causal, float("-inf"))
        beta = torch.softmax(scores, dim=-1)
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)
        z_word_weighted = torch.bmm(beta.unsqueeze(1), z_words).squeeze(1)
        if z_word_weighted.dim() == 1:
            z_word_weighted = z_word_weighted.unsqueeze(0)
        return self.up_proj(z_word_weighted)

    def refine(self, z_words: torch.Tensor, n_iter: int = 5, rho: float = 0.85) -> torch.Tensor:
        z = self.up_proj(z_words.mean(dim=1))
        for _ in range(n_iter):
            z_new = self.aggregate(z, z_words)
            z = rho * z + (1 - rho) * z_new
            if torch.norm(z_new - z).item() < 1e-4:
                break
        return z


class TextAggregator(nn.Module):
    """Агрегирует предложения → текст через GRU."""

    def __init__(self, k_sent: int = 128, k_text: int = 128, h_dim: int = 128):
        super().__init__()
        self.rnn = nn.GRU(k_sent, h_dim, 1, batch_first=True)
        self.proj = nn.Linear(h_dim, k_text)

    def forward(self, z_sents: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(z_sents)
        return self.proj(h[-1])


class FractalHierarchy(nn.Module):
    """Фрактальная иерархия: символ → слово → предложение → текст."""

    def __init__(self, d_model: int = 2560, config: HierarchyConfig = None):
        super().__init__()
        self.cfg = config or HierarchyConfig()
        self.sym_enc = SymbolEncoder(d_model, self.cfg.k_sym)
        self.word_agg = WordAggregator(self.cfg.k_sym, self.cfg.k_word, self.cfg.d_head)
        self.sent_agg = SentenceAggregator(self.cfg.k_word, self.cfg.k_sent, self.cfg.d_head)
        self.text_agg = TextAggregator(self.cfg.k_sent, self.cfg.k_text)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, D = embeddings.shape
        z_sym = self.sym_enc(embeddings)
        z_word = self.word_agg.refine(z_sym, self.cfg.num_iterations)
        if z_word.dim() == 1:
            z_word = z_word.unsqueeze(0)
        z_sent = self.sent_agg.refine(z_word.unsqueeze(1), self.cfg.num_iterations)
        if z_sent.dim() == 1:
            z_sent = z_sent.unsqueeze(0)
        z_text = self.text_agg(z_sent.unsqueeze(1))
        return {"z_sym": z_sym, "z_word": z_word, "z_sent": z_sent, "z_text": z_text}

    def hierarchy_loss(self, z_sym, z_word, z_sent) -> torch.Tensor:
        loss = torch.tensor(0.0, device=z_sym.device)
        z_w2 = self.word_agg.refine(z_sym, 1)
        loss += F.mse_loss(z_word, z_w2)
        z_s2 = self.sent_agg.refine(z_word.unsqueeze(0), 1)
        loss += F.mse_loss(z_sent, z_s2)
        return loss

    def contrastive_loss(self, a, b, similar, temp=0.1) -> torch.Tensor:
        sim = F.cosine_similarity(a, b, dim=-1) / temp
        return F.binary_cross_entropy_with_logits(sim, similar.float())

    def recursive_loss(self, history: List[float]) -> torch.Tensor:
        if len(history) < 2:
            return torch.tensor(0.0)
        return torch.tensor(sum(h * h for h in history[1:]) / len(history[1:]))

    def summary(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"FractalHierarchy(sym={self.cfg.k_sym}, word={self.cfg.k_word}, sent={self.cfg.k_sent}, text={self.cfg.k_text}, params={p:,})"

    def __repr__(self) -> str:
        return self.summary()
