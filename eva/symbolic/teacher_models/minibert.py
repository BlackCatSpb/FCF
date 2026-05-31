"""
MiniBERT — маленький BERT-style teacher для дистилляции в EVA.

Архитектура:
- Learned embeddings (nn.Embedding, НЕ CoordinateEmbedding)
- 4 слоя TransformerEncoder, 128 dim, 4 heads, GELU
- MLM head: плотный → LayerNorm → GELU → LayerNorm → плотный

Зачем:
- Teacher должен быть ДРУГИМ (learned embeddings), чтобы EVA
  училась отображать "стандартное трансформерное пространство" в ℝ¹²⁸
- MiniBERT → EVA distillation: MSE(h_eva, proj(h_minibert))
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math


class MiniBERT(nn.Module):
    def __init__(self, vocab_size=161, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, max_len=512, pad_token_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm_embed = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLM head
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model)
        self.mlm_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if self.embed.padding_idx is not None:
            with torch.no_grad():
                self.embed.weight[self.embed.padding_idx].zero_()

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embed(pos)
        x = self.norm_embed(x)
        x = self.dropout(x)

        if attention_mask is None:
            src_key_padding_mask = (input_ids == self.pad_token_id)
        else:
            src_key_padding_mask = ~attention_mask.bool()

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return h

    def get_hidden(self, input_ids):
        return self.forward(input_ids)

    def mlm_loss(self, input_ids, masked_positions):
        """
        MLM loss: predict masked tokens.

        Args:
            input_ids: [B, L] — с MASK на masked_positions
            masked_positions: [B, L] — boolean mask позиций для предсказания
        Returns:
            loss: scalar
            accuracy: scalar
        """
        h = self.forward(input_ids)
        logits = self.mlm_head(h)
        
        labels = input_ids.clone()
        labels[~masked_positions] = -100  # игнорируем не-MASK позиции

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1),
                                ignore_index=-100)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == input_ids) & masked_positions
            accuracy = correct.sum().float() / masked_positions.sum().float()

        return loss, accuracy

    def mlm_head(self, h):
        h = self.mlm_dense(h)
        h = self.mlm_norm(h)
        h = F.gelu(h)
        return self.mlm_out(h)

    @torch.no_grad()
    def generate_mlm_batch(self, input_ids, mask_prob=0.15, mask_token_id=4):
        """
        Создаёт MLM батч: маскирует mask_prob процентов токенов.

        Returns:
            masked_ids: input_ids с заменой на MASK
            mask: boolean mask — какие позиции замаскированы
        """
        masked_ids = input_ids.clone()
        mask = torch.rand_like(input_ids.float()) < mask_prob
        # Не маскируем pad токены
        mask[input_ids == self.pad_token_id] = False
        masked_ids[mask] = mask_token_id
        return masked_ids, mask


def train_minibert(model, data, config, device):
    """
    Быстрое pre-train MiniBERT с MLM.

    Args:
        model: MiniBERT
        data: np.array токенов [N]
        config: TrainingConfig-like (batch_size, seq_len, lr, etc.)
        device: torch.device
    Returns:
        model: обученный MiniBERT
    """
    from .train_v3 import TrainingConfig
    if not isinstance(config, TrainingConfig):
        class SimpleConfig:
            pass
        cfg = SimpleConfig()
        cfg.batch_size = getattr(config, 'batch_size', 8)
        cfg.seq_len = getattr(config, 'seq_len', 128)
        cfg.lr = getattr(config, 'lr', 3e-4)
        cfg.warmup_steps = getattr(config, 'warmup_steps', 500)
        cfg.save_every = getattr(config, 'save_every', 1000)
        cfg.clip_grad_norm = getattr(config, 'clip_grad_norm', 1.0)
        cfg.log_every = getattr(config, 'log_every', 50)
        config = cfg

    import numpy as np, time
    from .train_v3 import create_batch

    model.train()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10000)

    data = np.array(data, dtype=np.int64)
    step = 0
    t0 = time.time()

    print(f'[MiniBERT] Training on {len(data)} tokens...')
    while True:
        batch = create_batch(data, config, device)
        if batch is None:
            break

        optim.zero_grad()
        masked_ids, mask = model.generate_mlm_batch(batch['input_ids'])
        loss, acc = model.mlm_loss(masked_ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optim.step()
        scheduler.step()

        if step % config.log_every == 0:
            elapsed = time.time() - t0
            print(f'[MiniBERT Step {step}] loss={loss.item():.4f} acc={acc.item():.3f} '
                  f'| {elapsed:.0f}s')

        if step % config.save_every == 0 and step > 0:
            path = f'checkpoints/minibert_step_{step}.pt'
            import os
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({'step': step, 'model_state': model.state_dict()}, path)
            print(f'[MiniBERT Save] {path}')

        step += 1

    print(f'[MiniBERT] Done. {step} steps, {time.time()-t0:.0f}s')
    return model
