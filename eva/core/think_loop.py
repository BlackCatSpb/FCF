"""
think_loop.py — autonomous thinking loop.

Phases:
  THINK    → generate text, collect trajectories
  ANALYZE  → find patterns, concepts, contradictions
  LEARN    → self-train WeightTransformer on real + generated data
  OPTIMIZE → prune heads, update attractors, compress storage

Runs continuously. State published to dashboard via set_state().
"""
import sys, os, time, math, random, pickle, threading
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import torch.nn.functional as F

from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.weight_transformer import WeightTransformer, count_params
from eva.symbolic.generation_loop import GenerationLoop
from eva.core.dashboard import set_state, set_state_batch, log, start_server
from eva.core.database import Database

V = 4101
NORM = {'word_len': 19, 'pos_in_word': 18, 'word_num': 275, 'pos_in_sent': 587, 'sent_len': 587}
META = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'v5', 'heads_meta.pkl'))
CSR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'v5', 'hierarchical'))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
MODEL_PATH = os.path.join(MODEL_DIR, 'weight_transformer_best.pt')


class ThinkLoop:
    def __init__(self):
        self.db = Database()
        self.heads: HeadsEnsemble = None
        self.transformer: WeightTransformer = None
        self.generator: GenerationLoop = None
        self.optimizer: torch.optim.Optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phase = 'INIT'

        # State tracking
        self.tokens_generated = 0
        self.sentences_generated = 0
        self.contradictions_found = 0
        self.concepts_discovered = 0
        self.acc_history = [0.0] * 100
        self.rate_history = [0.0] * 100
        self.generation_times = []
        self.start_time = time.time()
        self.gen_count = 0
        self.last_acc_update = 0

        # Self-training buffer
        self.train_buffer = []  # list of (context, next_token)
        self.buffer_max = 10000
        self.train_batch_size = 16

        # Concept discovery
        self.concept_clusters = defaultdict(set)  # concept_id → {token_ids}
        self.next_concept_id = 0

    # ─── Init ──────────────────────────────────────────────
    def init(self):
        log('Loading database...')
        self.db.load()
        s = self.db.summary()
        set_state_batch({
            'db_sentences': s['sentences'],
            'db_tokens': s['tokens'],
            'db_words': s['words'],
            'disk_usage': s['disk'],
            'rate_history': [0.0] * 100,
        })
        log(f'DB: {s["sentences"]} sentences (WP={self.db.n_sentences:,} + Wiki={self.db.n_wiki_sentences:,}), {s["tokens"]} tokens, {s["disk"]}')

        log('Loading heads...')
        self.heads = HeadsEnsemble(META, CSR)
        log('Heads loaded.')

        log(f'Using device: {self.device}')
        log('Loading transformer...')
        self.transformer = WeightTransformer()
        if os.path.exists(MODEL_PATH):
            self.transformer.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=self.device))
            log(f'Transformer loaded ({count_params(self.transformer):,} params).')
        else:
            log('New transformer initialized.')
        self.transformer.to(self.device)
        self.transformer.train()
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)

        self.generator = GenerationLoop(self.heads, transformer=self.transformer,
                                        max_tokens=80, device=self.device)
        self.phase = 'IDLE'

    # ─── Train step ────────────────────────────────────────
    def train_step(self, contexts, next_tokens):
        """Single training step on a batch."""
        batch = len(contexts)
        device = self.device
        prev_ids = torch.zeros(batch, dtype=torch.long, device=device)
        wl = torch.zeros(batch, dtype=torch.float32, device=device)
        piw = torch.zeros(batch, dtype=torch.float32, device=device)
        wn = torch.zeros(batch, dtype=torch.float32, device=device)
        pis = torch.zeros(batch, dtype=torch.float32, device=device)
        sl = torch.zeros(batch, dtype=torch.float32, device=device)
        fl = torch.zeros(batch, dtype=torch.float32, device=device)
        targets = torch.zeros(batch, dtype=torch.long, device=device)

        for i, ctx in enumerate(contexts):
            prev_ids[i] = ctx['prev_token_id']
            wl[i] = ctx['word_len'] / max(NORM['word_len'], 1)
            piw[i] = ctx['pos_in_word'] / max(NORM['pos_in_word'], 1)
            wn[i] = ctx['word_num'] / max(NORM['word_num'], 1)
            pis[i] = ctx['pos_in_sent'] / max(NORM['pos_in_sent'], 1)
            sl[i] = ctx['sent_len'] / max(NORM['sent_len'], 1)
            fl[i] = ctx['flags'] / 255.0
            targets[i] = next_tokens[i]

        head_scores = np.zeros((batch, 6, V), dtype=np.float32)
        for i, ctx in enumerate(contexts):
            head_scores[i] = self.heads.individual_scores(ctx)
        head_t = torch.from_numpy(head_scores).to(device)

        weights = self.transformer(prev_ids, wl, piw, wn, pis, sl, fl)
        final = torch.einsum('bi,biv->bv', weights, head_t)
        loss = F.cross_entropy(final, targets)

        # L2-регуляризация: штраф за концентрацию весов на одной голове
        # sum(weights) = 6.0 после нормализации, L2 минимален при равномерном распределении
        l2_lambda = 0.005
        l2_penalty = l2_lambda * weights.pow(2).sum(dim=-1).mean()
        loss = loss + l2_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            acc = (final.argmax(dim=-1) == targets).float().mean().item()
        return loss.item(), acc

    # ─── Phases ─────────────────────────────────────────────

    def phase_THINK(self):
        """Generate sentences, collect training data."""
        self.phase = 'THINK'
        log('Phase: THINK')
        n_gen = 0
        t0 = time.time()

        while time.time() - t0 < 15.0:  # 15 seconds of generation
            temperature = random.uniform(0.3, 1.5)
            result = self.generator.generate(
                temperature=temperature, seed=random.randint(0, 99999),
                return_coords=True)
            tokens = result['tokens']
            self.tokens_generated += len(tokens)
            self.sentences_generated += 1
            n_gen += 1

            # Extract training data from generation
            for pos in range(1, len(tokens) - 1):
                ctx = self._context_from_tokens(tokens, pos)
                if ctx is not None:
                    self.train_buffer.append((ctx, tokens[pos + 1]))
                    if len(self.train_buffer) > self.buffer_max:
                        self.train_buffer.pop(0)

        elapsed = time.time() - t0
        rate = self.tokens_generated / max(time.time() - self.start_time, 1)
        self.rate_history.append(rate)
        if len(self.rate_history) > 100:
            self.rate_history.pop(0)
        set_state_batch({
            'tokens_generated': self.tokens_generated,
            'sentences_generated': self.sentences_generated,
            'gen_rate': rate,
            'rate_history': list(self.rate_history),
            'buffer_size': len(self.train_buffer),
        })
        log(f'Generated {n_gen} sentences ({len(tokens)} tok) in {elapsed:.1f}s')

    def _context_from_tokens(self, tokens, pos):
        """Build context dict from a position in a token list."""
        if pos < 1 or pos >= len(tokens):
            return None
        tok_id = tokens[pos]
        piw = 0
        wl = 5
        wn = 0
        flags = 0
        prev = tokens[pos - 1] if pos > 0 else 0
        ctx_toks = tokens[max(0, pos - 3):pos]
        if prev == 157:  # WORD_OPEN → word start
            piw = 0
            flags = 1
            wn = sum(1 for t in tokens[:pos] if t == 158)
        elif prev == 158:  # WORD_CLOSE
            flags = 1 << 5
        else:
            # Count back from pos to find word start
            bpos = pos - 1
            piw = 1
            while bpos > 0 and tokens[bpos] != 157:
                bpos -= 1
                piw += 1
            piw = max(0, piw - 1)
        return {
            'token_id': tok_id, 'pos_in_word': piw, 'word_len': wl,
            'word_num': wn, 'pos_in_sent': pos, 'sent_len': len(tokens),
            'prev_token_id': prev, 'flags': flags, 'context_tokens': ctx_toks,
        }

    def phase_ANALYZE(self):
        """Analyze: find patterns, concepts, contradictions."""
        self.phase = 'ANALYZE'
        log('Phase: ANALYZE')

        # Concept discovery: find tokens with similar transition patterns
        trans_sim = self.db.meta.get('trans_sim_sparse', {})
        if len(trans_sim) > 0:
            n_clusters = 0
            for tid, neighbors in list(trans_sim.items())[:50]:
                high_sim = [(n, s) for n, s in neighbors if s > 0.5]
                if len(high_sim) >= 2:
                    cid = self.next_concept_id
                    self.next_concept_id += 1
                    self.concept_clusters[cid].add(tid)
                    for n, _ in high_sim[:5]:
                        self.concept_clusters[cid].add(n)
                    n_clusters += 1
            self.concepts_discovered = len(self.concept_clusters)
            log(f'Found {n_clusters} concept clusters ({self.concepts_discovered} total)')

        # Contradiction audit in generated text
        contra_pairs = self.db.meta.get('contra_pairs', [])
        self.contradictions_found = len(contra_pairs)
        log(f'Contradiction pairs: {self.contradictions_found}')

        set_state_batch({
            'contradictions': self.contradictions_found,
            'concepts': self.concepts_discovered,
        })

    def _sample_real_data(self, n_samples):
        """Sample (context, next_token) pairs from real data (WP + Wikipedia)."""
        pairs = []
        n_wp = min(self.db.n_sentences, 5000)
        n_wiki = min(self.db.n_wiki_sentences, 5000)
        if n_wp + n_wiki < 2:
            return pairs
        for _ in range(n_samples * 3):
            if n_wiki > 0 and random.random() < 0.5:
                sent_idx = random.randint(0, n_wiki - 1)
                sent = self.db.get_wiki_sentence(sent_idx)
            else:
                sent_idx = random.randint(0, n_wp - 1)
                sent = self.db.get_sentence(sent_idx)
            if sent is None or len(sent['tokens']) < 3:
                continue
            tokens = sent['tokens']
            pos = random.randint(1, len(tokens) - 2)
            ctx = self._context_from_tokens(tokens, pos)
            if ctx is not None:
                pairs.append((ctx, tokens[pos + 1]))
            if len(pairs) >= n_samples:
                break
        return pairs[:n_samples]

    def phase_LEARN(self):
        """Self-train transformer on accumulated buffer + real data."""
        self.phase = 'LEARN'
        log('Phase: LEARN')

        if len(self.train_buffer) < self.train_batch_size:
            log(f'Skipping: only {len(self.train_buffer)} samples')
            return

        # Mix real + generated data (50/50)
        n_samples = min(500, len(self.train_buffer))
        n_real = min(n_samples // 2, 250)
        n_gen = n_samples - n_real

        real_pairs = self._sample_real_data(n_real) if n_real > 0 else []
        gen_pairs = random.sample(self.train_buffer, min(n_gen, len(self.train_buffer)))
        batch = real_pairs + gen_pairs
        random.shuffle(batch)
        contexts = [b[0] for b in batch]
        next_tokens = [b[1] for b in batch]

        # Mini-batch training
        losses = []
        accs = []
        t0 = time.time()
        for i in range(0, n_samples, self.train_batch_size):
            end = min(i + self.train_batch_size, n_samples)
            ctx_batch = contexts[i:end]
            tok_batch = next_tokens[i:end]
            loss, acc = self.train_step(ctx_batch, tok_batch)
            losses.append(loss)
            accs.append(acc)

        avg_loss = np.mean(losses) if losses else 0
        avg_acc = np.mean(accs) if accs else 0

        # Track accuracy history
        self.acc_history.append(avg_acc)
        if len(self.acc_history) > 100:
            self.acc_history.pop(0)

        elapsed = time.time() - t0
        log(f'Trained on {n_samples} samples ({n_real} real/{n_gen} gen): loss={avg_loss:.4f}, acc={avg_acc:.4f} ({elapsed:.1f}s)')

        set_state_batch({
            'transformer_acc': avg_acc * 100,
            'acc_history': self.acc_history,
            'head_weights': self._get_current_weights(),
        })

        # Save model periodically
        if self.sentences_generated % 50 < 10:
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(self.transformer.state_dict(), MODEL_PATH)
            log('Model saved.')

    def _get_current_weights(self):
        """Extract current head weights from transformer for a typical context."""
        with torch.no_grad():
            d = self.device
            sample_ctx = self._sample_context()
            if sample_ctx is None:
                return [1.0, 1.0, 2.0, 0.5, 0.2, 0.5]
            prev = torch.tensor([sample_ctx['prev_token_id']], dtype=torch.long, device=d)
            wl = torch.tensor([sample_ctx['word_len'] / max(NORM['word_len'], 1)], dtype=torch.float32, device=d)
            piw = torch.tensor([sample_ctx['pos_in_word'] / max(NORM['pos_in_word'], 1)], dtype=torch.float32, device=d)
            wn = torch.tensor([sample_ctx['word_num'] / max(NORM['word_num'], 1)], dtype=torch.float32, device=d)
            pis = torch.tensor([sample_ctx['pos_in_sent'] / max(NORM['pos_in_sent'], 1)], dtype=torch.float32, device=d)
            sl = torch.tensor([sample_ctx['sent_len'] / max(NORM['sent_len'], 1)], dtype=torch.float32, device=d)
            fl = torch.tensor([sample_ctx['flags'] / 255.0], dtype=torch.float32, device=d)
            w = self.transformer(prev, wl, piw, wn, pis, sl, fl)
            return w.squeeze(0).cpu().tolist()

    def _sample_context(self):
        if len(self.train_buffer) > 0:
            return random.choice(self.train_buffer)[0]
        return {'prev_token_id': 158, 'word_len': 5, 'pos_in_word': 0,
                'word_num': 0, 'pos_in_sent': 0, 'sent_len': 10, 'flags': 0}

    def phase_OPTIMIZE(self):
        """Optimize: prune, compress, update metadata."""
        self.phase = 'OPTIMIZE'
        log('Phase: OPTIMIZE')

        # Store concept clusters in meta
        if self.concept_clusters:
            cc = {str(k): list(v) for k, v in self.concept_clusters.items()}
            self.db.meta['concept_clusters'] = cc

            # Update concept_head scores dynamically from clusters
            cs = np.ones(V, dtype=np.float32) * 0.2
            for cid, members in self.concept_clusters.items():
                for tid in members:
                    if tid < V:
                        cs[tid] += 0.5
            cs = cs / cs.max()  # normalize to [0, 1]
            self.heads.concept_scores = cs
            self.db.meta['concept_scores'] = cs.tolist()
            self.db.save_meta()
            log(f'Saved {len(cc)} concept clusters + scores to DB.')

        # Update stats
        set_state_batch({
            'contradictions': self.contradictions_found,
            'concepts': self.concepts_discovered,
            'buffer_size': len(self.train_buffer),
        })

    def run(self, cycles=-1):
        """Main loop. cycles=-1 = forever."""
        log('Think loop started.')
        n_cycle = 0
        while cycles < 0 or n_cycle < cycles:
            n_cycle += 1
            uptime = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))
            set_state_batch({
                'phase': f'CYCLE {n_cycle}',
                'uptime': uptime,
                'cycle': n_cycle,
            })

            self.phase_THINK()
            self.phase_ANALYZE()
            self.phase_LEARN()
            self.phase_OPTIMIZE()

            self.phase = 'IDLE'
            time.sleep(1)

        log('Think loop finished.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8383)
    parser.add_argument('--cycles', type=int, default=-1)
    args = parser.parse_args()

    # Start dashboard
    start_server(args.port)
    log(f'Starting EVA on http://127.0.0.1:{args.port}')

    # Start think loop
    loop = ThinkLoop()
    loop.init()
    loop.run(cycles=args.cycles)


if __name__ == '__main__':
    main()
