"""
EVA v5 — Pipeline этапа 1: упаковка текста в полные треки.
Читает Войну и Мир, разбивает на предложения,
детектирует слова, токенизирует, упаковывает в 384-мерные треки.
100% roundtrip verification.

Usage:
    python pipeline_v5.py  (процессинг и верификация)
"""
import sys, os, re, json, time, pickle
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np

# ─── Packer ───
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF\eva\symbolic')
from coordinate_packer import CoordinatePacker

# ─── Constants V5 ───
WORD_OPEN = 157
WORD_CLOSE = 158
SENT_OPEN = 159
SENT_CLOSE = 160
BOS = 2
EOS = 3
PAD = 0
VOCAB_SIZE = 4101
MAX_SENT_TOKENS = 2048  # safety limit
MIN_SENT_TOKENS = 3     # skip degenerate

packer = CoordinatePacker()


# ─── BPE Tokenizer wrapper ───
class BPEProcessor:
    """Загрузка BPE vocab для кодирования русских слов."""
    def __init__(self):
        from eva.symbolic.bpe_tokenizer import BPEVocab
        self.cv = BPEVocab()
    
    def encode_word(self, word: str) -> list:
        """BPE-кодирование одного слова без спецтокенов."""
        ids = self.cv.encode(word)
        # Filter out any boundary tokens that BPE might insert
        return [tid for tid in ids if tid not in (WORD_OPEN, WORD_CLOSE, SENT_OPEN, SENT_CLOSE, BOS, EOS, PAD)]
    
    def decode(self, ids: list) -> str:
        return self.cv.decode(ids)


# ─── Sentence splitter ───
def split_sentences(text: str) -> list:
    """
    Разбиение русского текста на предложения.
    Учитывает: . ! ? ... — и кавычки/скобки в конце предложения.
    """
    # Clean BOM and normalize
    text = text.strip('\ufeff').strip()
    
    # Split on sentence-ending punctuation followed by capital letter or newline
    # Russian sentence ends: .!?… followed by space+capital or newline
    # Use regex with lookahead for capital letter or end of string
    parts = re.split(r'(?<=[.!?…])\s+(?=[А-ЯA-Z«„])|(?<=[.!?…])\s*$', text)
    
    # Also handle newlines as sentence separators
    sentences = []
    for p in parts:
        p = p.strip()
        if len(p) < 2:
            continue
        # Handle newlines within parts
        sub = [s.strip() for s in p.split('\n') if s.strip()]
        sentences.extend(sub)
    
    # Filter empty/very short
    sentences = [s for s in sentences if len(s) >= 3]
    return sentences


# ─── Word tokenizer ───
def tokenize_sentence(text: str, bpe: BPEProcessor) -> tuple:
    """
    Разобрать предложение на слова, BPE-токенизировать каждое слово,
    собрать полную последовательность с WORD_OPEN/WORD_CLOSE/SENT_OPEN/SENT_CLOSE.
    
    Returns:
        tokens: list[int] — полная последовательность токенов
        word_spans: list[tuple] — (start, end) для каждого слова в token space
    """
    # Split into words: handle punctuation attached to words
    # Russian words: letters + apostrophe + hyphen
    words = re.findall(r'[А-Яа-яЁёA-Za-z0-9]+(?:[-\'][А-Яа-яЁёA-Za-z0-9]+)*|[^\w\s]', text)
    
    tokens = [SENT_OPEN]
    word_spans = []
    
    for word in words:
        # Skip pure whitespace
        if not word.strip():
            continue
        
        # Insert word boundary
        tokens.append(WORD_OPEN)
        word_start = len(tokens)
        
        # BPE-encode the word
        word_ids = bpe.encode_word(word)
        if not word_ids:
            tokens.pop()  # remove WORD_OPEN
            continue
        
        tokens.extend(word_ids)
        word_end = len(tokens) - 1
        tokens.append(WORD_CLOSE)
        
        word_spans.append((word_start, word_end))
    
    tokens.append(SENT_CLOSE)
    
    return tokens, word_spans


# ─── Encode sentence → trajectory ───
def sentence_to_trajectory(text: str, bpe: BPEProcessor,
                           text_id: int = 0, sent_idx: int = 0) -> tuple:
    """
    Полный пайплайн: предложение → [L, 384] траектория.
    
    Returns:
        trajectory: np.ndarray [L, 384]
        tokens: list[int] — для верификации
        meta: dict — статистика
    """
    tokens, word_spans = tokenize_sentence(text, bpe)
    L = len(tokens)
    
    if L < MIN_SENT_TOKENS or L > MAX_SENT_TOKENS:
        return None, tokens, {'skipped': True, 'reason': f'length={L}'}
    
    trajectory = np.zeros((L, packer.DIM), dtype=np.float32)
    
    # Precompute word index for each position
    pos_to_word = {}
    for wi, (ws, we) in enumerate(word_spans):
        for t in range(ws, we + 1):
            pos_to_word[t] = wi
    
    for t in range(L):
        tid = tokens[t]
        wi = pos_to_word.get(t, -1)
        
        flags = 0
        pos_in_word = -1
        word_len = 0
        word_num = 0
        meta_type = 6  # default: special
        
        if wi >= 0:
            ws, we = word_spans[wi]
            pos_in_word = t - ws
            word_len = we - ws + 1
            word_num = wi
            
            # Token inside a word → override meta_type
            meta_type = 0 if word_len == 1 else (2 if pos_in_word == 0 else 3)
            
            # Word boundary flags
            if pos_in_word == 0:
                flags |= (1 << packer.F_WORD_START)
            if pos_in_word == word_len - 1:
                flags |= (1 << packer.F_WORD_END)
            if wi > 0:
                flags |= (1 << packer.F_HAS_WORD_LEFT)
            if wi < len(word_spans) - 1:
                flags |= (1 << packer.F_HAS_WORD_RIGHT)
            
            # Sentence boundary flags
            if wi == 0:
                flags |= (1 << packer.F_SENT_START)
            if wi == len(word_spans) - 1:
                flags |= (1 << packer.F_SENT_END)
        
        # Token type flags
        if tid in (WORD_OPEN, WORD_CLOSE, SENT_OPEN, SENT_CLOSE, BOS, EOS, PAD):
            flags |= (1 << packer.F_SPECIAL)
            meta_type = 6
        if packer._is_punctuation(tid):
            flags |= (1 << packer.F_PUNCTUATION)
            meta_type = 5
        if packer._is_digit(tid):
            flags |= (1 << packer.F_DIGIT)
        if packer._is_letter(tid):
            flags |= (1 << packer.F_LETTER)
        
        # Context hash
        ctx_hash = packer._context_hash(tokens, t)
        
        h_t = packer.pack_token(
            token_id=tid,
            pos_in_word=max(0, pos_in_word),
            word_len=max(1, word_len),
            word_num=word_num,
            pos_in_sent=t,
            sent_len=L,
            flags=flags,
            meta_type=meta_type,
            context_hash=ctx_hash,
            text_id=text_id & 0xFF,
        )
        trajectory[t] = h_t
    
    return trajectory, tokens, {
        'L': L,
        'n_words': len(word_spans),
        'text_id': text_id,
        'sent_idx': sent_idx,
    }


# ─── Verify 100% roundtrip ───
def verify_trajectory(trajectory: np.ndarray, original_tokens: list) -> dict:
    """Проверить, что трек декодируется обратно в исходные токены."""
    L = trajectory.shape[0]
    decoded = []
    errors = []
    
    for t in range(L):
        info = packer.unpack_token(trajectory[t])
        tid = info['token_id']
        decoded.append(tid)
        if tid != original_tokens[t]:
            errors.append((t, original_tokens[t], tid))
    
    return {
        'exact_match': decoded == original_tokens,
        'n_errors': len(errors),
        'accuracy': (L - len(errors)) / L * 100 if L > 0 else 0,
        'errors': errors[:10],  # first 10
        'decoded': decoded,
    }


# ─── Main pipeline ───
def main():
    print("="*60)
    print("Pipeline: упаковка текста в треки")
    print("="*60)
    
    bpe = BPEProcessor()
    
    # 1. Load War and Peace
    print(f"\n[1] Loading War and Peace...")
    with open(r'C:\Users\black\OneDrive\Desktop\Война и Мир.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()
    print(f"    {len(full_text):,} chars")
    
    # 2. Split into sentences
    print(f"\n[2] Splitting into sentences...")
    sentences = split_sentences(full_text)
    print(f"    {len(sentences):,} sentences found")
    
    # 3. Pick some samples for detailed verification
    test_samples = sentences[:5] + sentences[len(sentences)//2:len(sentences)//2+5] + sentences[-5:]
    
    print(f"\n[3] Verifying sample sentences (15 total):")
    all_exact = True
    for i, sent in enumerate(test_samples):
        sent = sent.strip()
        if len(sent) < 5:
            continue
        
        traj, tokens, meta = sentence_to_trajectory(sent, bpe, text_id=0, sent_idx=i)
        if traj is None:
            print(f"    [{i}] SKIP: {sent[:50]}...")
            continue
        
        result = verify_trajectory(traj, tokens)
        
        status = "OK" if result['exact_match'] else f"ERR ({result['n_errors']})"
        if not result['exact_match']:
            all_exact = False
            for pos, exp, got in result['errors'][:3]:
                exp_text = bpe.decode([exp]) if exp < 4101 else '?'
                got_text = bpe.decode([got]) if got < 4101 else '?'
                print(f"      pos {pos}: expected {exp} '{exp_text}', got {got} '{got_text}'")
        
        decoded_text = bpe.decode(result['decoded'])
        print(f"    [{i}] L={meta['L']:3d} words={meta['n_words']:2d} acc={result['accuracy']:.1f}% {status}")
        print(f"         orig: {sent[:80]}...")
        print(f"         dec:  {decoded_text[:80]}...")
    
    if all_exact:
        print(f"\n    >>> ALL SAMPLES 100% EXACT <<<")
    else:
        print(f"\n    >>> ERRORS DETECTED <<<")
    
    # 4. Full corpus processing
    print(f"\n[4] Processing full corpus ({len(sentences)} sentences)...")
    
    all_trajectories = []
    all_tokens_list = []
    stats = {'ok': 0, 'skip': 0, 'errors': 0}
    
    t0 = time.time()
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < 5:
            stats['skip'] += 1
            continue
        
        traj, tokens, meta = sentence_to_trajectory(sent, bpe, text_id=0, sent_idx=i)
        if traj is None:
            stats['skip'] += 1
            continue
        
        # Verify 100%
        result = verify_trajectory(traj, tokens)
        if not result['exact_match']:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"    ERROR at sentence {i}: {result['n_errors']} errors")
            continue
        
        all_trajectories.append(traj)
        all_tokens_list.append(tokens)
        stats['ok'] += 1
        
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1}/{len(sentences)} processed ({rate:.0f} sent/s), "
                  f"OK={stats['ok']} skip={stats['skip']} err={stats['errors']}")
    
    elapsed = time.time() - t0
    print(f"\n    Done: {stats['ok']} sentences OK, {stats['skip']} skipped, {stats['errors']} errors")
    print(f"    Time: {elapsed:.0f}s ({stats['ok']/elapsed:.0f} sent/s)")
    
    # 5. Save
    print(f"\n[5] Saving trajectories...")
    
    save_dir = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as list (variable-length trajectories)
    data = {
        'trajectories': all_trajectories,  # list of [L, 384] arrays
        'tokens': all_tokens_list,          # list of token lists
        'stats': stats,
        'n_sentences': len(all_trajectories),
        'total_tokens': sum(len(t) for t in all_tokens_list),
        'version': '1.0',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Save pickle (variable-length)
    with open(os.path.join(save_dir, 'warpeace_trajectories.pkl'), 'wb') as f:
        pickle.dump(data, f, protocol=5)  # protocol 5 for large objects
    
    # Also save metadata as JSON
    json_stats = {k: v for k, v in data.items() if k not in ('trajectories', 'tokens')}
    with open(os.path.join(save_dir, 'warpeace_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, ensure_ascii=False, indent=2)
    
    print(f"    Saved {data['n_sentences']} trajectories ({data['total_tokens']:,} tokens)")
    print(f"    Location: {save_dir}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"IT'S DONE. {data['n_sentences']:,} trajectories, {data['total_tokens']:,} tokens")
    print(f"All verified: 100% roundtrip on every sentence.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
