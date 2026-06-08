"""HierarchicalCompressor — text → semantic tree with centroid + residuals.

Compresses text into hierarchical structure:
  word → phrase → sentence → paragraph

Each level:
  - centroid: mean vector of all constituents (the "gist")
  - residuals: what each constituent contributes beyond the centroid
  - children: sub-level nodes

Usage:
  compressor = HierarchicalCompressor(cs)
  tree = compressor.compress("Вчера шёл сильный дождь.")
  # tree = {
  #   'level': 'sentence', 'centroid': [...], 'residuals': [...],
  #   'children': [
  #     {'level': 'phrase', 'centroid': [...], 'words': ['шёл', 'дождь']},
  #     ...
  #   ]
  # }
"""

import numpy as np
from eva.symbolic.concept_tokenizer import ConceptTokenizer
from eva.symbolic.pos_tagger import get_pos


class HierarchicalCompressor:
    """Compress text into hierarchical semantic tree.

    Each level captures the essential meaning (centroid) and what
    each part adds beyond the centroid (residuals).
    """

    def __init__(self, cs, lattice=None):
        self.cs = cs
        self.lattice = lattice

    def compress(self, text):
        """Compress text into hierarchical semantic tree.

        Args:
            text: input text (single sentence or paragraph)

        Returns:
            dict tree: {level, centroid, residuals, children, words}
        """
        sentences = self._split_sentences(text)
        if len(sentences) == 1:
            return self._compress_sentence(sentences[0])
        return self._compress_paragraph(sentences)

    def _split_sentences(self, text):
        import re
        sents = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', text.strip())
        return [s.strip() for s in sents if s.strip()]

    def _compress_sentence(self, sentence):
        """Single sentence → tree of word→phrase→sentence."""
        words = sentence.split()
        word_nodes = []

        for w in words:
            clean = w.strip('.,!?;:()[]{}«»—–-…\'\"')
            if not clean:
                continue
            morph = ConceptTokenizer.morph_parse(clean)
            cid, v = ConceptTokenizer.morph_root_vector(clean, self.cs)
            if v is None:
                continue

            word_nodes.append({
                'word': clean,
                'root': morph['normal_form'] if morph else clean,
                'prefix': morph['prefix'] if morph else '',
                'suffix': morph['suffix'] if morph else '',
                'ending': morph['ending'] if morph else '',
                'pos': morph['pos'] if morph else get_pos(clean),
                'cid': cid,
                'vector': v.copy(),
            })

        if not word_nodes:
            return {'level': 'word', 'centroid': None, 'words': []}

        # Group into phrases by POS patterns
        phrases = self._group_phrases(word_nodes)

        # Compute phrase vectors
        phrase_nodes = []
        for phrase in phrases:
            vecs = [n['vector'] for n in phrase if n['vector'] is not None]
            if not vecs:
                continue
            centroid = np.mean(vecs, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Residuals: difference between each word and phrase centroid
            residuals = []
            for n in phrase:
                if n['vector'] is not None:
                    resid = n['vector'] - centroid
                    residuals.append(resid)

            phrase_nodes.append({
                'level': 'phrase',
                'centroid': centroid,
                'residuals': residuals,
                'words': [n['word'] for n in phrase],
                'cids': [n['cid'] for n in phrase if n['cid'] is not None],
                'children': word_nodes,
            })

        if not phrase_nodes:
            return {'level': 'sentence', 'centroid': None, 'words': words}

        # Sentence vector = mean of phrase centroids
        sent_centroid = np.mean(
            [p['centroid'] for p in phrase_nodes if p['centroid'] is not None],
            axis=0
        ).astype(np.float32)
        sent_norm = np.linalg.norm(sent_centroid)
        if sent_norm > 0:
            sent_centroid /= sent_norm

        # Sentence-level residuals
        sent_residuals = []
        for p in phrase_nodes:
            if p['centroid'] is not None:
                sent_residuals.append(p['centroid'] - sent_centroid)

        return {
            'level': 'sentence',
            'centroid': sent_centroid,
            'residuals': sent_residuals,
            'words': [n['word'] for n in word_nodes],
            'cids': [n['cid'] for n in word_nodes if n['cid'] is not None],
            'children': phrase_nodes,
        }

    def _compress_paragraph(self, sentences):
        """Multiple sentences → paragraph tree."""
        sent_nodes = [self._compress_sentence(s) for s in sentences]
        sent_nodes = [s for s in sent_nodes if s.get('centroid') is not None]

        if not sent_nodes:
            return {'level': 'paragraph', 'centroid': None, 'sentences': []}

        para_centroid = np.mean(
            [s['centroid'] for s in sent_nodes], axis=0
        ).astype(np.float32)
        norm = np.linalg.norm(para_centroid)
        if norm > 0:
            para_centroid /= norm

        return {
            'level': 'paragraph',
            'centroid': para_centroid,
            'sentences': sent_nodes,
            'words': [w for s in sent_nodes for w in s.get('words', [])],
        }

    def _group_phrases(self, word_nodes):
        """Group words into phrases based on POS patterns.

        Simple heuristic grouping:
        - ADJ + NOUN → noun phrase
        - ADV + VERB → verb phrase
        - PREP + NOUN → prepositional phrase
        - NOUN + VERB → clause (separate phrases)
        """
        if not word_nodes:
            return []

        phrases = []
        current = [word_nodes[0]]

        for node in word_nodes[1:]:
            prev = current[-1]
            pos = node['pos']
            prev_pos = prev['pos']

            # Same phrase if: ADJ→NOUN, ADV→VERB, PREP→NOUN/VERB, DET→NOUN
            if (prev_pos in ('ADJ', 'DET') and pos == 'NOUN') or \
               (prev_pos == 'ADV' and pos == 'VERB') or \
               (prev_pos == 'PREP' and pos in ('NOUN', 'VERB', 'ADJ')) or \
               (prev_pos == 'NUM' and pos == 'NOUN'):
                current.append(node)
            else:
                # New phrase starts at noun/verb
                if current:
                    phrases.append(current)
                current = [node]

        if current:
            phrases.append(current)

        return phrases

    def centroid_text(self, tree, depth=0):
        """Get text representation of the compression tree."""
        indent = '  ' * depth
        lines = []
        if tree.get('centroid') is not None:
            lines.append(f"{indent}{tree['level']}: centroid_norm={np.linalg.norm(tree['centroid']):.3f}")
        if tree.get('words'):
            lines.append(f"{indent}  words: {' '.join(tree['words'][:8])}")
        if tree.get('children'):
            for c in tree['children']:
                lines.append(self.centroid_text(c, depth + 1))
        if tree.get('sentences'):
            for s in tree['sentences']:
                lines.append(self.centroid_text(s, depth + 1))
        return '\n'.join(lines)
