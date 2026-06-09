import sys
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import re

with open('eva/symbolic/concept_space.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = 'def _init_affix_shifts(self):'
end_marker = '# ---- STDP: Spike-Timing-Dependent Plasticity on fractal codes ----'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx == -1 or end_idx == -1:
    print(f"Markers not found: start={start_idx}, end={end_idx}")
    sys.exit(1)

new_block = '''    # ---- Initialization ----

    def init_concepts(self):
        """Initialize all concept vectors (0..vocab_size-1) via fractal field."""
        for cid in range(self.vocab_size):
            v = self.fractal.init_concept(cid)
            if v is not None:
                self.concept_vectors[cid] = v
            else:
                v = self.rng.randn(self.dim).astype(np.float32)
                v /= max(np.linalg.norm(v), 1e-10)
                self.concept_vectors[cid] = v
        print(f"  Initialized {len(self.concept_vectors)} concepts via fractal")

    def _sync_concept_vectors_from_fractal(self):
        """Rebuild concept_vectors dict from fractal latent codes."""
        for cid in list(self.fractal.codes.keys()):
            v = self.fractal.compute_vector(cid)
            if v is not None:
                self.concept_vectors[cid] = v
        self.mark_matrix_dirty()

    def reinit_fractal(self, cid_list=None):
        """Reinitialize all fractal latent codes (resets all vectors)."""
        cids = cid_list if cid_list is not None else list(self.concept_vectors.keys())
        self.fractal.reinitialize_all(cids)
        self._sync_concept_vectors_from_fractal()
        print(f"  Reinitialized {len(cids)} concepts via fractal field")

    def fluctuate_fractal(self, noise_scale=0.003, decay=0.9995):
        """Autonomous drift of all concept vectors."""
        self.fractal.fluctuate(noise_scale=noise_scale, decay=decay)
        self._sync_concept_vectors_from_fractal()

    # ---- STDP: Spike-Timing-Dependent Plasticity on fractal codes ----
'''

new_content = content[:start_idx] + new_block + content[end_idx:]
with open('eva/symbolic/concept_space.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Replaced successfully!')
print(f"Removed {end_idx - start_idx} chars")

# Verify
with open('eva/symbolic/concept_space.py', 'r', encoding='utf-8') as f:
    verify = f.read()
print(f"File size: {len(verify)} chars")
print(f"Contains _init_affix_shifts: {'_init_affix_shifts' in verify}")
print(f"Contains init_concepts: {'init_concepts' in verify}")
print(f"Contains _build_concept_transitions: {'_build_concept_transitions' in verify}")
print(f"Contains _apply_conceptnet_constraints: {'_apply_conceptnet_constraints' in verify}")
