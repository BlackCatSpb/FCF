"""
fill_reserved — fill 287 reserved dimensions (97-383) during generation.

Reserved dim layout:
  dim   97-102: 6 head weights
  dim  103-108: 6 head contribution scores for selected token
  dim      109: winning head index (0-5)
  dim      110: concept score (rare=1, freq=0)
  dim      111: max contradiction penalty
  dim      112: attractor potential
  dim  113-287: spare
"""
import numpy as np


# Dim offsets
D_W_HEAD = 97      # 6 dims
D_CONTRIB = 103     # 6 dims
D_WINNER = 109      # 1 dim
D_CONCEPT = 110     # 1 dim
D_CONTRA = 111      # 1 dim
D_ATTRACTOR = 112   # 1 dim
D_SPARE = 113       # spare start


def fill_reserved(h: np.ndarray, head_weights: np.ndarray,
                  head_contributions: np.ndarray, selected_token: int,
                  heads_obj=None) -> np.ndarray:
    """
    Fill reserved dims 97-383 in coordinate vector h.

    Args:
        h: 384-dim coordinate array (modified in-place and returned)
        head_weights: (6,) array of head weights used for this selection
        head_contributions: (6,) array — each head's score for the selected token
        selected_token: the token ID that was selected
        heads_obj: HeadsEnsemble instance (for concept, contra, etc.)
    Returns:
        Modified h (same array, for chaining)
    """
    if h.shape[0] < 384:
        return h

    # dim 97-102: head weights (clamped 0.0-5.0, normalized)
    for i in range(6):
        val = float(head_weights[i]) if i < len(head_weights) else 0.0
        h[D_W_HEAD + i] = min(max(val / 5.0, 0.0), 1.0)

    # dim 103-108: head contributions for selected token
    for i in range(6):
        val = float(head_contributions[i]) if i < len(head_contributions) else 0.0
        h[D_CONTRIB + i] = min(max(val / 5.0 + 0.5, 0.0), 1.0)

    # dim 109: winning head
    if len(head_contributions) > 0:
        winner = int(np.argmax(head_contributions))
        h[D_WINNER] = (winner + 1) / 6.0  # 1/6 to 6/6 range
    else:
        h[D_WINNER] = 0.0

    # dim 110: concept score
    if heads_obj is not None and selected_token < heads_obj.V:
        h[D_CONCEPT] = float(heads_obj.concept_scores[selected_token])
    else:
        h[D_CONCEPT] = 0.5

    # dim 111: max contradiction penalty
    h[D_CONTRA] = 0.0

    # dim 112: attractor potential (spare for now)
    h[D_ATTRACTOR] = 0.0

    # dim 113-287: spare
    h[D_SPARE:] = 0.0

    return h
