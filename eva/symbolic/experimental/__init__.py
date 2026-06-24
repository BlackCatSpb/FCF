"""Experimental VSA utilities — not integrated into training pipeline.

These classes and functions are prototypes from Fibonacci and Z8^d research.
They are maintained but NOT called by STDPTrainer, ConceptSpace, or CrystalGenerator.
"""

from eva.symbolic.experimental.vsa_grid import VSAGrid, VSAConvLayer, VSACNN
from eva.symbolic.experimental.residue_encoder import ResidueEncoder
from eva.symbolic.experimental.vsa_utils import (
    _make_kernel, _fractal_convolution, _compute_dim_importance,
    _analogy, _quantize_adaptive, _random_masks,
)

__all__ = [
    'VSAGrid', 'VSAConvLayer', 'VSACNN',
    'ResidueEncoder',
    '_make_kernel', '_fractal_convolution',
    '_compute_dim_importance', '_analogy',
    '_quantize_adaptive', '_random_masks',
]
