"""vRBA local loss weighting + PDF-based adaptive sampling (RBA-R).

Copied verbatim from athena.optimization.loss_function.local_weights_sampling
so NABLA-SciML projects can use them without depending on athena.
"""
from .vrba import update_vRBA, get_exact_epsilon, get_gamma
from .sampling import sample_indices_pdf, sample_points_pdf, sample_points_pdf_coords
