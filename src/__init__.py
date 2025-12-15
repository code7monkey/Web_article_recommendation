"""Top‑level package for the NewsRec hybrid recommendation project.

This package exposes helper modules for data loading, model construction,
training and evaluation. See the individual modules for more details.
"""

from .dataset import (
    load_view_log,
    load_article_info,
    load_sample_submission,
    create_user_article_matrix,
    create_cosine_sim_df,
)
from .model import (
    compute_user_similarity,
    compute_collaborative_scores,
    compute_content_based_scores,
)
from .trainer import (
    combine_scores,
    prepare_merged_df,
    compute_user_article_count,
    generate_recommendations,
)
from .utils import seed_everything

__all__ = [
    'load_view_log', 'load_article_info', 'load_sample_submission',
    'create_user_article_matrix', 'create_cosine_sim_df',
    'compute_user_similarity', 'compute_collaborative_scores',
    'compute_content_based_scores', 'combine_scores',
    'prepare_merged_df', 'compute_user_article_count',
    'generate_recommendations', 'seed_everything'
]