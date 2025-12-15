"""Model components for the NewsRec project.

This module implements collaborative filtering and content‑based
recommendation logic. The functions operate on pandas DataFrames and
NumPy arrays, making them easy to integrate into the training and
inference pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import create_cosine_sim_df


def compute_user_similarity(user_article_matrix: pd.DataFrame) -> np.ndarray:
    """Compute cosine similarity between users based on their interaction matrix.

    Args:
        user_article_matrix: A DataFrame with users as rows and articles as
            columns. Each entry should represent the number of times a
            user has viewed an article.

    Returns:
        A 2D NumPy array containing pairwise user similarities.
    """
    return cosine_similarity(user_article_matrix)


def compute_collaborative_scores(
    user_article_matrix: pd.DataFrame, user_similarity: np.ndarray
) -> np.ndarray:
    """Compute collaborative filtering scores for each user–article pair.

    The predicted score for a user–article pair is computed by
    multiplying the user similarity matrix with the interaction matrix
    and normalising by the sum of absolute similarities for each user.

    Args:
        user_article_matrix: User–article interaction matrix.
        user_similarity: Pairwise user similarity matrix.

    Returns:
        A 2D NumPy array of predicted collaborative scores with the
        same shape as `user_article_matrix`.
    """
    # Dot product between similarity matrix and interaction matrix
    raw_scores = user_similarity.dot(user_article_matrix.values)
    # Normalise by the sum of absolute similarities to avoid favouring users
    normalisation = np.abs(user_similarity).sum(axis=1, keepdims=True)
    # Prevent division by zero
    normalisation[normalisation == 0] = 1.0
    collab_scores = raw_scores / normalisation
    return collab_scores


def compute_content_based_scores(
    user_article_matrix: pd.DataFrame, cosine_sim_df: pd.DataFrame
) -> np.ndarray:
    """Compute content‑based recommendation scores for each user.

    For each user, the scores for candidate articles are obtained by
    averaging the cosine similarity of all articles the user has viewed.
    Articles the user has never viewed receive a score of zero.

    Args:
        user_article_matrix: User–article interaction matrix.
        cosine_sim_df: Square DataFrame of article–article similarities.

    Returns:
        A NumPy array with the same shape as `user_article_matrix`, where
        each row corresponds to a user and each column corresponds to an
        article.
    """
    num_users, num_articles = user_article_matrix.shape
    content_scores = np.zeros((num_users, num_articles), dtype=float)
    # Pre-convert column order to align with similarity matrix index order
    article_order = user_article_matrix.columns
    # Ensure the similarity DataFrame has the same order of articles
    sim_df = cosine_sim_df.reindex(index=article_order, columns=article_order, fill_value=0)
    for idx, user in enumerate(user_article_matrix.index):
        user_history = user_article_matrix.loc[user]
        viewed_articles = user_history[user_history > 0].index
        if len(viewed_articles) == 0:
            continue
        # Compute the mean similarity across all viewed articles
        user_sim_scores = sim_df.loc[viewed_articles].mean(axis=0)
        # Reindex to align with article order and fill missing with zero
        content_scores[idx, :] = user_sim_scores.reindex(article_order, fill_value=0).values
    return content_scores
