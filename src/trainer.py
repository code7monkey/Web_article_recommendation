"""Training and recommendation routines for the NewsRec project."""

from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .model import (
    compute_user_similarity,
    compute_collaborative_scores,
    compute_content_based_scores,
)
from .dataset import (
    create_user_article_matrix,
    create_cosine_sim_df,
    load_view_log,
    load_article_info,
    load_sample_submission,
)
from .utils import seed_everything


def combine_scores(
    collab_scores: np.ndarray,
    content_scores: np.ndarray,
    weight_collab: float = 1.0,
    weight_content: float = 1.0,
) -> np.ndarray:
    """Combine collaborative and content scores via a weighted sum.

    Args:
        collab_scores: Scores from collaborative filtering.
        content_scores: Scores from content‑based filtering.
        weight_collab: Weight for collaborative scores.
        weight_content: Weight for content scores.

    Returns:
        A NumPy array with the same shape as the input score matrices.
    """
    return weight_collab * collab_scores + weight_content * content_scores


def prepare_merged_df(view_log: pd.DataFrame, article_info: pd.DataFrame) -> pd.DataFrame:
    """Merge view log and article info to identify self‑written articles.

    Args:
        view_log: DataFrame with `userID` and `articleID` columns.
        article_info: DataFrame with `userID` (author) and `articleID` columns.

    Returns:
        A merged DataFrame with suffixes `_view` and `_article`.
    """
    merged_df = pd.merge(
        view_log,
        article_info,
        on='articleID',
        suffixes=('_view', '_article'),
    )
    return merged_df


def compute_user_article_count(merged_df: pd.DataFrame) -> pd.Series:
    """Compute the number of articles each user has authored and viewed themselves.

    Args:
        merged_df: Merged DataFrame returned by `prepare_merged_df`.

    Returns:
        A pandas Series indexed by user ID indicating how many of their own
        articles they have viewed.
    """
    # Filter rows where the viewer is also the author
    self_viewed = merged_df[merged_df['userID_view'] == merged_df['userID_article']]
    return self_viewed.groupby('userID_view').size()


def generate_recommendations(
    user_article_matrix: pd.DataFrame,
    combined_scores: np.ndarray,
    view_log: pd.DataFrame,
    merged_df: pd.DataFrame,
    user_article_count: pd.Series,
    top_k: int = 5,
) -> List[Tuple[int, int]]:
    """Generate top‑k recommendations for each user.

    This function implements a hybrid recommendation strategy that
    prioritises articles the user has authored and articles they have
    repeatedly viewed, before filling the list with highest‑scoring
    recommendations from the hybrid score matrix.

    Args:
        user_article_matrix: Interaction matrix indexed by user IDs.
        combined_scores: Combined recommendation scores.
        view_log: Original view log DataFrame.
        merged_df: DataFrame produced by `prepare_merged_df`.
        user_article_count: Series of counts of self‑viewed articles.
        top_k: Number of recommendations per user.

    Returns:
        A list of (userID, articleID) tuples representing the
        recommendations.
    """
    recommendations: List[Tuple[int, int]] = []
    users = user_article_matrix.index.tolist()
    articles = user_article_matrix.columns.tolist()
    # Precompute per‑user repeated articles counts
    for idx, user in enumerate(users):
        # Articles authored by the user
        authored = merged_df[merged_df['userID_article'] == user][['articleID', 'userID_view']]
        # Count how many times each authored article was viewed
        # The result is a Series indexed by articleID
        article_view_counts = authored['articleID'].value_counts().sort_values(ascending=False)
        # Decide how many authored articles to include
        if user in user_article_count.index:
            num_self = user_article_count[user]
            # Users with many self‑views get up to 2 authored articles
            threshold = user_article_count.quantile(0.9)
            max_self_articles = 2 if num_self >= threshold else 1
        else:
            max_self_articles = 1
        # Extract top authored article IDs (index of the Series)
        top_authored = article_view_counts.head(max_self_articles).index.tolist()
        # Articles the user has viewed multiple times
        user_views = view_log[view_log['userID'] == user]['articleID'].value_counts()
        repeated_articles = user_views[user_views > 1].index.tolist()
        # Hybrid recommendations sorted by combined score
        sorted_indices = np.argsort(combined_scores[idx])[::-1]
        ranked_articles = [articles[i] for i in sorted_indices]
        # Build final list: repeated > authored > top‑score
        unique_recommendations: List[int] = []
        for art in repeated_articles + top_authored + ranked_articles:
            if art not in unique_recommendations:
                unique_recommendations.append(art)
            if len(unique_recommendations) >= top_k:
                break
        # Append to global list
        for art in unique_recommendations[:top_k]:
            recommendations.append((user, art))
    return recommendations


def train_pipeline(config: Dict) -> None:
    """Run the training pipeline according to the provided configuration.

    This high‑level function loads data, computes recommendation scores,
    combines them, and saves intermediate results for later inference.

    Args:
        config: Dictionary parsed from a YAML file specifying paths and
            hyperparameters.
    """
    seed = config.get('seed', 42)
    seed_everything(seed)

    # Load data
    data_cfg = config['data']
    view_log = load_view_log(data_cfg['view_log_path'])
    article_info = load_article_info(data_cfg['article_info_path'])

    # Build user‑article matrix
    user_article_matrix = create_user_article_matrix(view_log)
    # Compute user similarity and collaborative scores
    user_sim = compute_user_similarity(user_article_matrix)
    collab_scores = compute_collaborative_scores(user_article_matrix, user_sim)
    # Compute article cosine similarity and content‑based scores
    stop_words = None  # Use default; can be extended via config
    cos_df = create_cosine_sim_df(article_info, text_column=data_cfg.get('text_column', 'Content'), stop_words=stop_words)
    content_scores = compute_content_based_scores(user_article_matrix, cos_df)
    # Combine scores
    model_cfg = config.get('model', {})
    weight_collab = model_cfg.get('weight_collaborative', 1.0)
    weight_content = model_cfg.get('weight_content', 1.0)
    combined = combine_scores(collab_scores, content_scores, weight_collab, weight_content)
    # Prepare merged DataFrame and user article counts
    merged = prepare_merged_df(view_log, article_info)
    user_counts = compute_user_article_count(merged)
    # Save intermediate artifacts
    assets_cfg = config['assets']
    combined_path = assets_cfg['combined_scores_path']
    user_data_path = assets_cfg['user_data_path']
    # Ensure directories exist
    import os
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    os.makedirs(os.path.dirname(user_data_path), exist_ok=True)
    # Save combined scores as a NumPy file
    np.save(combined_path, combined)
    # Save additional objects needed for inference
    with open(user_data_path, 'wb') as f:
        pickle.dump({
            'user_article_matrix': user_article_matrix,
            'view_log': view_log,
            'merged_df': merged,
            'user_article_count': user_counts,
        }, f)
    # Optionally generate a submission during training
    training_cfg = config.get('training', {})
    if training_cfg.get('generate_submission', False):
        sample_sub = load_sample_submission(data_cfg['sample_submission_path'])
        top_k = training_cfg.get('top_k', 5)
        recs = generate_recommendations(user_article_matrix, combined, view_log, merged, user_counts, top_k=top_k)
        # Build submission DataFrame
        submission = pd.DataFrame(recs, columns=['userID', 'articleID'])
        # Align with sample submission order
        submission = sample_sub[['userID']].merge(submission, on='userID', how='left')
        output_path = assets_cfg.get('output_submission_path', 'data/final_submission.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission.to_csv(output_path, index=False)


def inference_pipeline(config: Dict) -> None:
    """Run the inference pipeline using saved artifacts.

    Args:
        config: Dictionary parsed from a YAML file specifying paths and
            hyperparameters.
    """
    seed_everything(config.get('seed', 42))
    data_cfg = config['data']
    assets_cfg = config['assets']
    model_cfg = config.get('model', {})
    inference_cfg = config.get('inference', {})
    # Load sample submission to get user list and output path
    sample_sub = load_sample_submission(data_cfg['sample_submission_path'])
    # Load combined scores
    combined = np.load(assets_cfg['combined_scores_path'])
    # Load user data
    import pickle
    with open(assets_cfg['user_data_path'], 'rb') as f:
        user_data = pickle.load(f)
    user_article_matrix = user_data['user_article_matrix']
    view_log = user_data['view_log']
    merged_df = user_data['merged_df']
    user_article_count = user_data['user_article_count']
    # Generate recommendations
    top_k = inference_cfg.get('top_k', 5)
    recs = generate_recommendations(
        user_article_matrix,
        combined,
        view_log,
        merged_df,
        user_article_count,
        top_k=top_k,
    )
    submission = pd.DataFrame(recs, columns=['userID', 'articleID'])
    # Align with sample submission order
    submission = sample_sub[['userID']].merge(submission, on='userID', how='left')
    output_path = assets_cfg.get('output_submission_path', 'data/final_submission.csv')
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
