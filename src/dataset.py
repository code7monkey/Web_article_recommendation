"""Data loading and preprocessing utilities for the NewsRec project."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Optional


def load_view_log(path: str) -> pd.DataFrame:
    """Load the user view log from a CSV file.

    The view log should contain at least two columns: `userID` and
    `articleID`. Each row represents a user viewing an article.

    Args:
        path: Path to the CSV file containing the view log.

    Returns:
        A pandas DataFrame of the view log.
    """
    return pd.read_csv(path)


def load_article_info(path: str) -> pd.DataFrame:
    """Load the article information from a CSV file.

    The article info file should contain a column `articleID` and a
    text column (e.g. `Content`) which will be used for content‑based
    similarity. Additional metadata columns are ignored in this helper.

    Args:
        path: Path to the CSV file containing article information.

    Returns:
        A pandas DataFrame of the article information.
    """
    return pd.read_csv(path)


def load_sample_submission(path: str) -> pd.DataFrame:
    """Load the sample submission from a CSV file.

    The sample submission should contain a `userID` column and an
    `articleID` column (which may be empty or contain placeholders).

    Args:
        path: Path to the CSV file containing the sample submission.

    Returns:
        A pandas DataFrame of the sample submission.
    """
    return pd.read_csv(path)


def create_user_article_matrix(view_log: pd.DataFrame) -> pd.DataFrame:
    """Create a user–article interaction matrix.

    The resulting matrix has users as rows and articles as columns.
    Each entry is the count of how many times the user has viewed the
    article. Missing interactions are filled with zeros.

    Args:
        view_log: DataFrame with at least `userID` and `articleID` columns.

    Returns:
        A pivot table DataFrame indexed by user IDs and with article IDs
        as columns.
    """
    user_article_matrix = view_log.groupby(['userID', 'articleID']).size().unstack(fill_value=0)
    # Ensure deterministic column ordering
    user_article_matrix = user_article_matrix.sort_index(axis=0).sort_index(axis=1)
    return user_article_matrix


def _get_default_stop_words() -> List[str]:
    """Return a combined list of English and Portuguese stop words.

    This helper replicates the stop words list used in the original
    notebook. It can be extended or replaced via the `stop_words`
    argument of `create_cosine_sim_df`.
    """
    english_stop_words = [
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him",
        "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very",
        "s", "t", "can", "will", "just", "don", "should", "now",
    ]
    portuguese_stop_words = [
        "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles",
        "aquilo", "as", "até", "com", "como", "da", "das", "de", "dela",
        "delas", "dele", "deles", "depois", "do", "dos", "e", "ela",
        "elas", "ele", "eles", "em", "entre", "era", "eram", "essa",
        "essas", "esse", "esses", "esta", "está", "estão", "estas",
        "estava", "estavam", "este", "estes", "eu", "foi", "foram",
        "fui", "há", "isso", "isto", "já", "lhe", "lhes", "mas", "me",
        "mesmo", "meu", "meus", "minha", "minhas", "muito", "na",
        "não", "nas", "nem", "no", "nos", "nossa", "nossas", "nosso",
        "nossos", "num", "numa", "o", "os", "ou", "para", "pela",
        "pelas", "pelo", "pelos", "por", "qual", "quando", "que",
        "quem", "se", "seu", "seus", "só", "suas", "também", "te",
        "tem", "tinha", "tive", "tivemos", "tiveram", "tua", "tuas",
        "tudo", "um", "uma", "você", "vocês",
    ]
    return list(english_stop_words) + list(portuguese_stop_words)


def create_cosine_sim_df(
    article_info: pd.DataFrame,
    text_column: str = 'Content',
    stop_words: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute a cosine similarity DataFrame for articles based on TF‑IDF.

    Args:
        article_info: DataFrame containing at least an `articleID` column and
            a text column specified by `text_column`.
        text_column: The column in `article_info` containing the textual
            content to build the TF‑IDF representation.
        stop_words: Optional list of stop words to remove from the text.
            If None, a default combination of English and Portuguese stop
            words is used.

    Returns:
        A square DataFrame where both the index and columns are `articleID`
        and each entry represents the cosine similarity between the
        corresponding articles.
    """
    if stop_words is None:
        stop_words = _get_default_stop_words()
    # Replace missing content with empty strings
    texts = article_info[text_column].fillna('')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Construct a DataFrame with article IDs as index/columns
    sim_df = pd.DataFrame(
        cosine_sim,
        index=article_info['articleID'],
        columns=article_info['articleID'],
    )
    return sim_df
