"""Reusable helpers for the opium snippets topic-modelling notebooks.

The functions here keep the notebooks focused on interpretation: prepare a
document table once, run LDA or BERTopic on either the whole corpus or grouped
subsets, and return compact topic summaries plus document assignments.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


DEFAULT_TOKEN_PATTERN = r"(?u)\b[a-z][a-z-]{2,}\b"
DEFAULT_PROJECT_STOP_WORDS = {
    "said",
    "say",
    "thou",
    "thy",
    "thee",
    "shall",
    "would",
    "could",
    "may",
    "like",
    "did",
    "little",
    "mr",
    "mrs",
    "miss",
    "man",
    "men",
    "came",
    "come",
    "way",
    "went",
    "just",
    "make",
    "took",
    "don",
    "yes",
    "tell",
    "thing",
    "things",
    "know",
    "kenw",
    "thought",
    "think",
}


def load_snippets(parquet_path: str | Path, csv_path: str | Path) -> pd.DataFrame:
    """Load snippets, preferring parquet when available."""
    parquet_path = Path(parquet_path)
    csv_path = Path(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def clean_text(text: object) -> str:
    """Normalize nineteenth-century snippet text for count-based models."""
    text = str(text).lower()
    text = re.sub(r"[_\u2018\u2019\u201c\u201d]", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[^a-z\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def target_terms_from_keywords(keywords: Iterable[object]) -> set[str]:
    """Turn opium keyword variants into stop-word tokens."""
    terms: set[str] = set()
    for keyword in keywords:
        terms.update(re.findall(r"\b[a-z][a-z-]{2,}\b", str(keyword).lower()))
    return terms


def corpus_frequent_words(
    texts: Sequence[str],
    base_stop_words: Iterable[str] = ENGLISH_STOP_WORDS,
    min_document_share: float = 0.25,
) -> pd.Series:
    """Find terms that are too frequent in this corpus to be distinctive."""
    cleaned = pd.Series(texts, dtype="object").map(clean_text)
    base_stop_words = set(base_stop_words)
    words = cleaned.apply(
        lambda text: [word for word in text.split() if word not in base_stop_words]
    )
    counts = words.explode().value_counts()
    threshold = max(1, int(np.ceil(min_document_share * len(cleaned))))
    return counts.loc[counts >= threshold]


def build_stop_words(
    texts: Sequence[str],
    keywords: Iterable[object] = (),
    project_stop_words: Iterable[str] = DEFAULT_PROJECT_STOP_WORDS,
    min_document_share: float = 0.25,
) -> list[str]:
    """Combine sklearn English stop words, project words, target terms and common corpus words."""
    target_terms = target_terms_from_keywords(keywords)
    common_terms = set(
        corpus_frequent_words(
            texts,
            base_stop_words=ENGLISH_STOP_WORDS | target_terms,
            min_document_share=min_document_share,
        ).index
    )
    return sorted(
        set(ENGLISH_STOP_WORDS)
        | set(project_stop_words)
        | target_terms
        | common_terms
    )


def prepare_documents(
    df: pd.DataFrame,
    text_col: str = "Snippet",
    id_col: str = "snippet_id",
    clean_col: str = "document",
) -> pd.DataFrame:
    """Add a stable id and cleaned document column, dropping empty documents."""
    docs = df.copy()
    if id_col not in docs.columns:
        docs[id_col] = docs.index
    docs[clean_col] = docs[text_col].map(clean_text)
    docs = docs.loc[docs[clean_col].str.len() > 0].reset_index(drop=True)
    return docs


def split_snippets_into_sentences(
    df: pd.DataFrame,
    text_col: str = "Snippet",
    snippet_id_col: str = "snippet_id",
    min_words: int = 6,
) -> pd.DataFrame:
    """Split each snippet into sentence-like documents while keeping snippet metadata."""
    if snippet_id_col not in df.columns:
        df = df.copy()
        df[snippet_id_col] = df.index

    rows = []
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    for _, row in df.iterrows():
        raw_sentences = sentence_pattern.split(str(row[text_col]))
        sentence_number = 0
        for sentence in raw_sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence.split()) < min_words:
                continue
            new_row = row.to_dict()
            new_row["sentence_id"] = f"{row[snippet_id_col]}_{sentence_number}"
            new_row["sentence_number"] = sentence_number
            new_row[text_col] = sentence
            rows.append(new_row)
            sentence_number += 1

    return pd.DataFrame(rows)


def build_keyword_pattern(keywords: Iterable[object]) -> re.Pattern:
    """Build a case-insensitive regex for opium-related keyword matching."""
    terms = sorted(
        {str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()},
        key=len,
        reverse=True,
    )
    escaped_terms = [re.escape(term).replace(r"\ ", r"\s+") for term in terms]
    if not escaped_terms:
        return re.compile(r"a\A")
    return re.compile(r"\b(?:" + "|".join(escaped_terms) + r")\b", flags=re.IGNORECASE)


def add_keyword_context_flags(
    sentences: pd.DataFrame,
    keywords: Iterable[object],
    text_col: str = "Snippet",
    snippet_id_col: str = "snippet_id",
    sentence_number_col: str = "sentence_number",
    window: int = 1,
) -> pd.DataFrame:
    """Mark keyword-bearing sentences and sentences within a local context window."""
    out = sentences.copy()
    pattern = build_keyword_pattern(keywords)
    out["contains_target_term"] = out[text_col].astype(str).str.contains(
        pattern,
        regex=True,
        na=False,
    )

    out["near_target_term"] = False
    for _, group in out.groupby(snippet_id_col, sort=False):
        target_numbers = set(
            group.loc[group["contains_target_term"], sentence_number_col].astype(int)
        )
        if not target_numbers:
            continue
        context_numbers = {
            sentence_number + offset
            for sentence_number in target_numbers
            for offset in range(-window, window + 1)
        }
        context_index = group.loc[
            group[sentence_number_col].astype(int).isin(context_numbers)
        ].index
        out.loc[context_index, "near_target_term"] = True

    return out


def add_locc_group(
    df: pd.DataFrame,
    locc_col: str = "LoCC",
    output_col: str = "LoCC_group",
) -> pd.DataFrame:
    """Label PR as English Literature and PS as American Literature."""
    labels = {
        "PR": "English Literature",
        "PS": "American Literature",
    }
    out = df.copy()
    out[output_col] = out[locc_col].astype("string").str.strip().map(labels)
    return out


def add_period_column(
    df: pd.DataFrame,
    year_col: str = "Published Year",
    output_col: str = "Period",
    start_year: int = 1850,
    period_years: int = 20,
    n_periods: int = 4,
    period_bins: Sequence[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Add fixed period labels, inclusive of both period endpoints."""
    out = df.copy()
    years = pd.to_numeric(out[year_col], errors="coerce")
    if period_bins is None:
        period_bins = [
            (start_year + period_years * i, start_year + period_years * (i + 1) - 1)
            for i in range(n_periods)
        ]
    bins = [period_bins[0][0]] + [end_year + 1 for _, end_year in period_bins]
    labels = [f"{start_year}-{end_year}" for start_year, end_year in period_bins]
    out[output_col] = pd.cut(
        years,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    out[output_col] = out[output_col].astype("string")
    return out


def vectorize_documents(
    documents: Sequence[str],
    stop_words: Iterable[str],
    min_df: int = 5,
    max_df: float = 0.65,
    ngram_range: tuple[int, int] = (1, 2),
    token_pattern: str = DEFAULT_TOKEN_PATTERN,
):
    """Build a document-term matrix and matching feature names."""
    vectorizer = CountVectorizer(
        stop_words=list(stop_words),
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
    )
    matrix = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())
    return vectorizer, matrix, feature_names


def fit_lda_candidates(
    matrix,
    candidate_topic_counts: Iterable[int],
    random_state: int = 23,
    max_iter: int = 25,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[int, tuple[LatentDirichletAllocation, np.ndarray]]]:
    """Fit LDA models for a range of topic counts and collect diagnostics."""
    scores = []
    models = {}
    for n_topics in candidate_topic_counts:
        model = LatentDirichletAllocation(
            n_components=int(n_topics),
            learning_method="batch",
            max_iter=max_iter,
            random_state=random_state,
            evaluate_every=-1,
            n_jobs=n_jobs,
        )
        doc_topic = model.fit_transform(matrix)
        models[int(n_topics)] = (model, doc_topic)
        scores.append(
            {
                "n_topics": int(n_topics),
                "perplexity": model.perplexity(matrix),
                "log_likelihood": model.score(matrix),
            }
        )
    return pd.DataFrame(scores), models


def top_terms_by_topic(
    model: LatentDirichletAllocation,
    feature_names: np.ndarray,
    n_terms: int = 12,
) -> pd.DataFrame:
    """Return top weighted terms for each LDA topic."""
    rows = []
    for topic_idx, weights in enumerate(model.components_):
        top_indices = weights.argsort()[::-1][:n_terms]
        rows.append(
            {
                "topic": topic_idx,
                "top_terms": ", ".join(feature_names[top_indices]),
            }
        )
    return pd.DataFrame(rows)


def lda_topic_summary(
    docs: pd.DataFrame,
    doc_topic_matrix: np.ndarray,
    topics_df: pd.DataFrame,
    group_label: str = "all",
    id_col: str = "snippet_id",
    book_col: str = "Book_ID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach dominant LDA topic assignments and summarize topic sizes."""
    topic_columns = [f"topic_{i}" for i in range(doc_topic_matrix.shape[1])]
    doc_topics = pd.DataFrame(doc_topic_matrix, columns=topic_columns)
    modelled = pd.concat([docs.reset_index(drop=True), doc_topics], axis=1)
    modelled["dominant_topic"] = doc_topic_matrix.argmax(axis=1)
    modelled["dominant_topic_weight"] = doc_topic_matrix.max(axis=1)
    modelled["model_group"] = group_label

    aggregations = {
        "n_documents": (id_col, "count"),
        "mean_weight": ("dominant_topic_weight", "mean"),
    }
    if book_col in modelled.columns:
        aggregations["n_books"] = (book_col, "nunique")

    summary = (
        modelled.groupby("dominant_topic")
        .agg(**aggregations)
        .reset_index()
        .merge(topics_df, left_on="dominant_topic", right_on="topic", how="left")
        .drop(columns="topic")
        .sort_values("n_documents", ascending=False)
    )
    summary.insert(0, "model_group", group_label)
    return modelled, summary


def run_lda_topic_model(
    docs: pd.DataFrame,
    stop_words: Iterable[str],
    candidate_topic_counts: Iterable[int] = range(2, 70, 2),
    selected_n_topics: int | None = None,
    group_label: str = "all",
    text_col: str = "document",
    id_col: str = "snippet_id",
    min_docs: int = 20,
    min_df: int = 5,
    max_df: float = 0.65,
    ngram_range: tuple[int, int] = (1, 2),
    random_state: int = 23,
    max_iter: int = 25,
    n_jobs: int = 1,
    n_terms: int = 12,
) -> dict[str, object]:
    """Run a complete LDA workflow for one document table."""
    docs = docs.loc[docs[text_col].notna()].copy().reset_index(drop=True)
    docs = docs.loc[docs[text_col].str.len() > 0].reset_index(drop=True)
    if len(docs) < min_docs:
        return {
            "status": f"skipped: only {len(docs)} documents",
            "group": group_label,
            "docs": docs,
            "topics": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "assignments": pd.DataFrame(),
            "scores": pd.DataFrame(),
        }

    adaptive_min_df = min(min_df, max(1, len(docs) // 10))
    try:
        vectorizer, matrix, feature_names = vectorize_documents(
            docs[text_col].tolist(),
            stop_words=stop_words,
            min_df=adaptive_min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )
    except ValueError as exc:
        return {
            "status": f"skipped: {exc}",
            "group": group_label,
            "docs": docs,
            "topics": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "assignments": pd.DataFrame(),
            "scores": pd.DataFrame(),
        }

    valid_topic_counts = [
        int(n)
        for n in candidate_topic_counts
        if int(n) >= 2 and int(n) <= max(2, matrix.shape[0] - 1)
    ]
    if not valid_topic_counts:
        valid_topic_counts = [2]

    scores, models = fit_lda_candidates(
        matrix,
        valid_topic_counts,
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )
    if selected_n_topics is None:
        selected_n_topics = int(scores.sort_values("perplexity").iloc[0]["n_topics"])
    elif selected_n_topics not in models:
        selected_n_topics = min(models, key=lambda n: abs(n - selected_n_topics))

    model, doc_topic_matrix = models[selected_n_topics]
    topics = top_terms_by_topic(model, feature_names, n_terms=n_terms)
    assignments, summary = lda_topic_summary(
        docs,
        doc_topic_matrix,
        topics,
        group_label=group_label,
        id_col=id_col,
    )
    return {
        "status": "fit",
        "group": group_label,
        "model": model,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "feature_names": feature_names,
        "selected_n_topics": selected_n_topics,
        "topics": topics,
        "summary": summary,
        "assignments": assignments,
        "scores": scores,
    }


def run_lda_by_group(
    docs: pd.DataFrame,
    group_cols: str | Sequence[str],
    stop_words: Iterable[str],
    **kwargs,
) -> dict[str, dict[str, object]]:
    """Run LDA independently for each group value or group-value combination."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    results = {}
    grouped = docs.dropna(subset=list(group_cols)).groupby(list(group_cols), dropna=False)
    for keys, group_df in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join(f"{col}={value}" for col, value in zip(group_cols, keys))
        results[label] = run_lda_topic_model(
            group_df.reset_index(drop=True),
            stop_words=stop_words,
            group_label=label,
            **kwargs,
        )
    return results


def combine_result_frames(
    results: Mapping[str, Mapping[str, object]],
    frame_name: str,
) -> pd.DataFrame:
    """Concatenate a named result frame from grouped model outputs."""
    frames = []
    for label, result in results.items():
        frame = result.get(frame_name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame = frame.copy()
            if "model_group" not in frame.columns:
                frame.insert(0, "model_group", label)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_bertopic_vectorizer(
    stop_words: Iterable[str],
    min_df: int = 1,
    max_df: float = 1.0,
    ngram_range: tuple[int, int] = (1, 2),
    token_pattern: str = DEFAULT_TOKEN_PATTERN,
) -> CountVectorizer:
    """Create the vectorizer used by BERTopic's c-TF-IDF description step.

    BERTopic applies this vectorizer to topic-level documents, not to the
    original snippets. Grouped models can discover only one or two clusters, so
    stricter min_df/max_df settings may become mathematically impossible.
    """
    return CountVectorizer(
        stop_words=list(stop_words),
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
    )


def get_bertopic_terms(model, topic_id: int, n_terms: int = 12) -> str:
    """Return BERTopic terms as a readable comma-separated string."""
    terms = model.get_topic(topic_id)
    if not terms:
        return ""
    return ", ".join(term for term, _ in terms[:n_terms])


def bertopic_topic_tables(
    model,
    docs: pd.DataFrame,
    topics: Sequence[int],
    probabilities=None,
    group_label: str = "all",
    id_col: str = "snippet_id",
    book_col: str = "Book_ID",
    n_terms: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach BERTopic assignments and build topic term and summary tables."""
    terms = pd.DataFrame({"topic": list(model.get_topics().keys())})
    terms["top_terms"] = terms["topic"].map(
        lambda topic_id: get_bertopic_terms(model, topic_id, n_terms=n_terms)
    )
    terms = terms.sort_values("topic").reset_index(drop=True)

    assignments = docs.copy().reset_index(drop=True)
    assignments["bertopic_topic"] = list(topics)
    if probabilities is not None and np.ndim(probabilities) == 2:
        assignments["bertopic_topic_probability"] = np.asarray(probabilities).max(axis=1)
    elif probabilities is not None:
        assignments["bertopic_topic_probability"] = probabilities
    else:
        assignments["bertopic_topic_probability"] = np.nan
    assignments["model_group"] = group_label

    aggregations = {
        "n_documents": (id_col, "count"),
        "mean_probability": ("bertopic_topic_probability", "mean"),
    }
    if book_col in assignments.columns:
        aggregations["n_books"] = (book_col, "nunique")

    summary = (
        assignments.groupby("bertopic_topic")
        .agg(**aggregations)
        .reset_index()
        .merge(terms, left_on="bertopic_topic", right_on="topic", how="left")
        .drop(columns="topic")
        .sort_values("n_documents", ascending=False)
    )
    summary.insert(0, "model_group", group_label)
    return assignments, terms, summary


def adaptive_min_topic_size(n_documents: int, preferred: int = 50, floor: int = 8) -> int:
    """Choose a BERTopic min_topic_size that still works for smaller subsets."""
    if n_documents <= preferred * 2:
        return max(floor, n_documents // 10)
    return preferred


def run_bertopic_model(
    docs: pd.DataFrame,
    stop_words: Iterable[str],
    embedding_model: str = "emanjavacas/MacBERTh",
    group_label: str = "all",
    text_col: str = "document",
    id_col: str = "snippet_id",
    min_docs: int = 30,
    min_topic_size: int | None = None,
    nr_topics: int | str | None = None,
    calculate_probabilities: bool = True,
    n_terms: int = 12,
    verbose: bool = True,
    embeddings=None,
    vectorizer_min_df: int = 1,
    vectorizer_max_df: float = 1.0,
    vectorizer_ngram_range: tuple[int, int] = (1, 2),
    bertopic_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run BERTopic and return model, topic terms, summary and assignments."""
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise ImportError(
            "BERTopic is not installed in this environment yet. "
            "Install the requirements with: pip install -r requirements.txt"
        ) from exc

    docs = docs.loc[docs[text_col].notna()].copy().reset_index(drop=True)
    docs = docs.loc[docs[text_col].str.len() > 0].reset_index(drop=True)
    if len(docs) < min_docs:
        return {
            "status": f"skipped: only {len(docs)} documents",
            "group": group_label,
            "docs": docs,
            "topics": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "assignments": pd.DataFrame(),
            "topic_info": pd.DataFrame(),
        }

    if min_topic_size is None:
        min_topic_size = adaptive_min_topic_size(len(docs))

    if embeddings is not None:
        embeddings = np.asarray(embeddings)
        if len(embeddings) != len(docs):
            raise ValueError(
                "The number of BERTopic embeddings must match the number of documents."
            )

    vectorizer = make_bertopic_vectorizer(
        stop_words=stop_words,
        min_df=vectorizer_min_df,
        max_df=vectorizer_max_df,
        ngram_range=vectorizer_ngram_range,
    )
    kwargs = dict(bertopic_kwargs or {})
    model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        top_n_words=n_terms,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=calculate_probabilities,
        verbose=verbose,
        **kwargs,
    )
    documents = docs[text_col].tolist()
    if embeddings is None:
        topics, probabilities = model.fit_transform(documents)
    else:
        topics, probabilities = model.fit_transform(documents, embeddings=embeddings)
    assignments, terms, summary = bertopic_topic_tables(
        model,
        docs,
        topics,
        probabilities=probabilities,
        group_label=group_label,
        id_col=id_col,
        n_terms=n_terms,
    )
    topic_info = model.get_topic_info()
    topic_info.insert(0, "model_group", group_label)
    return {
        "status": "fit",
        "group": group_label,
        "model": model,
        "topics": terms,
        "summary": summary,
        "assignments": assignments,
        "topic_info": topic_info,
        "min_topic_size": min_topic_size,
    }


def run_bertopic_by_group(
    docs: pd.DataFrame,
    group_cols: str | Sequence[str],
    stop_words: Iterable[str],
    **kwargs,
) -> dict[str, dict[str, object]]:
    """Run BERTopic independently for each group value or group-value combination."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    results = {}
    grouped = docs.dropna(subset=list(group_cols)).groupby(list(group_cols), dropna=False)
    for keys, group_df in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join(f"{col}={value}" for col, value in zip(group_cols, keys))
        results[label] = run_bertopic_model(
            group_df.reset_index(drop=True),
            stop_words=stop_words,
            group_label=label,
            **kwargs,
        )
    return results


def bertopic_size_diagnostics(
    summary: pd.DataFrame,
    total_documents: int | None = None,
    topic_col: str = "bertopic_topic",
    count_col: str = "n_documents",
    outlier_topic: int = -1,
    rich_topic_min_size: int = 40,
    rich_topic_max_size: int = 500,
) -> dict[str, object]:
    """Summarize whether BERTopic produced useful small clusters or huge buckets."""
    if summary.empty:
        return {
            "n_topics": 0,
            "outlier_share": np.nan,
            "largest_topic_share": np.nan,
            "n_rich_sized_topics": 0,
        }

    if total_documents is None:
        total_documents = int(summary[count_col].sum())
    non_outlier = summary.loc[summary[topic_col] != outlier_topic].copy()
    outlier_count = int(
        summary.loc[summary[topic_col] == outlier_topic, count_col].sum()
    )
    largest_topic_count = int(non_outlier[count_col].max()) if not non_outlier.empty else 0
    rich_sized = non_outlier.loc[
        non_outlier[count_col].between(rich_topic_min_size, rich_topic_max_size)
    ]

    return {
        "n_topics": int(non_outlier.shape[0]),
        "outlier_documents": outlier_count,
        "outlier_share": outlier_count / total_documents if total_documents else np.nan,
        "largest_topic_documents": largest_topic_count,
        "largest_topic_share": (
            largest_topic_count / total_documents if total_documents else np.nan
        ),
        "n_rich_sized_topics": int(rich_sized.shape[0]),
        "median_topic_size": (
            float(non_outlier[count_col].median()) if not non_outlier.empty else np.nan
        ),
    }


def save_model_outputs(
    output_dir: str | Path,
    prefix: str,
    topics: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    assignments: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    topic_info: pd.DataFrame | None = None,
) -> None:
    """Save non-empty model output tables with a consistent prefix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "topics_terms": topics,
        "topic_summary": summary,
        "document_topics": assignments,
        "model_scores": scores,
        "topic_info": topic_info,
    }
    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(output_dir / f"{prefix}_{name}.csv", index=False)


def representative_lda_documents(
    assignments: pd.DataFrame,
    topic_id: int,
    n: int = 5,
    text_col: str = "Snippet",
) -> pd.DataFrame:
    """Return the highest-weight documents for one LDA topic."""
    topic_col = f"topic_{topic_id}"
    default_cols = [
        "model_group",
        "snippet_id",
        "Book_ID",
        "Keyword",
        topic_col,
        text_col,
    ]
    cols = [col for col in default_cols if col in assignments.columns]
    return assignments.sort_values(topic_col, ascending=False)[cols].head(n)


def representative_bertopic_documents(
    assignments: pd.DataFrame,
    topic_id: int,
    n: int = 5,
    text_col: str = "Snippet",
) -> pd.DataFrame:
    """Return the highest-probability documents assigned to one BERTopic topic."""
    default_cols = [
        "model_group",
        "snippet_id",
        "Book_ID",
        "Keyword",
        "bertopic_topic_probability",
        text_col,
    ]
    cols = [col for col in default_cols if col in assignments.columns]
    return (
        assignments.loc[assignments["bertopic_topic"] == topic_id]
        .sort_values("bertopic_topic_probability", ascending=False)
        [cols]
        .head(n)
    )


def make_topic_labels(
    topics: pd.DataFrame,
    topic_col: str = "topic",
    terms_col: str = "top_terms",
    max_terms: int = 5,
) -> pd.DataFrame:
    """Create compact labels such as '12: sleep, dream, night' for plots."""
    labels = topics[[topic_col, terms_col]].copy()
    labels[topic_col] = pd.to_numeric(labels[topic_col], errors="coerce").astype("Int64")

    def label_topic(row) -> str:
        topic = row[topic_col]
        terms = [
            term.strip()
            for term in str(row[terms_col]).split(",")
            if term.strip()
        ][:max_terms]
        if pd.isna(topic):
            topic_name = "NA"
        elif int(topic) == -1:
            topic_name = "-1"
        else:
            topic_name = str(int(topic))
        return f"{topic_name}: {', '.join(terms)}" if terms else topic_name

    labels["topic_label"] = labels.apply(label_topic, axis=1)
    return labels


def add_topic_labels(
    assignments: pd.DataFrame,
    topics: pd.DataFrame,
    assignment_topic_col: str = "bertopic_topic",
    topic_col: str = "topic",
    terms_col: str = "top_terms",
    max_terms: int = 5,
) -> pd.DataFrame:
    """Attach compact topic labels to BERTopic assignment rows."""
    out = assignments.copy()
    out[assignment_topic_col] = pd.to_numeric(
        out[assignment_topic_col],
        errors="coerce",
    ).astype("Int64")
    labels = make_topic_labels(
        topics,
        topic_col=topic_col,
        terms_col=terms_col,
        max_terms=max_terms,
    )
    return out.merge(
        labels[[topic_col, "topic_label"]],
        left_on=assignment_topic_col,
        right_on=topic_col,
        how="left",
    ).drop(columns=[topic_col])


def topic_distribution(
    assignments: pd.DataFrame,
    group_cols: str | Sequence[str],
    topic_col: str = "bertopic_topic",
    label_col: str = "topic_label",
    unit_col: str = "sentence_id",
    include_outlier: bool = False,
) -> pd.DataFrame:
    """Count and normalize final-topic assignments within metadata groups."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    df = assignments.copy()
    df[topic_col] = pd.to_numeric(df[topic_col], errors="coerce")
    if not include_outlier:
        df = df.loc[df[topic_col] != -1].copy()
    df = df.dropna(subset=list(group_cols) + [topic_col])

    if unit_col in df.columns:
        grouped = (
            df.groupby(list(group_cols) + [topic_col, label_col], dropna=False)[unit_col]
            .nunique()
            .rename("n_documents")
            .reset_index()
        )
    else:
        grouped = (
            df.groupby(list(group_cols) + [topic_col, label_col], dropna=False)
            .size()
            .rename("n_documents")
            .reset_index()
        )

    grouped["group_total"] = grouped.groupby(list(group_cols))["n_documents"].transform(
        "sum"
    )
    grouped["topic_share"] = grouped["n_documents"] / grouped["group_total"]
    return grouped.sort_values(list(group_cols) + ["topic_share"], ascending=False)


def top_topics_by_group(
    distribution: pd.DataFrame,
    group_cols: str | Sequence[str],
    n_topics: int = 10,
) -> pd.DataFrame:
    """Keep the largest topic shares within each group."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    return (
        distribution.sort_values(list(group_cols) + ["topic_share"], ascending=False)
        .groupby(list(group_cols), group_keys=False)
        .head(n_topics)
        .reset_index(drop=True)
    )
