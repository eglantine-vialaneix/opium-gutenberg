# Feminine Narratives of Opium Use in the Long Nineteenth Century (1850–1930)

This repository contains the computational pipeline, data resources, and analysis scripts for investigating the representation of female characters and the construction of feminine narratives surrounding opium in British and American literature between 1850 and 1930. The project combines corpus-scale computational linguistics (lexical analysis, sentence-level topic modeling, and connotation frames) with qualitative close readings.

---

## Repository Structure

*   `code/`: Core Python notebooks and utility scripts organized by pipeline phase.
*   `data/`: Scoped datasets, enriched metadata, extracted snippets, and trained models.
*   `plots/`: Dedicated directory containing all visualization plots generated during exploration and analysis.
*   `frameworks/`: Lexical lexicons for connotation frame analysis (Rashkin et al. 2016; Sap et al. 2017).

> [!NOTE]
> **Figure Path Migration:** Historically, the notebooks and scripts in this repository were configured to save plots directly into `data/snippets/`. To organize the repository, these visualizations have been migrated to the dedicated `plots/` folder (with key paper figures synced to `Figures/`). Note that some exploration code may still write to the old path, but all final figures are maintained in `plots/` and `Figures/`.

---

## Project Pipeline & Execution Order

The analytical workflow is executed in the following sequence:

### 1. Preprocessing
*   **Purpose**: Corpus collection, publication year prediction, and generic filters.
*   **Execution Order**:
    1.  `code/1_preprocessing/data_download.ipynb`: Harvest Gutenberg metadata and download full-text novels.
    2.  `code/1_preprocessing/data_preparation.ipynb`: Enrich raw Gutenberg catalog metadata with estimated publication years (using preface/text matching) and Wikidata author gender records.
    3.  `code/1_preprocessing/data_cleaning.ipynb`: Filter texts to British (PR) and American (PS) literature between 1850 and 1930.

### 2. Exploration
*   **Purpose**: Basic corpus analytics, distribution tracking, and initial context extraction.
*   **Execution Order**:
    1.  `code/2_exploration/descriptive_analysis.ipynb`: Analyze corpus distributions across genres, decades, and keywords.
    2.  `code/2_exploration/sample_exploration.ipynb`: Extract 200-word context windows (snippets) surrounding opium keywords for distant reading.

### 3. Analysis & Modeling
*   **Purpose**: NLP pipelines for gender attribution, lexicon skewness, topic extraction, and connotation frames.
*   **Execution Order**:
    1.  `code/3_analysis/gender_extraction.ipynb`: Execute the three-stage gender attribution pipeline (syntactic dependency, speaker tagging, and contextual fallback).
    2.  `code/3_analysis/word_analysis.ipynb`: Compute decay-weighted gender-skewed vocabularies (nouns, verbs, adjectives).
    3.  `code/3_analysis/topic_modelling.ipynb`: Train sentence-level BERTopic model utilizing MacBERTh embeddings on the extracted snippet contexts.
    4.  `code/3_analysis/connotation_frame.ipynb`: Resolve grammatical roles (Agent vs. Theme) and project agency, power, and sentiment scores onto characters.

---

## Key Technical Dependencies

To set up the project python environment, refer to `requirements.txt`. Core dependencies include:
- Python 3.9+
- `spacy` (with the `en_core_web_sm` model)
- `bertopic` (with `sentence-transformers` and `umap-learn`)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `pyarrow` (for Parquet data representations)
