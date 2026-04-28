# Opium in Project Gutenberg: A Literary Analysis

This project investigates the depiction and evolution of opium-related themes in 19th and early 20th-century literature (1860–1920) using the Project Gutenberg corpus.

## Project Pipeline & Execution Order

The workflow is organized into distinct phases, moving from raw data collection to experimental linguistic analysis.

### 1. Preprocessing
*   **Purpose**: Acquisition, metadata extraction, and dataset scoping.
*   **Key Files**:
    - `code/1_preprocessing/data_download.ipynb`: Harvest Gutenberg index and full texts.
    - `code/1_preprocessing/data_preparation.ipynb`: Enrich with publication years and demographic data.
    - `code/1_preprocessing/data_cleaning.ipynb`: Filter by LoCC PR/PS and remove non-literary texts.
*   **Outputs**: `data/GP_opium_filtered.csv`, `data/GP_opium_filtered.parquet`, `data/metadata_with_years.csv`

### 2. Exploration (Pre-Analysis)
*   **Purpose**: Broad statistical observation and qualitative sampling for validation.
*   **Key Files**:
    - `code/2_exploration/descriptive_analysis.ipynb`: Visualize distributions across authors, genres, and decades.
    - `code/2_exploration/sample_exploration.ipynb`: Extract keyword context windows for close reading.
*   **Outputs**: `data/pr_ps_snippets.csv`, `data/extended_sample_snippets.csv`

### 3. Core Analysis
*   **Purpose**: The heavy lifting—quantifying gender, agency, and power dynamics in the text.
*   **Key Files**:
    - `code/3_analysis/gender_pilot.ipynb`: NLP-driven (SpaCy) gender attribution pipeline.
*   **Pending Features**:
    - **Topic Modeling**: Identifying semantic shifts in opium discourses.
    - **Connotation Framework**: Measuring agency/power/sentiment scores.
    - **Authorial Lens**: Analyzing the convergence/divergence of opium depictions across author demographics (gender, country).

### Utilities
*   **Location**: `code/utils/`
*   Contains core `.py` scripts (`extract_snippets.py`, `date_extraction.py`, etc.) used across the notebook pipeline.

---

## Technical Dependencies
- Python 3.9+
- pandas, numpy, matplotlib, spacy (en_core_web_sm)
- pyarrow (for parquet support)
