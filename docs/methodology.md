# Methodology

## Overview

This study builds a local, reproducible pipeline to identify peer firms from SEC disclosures at the item level. The workflow is designed to preserve firm-specific disclosure content, reduce generic filing boilerplate, and combine efficient vector retrieval with higher-precision large language model reranking. The full process consists of five stages: item extraction, item summarization, vector database construction, peer retrieval and reranking, and evaluation.

## 1. Item Extraction

We begin from pre-extracted SEC filing artifacts stored as structured JSON files. For each firm-year filing, the pipeline reads itemized disclosure content from `*_item.json` files and heading structures from `*_str.json` files. Item identifiers are normalized so that comparable sections such as `Item 1A`, `Item 7`, and `Item 7A` can be aligned across firms and years. This produces the base text corpus for downstream representation learning and peer comparison.

## 2. Retrieval-Oriented Item Summarization

Because many disclosure items are long and contain repetitive compliance language, we generate an additional summary representation for each `(firm, year, item)`. A local language model is used to write approximately 1,000-word summaries and save them as sibling `_summ` artifacts next to the original item files. The prompt is not a generic summarization instruction. Instead, it explicitly emphasizes information that is useful for peer matching, including business model, end markets, geography, strategic changes, operating constraints, supply-chain exposures, regulation, capital allocation, and salient differences from the previous year when prior-year text is available.

For long inputs, the summarization stage uses a map-reduce strategy. Item text is divided into model-appropriate windows based on the target model capacity, each window is summarized independently, and a second-stage reduction prompt merges the partial outputs into one retrieval-oriented item summary. This design preserves coverage for long sections while keeping the summarization process within practical local-model context limits.

## 3. Vector Database Construction

We then build a local vector database using FAISS. The database is organized at the item-year level and stores exactly one pooled vector per `(firm, year, item)` for each scope. Three scopes are supported: `all` for full item text, `heading` for heading-only text, and `summary` for retrieval-oriented item summaries.

For the `all` and `heading` scopes, long item text is split into overlapping chunks. Chunk size is not fixed globally. Instead, it is assigned by item type using empirical disclosure-length statistics, and overlap size is automatically scaled with chunk size. This reduces noise from overly short windows in long narrative sections and avoids unnecessary mixing in short items. Each chunk is embedded locally with the `bge-m3` model, and chunk embeddings are combined into one item vector using distinctiveness-weighted pooling rather than plain averaging. This weighting increases the influence of chunks that deviate from the item centroid and therefore are more likely to encode firm-specific rather than boilerplate content.

For the `summary` scope, each summary is embedded directly into a single vector without chunk pooling. The resulting vectors are normalized and indexed with FAISS inner-product search, which is equivalent to cosine similarity under L2 normalization.

## 4. Candidate Retrieval and LLM Reranking

Peer identification uses a two-stage ranking design. First, for a focal firm and item, the pipeline retrieves the top candidate set by cosine similarity from the FAISS index. The initial candidate set is defined as the top 10% of firms within the same item-year universe, which provides high-recall screening at low cost.

Second, the candidate firms are reranked with a local language model using the retrieved item text itself rather than raw embedding vectors. The reranking prompt instructs the model to judge similarity strictly from the provided disclosure text and to reward overlap in business model, products, customer base, geography, strategic priorities, risk profile, operating constraints, and financial posture, while penalizing matches driven only by generic SEC language. The reranker is run at temperature zero to improve consistency across repeated evaluations.

When multiple items are supplied, reranking is performed separately for each item. The pipeline then aggregates item-level scores across matched items to generate a final peer ranking for each focal firm-year. This architecture uses vectors for efficient retrieval and language-model reasoning only where textual comparison adds value.

## 5. Evaluation

The final stage evaluates peer-firm rankings with weak labels. Candidate validation uses observable similarity signals such as shared SIC, NAICS, or GICS classifications, comparable market-capitalization bands, comparable revenue bands, and analyst peer sets when available. Performance is summarized using `Recall@50` to measure retrieval coverage, `NDCG@10` to measure ranking quality near the top of the list, and year-over-year stability based on overlap in top-ranked peers across adjacent years.

All ranking outputs are saved with timestamps to support experiment tracking and reproducibility. This makes it possible to compare alternative scopes, chunking choices, summary prompts, embedding models, or reranking models under a common evaluation protocol.

## Design Rationale

The methodology intentionally separates broad retrieval from fine-grained judgment. Cosine similarity over item vectors is computationally efficient and scalable for several thousand firms per year, but pooled vectors may blur decisive local passages. In contrast, direct large-language-model comparison across the entire firm universe would be prohibitively expensive and less reproducible. The adopted hybrid strategy therefore uses vectors to screen the search space and a deterministic reranking prompt to refine the top candidates. This balances scalability, interpretability, and ranking precision for disclosure-based peer identification.
