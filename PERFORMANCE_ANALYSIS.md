# RAG4PeerFirm Performance Analysis: Baseline Bottlenecks (Archived)

> Status: Historical baseline analysis record.

## Baseline Bottlenecks (Historical)

- Text conversion dominated runtime for large items.
- Item extraction incurred redundant HTML parsing in the earlier baseline flow.
- Position detection performed repeated regex scans across large documents.
- Header/footer cleanup logic was expensive for long pages.
- Sequential item handling limited throughput in broader workloads.

## Historical Recommendation

Prior optimization work prioritized parser speedups, regex precompilation, reduced parsing duplication, and better concurrency strategy.

## Current Canonical Docs

- Consolidated benchmark outcomes: `PERFORMANCE_COMPARISON.md`
- Optimization backlog and implementation ideas: `OPTIMIZATION_GUIDE.md`

## Note

Detailed legacy timing breakdowns and long-form explanatory sections were removed to keep archived content concise and reduce overlap with active docs.
