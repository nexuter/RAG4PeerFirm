# RAG4PeerFirm Structure Integration Notes (Archived)

This document previously contained a draft patch proposal for integrating structure extraction into the extraction pipeline.

## Current Status

Structure extraction is already integrated into the main extraction flow in `script/main.py` through:
- `StructureExtractor` initialization in `RAG4PeerFirmExtractor.__init__`
- Automatic generation of `*_xtr.json` files during item extraction

## Where to Look Now

- Active usage and output behavior: `README.md`
- Detailed extraction behavior and examples: `STRUCTURE_EXTRACTION_ENHANCEMENT.md`
- Source implementation: `script/main.py` and `src/itemextraction/structure_extractor.py`

## Why This File Exists

Retained as an archive marker to avoid duplicate maintenance and to preserve historical context for prior implementation planning.
