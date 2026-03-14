# TranslateBook

AI-powered modular translation pipeline for TXT documents, designed for incremental growth.

## Current Scope

TranslateBook currently supports:

- TXT extraction
- Text cleaning
- Paragraph-aware chunking
- Prompt construction
- Pluggable translators (`mock`, `openai`)
- Chunk assembly
- CLI orchestration
- Persistent translation cache (SQLite)
- Glossary-aware prompting
- Basic translation memory (SQLite exact match)
- Per-run JSON reporting

Backward compatibility is preserved: running the original MVP command still works.

## Project Structure

```text
TranslateBook/
├── main.py
├── config.py
├── models.py
├── txt_extractor.py
├── cleaner.py
├── chunker.py
├── prompt_builder.py
├── translator.py
├── assembler.py
├── pipeline.py
├── normalization.py
├── cache.py
├── glossary.py
├── translation_memory.py
├── reporting.py
├── tests/
│   ├── test_glossary.py
│   ├── test_cache.py
│   └── test_normalization.py
└── data/
    ├── input/
    ├── output/
    └── glossaries/
        └── example_fantasy_es.json
```

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

## CLI Usage

Basic:

```bash
python main.py --input data/input/book.txt --output data/output/book_es.txt
```

With OpenAI:

```bash
python main.py --input data/input/book.txt --output data/output/book_es.txt --translator openai --api-key sk-...
```

With glossary + reporting:

```bash
python main.py \
  --input data/input/book.txt \
  --output data/output/book_es.txt \
  --translator openai \
  --glossary data/glossaries/example_fantasy_es.json \
  --report data/output/book_es_report.json
```

Disable reuse layers:

```bash
python main.py --input data/input/book.txt --output data/output/book_es.txt --disable-cache --disable-tm
```

### Arguments

Required:

- `--input`, `-i`: input TXT path
- `--output`, `-o`: output TXT path

Core optional:

- `--chunk-size`: target words per chunk (100-5000, default `1000`)
- `--translator`: `mock` or `openai` (default `mock`)
- `--api-key`: OpenAI API key (or use `OPENAI_API_KEY`)
- `--model`: OpenAI model (default `OPENAI_MODEL` or `gpt-5.2`)
- `--source-language`: source language code (default `en`)
- `--target-language`: target language code (default `es`)
- `--verbose`, `-v`: verbose logs

Phase 2 optional:

- `--glossary`: glossary JSON path
- `--cache-db`: SQLite path for cache + TM (default `data/cache/translation_store.sqlite3`)
- `--disable-cache`: disable cache layer
- `--disable-tm`: disable translation memory layer
- `--report`: output JSON report path

## Reuse Layer Behavior

For each chunk, pipeline order is:

1. cache lookup
2. translation memory lookup
3. translator call on miss
4. store new translation in cache and TM

Notes:

- Cache key uses normalized source chunk + source language + target language + model/backend + glossary hash.
- If cache/TM initialization or operation fails, the error is logged and processing continues when possible.
- Output chunk order is always preserved.

## Glossary Format

Example file: `data/glossaries/example_fantasy_es.json`

```json
{
  "terms": [
    {
      "source": "Mana Core",
      "target": "Núcleo de maná",
      "notes": "Always use this term in fantasy context"
    },
    {
      "source": "sword aura",
      "target": "aura de espada"
    }
  ]
}
```

Glossary terms are injected deterministically into translation prompts.

## Run Report

When `--report` is provided, a JSON report is generated.

Included fields:

- `input_file`
- `output_file`
- `translator_backend`
- `model`
- `source_language`
- `target_language`
- `total_chunks`
- `translated_chunks`
- `cache_hits`
- `cache_misses`
- `translation_memory_hits`
- `estimated_token_usage` (if available)
- `started_at`
- `finished_at`
- `elapsed_seconds`
- `errors`

Sample:

```json
{
  "input_file": "data/input/book.txt",
  "output_file": "data/output/book_es.txt",
  "translator_backend": "openai",
  "model": "gpt-5.2",
  "source_language": "en",
  "target_language": "es",
  "total_chunks": 12,
  "translated_chunks": 12,
  "cache_hits": 5,
  "cache_misses": 7,
  "translation_memory_hits": 2,
  "estimated_token_usage": 9432,
  "started_at": "2026-03-13T16:11:02.219821+00:00",
  "finished_at": "2026-03-13T16:11:48.490283+00:00",
  "elapsed_seconds": 46.2705,
  "errors": []
}
```

## Testing

Run focused unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- This phase intentionally keeps translation memory to exact matching only.
- No EPUB/PDF support has been added yet.
- No web framework is used.
