# TranslateBook - AI-Powered Text Translation

A modular Python application for translating book-length TXT files from English to Spanish using AI. This MVP is designed as a scalable foundation for future enhancements.

## Features

- **Modular Architecture**: Clean separation of concerns with pluggable modules
- **Paragraph-Aware Chunking**: Intelligently splits text while preserving paragraph boundaries
- **OpenAI Translator**: Real chunk-by-chunk translation using OpenAI Responses API
- **Mock Translator**: Test the full pipeline without API calls
- **Configurable Chunk Size**: Adjust text segmentation for optimal performance
- **UTF-8 Support**: Proper encoding handling for international text
- **Extensible Design**: Easy to add new translator backends, formats, and features

## Project Structure

```
project/
├── main.py                 # CLI entry point
├── config.py              # Configuration management
├── models.py              # Data models (TextChunk, TranslatedChunk, etc.)
├── txt_extractor.py       # TXT file reading and extraction
├── cleaner.py             # Text cleaning and normalization
├── chunker.py             # Paragraph-aware text chunking
├── prompt_builder.py      # Translation prompt construction
├── translator.py          # Translator implementations (mock, OpenAI, extensible)
├── assembler.py           # Translated chunk reassembly
├── pipeline.py            # Main orchestration pipeline
├── requirements.txt       # Python dependencies (minimal for MVP)
├── README.md              # This file
├── data/
│   ├── input/            # Place input TXT files here
│   └── output/           # Translated files are saved here
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

1. Clone or download the project
2. Navigate to the project directory
3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Command

Translate a TXT file with default settings (mock translator):

```bash
python main.py --input data/input/book.txt --output data/output/book_es.txt
```

### Command-Line Arguments

```bash
python main.py --input INPUT_FILE --output OUTPUT_FILE [OPTIONS]
```

**Required Arguments:**
- `--input`, `-i PATH`: Path to input TXT file
- `--output`, `-o PATH`: Path for output TXT file

**Optional Arguments:**
- `--chunk-size SIZE`: Target word count per chunk (default: 1000, range: 100-5000)
- `--translator TYPE`: Translator backend (default: mock, options: mock, openai)
- `--api-key KEY`: API key for translation service (if required)
- `--model MODEL`: OpenAI model when using `--translator openai` (default: `OPENAI_MODEL` or `gpt-5.2`)
- `--source-language CODE`: Language code of input (default: en)
- `--target-language CODE`: Language code of output (default: es)
- `--verbose`, `-v`: Enable verbose logging for debugging

### Examples

**Using mock translator (for testing):**
```bash
python main.py --input samples/chapter1.txt --output output/chapter1_es.txt
```

**With custom chunk size:**
```bash
python main.py --input book.txt --output book_es.txt --chunk-size 1500
```

**Using OpenAI translator:**
```bash
python main.py --input book.txt --output book_es.txt --translator openai --api-key sk-...
```

**Using OpenAI translator with environment variables:**
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.2"
python main.py --input book.txt --output book_es.txt --translator openai
```

**Verbose logging:**
```bash
python main.py --input book.txt --output book_es.txt --verbose
```

## How It Works

### Pipeline Flow

1. **Extraction**: Reads the input TXT file using UTF-8 encoding
2. **Cleaning**: Normalizes text (line endings, whitespace) while preserving paragraphs
3. **Chunking**: Intelligently splits text into segments:
   - Respects paragraph boundaries (never splits a paragraph)
   - Targets ~1000 words per chunk (configurable)
   - Allows single large paragraphs to exceed target size
4. **Prompting**: Builds translation instructions for each chunk
5. **Translation**: Sends chunks to the translator backend
6. **Assembly**: Reconstructs the translated text in original order
7. **Output**: Saves final translated text to file

### Chunking Strategy

The chunker uses a smart algorithm:

- Splits text by blank lines to identify paragraphs
- Accumulates paragraphs until reaching target word count
- Respects paragraph boundaries (no splitting mid-paragraph)
- Allows oversized paragraphs to become their own chunk
- Maintains explicit chunk ordering

**Example:**
```
Input: 3 paragraphs
  - Para 1: 400 words
  - Para 2: 300 words  
  - Para 3: 800 words
  - Target: 1000 words

Output: 2 chunks
  - Chunk 0: Para 1 + Para 2 (700 words)
  - Chunk 1: Para 3 (800 words)
```

## Mock Translator

The **MockTranslator** is used for testing and development. It:

- Returns a fake translation without making API calls
- Prefixes output with `[ES MOCK TRANSLATION]`
- Performs simple keyword substitution for demonstration
- Allows testing the entire pipeline end-to-end

**Example output:**
```
Input:  "The quick brown fox jumps over the lazy dog"
Output: "[ES MOCK TRANSLATION]
el/la quick brown fox jumps over el/la lazy dog"
```

This allows you to:
- Test the full pipeline without API keys
- Verify chunking and assembly logic
- Develop/debug without waiting for real API calls
- Understand how data flows through the system

## OpenAI Translator

The **OpenAITranslator** performs real translations using the OpenAI Python SDK and the
Responses API.

- Reads API key from `--api-key` or `OPENAI_API_KEY`
- Uses `--model`, or `OPENAI_MODEL`, or defaults to `gpt-5.2`
- Preserves chunk order and paragraph structure in final assembly

**Example:**
```bash
python main.py --input data/input/book.txt --output data/output/book_es.txt --translator openai --model gpt-5.2
```

## Architecture & Design Decisions

### Modularity

Each module has a single responsibility:

- **txt_extractor**: File I/O only
- **cleaner**: Text normalization
- **chunker**: Segmentation logic
- **prompt_builder**: Prompt/payload construction
- **translator**: Translation abstraction layer
- **assembler**: Chunk reassembly
- **pipeline**: Orchestration and workflow

### Extensibility Points

The architecture is designed for these future additions without major refactoring:

1. **New Translators**: Implement `BaseTranslator` interface and register with `TranslatorFactory`
2. **File Formats**: Add new extractors (EPUBExtractor, PDFExtractor, etc.)
3. **Caching**: Add a cache layer to translator
4. **Translation Memory**: Store and reuse previous translations
5. **Formatting Preservation**: Extend cleaner to handle rich formatting
6. **Glossaries**: Add glossary-aware prompt builder
7. **Quality Assurance**: Add QA module to verify translations
8. **Export Formats**: Add exporters for DOCX, EPUB, etc.
9. **Parallel Processing**: Distribute chunk translation across workers
10. **Database**: Store document history, translations, metadata

### Type Safety

All code uses type hints for:
- Better IDE support and autocompletion
- Static type checking with mypy
- Self-documenting function signatures
- Reduced runtime errors

### Logging

Comprehensive logging at each pipeline stage:
- INFO: High-level progress
- DEBUG: Detailed process information
- ERROR: Failures with context
- Use `--verbose` flag to see DEBUG logs

## Configuration

Configuration is centralized in `config.py` and can be:

1. **Command-line arguments** (takes precedence)
2. **Configuration files** (can be added later)
3. **Environment variables** (can be added later)

## Performance Considerations

### Chunk Size

The default chunk size is **1000 words**:
- **Smaller chunks** (500-800): More granular, better for complex text, slower overall
- **Larger chunks** (1500-2000): Faster processing, better for APIs with per-request costs

### Processing

Current implementation is **sequential** for MVP simplicity. Future versions can add:
- Parallel chunk processing using `concurrent.futures` or `multiprocessing`
- Async/await for I/O-bound operations
- Batch API requests for efficiency

## Error Handling

The pipeline handles:
- Missing input files
- Invalid file encodings
- Invalid configuration parameters
- Empty text or no chunks
- Translation failures (with detailed logging)

All errors are logged with full context and stack traces (in verbose mode).

## Testing

For initial testing, use the mock translator:

```bash
# Create a test file
echo "This is a sample paragraph.

This is another paragraph with more content." > data/input/test.txt

# Run translation
python main.py --input data/input/test.txt --output data/output/test_es.txt --verbose

# Check results
cat data/output/test_es.txt
```

## Future Enhancements (Not in MVP)

### Immediate Next Steps

1. **Add Caching Layer**
   - Cache translations for identical chunks
   - Use SQLite for persistent cache
   - Reduce API costs and improve speed

2. **Implement Glossary Support**
   - Load custom term dictionaries
   - Inject glossary terms into prompts
   - Ensure consistent terminology

3. **Add Translation Memory**
   - Store all translations in a database
   - Reuse previous translations
   - Leverage fuzzy matching

4. **Enhance File Format Support**
   - Add EPUB reader
   - Add PDF extraction
   - Add DOCX support with formatting preservation

### Advanced Features (Later Phases)

- Quality assurance and automated checking
- Cost tracking and budget management
- Style profiles for different authors/domains
- Web interface for non-technical users
- Collaborative translation features
- Machine translation memory export
- Integration with CAT tools

## Troubleshooting

### "File not found" error
- Check that `--input` path is correct and file exists
- Use absolute paths if relative paths don't work

### Empty output file
- Check that input file contains text
- Verify encoding is UTF-8
- Use `--verbose` to see logs

### TypeError or AttributeError
- Ensure Python 3.8+ is installed
- Check that all files are in the project directory
- Verify no syntax errors with `python -m py_compile *.py`

### Mock translator prefix in output
- This is expected - the mock translator is for testing only
- Use `--translator openai` (with an API key) for real translations

## Contributing

To extend this project:

1. Add a new translator: Create class inheriting `BaseTranslator`, register with `TranslatorFactory`
2. Add a new extractor: Create class for new file format
3. Add a feature: Add new module with clear single responsibility
4. Maintain modularity: Avoid cross-module dependencies beyond what's necessary

## License

[Add your license here]

## Support

For issues, questions, or suggestions, please contact the development team.

---

**Version**: 1.0.0 (MVP)  
**Last Updated**: March 2026
