# Cross-Lingual RAG with Wikipedia Datasets

This project processes Wikipedia datasets for multiple languages, generates embeddings using EmbeddingGemma-300m, and stores them in Weaviate for cross-lingual retrieval.

## Features

- **Process local parquet files** or download Wikipedia datasets for 5 languages: English, Spanish, French, Hindi, and Chinese
- Combines multiple parquet files and gets total row count
- Filters to 100k records (prioritizing famous/comprehensive topics)
- Splits documents into chunks of 1000 tokens with 200 token overlap using tiktoken
- Generates embeddings using Google's EmbeddingGemma-300m model
- Stores data in Weaviate collections with schema: id, url, title, text, text_raw

## Prerequisites

1. **Docker** - For running Weaviate
2. **Python 3.8+**
3. **Weaviate instance** - See setup instructions below

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Weaviate

Start Weaviate using Docker:

```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

Or use docker-compose (create `docker-compose.yml`):

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate
volumes:
  weaviate_data:
```

Then run:
```bash
docker-compose up -d
```

### 3. Create Collections (Optional)

You can create the collections separately using:

```bash
python create_collections.py
```

Or the collections will be created automatically when running the main processing script.

## Usage

### Option 1: Process Parquet Files from Data Folder

If you have parquet files in the `data/` folder:

**Step 1: Process parquet files and create chunks**
```bash
python process_parquet_data.py
```

This will:
1. Combine all parquet files from the `data/` folder
2. Display total row count
3. Filter to 100k rows
4. Split documents into chunks (1000 tokens, 200 overlap)
5. Save processed chunks to `processed_data/` folder as JSON files

**Step 2: Insert processed data into Weaviate**
```bash
python insert_to_weaviate.py --collection wikipedia
```

This will:
1. Load the EmbeddingGemma-300m model
2. Create Weaviate collection
3. Load processed chunks from `processed_data/` folder
4. Generate embeddings for each chunk
5. Insert into Weaviate with vectors

**Custom options:**
```bash
python insert_to_weaviate.py \
  --collection my_collection \
  --data-folder processed_data \
  --batch-size 50
```

### Option 2: Process Wikipedia Datasets from HuggingFace

```bash
python process_wikipedia.py
```

This will:
1. Load the EmbeddingGemma-300m model
2. For each language:
   - Download the Wikipedia dataset from HuggingFace
   - Filter 100k famous topics
   - Split documents into chunks (1000 tokens, 200 overlap)
   - Generate embeddings
   - Store in Weaviate collection

### Process Single Language

You can modify the script to process only specific languages by editing the `LANGUAGE_CONFIG` dictionary in `process_wikipedia.py`.

### Environment Variables

- `WEAVIATE_URL`: Weaviate instance URL (default: `http://localhost:8080`)

```bash
export WEAVIATE_URL=http://localhost:8080
python process_wikipedia.py
```

## Configuration

### Language Codes

The script uses the following language codes:
- Spanish: `es`
- English: `en`
- French: `fr`
- Hindi: `hi`
- Chinese: `zh` (may need adjustment based on dataset availability)

### Processing Parameters

Edit these constants in `process_wikipedia.py`:
- `MAX_TOKENS`: Maximum tokens per chunk (default: 1000)
- `OVERLAP_TOKENS`: Overlap between chunks (default: 200)
- `TARGET_RECORDS`: Number of records per language (default: 100000)

## Collection Schema

Each Weaviate collection has the following schema:
- `id` (INT): Unique identifier
- `url` (TEXT): Wikipedia article URL
- `title` (TEXT): Article title
- `text` (TEXT): Chunk text (1000 tokens)
- `text_raw` (TEXT): Full original text (stored only for first chunk)

## Notes

1. **Model Download**: The EmbeddingGemma-300m model will be downloaded from HuggingFace on first run (~1.2GB).

2. **Processing Time**: Processing 100k records per language can take several hours depending on your hardware.

3. **Memory Requirements**: Ensure you have sufficient RAM (recommended: 16GB+) for processing large datasets.

4. **Chinese Language**: The Chinese dataset code may need adjustment. Check available subsets at: https://huggingface.co/datasets/wikimedia/wikipedia

5. **Famous Topics Filtering**: The script filters articles by:
   - Minimum text length (500 characters)
   - Excluding stubs, lists, categories, and other non-article pages
   - Prioritizing longer, more comprehensive articles

## Troubleshooting

### Weaviate Connection Error
- Ensure Weaviate is running: `docker ps`
- Check the URL: `curl http://localhost:8080/v1/meta`

### Dataset Loading Error
- Check internet connection
- Verify the dataset subset exists on HuggingFace
- Some language codes may differ (e.g., Chinese might use different codes)

### Memory Issues
- Process languages one at a time
- Reduce `TARGET_RECORDS` or `batch_size`
- Use streaming mode (already enabled)

## Data Management

### Why Data Files Are Not Included

The `data/` and `processed_data/` folders are excluded from this repository (see `.gitignore`) for the following reasons:

1. **Repository Size**: Large data files make cloning slow and can exceed GitHub's recommended limits
2. **Regenerability**: Data can be recreated using the provided scripts
3. **Privacy**: Your specific data may contain sensitive information
4. **Version Control**: Large binary files don't version well in Git

### Getting Data

**Option 1: Use Your Own Data**
- Place parquet files in the `data/` folder
- Run `python process_parquet_data.py` to process them

**Option 2: Download from HuggingFace**
- Run `python process_wikipedia.py` to download and process Wikipedia datasets

**Option 3: Share Data Separately (if needed)**
If you need to share processed data with others:
- Use cloud storage (Google Drive, Dropbox, AWS S3)
- Use Git LFS for large files (if you must include them)
- Provide download links in documentation
- Use data hosting services like:
  - [HuggingFace Datasets](https://huggingface.co/datasets)
  - [Kaggle Datasets](https://www.kaggle.com/datasets)
  - [Zenodo](https://zenodo.org/)

### Sample Data

If you want to include sample data for testing:
- Keep it small (< 10MB)
- Include only a few example files
- Document it clearly in the README

## References

- [Weaviate Documentation](https://docs.weaviate.io/)
- [EmbeddingGemma Model](https://huggingface.co/google/embeddinggemma-300m)
- [Wikipedia Datasets](https://huggingface.co/datasets/wikimedia/wikipedia)

