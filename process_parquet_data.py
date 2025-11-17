"""
Script to process parquet files from data folder:
1. Combine all parquet files
2. Get total row count
3. Filter to 100k rows
4. Process text chunks (1000 tokens, 200 overlap)
5. Prepare data for vector DB insertion
"""

import os
import glob
import tiktoken
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import json

# Processing parameters
MAX_TOKENS = 1000
OVERLAP_TOKENS = 200
TARGET_RECORDS = 100000
DATA_FOLDER = "data"
OUTPUT_FOLDER = "processed_data"

# Initialize tiktoken encoder
tiktoken_encoder = tiktoken.get_encoding("cl100k_base")


def combine_parquet_files(data_folder: str = DATA_FOLDER) -> pd.DataFrame:
    """Combine all parquet files from the data folder."""
    print(f"Combining parquet files from {data_folder}...")
    
    # Get all parquet files
    parquet_files = sorted(glob.glob(os.path.join(data_folder, "*.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_folder}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Read and combine all parquet files
    dataframes = []
    for file_path in tqdm(parquet_files, desc="Reading parquet files"):
        try:
            df = pd.read_parquet(file_path)
            dataframes.append(df)
            print(f"  Loaded {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No dataframes were successfully loaded")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n✓ Combined {len(parquet_files)} files into {len(combined_df)} total rows")
    
    return combined_df


def get_dataframe_info(df: pd.DataFrame):
    """Print information about the dataframe."""
    print("\n" + "="*60)
    print("DataFrame Information")
    print("="*60)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Show sample row
    if len(df) > 0:
        print("\nSample row:")
        sample = df.iloc[0].to_dict()
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")


def filter_to_target_rows(df: pd.DataFrame, target_count: int = TARGET_RECORDS) -> pd.DataFrame:
    """Filter dataframe to target number of rows."""
    total_rows = len(df)
    print(f"\nTotal rows available: {total_rows:,}")
    print(f"Target rows: {target_count:,}")
    
    if total_rows <= target_count:
        print(f"✓ Using all {total_rows:,} rows (less than target)")
        return df
    
    # Filter to target count
    filtered_df = df.head(target_count).copy()
    print(f"✓ Filtered to {len(filtered_df):,} rows")
    
    return filtered_df


def split_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    """Split text into chunks using tiktoken with specified overlap."""
    if not text or not isinstance(text, str):
        return []
    
    # Encode text to tokens
    tokens = tiktoken_encoder.encode(text)
    
    if len(tokens) == 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tiktoken_encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
            
        start = end - overlap
    
    return chunks


def process_dataframe_for_chunking(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process dataframe and create chunks ready for vector DB insertion."""
    print(f"\nProcessing {len(df):,} rows for chunking...")
    
    processed_data = []
    
    # Identify text column (common names: text, Text, content, Content, body, Body)
    text_column = None
    for col in ['text', 'Text', 'content', 'Content', 'body', 'Body']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        # Try to find any column with string data
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].iloc[0] if len(df) > 0 else None
                if isinstance(sample, str) and len(str(sample)) > 100:
                    text_column = col
                    break
    
    if text_column is None:
        raise ValueError("Could not find text column in dataframe. Available columns: " + str(df.columns.tolist()))
    
    print(f"Using '{text_column}' column for text processing")
    
    # Identify other columns
    title_column = None
    for col in ['title', 'Title', 'name', 'Name']:
        if col in df.columns:
            title_column = col
            break
    
    url_column = None
    for col in ['url', 'URL', 'link', 'Link']:
        if col in df.columns:
            url_column = col
            break
    
    print(f"Title column: {title_column or 'Not found'}")
    print(f"URL column: {url_column or 'Not found'}")
    
    # Process each row
    doc_id = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Get text
        text_raw = str(row[text_column]) if pd.notna(row[text_column]) else ""
        
        if not text_raw or len(text_raw.strip()) < 100:
            continue
        
        # Get title and URL
        title = str(row[title_column]) if title_column and pd.notna(row[title_column]) else f"Document {idx}"
        url = str(row[url_column]) if url_column and pd.notna(row[url_column]) else ""
        
        # Split text into chunks
        chunks = split_text(text_raw, MAX_TOKENS, OVERLAP_TOKENS)
        
        if not chunks:
            continue
        
        # Create chunk objects
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_obj = {
                "id": doc_id,
                "url": url,
                "title": f"{title} (chunk {chunk_idx + 1})" if len(chunks) > 1 else title,
                "text": chunk_text,
                "text_raw": text_raw if chunk_idx == 0 else "",  # Store full text only for first chunk
                "original_index": int(idx),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks)
            }
            
            processed_data.append(chunk_obj)
            doc_id += 1
    
    print(f"\n✓ Processed {len(df):,} rows into {len(processed_data):,} chunks")
    print(f"  Average chunks per document: {len(processed_data) / len(df):.2f}")
    
    return processed_data


def save_processed_data(processed_data: List[Dict[str, Any]], output_folder: str = OUTPUT_FOLDER):
    """Save processed data to JSON files for later insertion."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Save as JSON (split into multiple files if too large)
    chunk_size = 10000
    num_files = (len(processed_data) + chunk_size - 1) // chunk_size
    
    print(f"\nSaving processed data to {output_folder}...")
    
    for file_idx in range(num_files):
        start_idx = file_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(processed_data))
        chunk_data = processed_data[start_idx:end_idx]
        
        output_file = os.path.join(output_folder, f"processed_chunks_{file_idx:04d}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved {len(chunk_data):,} chunks to {output_file}")
    
    # Also save a summary
    summary = {
        "total_chunks": len(processed_data),
        "total_documents": len(set(item["original_index"] for item in processed_data)),
        "num_files": num_files,
        "max_tokens": MAX_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "target_records": TARGET_RECORDS
    }
    
    summary_file = os.path.join(output_folder, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary to {summary_file}")
    print(f"  Total chunks: {summary['total_chunks']:,}")
    print(f"  Total documents: {summary['total_documents']:,}")
    print(f"  Number of files: {summary['num_files']}")


def main():
    """Main processing pipeline."""
    print("="*60)
    print("Parquet Data Processing Pipeline")
    print("="*60)
    
    # Step 1: Combine parquet files
    combined_df = combine_parquet_files(DATA_FOLDER)
    
    # Step 2: Get dataframe info
    get_dataframe_info(combined_df)
    
    # Step 3: Filter to target rows
    filtered_df = filter_to_target_rows(combined_df, TARGET_RECORDS)
    
    # Step 4: Process for chunking
    processed_data = process_dataframe_for_chunking(filtered_df)
    
    # Step 5: Save processed data
    save_processed_data(processed_data, OUTPUT_FOLDER)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nProcessed data is ready in '{OUTPUT_FOLDER}' folder")
    print("You can now use this data to insert into Weaviate vector database.")


if __name__ == "__main__":
    main()

