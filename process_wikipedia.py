"""
Script to download, process, and store Wikipedia datasets in Weaviate
with EmbeddingGemma-300m embeddings.
"""

import os
import tiktoken
import numpy as np
from datasets import load_dataset
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import hashlib
from typing import List, Dict, Any
import json

# Language configuration
# Note: Chinese Wikipedia dataset may use different codes:
# - 'zh' for simplified Chinese
# - Check available subsets at: https://huggingface.co/datasets/wikimedia/wikipedia
#   Common variants: zh, zh-hans, zh-hant
LANGUAGE_CONFIG = {
    'spanish': {'code': 'es', 'collection': 'spanish'},
    'english': {'code': 'en', 'collection': 'english'},
    'chinese': {'code': 'zh', 'collection': 'chinese'},  # May need adjustment - check dataset
    'hindi': {'code': 'hi', 'collection': 'hindi'},
    'french': {'code': 'fr', 'collection': 'french'},
}

# Weaviate configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')

# Embedding model
EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m'

# Processing parameters
MAX_TOKENS = 1000
OVERLAP_TOKENS = 200
TARGET_RECORDS = 100000


class WikipediaProcessor:
    def __init__(self, weaviate_url: str = WEAVIATE_URL):
        """Initialize the processor with Weaviate client and embedding model."""
        # Weaviate v4 connection method
        if weaviate_url.startswith('http://localhost') or weaviate_url.startswith('http://127.0.0.1'):
            self.weaviate_client = weaviate.connect_to_local()
        else:
            # For custom URLs, parse and connect
            from urllib.parse import urlparse
            parsed = urlparse(weaviate_url)
            self.weaviate_client = weaviate.connect_to_custom(
                http_host=parsed.hostname,
                http_port=parsed.port or 8080,
                http_secure=parsed.scheme == 'https',
                grpc_host=parsed.hostname,
                grpc_port=50051,
                grpc_secure=parsed.scheme == 'https'
            )
        self.tokenizer = None
        self.embedding_model = None
        self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        
    def load_embedding_model(self):
        """Load the EmbeddingGemma model from HuggingFace."""
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        self.embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        self.embedding_model.eval()
        print("Embedding model loaded successfully")
        
    def create_weaviate_collection(self, collection_name: str):
        """Create a Weaviate collection with the specified schema."""
        print(f"Creating Weaviate collection: {collection_name}")
        
        # Check if collection exists, delete if it does
        if self.weaviate_client.collections.exists(collection_name):
            print(f"Collection {collection_name} already exists. Deleting...")
            self.weaviate_client.collections.delete(collection_name)
        
        # Create collection with schema
        # Note: Weaviate v4+ API
        try:
            self.weaviate_client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="text_raw", data_type=DataType.TEXT),
                ],
                vectorizer_config=None,  # We'll provide our own vectors
            )
            print(f"Collection {collection_name} created successfully")
        except Exception as e:
            print(f"Error creating collection (may already exist): {e}")
        
    def filter_famous_topics(self, dataset, language_code: str, target_count: int = TARGET_RECORDS):
        """
        Filter dataset for famous/popular topics.
        Strategy: Filter by text length (longer articles are often more comprehensive),
        exclude stubs, lists, and redirects.
        """
        print(f"Filtering dataset for {language_code}...")
        
        # Convert to list and filter
        records = []
        seen_titles = set()
        
        for item in dataset:
            text = item.get('text', '')
            title = item.get('title', '')
            
            # Skip if already seen
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Filter criteria:
            # 1. Text should be substantial (at least 500 characters)
            # 2. Title should not be a redirect, stub, or list
            # 3. Exclude common non-article patterns
            exclude_patterns = [
                'List of', 'Category:', 'Template:', 'File:', 'User:',
                'Wikipedia:', 'Help:', 'Portal:', 'Talk:', 'MediaWiki:'
            ]
            
            if (len(text) > 500 and 
                title and 
                not any(title.startswith(pattern) for pattern in exclude_patterns) and
                not title.startswith('List of')):
                records.append(item)
                
            if len(records) >= target_count * 1.5:  # Get more than needed for better filtering
                break
        
        # Sort by text length (longer articles are often more comprehensive/famous)
        records.sort(key=lambda x: len(x.get('text', '')), reverse=True)
        
        # Take top N records
        filtered_records = records[:target_count]
        print(f"Filtered to {len(filtered_records)} records for {language_code}")
        return filtered_records
    
    def split_text(self, text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
        """Split text into chunks using tiktoken with specified overlap."""
        # Encode text to tokens
        tokens = self.tiktoken_encoder.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tiktoken_encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
                
            start = end - overlap
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using EmbeddingGemma model."""
        with torch.no_grad():
            # Truncate text if too long (EmbeddingGemma has max length limits)
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            )
            outputs = self.embedding_model(**inputs)
            # Use mean pooling for sentence embedding
            # EmbeddingGemma outputs last_hidden_state, we pool it
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            # Convert to list
            if embedding.dim() == 0:
                embedding = embedding.unsqueeze(0)
            embedding = embedding.numpy().tolist()
        return embedding
    
    def process_language(self, language_name: str, language_code: str, collection_name: str):
        """Process a single language dataset."""
        print(f"\n{'='*60}")
        print(f"Processing {language_name} (code: {language_code})")
        print(f"{'='*60}")
        
        # Load dataset
        dataset_name = "wikimedia/wikipedia"
        subset_name = f"20231101.{language_code}"
        
        print(f"Loading dataset: {dataset_name}, subset: {subset_name}")
        dataset = None
        
        # Try streaming first (more memory efficient)
        try:
            dataset = load_dataset(dataset_name, subset_name, split="train", streaming=True)
            print("Loaded dataset in streaming mode")
        except Exception as e:
            print(f"Error loading dataset in streaming mode: {e}")
            # Try non-streaming
            try:
                print("Trying non-streaming mode...")
                dataset = load_dataset(dataset_name, subset_name, split="train")
                print("Loaded dataset in non-streaming mode")
            except Exception as e2:
                print(f"Failed to load dataset: {e2}")
                print(f"Please check if the subset '{subset_name}' exists in the dataset")
                return
        
        if dataset is None:
            print(f"Could not load dataset for {language_code}")
            return
        
        # Filter for famous topics
        filtered_records = self.filter_famous_topics(dataset, language_code, TARGET_RECORDS)
        
        # Create Weaviate collection
        self.create_weaviate_collection(collection_name)
        collection = self.weaviate_client.collections.get(collection_name)
        
        # Process and store records
        print(f"Processing and storing {len(filtered_records)} records...")
        batch_objects = []
        batch_size = 100
        doc_id = 0
        
        for record in tqdm(filtered_records, desc=f"Processing {language_name}"):
            # Extract fields with defaults
            title = record.get('title', '') or record.get('Title', '') or ''
            url = record.get('url', '') or record.get('URL', '') or ''
            text_raw = record.get('text', '') or record.get('Text', '') or ''
            
            if not text_raw or not title:
                continue
            
            # Split text into chunks
            chunks = self.split_text(text_raw, MAX_TOKENS, OVERLAP_TOKENS)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # Generate embedding
                embedding = self.generate_embedding(chunk_text)
                
                # Create object
                # Note: Weaviate auto-generates 'id', so we don't include it
                obj = {
                    "url": url,
                    "title": f"{title} (chunk {chunk_idx + 1})" if len(chunks) > 1 else title,
                    "text": chunk_text,
                    "text_raw": text_raw if chunk_idx == 0 else "",  # Store full text only for first chunk
                }
                
                # Store for batch insert
                batch_objects.append({
                    "properties": obj,
                    "vector": embedding
                })
                
                doc_id += 1
                
                # Batch insert
                if len(batch_objects) >= batch_size:
                    try:
                        # Weaviate v4 format: insert_many expects list of objects with properties and vector
                        collection.data.insert_many(batch_objects)
                        batch_objects = []
                    except Exception as e:
                        print(f"Error inserting batch: {e}")
                        # Try inserting one by one if batch fails
                        for obj_data in batch_objects:
                            try:
                                collection.data.insert(
                                    properties=obj_data["properties"], 
                                    vector=obj_data["vector"]
                                )
                            except Exception as e2:
                                print(f"Error inserting single object: {e2}")
                        batch_objects = []
        
        # Insert remaining objects
        if batch_objects:
            try:
                collection.data.insert_many(batch_objects)
            except Exception as e:
                print(f"Error inserting final batch: {e}")
                # Try inserting one by one
                for obj_data in batch_objects:
                    try:
                        collection.data.insert(
                            properties=obj_data["properties"], 
                            vector=obj_data["vector"]
                        )
                    except Exception as e2:
                        print(f"Error inserting single object: {e2}")
        
        print(f"Successfully stored {doc_id} chunks for {language_name}")
    
    def process_all_languages(self):
        """Process all configured languages."""
        # Load embedding model once
        self.load_embedding_model()
        
        # Process each language
        for language_name, config in LANGUAGE_CONFIG.items():
            try:
                self.process_language(
                    language_name,
                    config['code'],
                    config['collection']
                )
            except Exception as e:
                print(f"Error processing {language_name}: {e}")
                import traceback
                traceback.print_exc()
                continue


def main():
    """Main entry point."""
    processor = WikipediaProcessor()
    processor.process_all_languages()
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

