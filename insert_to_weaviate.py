"""
Script to insert processed chunks into Weaviate with embeddings.
This script reads the processed JSON files and inserts them into Weaviate.
"""

import os
import json
import glob
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.data import DataObject
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from typing import List, Dict, Any

# Configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m'
PROCESSED_DATA_FOLDER = "processed_data"
COLLECTION_NAME = "wikipedia"  # Default collection name, can be changed
BATCH_SIZE = 500  # Increased batch size for Weaviate insertion
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation (model-dependent)


class WeaviateInserter:
    def __init__(self, weaviate_url: str = WEAVIATE_URL, collection_name: str = COLLECTION_NAME):
        """Initialize the inserter with Weaviate client and embedding model."""
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
        self.collection_name = collection_name
        self.tokenizer = None
        self.embedding_model = None
        
    def load_embedding_model(self):
        """Load the EmbeddingGemma model from HuggingFace."""
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        print("Note: This model requires HuggingFace authentication.")
        print("If you haven't already, please:")
        print("1. Visit https://huggingface.co/google/embeddinggemma-300m and accept the terms")
        print("2. Run: huggingface-cli login")
        print("   Or set HUGGING_FACE_HUB_TOKEN environment variable")
        print()
        
        # Try to load with token if available
        token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        try:
            if token:
                self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, token=token)
                self.embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, token=token)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
                self.embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e) or "unauthorized" in str(e).lower():
                print("\n" + "="*60)
                print("AUTHENTICATION REQUIRED")
                print("="*60)
                print("The EmbeddingGemma model requires HuggingFace authentication.")
                print("\nTo fix this:")
                print("1. Visit: https://huggingface.co/google/embeddinggemma-300m")
                print("2. Click 'Agree and access repository' to accept the terms")
                print("3. Get your HuggingFace token from: https://huggingface.co/settings/tokens")
                print("4. Run one of these commands:")
                print("   huggingface-cli login")
                print("   OR")
                print("   export HUGGING_FACE_HUB_TOKEN=your_token_here")
                print("="*60)
            raise
        self.embedding_model.eval()
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.cuda()
            print("Using CUDA for embeddings")
        else:
            print("Using CPU for embeddings")
        print("✓ Embedding model loaded successfully")
        
    def create_collection(self, delete_existing: bool = True):
        """Create a Weaviate collection with the specified schema."""
        print(f"Creating Weaviate collection: {self.collection_name}")
        
        # Check if collection exists, delete if it does
        if self.weaviate_client.collections.exists(self.collection_name):
            if delete_existing:
                print(f"Collection {self.collection_name} already exists. Deleting...")
                self.weaviate_client.collections.delete(self.collection_name)
            else:
                print(f"Collection {self.collection_name} already exists. Skipping creation.")
                return
        
        # Create collection with schema
        # Note: 'id' is reserved in Weaviate, so we don't include it as a property
        try:
            self.weaviate_client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="text_raw", data_type=DataType.TEXT),
                ],
                vectorizer_config=None,  # We'll provide our own vectors
            )
            print(f"✓ Collection '{self.collection_name}' created successfully")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
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
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.embedding_model(**inputs)
            # Use mean pooling for sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # Convert to list
            if embedding.dim() == 0:
                embedding = embedding.unsqueeze(0)
            
            # Move back to CPU if needed
            if embedding.is_cuda:
                embedding = embedding.cpu()
            
            embedding = embedding.numpy().tolist()
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts (much faster)."""
        if not texts:
            return []
        
        with torch.no_grad():
            # Tokenize all texts at once
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_attention_mask=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.embedding_model(**inputs)
            # Use mean pooling for sentence embeddings
            # Shape: [batch_size, hidden_dim]
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Apply attention mask to handle padding
            if 'attention_mask' in inputs:
                attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                if torch.cuda.is_available():
                    attention_mask = attention_mask.cuda()
                # Sum of attention mask for each sequence
                mask_sum = attention_mask.sum(dim=1, keepdim=True)
                # Weighted mean
                embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / mask_sum
            
            # Move back to CPU if needed
            if embeddings.is_cuda:
                embeddings = embeddings.cpu()
            
            # Convert to list of lists
            embeddings_list = embeddings.numpy().tolist()
        
        return embeddings_list
    
    def load_processed_files(self, data_folder: str = PROCESSED_DATA_FOLDER) -> List[Dict[str, Any]]:
        """Load all processed JSON files."""
        print(f"Loading processed data from {data_folder}...")
        
        json_files = sorted(glob.glob(os.path.join(data_folder, "processed_chunks_*.json")))
        
        if not json_files:
            raise ValueError(f"No processed chunk files found in {data_folder}")
        
        print(f"Found {len(json_files)} processed files")
        
        all_data = []
        for json_file in tqdm(json_files, desc="Loading JSON files"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        print(f"✓ Loaded {len(all_data):,} chunks from {len(json_files)} files")
        return all_data
    
    def insert_data(self, processed_data: List[Dict[str, Any]], batch_size: int = BATCH_SIZE, embedding_batch_size: int = EMBEDDING_BATCH_SIZE, test_limit: int = None):
        """Insert processed data into Weaviate with embeddings using batch processing."""
        # Filter out empty texts
        valid_data = [d for d in processed_data if d.get('text', '').strip()]
        
        # Apply test limit if specified
        if test_limit:
            valid_data = valid_data[:test_limit]
            print(f"\n🧪 TEST MODE: Processing only {len(valid_data):,} chunks for testing")
        else:
            print(f"\nInserting {len(valid_data):,} chunks into Weaviate...")
        
        print(f"Using batch sizes: Embeddings={embedding_batch_size}, Weaviate={batch_size}")
        
        collection = self.weaviate_client.collections.get(self.collection_name)
        
        inserted_count = 0
        failed_count = 0
        
        # Accumulate objects across embedding batches for larger Weaviate insertions
        accumulated_objects = []
        
        # Process in batches: generate embeddings in batches, then insert in larger batches
        total_batches = (len(valid_data) + embedding_batch_size - 1) // embedding_batch_size
        
        for batch_idx in tqdm(range(0, len(valid_data), embedding_batch_size), desc="Processing batches"):
            batch_data = valid_data[batch_idx:batch_idx + embedding_batch_size]
            
            try:
                # Extract texts for batch embedding generation
                texts = [d.get('text', '') for d in batch_data]
                
                # Generate embeddings for the entire batch at once
                embeddings = self.generate_embeddings_batch(texts)
                
                # Prepare objects for Weaviate insertion
                for i, chunk_data in enumerate(batch_data):
                    if i < len(embeddings):
                        # embeddings[i] should already be a list from generate_embeddings_batch
                        vector = embeddings[i]
                        # Handle nested list case (sometimes embeddings come as [[...]] instead of [...])
                        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                            # It's nested, take the first element
                            vector = vector[0]
                        # Ensure it's a list of floats
                        if isinstance(vector, list):
                            vector = [float(x) for x in vector]
                        else:
                            # If it's a numpy array or tensor, convert it
                            import numpy as np
                            if isinstance(vector, np.ndarray):
                                vector = vector.tolist()
                            elif hasattr(vector, 'tolist'):
                                vector = vector.tolist()
                            else:
                                vector = [float(x) for x in vector]
                        # Store as dict with properties and vector
                        accumulated_objects.append({
                            "properties": {
                                "url": chunk_data.get('url', ''),
                                "title": chunk_data.get('title', ''),
                                "text": chunk_data.get('text', ''),
                                "text_raw": chunk_data.get('text_raw', ''),
                            },
                            "vector": vector
                        })
                
                # Insert into Weaviate when we accumulate enough objects
                while len(accumulated_objects) >= batch_size:
                    try:
                        batch_to_insert = accumulated_objects[:batch_size]
                        # Convert to DataObject list for insert_many
                        data_objects = [
                            DataObject(properties=obj["properties"], vector=obj["vector"])
                            for obj in batch_to_insert
                        ]
                        collection.data.insert_many(data_objects)
                        inserted_count += batch_size
                        accumulated_objects = accumulated_objects[batch_size:]
                    except Exception as e:
                        error_msg = str(e)
                        # If batch insert fails, try inserting one by one
                        print(f"\n⚠️  Batch insert failed, trying individual inserts...")
                        success_in_batch = 0
                        for idx, obj_data in enumerate(batch_to_insert):
                            try:
                                collection.data.insert(
                                    properties=obj_data["properties"],
                                    vector=obj_data["vector"]
                                )
                                success_in_batch += 1
                                inserted_count += 1
                            except Exception as e2:
                                if idx == 0:  # Print error only for first failure
                                    print(f"   Error: {str(e2)[:300]}")
                                failed_count += 1
                        accumulated_objects = accumulated_objects[batch_size:]
                        if success_in_batch == 0:
                            print(f"   All {batch_size} objects in batch failed")
            
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                failed_count += len(batch_data)
                continue
        
        # Insert any remaining accumulated objects
        while accumulated_objects:
            try:
                # Insert remaining objects in smaller batches if needed
                remaining_batch = accumulated_objects[:batch_size]
                # Convert to DataObject list
                data_objects = [
                    DataObject(properties=obj["properties"], vector=obj["vector"])
                    for obj in remaining_batch
                ]
                collection.data.insert_many(data_objects)
                inserted_count += len(remaining_batch)
                accumulated_objects = accumulated_objects[batch_size:]
            except Exception as e:
                print(f"\n⚠️  Error inserting final batch: {str(e)[:200]}...")
                # Try one by one as last resort
                for obj_data in accumulated_objects:
                    try:
                        collection.data.insert(
                            properties=obj_data["properties"],
                            vector=obj_data["vector"]
                        )
                        inserted_count += 1
                    except Exception as e2:
                        failed_count += 1
                accumulated_objects = []
        
        print(f"\n✓ Insertion complete!")
        print(f"  Successfully inserted: {inserted_count:,} chunks")
        print(f"  Failed: {failed_count:,} chunks")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Insert processed data into Weaviate')
    parser.add_argument('--collection', type=str, default=COLLECTION_NAME,
                        help='Weaviate collection name (default: wikipedia)')
    parser.add_argument('--data-folder', type=str, default=PROCESSED_DATA_FOLDER,
                        help='Folder containing processed JSON files')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for Weaviate insertion (default: {BATCH_SIZE})')
    parser.add_argument('--embedding-batch-size', type=int, default=EMBEDDING_BATCH_SIZE,
                        help=f'Batch size for embedding generation (default: {EMBEDDING_BATCH_SIZE})')
    parser.add_argument('--test-limit', type=int, default=None,
                        help='Limit number of chunks for testing (e.g., 200-300)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Weaviate Data Insertion")
    print("="*60)
    
    # Initialize inserter
    inserter = WeaviateInserter(
        weaviate_url=WEAVIATE_URL,
        collection_name=args.collection
    )
    
    # Load embedding model
    inserter.load_embedding_model()
    
    # Create collection
    inserter.create_collection()
    
    # Load processed data
    processed_data = inserter.load_processed_files(args.data_folder)
    
    # Insert data
    try:
        inserter.insert_data(
            processed_data, 
            batch_size=args.batch_size,
            embedding_batch_size=args.embedding_batch_size,
            test_limit=args.test_limit
        )
    finally:
        # Close Weaviate connection
        inserter.weaviate_client.close()
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()

