"""
Script to create Weaviate collections for Wikipedia datasets.
This script creates collections with the specified schema:
- id: number
- url: string
- title: string
- text: string
- text_raw: string
"""

import os
import weaviate
from weaviate.classes.config import Property, DataType

# Weaviate configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')

# Collection names
COLLECTIONS = ['english', 'chinese', 'hindi', 'french', 'spanish']


def create_collection(client: WeaviateClient, collection_name: str):
    """Create a Weaviate collection with the specified schema."""
    print(f"Creating collection: {collection_name}")
    
    # Check if collection exists
    if client.collections.exists(collection_name):
        print(f"Collection {collection_name} already exists. Deleting...")
        client.collections.delete(collection_name)
    
    # Create collection with schema
    # Note: 'id' is reserved in Weaviate, so we don't include it as a property
    client.collections.create(
        name=collection_name,
        properties=[
            Property(name="url", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="text_raw", data_type=DataType.TEXT),
        ],
        vectorizer_config=None,  # We'll provide our own vectors
    )
    print(f"✓ Collection '{collection_name}' created successfully\n")


def main():
    """Create all collections."""
    print(f"Connecting to Weaviate at {WEAVIATE_URL}")
    # Weaviate v4 connection method
    if WEAVIATE_URL.startswith('http://localhost') or WEAVIATE_URL.startswith('http://127.0.0.1'):
        client = weaviate.connect_to_local()
    else:
        from urllib.parse import urlparse
        parsed = urlparse(WEAVIATE_URL)
        client = weaviate.connect_to_custom(
            http_host=parsed.hostname,
            http_port=parsed.port or 8080,
            http_secure=parsed.scheme == 'https',
            grpc_host=parsed.hostname,
            grpc_port=50051,
            grpc_secure=parsed.scheme == 'https'
        )
    
    print(f"Creating {len(COLLECTIONS)} collections...\n")
    
    for collection_name in COLLECTIONS:
        try:
            create_collection(client, collection_name)
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("All collections created successfully!")


if __name__ == "__main__":
    main()

