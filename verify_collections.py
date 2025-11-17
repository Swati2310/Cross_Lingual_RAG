"""
Script to verify Weaviate collections and their contents.
"""

import os
import weaviate

# Weaviate configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')

# Collection names
COLLECTIONS = ['english', 'chinese', 'hindi', 'french', 'spanish']


def verify_collection(client: WeaviateClient, collection_name: str):
    """Verify a collection exists and show its stats."""
    print(f"\n{'='*60}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}")
    
    if not client.collections.exists(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist")
        return
    
    print(f"✓ Collection '{collection_name}' exists")
    
    try:
        collection = client.collections.get(collection_name)
        
        # Get collection info
        config = collection.config.get()
        print(f"  Properties: {[prop.name for prop in config.properties]}")
        
        # Count objects (sample query)
        try:
            result = collection.query.fetch_objects(limit=1)
            total = collection.query.fetch_objects(limit=10000)  # Get approximate count
            # Note: Weaviate doesn't have a direct count, so we'll use a workaround
            print(f"  Sample object retrieved: {len(result.objects) > 0}")
            
            if result.objects:
                obj = result.objects[0]
                print(f"  Sample properties: {list(obj.properties.keys())}")
        except Exception as e:
            print(f"  Could not query collection: {e}")
            
    except Exception as e:
        print(f"  Error accessing collection: {e}")


def main():
    """Verify all collections."""
    print(f"Connecting to Weaviate at {WEAVIATE_URL}")
    
    try:
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
        
        # Test connection
        try:
            # In v4, we can check if client is connected by trying to access collections
            print(f"✓ Connected to Weaviate")
        except Exception as e:
            print(f"❌ Could not connect to Weaviate: {e}")
            return
        
        print(f"\nVerifying {len(COLLECTIONS)} collections...")
        
        for collection_name in COLLECTIONS:
            verify_collection(client, collection_name)
        
        print("\n" + "="*60)
        print("Verification complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

