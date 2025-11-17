"""
Simple script to check what's in a Weaviate collection.
Use this to verify if data has been inserted.
"""

import os
import weaviate

# Weaviate configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')


def connect_to_weaviate():
    """Connect to Weaviate instance."""
    if WEAVIATE_URL.startswith('http://localhost') or WEAVIATE_URL.startswith('http://127.0.0.1'):
        return weaviate.connect_to_local()
    else:
        from urllib.parse import urlparse
        parsed = urlparse(WEAVIATE_URL)
        return weaviate.connect_to_custom(
            http_host=parsed.hostname,
            http_port=parsed.port or 8080,
            http_secure=parsed.scheme == 'https',
            grpc_host=parsed.hostname,
            grpc_port=50051,
            grpc_secure=parsed.scheme == 'https'
        )


def check_collection(collection_name: str, limit: int = 5):
    """Check what's in a collection."""
    print("="*60)
    print(f"Checking Collection: {collection_name}")
    print("="*60)
    print(f"Connecting to: {WEAVIATE_URL}")
    
    try:
        client = connect_to_weaviate()
        print("✓ Connected to Weaviate\n")
        
        # Check if collection exists
        if not client.collections.exists(collection_name):
            print(f"❌ Collection '{collection_name}' does NOT exist")
            print("   The collection is empty or hasn't been created yet.")
            client.close()
            return
        
        print(f"✓ Collection '{collection_name}' exists\n")
        
        collection = client.collections.get(collection_name)
        
        # Get collection config
        config = collection.config.get()
        print(f"Properties: {[prop.name for prop in config.properties]}\n")
        
        # Try to fetch some objects
        print(f"Fetching up to {limit} objects...\n")
        result = collection.query.fetch_objects(limit=limit)
        
        if not result.objects or len(result.objects) == 0:
            print("❌ Collection is EMPTY - No objects found")
            print("   You can proceed with insertion.")
        else:
            print(f"✓ Found {len(result.objects)} object(s) (showing up to {limit})")
            print(f"   There may be more objects in the collection.\n")
            print("-" * 60)
            
            for i, obj in enumerate(result.objects, 1):
                print(f"\n[{i}] Object ID: {obj.uuid}")
                print(f"    Properties:")
                for key, value in obj.properties.items():
                    if isinstance(value, str):
                        # Truncate long text
                        display_value = value[:100] + "..." if len(value) > 100 else value
                        print(f"      {key}: {display_value}")
                    else:
                        print(f"      {key}: {value}")
            
            # Try to get a rough count (sample)
            print(f"\n" + "-" * 60)
            print("Getting approximate count...")
            large_sample = collection.query.fetch_objects(limit=10000)
            count = len(large_sample.objects)
            if count == 10000:
                print(f"  At least {count:,} objects found (may be more)")
            else:
                print(f"  Approximately {count:,} objects found")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def list_all_collections():
    """List all collections in Weaviate."""
    print("="*60)
    print("Listing All Collections")
    print("="*60)
    print(f"Connecting to: {WEAVIATE_URL}")
    
    try:
        client = connect_to_weaviate()
        print("✓ Connected to Weaviate\n")
        
        collections = client.collections.list_all()
        
        if collections:
            print(f"Found {len(collections)} collection(s):\n")
            for collection_name in collections:
                print(f"  - {collection_name}")
                # Quick check of each collection
                try:
                    collection = client.collections.get(collection_name)
                    sample = collection.query.fetch_objects(limit=1)
                    if sample.objects:
                        large_sample = collection.query.fetch_objects(limit=10000)
                        count = len(large_sample.objects)
                        if count == 10000:
                            print(f"    → At least {count:,} objects")
                        else:
                            print(f"    → {count:,} objects")
                    else:
                        print(f"    → Empty")
                except:
                    print(f"    → (Could not check)")
        else:
            print("No collections found")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Weaviate collection contents')
    parser.add_argument('--collection', type=str, help='Collection name to check')
    parser.add_argument('--list', action='store_true', help='List all collections')
    parser.add_argument('--limit', type=int, default=5, help='Number of objects to show (default: 5)')
    
    args = parser.parse_args()
    
    if args.list:
        list_all_collections()
    elif args.collection:
        check_collection(args.collection, limit=args.limit)
    else:
        print("Please specify --collection <name> or use --list to see all collections")
        print("\nExample:")
        print("  python check_collection.py --collection spanish")
        print("  python check_collection.py --list")


if __name__ == "__main__":
    main()

