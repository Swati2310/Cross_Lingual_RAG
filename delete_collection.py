"""
Script to delete all data from a Weaviate collection.
Use this to clear a collection before re-inserting data.
"""

import os
import weaviate
import argparse

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


def delete_collection_data(client, collection_name: str):
    """Delete all data from a collection."""
    print(f"\n{'='*60}")
    print(f"Deleting data from collection: {collection_name}")
    print(f"{'='*60}")
    
    if not client.collections.exists(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist")
        return False
    
    collection = client.collections.get(collection_name)
    
    # Get approximate count before deletion
    try:
        sample = collection.query.fetch_objects(limit=1)
        print(f"Collection exists: ✓")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return False
    
    # Delete all objects in the collection
    print(f"\nDeleting all objects from '{collection_name}'...")
    
    try:
        # In Weaviate v4, we can delete by batch
        # First, get all object IDs in batches
        deleted_count = 0
        batch_size = 1000
        
        while True:
            # Fetch a batch of objects
            result = collection.query.fetch_objects(limit=batch_size)
            
            if not result.objects or len(result.objects) == 0:
                break
            
            # Extract UUIDs
            uuids = [obj.uuid for obj in result.objects]
            
            # Delete this batch
            collection.data.delete_many(where={"operator": "Or", "operands": [
                {"path": ["id"], "operator": "Equal", "value": str(uuid)} for uuid in uuids
            ]})
            
            # Alternative: delete by UUIDs directly
            # Weaviate v4 supports deleting by UUIDs
            try:
                collection.data.delete_many(where={"operator": "Or", "operands": [
                    {"path": ["id"], "operator": "Equal", "value": str(uuid)} for uuid in uuids
                ]})
            except:
                # Fallback: delete one by one
                for uuid in uuids:
                    try:
                        collection.data.delete_by_id(uuid)
                    except:
                        pass
            
            deleted_count += len(uuids)
            print(f"  Deleted {deleted_count:,} objects...", end='\r')
            
            # If we got fewer objects than batch_size, we're done
            if len(result.objects) < batch_size:
                break
        
        print(f"\n✓ Deleted {deleted_count:,} objects from '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"\nError deleting objects: {e}")
        print("Trying alternative deletion method...")
        
        # Alternative: Delete the entire collection and recreate it
        try:
            print("Deleting entire collection...")
            client.collections.delete(collection_name)
            print(f"✓ Collection '{collection_name}' deleted successfully")
            print("Note: You'll need to recreate the collection before inserting new data")
            return True
        except Exception as e2:
            print(f"Error deleting collection: {e2}")
            return False


def delete_entire_collection(client, collection_name: str):
    """Delete the entire collection (including schema)."""
    print(f"\n{'='*60}")
    print(f"Deleting entire collection: {collection_name}")
    print(f"{'='*60}")
    
    if not client.collections.exists(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist")
        return False
    
    try:
        client.collections.delete(collection_name)
        print(f"✓ Collection '{collection_name}' deleted successfully")
        print("Note: The collection schema has been removed. It will be recreated on next insertion.")
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Delete data from Weaviate collection')
    parser.add_argument('--collection', type=str, required=True,
                        help='Collection name to delete data from')
    parser.add_argument('--delete-schema', action='store_true',
                        help='Delete the entire collection (including schema)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Weaviate Collection Deletion")
    print("="*60)
    print(f"Connecting to: {WEAVIATE_URL}")
    
    try:
        client = connect_to_weaviate()
        print("✓ Connected to Weaviate")
        
        if args.delete_schema:
            success = delete_entire_collection(client, args.collection)
        else:
            success = delete_collection_data(client, args.collection)
        
        client.close()
        
        if success:
            print("\n" + "="*60)
            print("Deletion complete!")
            print("="*60)
            if not args.delete_schema:
                print(f"\nYou can now insert new data into '{args.collection}'")
            else:
                print(f"\nCollection '{args.collection}' has been deleted.")
                print("It will be automatically recreated when you run the insertion script.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

