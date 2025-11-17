#!/usr/bin/env python3
"""
Quick script to view Weaviate collections - simple and reliable
"""
import requests
import json
import sys

WEAVIATE_URL = "http://localhost:8080"

def query_collection(collection_name, limit=5):
    """Query a collection and display results."""
    query = {
        "query": f"""
        {{
            Get {{
                {collection_name}(limit: {limit}) {{
                    title
                    text
                    url
                    _additional {{
                        id
                    }}
                }}
            }}
        }}
        """
    }
    
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json=query,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        if "errors" in data:
            print("Error:", json.dumps(data["errors"], indent=2))
            return
        
        objects = data.get("data", {}).get("Get", {}).get(collection_name, [])
        
        if not objects:
            print(f"❌ Collection '{collection_name}' is EMPTY")
            print("   No objects found. Ready for insertion.")
        else:
            print(f"✓ Found {len(objects)} object(s) in '{collection_name}':\n")
            for i, obj in enumerate(objects, 1):
                print(f"[{i}] ID: {obj.get('_additional', {}).get('id', 'N/A')}")
                print(f"    Title: {obj.get('title', 'N/A')}")
                print(f"    URL: {obj.get('url', 'N/A')}")
                text = obj.get('text', '')
                if text:
                    print(f"    Text: {text[:100]}..." if len(text) > 100 else f"    Text: {text}")
                print()
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Weaviate at {WEAVIATE_URL}")
        print("   Make sure Weaviate is running: docker ps")
    except Exception as e:
        print(f"❌ Error: {e}")

def list_collections():
    """List all collections."""
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/schema")
        response.raise_for_status()
        schema = response.json()
        
        collections = schema.get("classes", [])
        if collections:
            print(f"Found {len(collections)} collection(s):\n")
            for cls in collections:
                name = cls.get("class", "Unknown")
                props = cls.get("properties", [])
                print(f"  - {name} ({len(props)} properties)")
        else:
            print("No collections found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_collections()
        else:
            collection = sys.argv[1]
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            query_collection(collection, limit)
    else:
        print("Usage:")
        print("  python quick_view.py --list")
        print("  python quick_view.py <collection_name> [limit]")
        print("\nExample:")
        print("  python quick_view.py Spanish 10")

