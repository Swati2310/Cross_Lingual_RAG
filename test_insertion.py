#!/usr/bin/env python3
"""
Simple test script to test insertion with 200-300 chunks.
This will test if the insertion works, then delete the collection.
"""

import os
import sys
import subprocess

# Configuration
COLLECTION = "spanish"
TEST_LIMIT = 250  # Test with 250 chunks
# HUGGING_FACE_TOKEN should be set as environment variable
# export HUGGING_FACE_HUB_TOKEN=your_token_here

def main():
    print("="*60)
    print("🧪 TEST INSERTION - 250 Chunks")
    print("="*60)
    print(f"Collection: {COLLECTION}")
    print(f"Test limit: {TEST_LIMIT} chunks")
    print()
    
    # Set environment - use token from environment variable or prompt user
    env = os.environ.copy()
    if 'HUGGING_FACE_HUB_TOKEN' not in env:
        token = os.getenv('HF_TOKEN') or input("Enter your HuggingFace token (or set HUGGING_FACE_HUB_TOKEN env var): ")
        env['HUGGING_FACE_HUB_TOKEN'] = token
    
    # Run insertion with test limit
    print("Step 1: Testing insertion...")
    print("-" * 60)
    
    cmd = [
        sys.executable,
        "insert_to_weaviate.py",
        "--collection", COLLECTION,
        "--test-limit", str(TEST_LIMIT),
        "--embedding-batch-size", "16",  # Smaller for CPU
        "--batch-size", "50"  # Smaller batches
    ]
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=False)
        print("\n✓ Insertion test completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Insertion failed: {e}")
        return
    
    # Check what was inserted
    print("\n" + "="*60)
    print("Step 2: Verifying insertion...")
    print("-" * 60)
    
    check_cmd = [sys.executable, "quick_view.py", COLLECTION, "5"]
    subprocess.run(check_cmd)
    
    # Ask if user wants to delete
    print("\n" + "="*60)
    print("Step 3: Cleanup")
    print("-" * 60)
    print(f"Collection '{COLLECTION}' has been tested.")
    print("Run this to delete it:")
    print(f"  python delete_collection.py --collection {COLLECTION}")
    print()
    print("Or delete it now? (y/n): ", end="")
    
    # Auto-delete for testing
    response = "y"  # Auto yes for testing
    if response.lower() == 'y':
        print("Deleting collection...")
        delete_cmd = [sys.executable, "delete_collection.py", "--collection", COLLECTION]
        subprocess.run(delete_cmd)
        print("✓ Collection deleted")
    else:
        print("Collection kept for inspection")

if __name__ == "__main__":
    main()

