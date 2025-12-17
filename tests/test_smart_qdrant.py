#!/usr/bin/env python3
"""
Test script for the Smart Qdrant Plugin.

Run this to verify the plugin works correctly with your Qdrant setup.

Usage:
    # From fitz project root after copying smart_qdrant_plugin.py:
    python -c "from fitz.vector_db.plugins.qdrant import QdrantVectorDB; print('OK')"

    # Or run this file directly (after installing):
    python test_smart_qdrant.py
"""

import os
import sys


def test_connection():
    """Test basic connection to Qdrant."""
    print("=" * 60)
    print("Testing Smart Qdrant Plugin")
    print("=" * 60)

    # Try to import from installed location first, then fall back to local
    try:
        from fitz.vector_db.plugins.qdrant import QdrantVectorDB
        print("✓ Plugin imported from fitz.vector_db.plugins.qdrant")
    except ImportError:
        try:
            from fitz.core.vector_db.plugins.qdrant import QdrantVectorDB
            print("✓ Plugin imported from fitz.core.vector_db.plugins.qdrant")
        except ImportError:
            print("✗ Failed to import plugin from fitz package")
            print("  Make sure you copied smart_qdrant_plugin.py to:")
            print("    fitz/vector_db/plugins/qdrant.py")
            print("  or:")
            print("    fitz/core/vector_db/plugins/qdrant.py")
            return False

    # Test connection
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    print(f"\nConnecting to Qdrant at {host}:{port}...")

    try:
        db = QdrantVectorDB(host=host, port=port)
        print(f"✓ Connected to Qdrant")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

    # List collections
    print("\nAvailable collections:")
    try:
        collections = db.list_collections()
        if collections:
            for c in collections:
                print(f"  - {c}")
        else:
            print("  (none)")
    except Exception as e:
        print(f"  Error listing collections: {e}")

    return True


def test_auto_create_collection():
    """Test auto-creation of collections."""
    print("\n" + "=" * 60)
    print("Testing Auto-Create Collection")
    print("=" * 60)

    try:
        from fitz.vector_db.plugins.qdrant import QdrantVectorDB
    except ImportError:
        from fitz.vector_db.plugins.qdrant import QdrantVectorDB

    db = QdrantVectorDB()

    test_collection = "_fitz_test_auto_create"

    # Clean up if exists
    if test_collection in db.list_collections():
        print(f"Cleaning up existing test collection...")
        db.delete_collection(test_collection)

    # Test auto-creation via upsert
    print(f"\nUpserting to non-existent collection '{test_collection}'...")

    test_points = [
        {
            "id": "test-doc-1",
            "vector": [0.1] * 1024,  # 1024 dimensions (Cohere embed size)
            "payload": {"content": "This is a test document", "doc_id": "doc1"}
        },
        {
            "id": "test-doc-2",
            "vector": [0.2] * 1024,
            "payload": {"content": "Another test document", "doc_id": "doc2"}
        }
    ]

    db.upsert(test_collection, test_points)
    print(f"✓ Auto-created collection and upserted {len(test_points)} points")

    # Verify collection exists
    stats = db.get_collection_stats(test_collection)
    print(f"\nCollection stats:")
    print(f"  Points: {stats.get('points_count', 0)}")
    print(f"  Status: {stats.get('status', 'unknown')}")

    # Test search
    print(f"\nTesting search...")
    results = db.search(
        collection_name=test_collection,
        query_vector=[0.15] * 1024,
        limit=2,
    )
    print(f"✓ Search returned {len(results)} results")
    for r in results:
        print(f"  - ID: {r.id}, Score: {r.score:.4f}")

    # Cleanup
    print(f"\nCleaning up test collection...")
    db.delete_collection(test_collection)
    print(f"✓ Deleted test collection")

    return True


def test_error_messages():
    """Test helpful error messages."""
    print("\n" + "=" * 60)
    print("Testing Error Messages")
    print("=" * 60)

    try:
        from fitz.vector_db.plugins.qdrant import QdrantVectorDB
    except ImportError:
        from fitz.vector_db.plugins.qdrant import QdrantVectorDB

    # Import error class
    try:
        from fitz.vector_db.plugins.qdrant import QdrantCollectionError
    except ImportError:
        try:
            from fitz.core.vector_db.plugins.qdrant import QdrantCollectionError
        except ImportError:
            # Define locally if not found
            QdrantCollectionError = Exception

    db = QdrantVectorDB()

    # Test searching non-existent collection
    print("\nSearching non-existent collection '_does_not_exist'...")
    try:
        db.search("_does_not_exist", [0.1] * 1024, limit=5)
        print("✗ Should have raised an error")
        return False
    except QdrantCollectionError as e:
        print(f"✓ Got helpful error message:")
        # Just show first few lines
        lines = str(e).strip().split('\n')[:5]
        for line in lines:
            print(f"  {line}")

    return True


def main():
    """Run all tests."""
    print("\n🔧 Smart Qdrant Plugin Test Suite\n")

    all_passed = True

    # Test 1: Connection
    if not test_connection():
        print("\n❌ Connection test failed - cannot continue")
        return 1

    # Test 2: Auto-create
    try:
        if not test_auto_create_collection():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Auto-create test failed: {e}")
        all_passed = False

    # Test 3: Error messages
    try:
        if not test_error_messages():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Error message test failed: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())