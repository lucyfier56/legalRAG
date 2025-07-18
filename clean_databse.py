from qdrant_client import QdrantClient

def clear_legal_documents_collection():
    """Delete the legal_documents collection from Qdrant"""
    try:
        # Connect to Qdrant
        client = QdrantClient(url="http://localhost:6333")
        
        # Delete the collection
        client.delete_collection("test_documents")
        print("✅ Collection 'test_documents' deleted successfully")
        
        # Verify deletion
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if "legal_documents" not in collection_names:
            print("✅ Deletion confirmed - collection no longer exists")
        else:
            print("❌ Collection still exists - deletion may have failed")
            
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")

if __name__ == "__main__":
    clear_legal_documents_collection()
