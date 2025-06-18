import chromadb
import numpy as np

# --- Configuration ---
CHROMA_DB_PATH = "./new_chroma_db_store"
COLLECTION_NAME = "face_embeddings_resnet_new"

def check_database():
    print("--- Checking ChromaDB Database ---")
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
        # Get total number of embeddings
        total_embeddings = collection.count()
        print(f"Total embeddings in collection '{COLLECTION_NAME}': {total_embeddings}")
        
        if total_embeddings == 0:
            print("Warning: No embeddings found in the database. Run face_recognition_new.py to generate embeddings.")
            return
        
        # Retrieve all entries (IDs, metadata, and optionally embeddings)
        results = collection.get(include=['metadatas', 'embeddings'])
        
        # Group by user_name for summary
        user_counts = {}
        for metadata in results['metadatas']:
            user_name = metadata['user_name']
            user_counts[user_name] = user_counts.get(user_name, 0) + 1
        
        print("\nSummary of embeddings by user:")
        for user_name, count in user_counts.items():
            print(f"User: {user_name}, Number of images: {count}")
        
        print("\nDetailed entries:")
        for i, (id, metadata, embedding) in enumerate(zip(results['ids'], results['metadatas'], results['embeddings'])):
            print(f"Entry {i+1}:")
            print(f"  ID: {id}")
            print(f"  User: {metadata['user_name']}")
            print(f"  Image: {metadata['image_name']}")
            # Print first few elements of embedding to verify
            embedding_preview = np.array(embedding)[:5]  # Show first 5 values
            print(f"  Embedding (first 5 values): {embedding_preview}")
            print()
        
        # Optional: Query a sample embedding to check similarity
        if results['embeddings']:
            sample_embedding = results['embeddings'][0]
            query_results = collection.query(
                query_embeddings=[sample_embedding],
                n_results=3
            )
            print("\nSample query results (top 3 matches for first embedding):")
            for dist, meta, id in zip(query_results['distances'][0], query_results['metadatas'][0], query_results['ids'][0]):
                print(f"Match: {meta['user_name']} ({meta['image_name']}), Distance: {dist:.4f}")
    
    except Exception as e:
        print(f"Error accessing database: {e}")

if __name__ == "__main__":
    check_database()
    print("--- Check Complete ---")