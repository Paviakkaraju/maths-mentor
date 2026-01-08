import chromadb
from pprint import pprint

def inspect_chroma():
    client = chromadb.PersistentClient(path="./chromadb")
    
    # 1. Check collection names
    collections = client.list_collections()
    print(f"Collections found: {[c.name for c in collections]}")
    
    if not collections:
        print("ERROR: No collections found. Did you persist the data?")
        return

    # 2. Peek into the collection
    # Change 'maths_docs' if your collection name is different
    collection = client.get_collection("maths_docs")
    count = collection.count()
    print(f"Total documents in 'maths_docs': {count}")

    if count > 0:
        print("\n--- PEEKING AT FIRST 2 DOCUMENTS ---")
        peek = collection.peek(limit=2)
        for i in range(len(peek['documents'])):
            print(f"\nDocument {i+1}: {peek['documents'][i][:100]}...")
            print(f"Metadata {i+1}: {peek['metadatas'][i]}")
            
    # 3. Test a query WITHOUT a filter
    print("\n--- TESTING QUERY WITHOUT FILTER ---")
    # We use a dummy vector or just a string if using a wrapper
    # Since we are using raw client, we can't easily embed here, 
    # but we can check if the metadata keys match our 'where' clause.

if __name__ == "__main__":
    inspect_chroma()