from .embeddings import get_vectorstore

def get_retriever(k=5):
    vectorstore = get_vectorstore()
    print(f"LOADED {vectorstore._collection.count()} DOCUMENTS")  # KEY LINE
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever

retriever = get_retriever()
print("QUERYING: 'Dice problem?'")
docs = retriever.invoke("Dice problem?")
print(f"FOUND {len(docs)} CHUNKS")

if docs:
    for i, d in enumerate(docs):
        print(f"\n CHUNK {i+1}:")
        print(d.page_content[:200])
        print("META:", d.metadata)
        print("-" * 40)
else:
    print(" EMPTY RESULTS - INGESTION FAILED!")
  

