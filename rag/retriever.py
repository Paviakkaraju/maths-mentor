from .embeddings import get_vectorstore

# def search(query, k=5):
#     vectorstore = get_vectorstore()

#     results = vectorstore.similarity_search(
#         query,
#         k=k, 
#         include_score=True            
#     )

#     return results

def get_retriever(k=5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


retriever = get_retriever()

docs = retriever.invoke("Dice problem?")

for d in docs:
    print("CHUNK:\n", d.page_content[:200])
    print("META:", d.metadata)
    print("-" * 40)
