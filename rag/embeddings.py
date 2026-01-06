from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def store_documents(text_chunks, metadata=None, persist_dir="chromadb"):
    embeddings = get_embeddings()

    # open persistent DB
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    # append documents + metadata
    vectorstore.add_texts(
        texts=text_chunks,
        metadata=metadata,
    )
    vectorstore.persist()
    return vectorstore

def get_vectorstore(persist_dir="chromadb"):
    embeddings = get_embeddings()

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    return vectorstore