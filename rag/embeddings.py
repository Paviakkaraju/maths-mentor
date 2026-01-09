# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# import torch # Add this import
# from langchain_huggingface import HuggingFaceEmbeddings

# # Global variable to cache the model
# _embeddings_instance = None

# def get_embeddings():
#     global _embeddings_instance
#     if _embeddings_instance is None:
#         print(" Loading Embedding Model into RAM...")
#         # Force the model to load on CPU and bypass the 'meta' device issue
#         model_kwargs = {'device': 'cpu'} 
#         encode_kwargs = {'normalize_embeddings': True}
        
#         _embeddings_instance = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2",
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs
#         )
#     return _embeddings_instance

import torch
from langchain_huggingface import HuggingFaceEmbeddings

# Global variable to cache the model
_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        # Determine the best available device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps' # For Mac M1/M2/M3
        else:
            device = 'cpu'
            
        # print(f"🚀 Initializing Embedding Model on {device.upper()}...")
        
        model_kwargs = {'device': device} 
        encode_kwargs = {'normalize_embeddings': True}
        
        # Load the model once
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        # print("Model loaded and cached.")
    return _embeddings_instance

# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def store_documents(
    text_chunks, metadata=None, persist_dir="chromadb", collection_name="maths_docs"
):
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,  
    )
    
    vectorstore.add_texts(texts=text_chunks, metadatas=metadata)

def get_vectorstore(
    persist_dir="chromadb",
    collection_name="maths_docs"
):
    embeddings = get_embeddings()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )