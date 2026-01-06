import os
from rag.splitter import split_text
from rag.embeddings import store_documents

data_dir = "knowledge_base"
files = os.listdir(data_dir)

for filename in files:
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(data_dir, filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # split this file 
    chunks = split_text(text, chunk_size=1000, chunk_overlap=150)

    # add metadata 
    metadata = [{"source": filename, "index": i} for i in range(len(chunks))]

    # store chunks
    store_documents(chunks, metadata=metadata)

    print(f"Indexed {filename} -> {len(chunks)} chunks")