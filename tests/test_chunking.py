from rag.splitter import split_text

file_path = "knowledge_base/probability.txt"
# Extract the text
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

chunks = split_text(text)

print(f"Total chunks: {len(chunks)}\n")
# for i, chunk in enumerate(chunks, 1):
#     print(f"--- Chunk {i} ---")
#     print(chunk[:00] + ("..." if len(chunk) > 200 else ""))