from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
    )


def split_text(text: str, **kwargs):
    splitter = get_text_splitter(**kwargs)
    return splitter.split_text(text)
