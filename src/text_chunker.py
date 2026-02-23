# src/text_chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(documents):
    """
    Optimized text chunking for Llama 3.2 1B model
    """
    if not documents:
        print("⚠️ No documents provided for chunking.")
        return []
    
    print(f"Chunking {len(documents)} document(s)...")
    print(f"  Strategy: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Separator priority optimized for natural language
        separators=[
            "\n\n",  # Paragraphs (highest priority)
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters (fallback)
        ],
        keep_separator=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✓ Created {len(chunks)} optimized chunks")
    
    if chunks:
        avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
        print(f"  Average chunk size: ~{avg_size} chars")
    
    return chunks

def chunk_text_batch(documents_list):
    """
    Batch processing for multiple documents
    """
    all_chunks = []
    
    for docs in documents_list:
        chunks = chunk_text(docs)
        all_chunks.extend(chunks)
    
    return all_chunks