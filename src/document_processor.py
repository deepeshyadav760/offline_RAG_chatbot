# src/document_processor.py

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
)
import os

def load_document(file_path):
    """
    Load a single document with optimized error handling
    """
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return []

    filename = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    print(f"📄 Loading: {filename} ({ext})")
    
    try:
        # Fast path selection based on extension
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = UnstructuredFileLoader(file_path, mode="single")
        else:
            loader = UnstructuredFileLoader(file_path)
        
        # Load documents
        documents = loader.load()
        
        if not documents:
            print(f"⚠️ No content extracted from {filename}")
            return []
        
        # Count total characters
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        print(f"✓ Loaded {len(documents)} page(s), {total_chars:,} characters")
        
        return documents
        
    except Exception as e:
        print(f"✗ Error loading {filename}: {e}")
        return []

def load_documents_batch(file_paths):
    """
    Load multiple documents efficiently
    """
    all_docs = []
    
    print(f"\n{'='*60}")
    print(f"BATCH LOADING {len(file_paths)} DOCUMENTS")
    print(f"{'='*60}\n")
    
    for i, path in enumerate(file_paths, 1):
        print(f"[{i}/{len(file_paths)}]", end=" ")
        docs = load_document(path)
        all_docs.extend(docs)
    
    print(f"\n{'='*60}")
    print(f"✓ LOADED {len(all_docs)} TOTAL PAGES")
    print(f"{'='*60}\n")
    
    return all_docs