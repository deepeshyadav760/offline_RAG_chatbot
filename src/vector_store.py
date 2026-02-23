# src/vector_store.py

import os
import pickle
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from langchain_community.vectorstores import FAISS
from src.model_loader import get_embedding_model
from src.config import (
    VECTOR_DB_PATH, 
    USE_IVF_INDEX, 
    IVF_NLIST, 
    IVF_NPROBE,
    EMBEDDING_BATCH_SIZE
)

def create_vector_store(text_chunks, index_name="document_index", progress_callback=None):
    """
    Create FAISS vector store with optional progress tracking
    """
    if not text_chunks:
        print("⚠️ No text chunks provided.")
        return None

    print(f"\n{'='*60}")
    print(f"CREATING VECTOR STORE")
    print(f"{'='*60}")
    print(f"Chunks to process: {len(text_chunks)}")
    
    try:
        embeddings = get_embedding_model()
        if not embeddings:
            print("✗ Failed to load embedding model")
            return None
        
        print("\nGenerating embeddings (GPU-accelerated)...")
        
        # Batch processing with progress tracking for large datasets
        if progress_callback and len(text_chunks) > 100:
            batch_size = 50
            
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                
                if i == 0:
                    # Create initial vector store with first batch
                    vector_store = FAISS.from_documents(
                        batch, 
                        embedding=embeddings,
                        distance_strategy="COSINE",
                    )
                else:
                    # Add subsequent batches
                    batch_store = FAISS.from_documents(
                        batch,
                        embedding=embeddings,
                        distance_strategy="COSINE",
                    )
                    vector_store.merge_from(batch_store)
                
                # Report progress
                current = min(i + batch_size, len(text_chunks))
                progress_callback(current, len(text_chunks))
        else:
            # Standard processing for small datasets
            vector_store = FAISS.from_documents(
                text_chunks, 
                embedding=embeddings,
                distance_strategy="COSINE",
            )
            if progress_callback:
                progress_callback(len(text_chunks), len(text_chunks))
        
        # IVF optimization (only for very large datasets)
        if USE_IVF_INDEX and len(text_chunks) > 50000:
            print(f"\nOptimizing index with IVF...")
            vector_store = _optimize_faiss_index(vector_store, IVF_NLIST, IVF_NPROBE)
        else:
            print("✓ Using Exact Search (Highest Accuracy)")
        
        # Save vector store
        save_path = os.path.join(VECTOR_DB_PATH, index_name)
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        print(f"\nSaving vector store to: {save_path}")
        vector_store.save_local(save_path)
        
        print(f"\n{'='*60}")
        print("✓ VECTOR STORE CREATED SUCCESSFULLY")
        print(f"  Total vectors: {vector_store.index.ntotal}")
        print(f"  Dimension: {vector_store.index.d}")
        print(f"{'='*60}\n")
        
        return vector_store
        
    except Exception as e:
        print(f"\n✗ Error creating vector store: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_vector_store(index_name="document_index"):
    """
    Load existing FAISS vector store
    """
    db_path = os.path.join(VECTOR_DB_PATH, index_name)
    
    if not os.path.exists(db_path):
        print(f"ℹ️ Vector store not found at: {db_path}")
        return None
    
    print(f"Loading vector store from: {db_path}")
    
    try:
        embeddings = get_embedding_model()
        if not embeddings:
            print("✗ Failed to load embedding model")
            return None
        
        vector_store = FAISS.load_local(
            db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        print(f"✓ Vector store loaded successfully")
        print(f"  Total vectors: {vector_store.index.ntotal}")
        
        return vector_store
        
    except Exception as e:
        print(f"✗ Error loading vector store: {e}")
        return None

def _optimize_faiss_index(vector_store, nlist, nprobe):
    """
    Optimize FAISS index with IVF (for very large datasets)
    """
    try:
        import faiss
        original_index = vector_store.index
        d = original_index.d
        
        # Create IVF index
        quantizer = faiss.IndexFlatIP(d)
        index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train and add vectors
        vectors = original_index.reconstruct_n(0, original_index.ntotal)
        index_ivf.train(vectors)
        index_ivf.add(vectors)
        index_ivf.nprobe = nprobe
        
        vector_store.index = index_ivf
        
        print(f"✓ Index optimized: {original_index.ntotal} vectors in {nlist} clusters")
        return vector_store
        
    except Exception as e:
        print(f"⚠️ Could not optimize index: {e}")
        return vector_store