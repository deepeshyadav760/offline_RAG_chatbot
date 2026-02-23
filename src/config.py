# src/config.py
import os
import torch

# ============================================================================
# PATHS
# ============================================================================
DOCUMENTS_DIR = "documents/"
VECTOR_DB_PATH = "vector_db/"
VECTOR_INDEX_NAME = "rag_index"
LOCAL_EMBEDDING_PATH = "./local_embedding_models/all-MiniLM-L6-v2"

# ============================================================================
# MODELS
# ============================================================================
EMBEDDING_MODEL_NAME = "custom-llama3.2" # Use the same Ollama model for embeddings
LLM_MODEL_NAME = "custom-llama3.2"  # Custom model name created in Ollama

# ============================================================================
# COMPUTATION
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# CHUNKING OPTIMIZATION
# ============================================================================
CHUNK_SIZE = 800  # Optimized for 1B model (smaller context)
CHUNK_OVERLAP = 150

# ============================================================================
# RETRIEVAL OPTIMIZATION
# ============================================================================
RETRIEVAL_K = 6  # Reduced for 1B model to avoid overwhelming it

# ============================================================================
# VECTOR STORE OPTIMIZATION
# ============================================================================
USE_IVF_INDEX = False  # Exact search for accuracy
IVF_NLIST = 100
IVF_NPROBE = 10

# ============================================================================
# LLM OPTIMIZATION (Optimized for Llama 3.2 1B)
# ============================================================================
CONTEXT_WINDOW_SIZE = 4096  # Llama 3.2 1B supports 4K context
MAX_TOKENS = 512  # Reduced for 1B model
TEMPERATURE = 0.2  # Lower temperature for more focused responses

USE_QUANTIZATION = True

# ============================================================================
# CACHING
# ============================================================================
ENABLE_QUERY_CACHE = True
CACHE_SIZE = 100

# ============================================================================
# BATCH PROCESSING
# ============================================================================
EMBEDDING_BATCH_SIZE = 32

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================
SERVER_PORT = 9999
BUFFER_SIZE = 16384

# ============================================================================
# OFFLINE MODE
# ============================================================================
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'

# ============================================================================
# CONSOLE UTF-8 ENCODING (Windows compatibility)
# ============================================================================
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass