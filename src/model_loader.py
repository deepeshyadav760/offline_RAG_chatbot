# src/model_loader.py

from langchain_ollama.llms import OllamaLLM
from langchain_ollama.llms import OllamaLLM
from src.config import (
    LLM_MODEL_NAME, 
    EMBEDDING_MODEL_NAME, 
    DEVICE,
    MAX_TOKENS,
    TEMPERATURE,
    CONTEXT_WINDOW_SIZE
)
import subprocess
import sys

_llm_instance = None
_embedding_instance = None

def is_ollama_running():
    """Check if Ollama service is running"""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            timeout=5,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def check_ollama_model(model_name):
    """Check if specific model is available in Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            text=True
        )
        return model_name.split(':')[0] in result.stdout
    except:
        return False



def load_llm(model_name=LLM_MODEL_NAME):
    """
    Load Ollama LLM with optimized settings for Llama 3.2 1B Instruct
    """
    global _llm_instance
    
    if _llm_instance is not None:
        print("✓ Using cached LLM instance.")
        return _llm_instance
    
    # USE OLLAMA (Local Custom Model)
    try:
        # Check if Ollama is running
        if not is_ollama_running():
            print("✗ Ollama service is not running!")
            print("  Please start Ollama first:")
            print("  Windows: Start Ollama from Start Menu")
            print("  Linux/Mac: ollama serve")
            return None
        
        # Check if model exists
        if not check_ollama_model(model_name):
            print(f"✗ Model '{model_name}' not found in Ollama!")
            print(f"  Please run 'create_model.bat' to import your GGUF file first!")
            return None
        
        print(f"Loading Ollama model: '{model_name}' (one-time initialization)...")
        
        # Optimized for Llama 3.2 1B Instruct
        _llm_instance = OllamaLLM(
            model=model_name,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,  # Response length limit
            num_ctx=CONTEXT_WINDOW_SIZE,  # 4K context window
            top_k=40,  # Balance between diversity and focus
            top_p=0.9,  # Nucleus sampling
            repeat_penalty=1.2,  # Prevent repetition
            repeat_last_n=64,  # Look back 64 tokens for repetition detection
            num_thread=8,  # Multi-threading for CPU
        )
        
        print("Testing Ollama connection...")
        test_response = _llm_instance.invoke("Hi")
        
        print("✓ LLM loaded and cached successfully.")
        print(f"  Model: {model_name}")
        print(f"  Device: {DEVICE}")
        print(f"  Context Window: {CONTEXT_WINDOW_SIZE} tokens")
        
        return _llm_instance
        
    except Exception as e:
        print(f"✗ Error loading LLM: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print(f"2. Pull the model: ollama pull {model_name}")
        print("3. Check Ollama status: ollama list")
        return None

from langchain_ollama import OllamaEmbeddings

def get_embedding_model():
    """
    Load Ollama embedding model (cached)
    """
    global _embedding_instance
    
    if _embedding_instance is not None:
        print("✓ Using cached embedding model.")
        return _embedding_instance
    
    try:
        print(f"Loading Ollama embedding model '{EMBEDDING_MODEL_NAME}'...")
        
        _embedding_instance = OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME,
        )
        
        # Test embedding generation
        _embedding_instance.embed_query("test")
        
        print(f"✓ Embedding model loaded successfully.")
        return _embedding_instance
        
    except Exception as e:
        print(f"✗ Error loading embedding model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running")
        print("2. Ensure the model exists in Ollama")
        return None

def preload_models():
    """
    Preload all models at startup for optimal performance
    """
    print("\n" + "="*60)
    print("PRELOADING MODELS FOR OPTIMAL PERFORMANCE")
    print("="*60)
    
    llm = load_llm()
    embeddings = get_embedding_model()
    
    if llm and embeddings:
        print("\n✓ All models preloaded successfully!")
        print("="*60 + "\n")
        return True
    else:
        print("\n✗ Model preloading failed!")
        print("="*60 + "\n")
        return False

def cleanup_models():
    """
    Clean up models from memory (for server shutdown)
    """
    global _llm_instance, _embedding_instance
    _llm_instance = None
    _embedding_instance = None
    print("✓ Models cleaned up from memory")