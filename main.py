# main.py

import os
import sys
from src.model_loader import load_llm
from src.document_processor import load_documents_batch
from src.text_chunker import chunk_text
from src.vector_store import create_vector_store
from src.chatbot import create_chatbot, ask_question
from src.config import DOCUMENTS_DIR, VECTOR_INDEX_NAME

def get_all_documents(doc_dir):
    """
    Get all supported document paths from the directory
    """
    supported_extensions = ('.pdf', '.docx', '.txt')
    files = []
    
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        return []
        
    for f in os.listdir(doc_dir):
        if f.lower().endswith(supported_extensions):
            files.append(os.path.join(doc_dir, f))
            
    return files

def main():
    print("--- Offline RAG Chatbot (Batch Processing) ---")
    
    # 1. Load the LLM (Supports Local GGUF)
    llm = load_llm()
    if llm is None:
        print("Exiting: Could not load the LLM.")
        return

    # 2. auto-detect all documents
    print(f"\nScanning '{DOCUMENTS_DIR}' for documents...")
    doc_paths = get_all_documents(DOCUMENTS_DIR)
    
    if not doc_paths:
        print(f"⚠️ No documents found in '{DOCUMENTS_DIR}'!")
        print("Please add .pdf, .docx, or .txt files to this folder.")
        return
        
    print(f"✓ Found {len(doc_paths)} documents.")
    
    # 3. Process documents (Batch)
    documents = load_documents_batch(doc_paths)
    if not documents:
        print("Exiting: No content could be loaded.")
        return
        
    text_chunks = chunk_text(documents)
    
    # 4. Create Vector Store
    # Note: creates a new index each time main.py is run.
    # For persistence, we should load existing if available, but for this CLI tool,
    # rebuilding ensures fresh state.
    vector_store = create_vector_store(text_chunks, index_name=VECTOR_INDEX_NAME)
    
    if vector_store is None:
        print("Exiting: Failed to create the vector store.")
        return

    # 5. Create Chatbot Chain
    qa_chain = create_chatbot(llm, vector_store)
    if qa_chain is None:
        print("Exiting: Could not create the chatbot chain.")
        return

    # 6. Chat Loop
    print("\n" + "="*50)
    print("🤖 Chatbot Ready! (Type 'exit' to quit)")
    print("="*50)
    
    while True:
        try:
            user_query = input("\nYour Question: ")
            if user_query.lower() in ['exit', 'quit']:
                break
            
            if not user_query.strip():
                continue
                
            response = ask_question(qa_chain, user_query)
            print("\nAnswer:", response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
        
    print("\n--- Goodbye! ---")

if __name__ == "__main__":
    main()
    
