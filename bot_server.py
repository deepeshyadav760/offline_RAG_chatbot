# bot_server.py

import socket
import threading
import json
import time
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.config import VECTOR_INDEX_NAME, RETRIEVAL_K, SERVER_PORT, BUFFER_SIZE
from src.model_loader import load_llm, preload_models, cleanup_models
from src.vector_store import load_vector_store
from src.chatbot import create_chatbot, ask_question

# Configuration
SERVER_IP = "0.0.0.0"  # Listen on all network interfaces

# Global RAG variables
rag_chain = None
is_ready = False

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def handle_client(client_socket, client_address):
    """
    Handles a single client connection in a separate thread
    """
    global rag_chain, is_ready
    
    print(f"🔗 Connection from {client_address}")
    
    try:
        # 1. Receive Request
        request_data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
        
        if not request_data:
            return

        try:
            request_json = json.loads(request_data)
            question = request_json.get("question", "")
        except json.JSONDecodeError:
            question = ""

        response = {}

        # 2. Process Logic
        if not question:
            response = {"status": "error", "answer": "Empty question received."}
        elif not is_ready:
            response = {"status": "error", "answer": "System is still loading models. Please wait."}
        else:
            print(f"📩 Query: {question}")
            start_time = time.time()
            try:
                # Ask the RAG Chatbot
                answer_text = ask_question(rag_chain, question)
                elapsed = round(time.time() - start_time, 2)
                
                response = {
                    "status": "success",
                    "answer": answer_text,
                    "time": elapsed
                }
                print(f"✓ Answered in {elapsed}s")
                
            except Exception as e:
                print(f"✗ Processing Error: {e}")
                response = {"status": "error", "answer": f"Internal Error: {str(e)}"}

        # 3. Send Response
        response_data = json.dumps(response).encode('utf-8')
        client_socket.sendall(response_data)

    except Exception as e:
        print(f"✗ Connection Error: {e}")
    finally:
        # 4. Clean up
        client_socket.close()

def start_server():
    """
    Main server loop using Raw TCP Sockets
    """
    global rag_chain, is_ready
    
    print("\n" + "="*60)
    print(f"🚀 STARTING RAG SERVER (Llama 3.2 1B Instruct)")
    print("="*60)
    
    local_ip = get_local_ip()
    print(f"\n📡 Server IP: {local_ip}")
    print(f"📡 Server Port: {SERVER_PORT}")
    print(f"\n💡 Clients should connect to: {local_ip}:{SERVER_PORT}")

    # --- LOAD MODELS ---
    try:
        print("\n" + "="*60)
        print("INITIALIZING RAG SYSTEM")
        print("="*60)
        
        if preload_models():
            print("📂 Loading Vector Store...")
            vector_store = load_vector_store(VECTOR_INDEX_NAME)
            
            if vector_store:
                print("🔗 Creating RAG Chain...")
                llm = load_llm()
                rag_chain = create_chatbot(llm, vector_store, k=RETRIEVAL_K)
                is_ready = True
                
                print("\n" + "="*60)
                print("✅ SYSTEM READY - Accepting Client Connections")
                print("="*60)
            else:
                print("\n⚠️ Vector store not found!")
                print("Please run bot_server_gui.py first to:")
                print("  1. Upload documents")
                print("  2. Process documents (Steps 1→2→3)")
                print("\nThen restart this server.")
                return
        else:
            print("\n✗ Model preload failed!")
            print("\nTroubleshooting:")
            print("1. Ensure Ollama is running: ollama serve")
            print("2. Pull the model: ollama pull llama3.2:1b-instruct-q4_K_M")
            print("3. Check embedding model in: local_embedding_models/")
            return
            
    except Exception as e:
        print(f"\n✗ Startup Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- START SOCKET LISTENER ---
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Allow port reuse immediately after stop
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((SERVER_IP, SERVER_PORT))
        server.listen(5)
        
        print(f"\n🎧 Listening for TCP connections on {SERVER_IP}:{SERVER_PORT}...")
        print(f"💡 Press Ctrl+C to stop the server\n")
        
        while True:
            client_sock, addr = server.accept()
            # Handle client in a new thread
            client_handler = threading.Thread(
                target=handle_client,
                args=(client_sock, addr),
                daemon=True
            )
            client_handler.start()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopping...")
    except Exception as e:
        print(f"\n✗ Server Crash: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        server.close()
        cleanup_models()
        print("✓ Server shutdown complete")

if __name__ == "__main__":
    start_server()