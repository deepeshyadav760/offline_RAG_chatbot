# 🤖 Offline RAG Chatbot - Client-Server Architecture

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Llama 3.2](https://img.shields.io/badge/Model-Llama%203.2%201B-green.svg)](https://ollama.ai/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-red.svg)](https://faiss.ai/)

> 🔒 **Privacy-First** • 🚀 **Completely Offline** • 🌐 **Network-Ready** • 🧠 **Powered by Llama 3.2 1B**

A high-performance, privacy-focused RAG (Retrieval-Augmented Generation) chatbot with **client-server architecture**. One server PC runs the AI models and processes documents, while multiple client PCs can connect and chat **without installing any dependencies** - just a simple .exe file!

---

## 🎯 Key Features

### 🏗️ Client-Server Architecture
- **Server PC**: Runs AI models, processes documents, hosts knowledge base
- **Client PCs**: Lightweight chat interface via simple .exe file
- **No Installation Required**: Clients need ZERO dependencies - just run the .exe!
- **Multi-Client Support**: Multiple users can connect simultaneously

### 🔒 Privacy & Security
- **100% Offline Operation** - No data sent to external servers
- **Local Network Only** - Data stays within your LAN
- **No Internet Required** - Works completely air-gapped

### ⚡ Performance
- **GPU Accelerated** - CUDA support for fast embeddings
- **Optimized for Llama 3.2 1B** - Efficient 1-billion parameter model
- **Smart Caching** - LRU cache for repeated queries
- **Batch Processing** - Efficient document handling

### 📄 Document Support
- **PDF** - Research papers, reports, manuals
- **DOCX** - Word documents
- **TXT** - Plain text files
- **Multi-Document** - Process multiple files together

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│             SERVER PC (Your Computer)                    │
│  ┌────────────────────┐    ┌─────────────────────┐     │
│  │ bot_server_gui.py  │    │   bot_server.py     │     │
│  │ (Management GUI)   │    │   (RAG Backend)     │     │
│  │                    │    │                     │     │
│  │ • Upload Docs      │───▶│ • Llama 3.2 1B     │     │
│  │ • Process Pipeline │    │ • FAISS Vector DB  │     │
│  │ • Test Queries     │    │ • TCP Server 9999  │     │
│  └────────────────────┘    └─────────────────────┘     │
│         (Optional)              (Required)              │
└─────────────────────────────────────────────────────────┘
                          │
                   (Local Network)
                          │
         ┌────────────────┼────────────────┐
         │                │                │
   ┌─────▼────┐     ┌────▼─────┐    ┌────▼─────┐
   │ PC #1    │     │ PC #2    │    │ PC #3    │
   │          │     │          │    │          │
   │ .exe     │     │ .exe     │    │ .exe     │
   │ (Client) │     │ (Client) │    │ (Client) │
   └──────────┘     └──────────┘    └──────────┘
```

---

## 🛠️ Installation Guide

### Prerequisites

**Server PC Requirements:**
- Windows/Linux/macOS
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- CUDA-capable GPU (optional, but recommended)
- Ollama installed

**Client PC Requirements:**
- Windows/Linux/macOS
- **NO Python, NO dependencies** - just the .exe file!

---

### Step 1: Install Ollama (Server PC Only)

**Download & Install Ollama:**
- Visit [https://ollama.ai/](https://ollama.ai/)
- Download for your platform (Windows/Mac/Linux)
- Install and start the Ollama service

**Pull the Llama 3.2 1B Model:**
```bash
# Open terminal/command prompt
ollama pull llama3.2:1b-instruct-q4_K_M
```

**Verify Installation:**
```bash
ollama list
# Should show: llama3.2:1b-instruct-q4_K_M
```

---

### Step 2: Setup Server PC

**1. Clone/Download the Project:**
```bash
git clone https://github.com/yourusername/offline-rag-chatbot.git
cd offline-rag-chatbot
```

**2. Create Virtual Environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download Embedding Model:**
```bash
# Create directory for embedding model
mkdir -p local_embedding_models

# The embedding model will be auto-downloaded on first run
# Or manually download 'sentence-transformers/all-MiniLM-L6-v2' 
# and place it in: local_embedding_models/all-MiniLM-L6-v2/
```

**5. Create Required Folders:**
```bash
mkdir documents
mkdir vector_db
mkdir removed_doc
```

---

### Step 3: Find Your Server IP Address

**Windows:**
```bash
ipconfig
# Look for "IPv4 Address" under your network adapter
# Example: 192.168.1.100
```

**Linux/Mac:**
```bash
ifconfig
# or
ip addr show
# Look for inet address, e.g., 192.168.1.100
```

**Important:** Note down this IP address - clients will need it!

---

### Step 4: Configure Firewall (Server PC)

**Windows Firewall:**
```powershell
# Run PowerShell as Administrator
netsh advfirewall firewall add rule name="RAG Server" dir=in action=allow protocol=TCP localport=9999
```

**Linux (UFW):**
```bash
sudo ufw allow 9999/tcp
```

**macOS:**
```bash
# System Preferences > Security & Privacy > Firewall > Firewall Options
# Add Python and allow incoming connections
```

---

### Step 5: Setup Client PCs

**Option A: Create .exe File (Recommended)**

On the **server PC**, create a standalone executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Create the client .exe
pyinstaller --onefile --windowed --name "RAG_Chatbot_Client" bot_client.py

# The .exe will be in: dist/RAG_Chatbot_Client.exe
```

**Share the .exe:**
1. Copy `dist/RAG_Chatbot_Client.exe` to a USB drive
2. Give it to your friends/colleagues
3. They just double-click to run - **NO installation needed!**

**Option B: Run Python Script (If Python is installed)**

```bash
python bot_client.py
```

---

## 🚀 Usage Guide

### Server-Side Workflow

#### Step 1: Start Server Management GUI

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Start the management GUI
python bot_server_gui.py
```

**What it does:**
- Loads AI models (one-time, ~2 minutes)
- Provides interface to upload and process documents
- Shows processing progress
- Allows testing queries locally

#### Step 2: Upload Documents

1. Click **"📤 Upload Documents"** button
2. Select PDF, DOCX, or TXT files
3. Documents are copied to `documents/` folder
4. You'll see them in the document list

#### Step 3: Process Documents (Sequential)

**⚠️ IMPORTANT: Steps MUST be completed in order!**

**Step 1: Chunking**
- Click **"Chunking"** button
- Splits documents into manageable chunks
- Progress bar shows completion (0-100%)
- Wait for "✅ Created X chunks..." message

**Step 2: Embeddings**
- Click **"Embeddings"** button (only available after Step 1)
- Prepares embedding model
- Quick step (~10 seconds)

**Step 3: Vector DB**
- Click **"Vector DB"** button (only available after Step 2)
- Creates FAISS vector database
- Generates embeddings (GPU accelerated if available)
- Wait for "✅ Vector DB Created..." message

**After all 3 steps:** The chat interface becomes active!

#### Step 4: Test Queries (Optional)

You can test queries directly in the GUI:
1. Type question in the input box
2. Press Enter or click **"💬 Ask"**
3. See bot response with timing

#### Step 5: Start Network Server

```bash
# In a new terminal (keep GUI running or close it)
python bot_server.py
```

**Output:**
```
🚀 STARTING RAG SERVER (Llama 3.2 1B Instruct)
📡 Server IP: 192.168.1.100
📡 Server Port: 9999

💡 Clients should connect to: 192.168.1.100:9999

✅ SYSTEM READY - Accepting Client Connections
🎧 Listening for TCP connections...
```

**Keep this terminal open** - it's your server!

---

### Client-Side Workflow

#### Method 1: Using .exe File (Recommended)

1. **Double-click** `RAG_Chatbot_Client.exe`
2. **Enter server IP** when prompted (e.g., `192.168.1.100`)
3. Wait for connection (green "✅ Connected" status)
4. **Start chatting!**

#### Method 2: Using Python Script

```bash
python bot_client.py
```

Then follow same steps as .exe method.

#### Using the Client

**Status Indicators:**
- 🟢 **Green**: Connected to server
- 🔴 **Red**: Server unreachable

**Chatting:**
1. Type your question in the input box
2. Press **Enter** or click **"Send ➤"**
3. Wait for bot response (shows response time)
4. Click **"Clear"** to reset conversation

**Example Questions:**
- "What is this document about?"
- "Summarize the main points"
- "What are the key findings?"
- "Explain the methodology used"

---

## 📁 Project Structure

```
offline-rag-chatbot/
├── 📄 bot_server.py           # TCP server backend
├── 📄 bot_server_gui.py       # Server management GUI
├── 📄 bot_client.py           # Client chat interface
├── 📄 requirements.txt        # Python dependencies
├── 📖 README.md              # This file
│
├── 📁 src/                    # Backend modules
│   ├── config.py             # Configuration settings
│   ├── model_loader.py       # LLM and embedding loader
│   ├── document_processor.py # Document loading
│   ├── text_chunker.py       # Text chunking
│   ├── vector_store.py       # FAISS operations
│   └── chatbot.py            # RAG chain logic
│
├── 📁 documents/              # Upload documents here
├── 📁 vector_db/              # Generated vector database
├── 📁 removed_doc/            # Deleted documents backup
└── 📁 local_embedding_models/ # Embedding model storage
```

---

## ⚙️ Configuration

### Model Configuration (src/config.py)

```python
# LLM Model
LLM_MODEL_NAME = "llama3.2:1b-instruct-q4_K_M"

# Embeddings
EMBEDDING_MODEL_NAME = "./local_embedding_models/all-MiniLM-L6-v2"

# Computation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
RETRIEVAL_K = 6

# Network
SERVER_PORT = 9999
```

### Switching Models

**To use a different Ollama model:**

1. Pull the model:
```bash
ollama pull mistral:7b-instruct
```

2. Edit `src/config.py`:
```python
LLM_MODEL_NAME = "mistral:7b-instruct"
```

3. Restart the server

---

## 🐛 Troubleshooting

### Server Issues

**Problem: "Ollama service is not running!"**
```bash
# Solution: Start Ollama
# Windows: Start from Start Menu
# Linux/Mac: ollama serve
```

**Problem: "Model not found!"**
```bash
# Solution: Pull the model
ollama pull llama3.2:1b-instruct-q4_K_M
```

**Problem: "Failed to load embedding model"**
```
Solution:
1. Check internet connection (first run only)
2. Embedding model will auto-download
3. Or manually download and place in: local_embedding_models/
```

**Problem: "CUDA out of memory"**
```python
# Solution: Use CPU instead
# In src/config.py:
DEVICE = 'cpu'
```

### Client Issues

**Problem: "Connection refused"**
```
Solutions:
1. Check server is running (bot_server.py)
2. Verify correct IP address
3. Check firewall allows port 9999
4. Ping server: ping 192.168.1.100
```

**Problem: "Server unreachable"**
```
Solutions:
1. Ensure both PCs on same network
2. Check IP address is correct
3. Try localhost (127.0.0.1) if on same PC
```

**Problem: ".exe won't run on client PC"**
```
Solutions:
1. Check antivirus didn't block it
2. Right-click → Properties → Unblock
3. Run as administrator
```

### Processing Issues

**Problem: "Empty question received"**
```
Solution: Processing steps weren't completed
- Complete all 3 steps in order on server
- Restart bot_server.py
```

**Problem: "I don't have that information"**
```
Solution: Question not covered in documents
- Check documents contain relevant info
- Try rephrasing question
- Upload more relevant documents
```

---

## 📊 Performance Tips

### For Faster Processing

1. **Use GPU**: Install CUDA and PyTorch with GPU support
2. **Reduce Chunk Size**: Edit `CHUNK_SIZE` in config.py (600-800)
3. **Lower Retrieval K**: Reduce `RETRIEVAL_K` to 4-5
4. **Smaller Model**: Use `llama3.2:1b-instruct-q4_K_M` (already set)

### For Better Accuracy

1. **Increase Retrieval K**: Set to 8-10 in config.py
2. **Larger Chunks**: Increase `CHUNK_SIZE` to 1000-1200
3. **More Overlap**: Increase `CHUNK_OVERLAP` to 200-250
4. **Larger Model**: Use `llama3.2:3b-instruct` or `llama3:8b-instruct`

---

## 🔒 Security Considerations

### Network Security

- Server binds to `0.0.0.0` (all interfaces) - **only use on trusted networks**
- No authentication implemented - anyone on LAN can connect
- No encryption - data sent in plain text over TCP

**For production use, consider:**
- VPN for remote access
- SSH tunneling
- Authentication layer
- TLS/SSL encryption

### Data Privacy

- All data stays local - nothing sent to internet
- Documents stored unencrypted in `documents/` folder
- Vector DB stored in `vector_db/` folder
- Deleted docs moved to `removed_doc/` (not permanently deleted)

---

## 🎓 Example Use Cases

### 1. Academic Research
```
Documents: Research papers, theses
Questions:
- "What methodology did the authors use?"
- "Summarize the key findings"
- "What are the limitations mentioned?"
```

### 2. Technical Documentation
```
Documents: User manuals, API docs
Questions:
- "How do I configure X?"
- "What are the system requirements?"
- "Explain the installation process"
```

### 3. Legal Documents
```
Documents: Contracts, policies
Questions:
- "What are the termination clauses?"
- "Explain the liability section"
- "What are my obligations?"
```

### 4. Medical Records
```
Documents: Reports, studies
Questions:
- "What were the test results?"
- "What treatment was recommended?"
- "What are the side effects?"
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork the repository**
2. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

**Areas for contribution:**
- Authentication system
- Web-based client interface
- Support for more document types
- Multi-language support
- Performance optimizations

---

## 📝 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

- **Ollama Team** - For making local LLMs accessible
- **Meta AI** - For Llama 3.2 models
- **LangChain** - For the RAG framework
- **FAISS** - For efficient vector search
- **HuggingFace** - For embedding models
- **Python Community** - For the amazing ecosystem

---

## 📧 Support

**Need help?**
- 📖 Check this README thoroughly
- 🐛 Check Troubleshooting section
- 💬 Open an issue on GitHub
- 📧 Email: your-email@example.com

---

## 🗺️ Roadmap

### Version 1.1 (Coming Soon)
- [ ] Web-based client interface
- [ ] User authentication
- [ ] Session management
- [ ] Query history

### Version 1.2 (Future)
- [ ] Multi-user document permissions
- [ ] Real-time collaboration
- [ ] Mobile app clients
- [ ] Cloud deployment option

---

<div align="center">

**⭐ Star this project if you find it helpful!**

Made with ❤️ for Privacy-Conscious Users

[Report Bug](https://github.com/yourusername/offline-rag-chatbot/issues) • [Request Feature](https://github.com/yourusername/offline-rag-chatbot/issues)

</div>

---

## 📸 Screenshots

### Server Management GUI
![Server GUI](assets/server_gui.png)
*Upload documents, run processing pipeline, and test queries*

### Client Chat Interface
![Client GUI](assets/client_gui.png)
*Lightweight chat interface for asking questions*

### Terminal Output
![Terminal](assets/terminal.png)
*Server backend showing active connections*

---

## 🎯 Quick Start Checklist

### Server Setup
- [ ] Install Ollama
- [ ] Pull Llama 3.2 1B model
- [ ] Install Python 3.8+
- [ ] Install requirements
- [ ] Find your IP address
- [ ] Configure firewall
- [ ] Start `bot_server_gui.py`
- [ ] Upload documents
- [ ] Run processing (Steps 1→2→3)
- [ ] Start `bot_server.py`

### Client Setup
- [ ] Get .exe file from server PC
- [ ] Know server IP address
- [ ] Double-click .exe
- [ ] Enter server IP
- [ ] Start chatting!

---

**That's it! You now have a fully functional offline RAG chatbot with client-server architecture! 🎉**