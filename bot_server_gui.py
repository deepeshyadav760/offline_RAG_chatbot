# bot_server_gui.py

import os
import sys

# === CRITICAL: Proxy fix for corporate networks ===
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import shutil
from pathlib import Path
import time

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from src.config import DOCUMENTS_DIR, VECTOR_INDEX_NAME, RETRIEVAL_K, VECTOR_DB_PATH
    from src.model_loader import load_llm, get_embedding_model, preload_models
    from src.document_processor import load_document, load_documents_batch
    from src.text_chunker import chunk_text
    from src.vector_store import create_vector_store, load_vector_store
    from src.chatbot import create_chatbot, ask_question, clear_query_cache
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import backend modules: {e}")
    sys.exit(1)


class ServerSideRAGGUI:
    """
    Server-Side RAG Chatbot GUI - Document Management & Processing
    """
    
    def __init__(self, master):
        self.master = master
        master.title("RAG Server Control Panel (Llama 3.2 1B Instruct)")
        master.geometry("1200x800")
        master.resizable(True, True)
        
        # Colors
        self.bg_color = '#F5F5F5'
        self.primary_color = '#2196F3'
        self.success_color = '#4CAF50'
        self.warning_color = '#FF9800'
        self.danger_color = '#F44336'
        
        master.configure(bg=self.bg_color)
        
        # Backend variables
        self.documents_dir = DOCUMENTS_DIR
        self.vector_db_index = VECTOR_INDEX_NAME
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        
        # Processing state
        self.loaded_documents = []
        self.text_chunks = []
        self.embeddings_ready = False
        self.models_preloaded = False
        self.query_cancelled = False
        
        # Session step tracking (forces re-processing on document changes)
        self.session_steps_completed = {
            'chunking': False,
            'embeddings': False,
            'vectordb': False
        }
        
        # Processing lock to prevent concurrent operations
        self._processing_lock = threading.Lock()
        self.is_processing = False
        
        # Timer for query processing
        self._timer_running = False
        
        # Create UI
        self._create_ui()
        
        # Initialize backend in background
        self._init_backend_async()
    
    def _create_ui(self):
        """Create the main UI layout"""
        main_container = tk.Frame(self.master, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Document Management
        left_panel = tk.Frame(main_container, bg='white', relief='ridge', bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        left_panel.configure(width=400)
        
        # Right Panel - Chat Interface
        right_panel = tk.Frame(main_container, bg='white', relief='ridge', bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self._create_left_panel(left_panel)
        self._create_right_panel(right_panel)
    
    def _create_left_panel(self, parent):
        """Create document management controls"""
        header = tk.Label(parent, text="📚 Knowledge Base", 
                         font=('Arial', 14, 'bold'), bg='white', fg='#333')
        header.pack(pady=(10, 5), padx=10, anchor='w')
        
        self.upload_btn = tk.Button(
            parent, 
            text="📤 Upload Documents", 
            command=self._upload_document,
            font=('Arial', 10, 'bold'), 
            bg=self.primary_color, 
            fg='white',
            relief='flat', 
            padx=20, 
            pady=8, 
            cursor='hand2', 
            state='disabled'
        )
        self.upload_btn.pack(pady=5, padx=10, fill=tk.X)
        
        # Document list
        list_frame = tk.Frame(parent, bg='white')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.doc_listbox = tk.Listbox(
            list_frame, 
            yscrollcommand=scrollbar.set,
            font=('Arial', 10), 
            bg='#FAFAFA', 
            selectmode=tk.EXTENDED, 
            relief='flat', 
            bd=1
        )
        self.doc_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.doc_listbox.yview)
        
        self.delete_btn = tk.Button(
            parent, 
            text="🗑️ Remove Selected", 
            command=self._delete_selected,
            font=('Arial', 9), 
            bg=self.danger_color, 
            fg='white',
            relief='flat', 
            padx=15, 
            pady=5, 
            cursor='hand2', 
            state='disabled'
        )
        self.delete_btn.pack(pady=5, padx=10, fill=tk.X)
        
        # Processing steps
        steps_label = tk.Label(parent, text="⚙️ Processing Pipeline", 
                              font=('Arial', 12, 'bold'), bg='white', fg='#333')
        steps_label.pack(pady=(15, 10), padx=10, anchor='w')
        
        self._create_step_button(parent, "Step 1", "Chunking", self._do_chunking, '#00BCD4')
        self._create_step_button(parent, "Step 2", "Embeddings", self._do_embeddings, '#9C27B0')
        self._create_step_button(parent, "Step 3", "Vector DB", self._do_vector_db, '#4CAF50')
        
        # Status label
        self.status_label = tk.Label(
            parent, 
            text="⏳ Initializing system...", 
            font=('Arial', 9, 'italic'), 
            bg='white', 
            fg='#666', 
            wraplength=350,
            justify=tk.LEFT
        )
        self.status_label.pack(pady=10, padx=10, fill=tk.X)
    
    def _create_step_button(self, parent, step_text, button_text, command, color):
        """Create a processing step button with progress bar"""
        step_frame = tk.Frame(parent, bg='white')
        step_frame.pack(fill=tk.X, padx=10, pady=8)
        
        step_label = tk.Label(
            step_frame, 
            text=step_text, 
            font=('Arial', 10), 
            bg='white', 
            fg='#666', 
            width=8
        )
        step_label.pack(side=tk.LEFT)
        
        # Progress bar container
        progress_container = tk.Frame(step_frame, bg='white')
        progress_container.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Style for progress bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Custom.Horizontal.TProgressbar", 
            troughcolor='#E0E0E0', 
            background=color, 
            thickness=20
        )
        
        progress = ttk.Progressbar(
            progress_container, 
            style="Custom.Horizontal.TProgressbar",
            mode='determinate', 
            length=150
        )
        progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        progress['value'] = 0
        
        # Percentage label
        percent_label = tk.Label(
            progress_container, 
            text="0%", 
            font=('Arial', 9, 'bold'), 
            bg='white', 
            fg=color, 
            width=5
        )
        percent_label.pack(side=tk.LEFT, padx=5)
        
        # Action button
        btn = tk.Button(
            step_frame, 
            text=button_text, 
            command=command,
            font=('Arial', 10, 'bold'), 
            bg=color, 
            fg='white',
            relief='flat', 
            padx=15, 
            pady=6, 
            cursor='hand2', 
            width=12, 
            state='disabled'
        )
        btn.pack(side=tk.RIGHT)
        
        # Store references
        if button_text == "Chunking":
            self.chunking_btn = btn
            self.chunking_progress = progress
            self.chunking_percent_label = percent_label
        elif button_text == "Embeddings":
            self.embeddings_btn = btn
            self.embeddings_progress = progress
            self.embeddings_percent_label = percent_label
        elif button_text == "Vector DB":
            self.vectordb_btn = btn
            self.vectordb_progress = progress
            self.vectordb_percent_label = percent_label

    def _create_right_panel(self, parent):
        """Create chat interface for testing"""
        header_frame = tk.Frame(parent, bg='white')
        header_frame.pack(pady=(10, 5), padx=10, fill=tk.X)

        header = tk.Label(
            header_frame, 
            text="💬 Test Chat Interface", 
            font=('Arial', 14, 'bold'), 
            bg='white', 
            fg='#333'
        )
        header.pack(side=tk.LEFT)

        self.clear_btn = tk.Button(
            header_frame, 
            text="🔄 Clear", 
            command=self._clear_chat,
            font=('Arial', 9, 'bold'), 
            bg=self.warning_color, 
            fg='white',
            relief='flat', 
            padx=15, 
            pady=5, 
            cursor='hand2'
        )
        self.clear_btn.pack(side=tk.RIGHT)
        
        # Chat history
        chat_frame = tk.Frame(parent, bg='white')
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_history = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            state='disabled',
            font=('Arial', 10), 
            bg='#FAFAFA', 
            fg='#333',
            relief='flat', 
            bd=1, 
            padx=10, 
            pady=10
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags
        self.chat_history.tag_config('user', foreground='#1976D2', font=('Arial', 10, 'bold'))
        self.chat_history.tag_config('bot', foreground='#388E3C', font=('Arial', 10))
        self.chat_history.tag_config('system', foreground='#F57C00', font=('Arial', 9, 'italic'))
        self.chat_history.tag_config('error', foreground='#D32F2F', font=('Arial', 10, 'bold'))
        self.chat_history.tag_config('success', foreground='#4CAF50', font=('Arial', 9, 'bold'))
        self.chat_history.tag_config('timer', foreground='#9E9E9E', font=('Arial', 9, 'italic'))

        # Input frame
        input_frame = tk.Frame(parent, bg='white')
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        query_frame = tk.Frame(input_frame, bg='white')
        query_frame.pack(fill=tk.X)

        self.query_entry = tk.Entry(
            query_frame, 
            font=('Arial', 11), 
            relief='flat', 
            bd=1, 
            bg='#FAFAFA'
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.query_entry.bind('<Return>', lambda e: self._ask_question())

        # Placeholder functionality
        self.placeholder_text = "Enter your test question here..."
        self.query_entry.insert(0, self.placeholder_text)
        self.query_entry.config(fg='#999999')
        self.placeholder_active = True

        self.query_entry.bind('<FocusIn>', self._on_entry_click)
        self.query_entry.bind('<FocusOut>', self._on_focus_out)

        self.ask_btn = tk.Button(
            query_frame, 
            text="💬 Ask", 
            command=self._ask_question,
            font=('Arial', 10, 'bold'), 
            bg=self.success_color, 
            fg='white', 
            relief='flat', 
            padx=25, 
            pady=8, 
            cursor='hand2',
            state='disabled'
        )
        self.ask_btn.pack(side=tk.RIGHT)

    # Placeholder handlers
    def _on_entry_click(self, event):
        if self.placeholder_active:
            self.query_entry.delete(0, tk.END)
            self.query_entry.config(fg='#333333')
            self.placeholder_active = False

    def _on_focus_out(self, event):
        if self.query_entry.get() == "":
            self.query_entry.insert(0, self.placeholder_text)
            self.query_entry.config(fg='#999999')
            self.placeholder_active = True

    # Backend initialization
    def _init_backend_async(self):
        self._update_status("⏳ Loading AI models... Please wait.", '#FF9800')
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            success = preload_models()
            if success:
                self.llm = load_llm()
                self.vector_store = load_vector_store(self.vector_db_index)
                self.master.after(0, self._on_models_loaded)
            else:
                self.master.after(0, self._on_models_failed)
        except Exception as e:
            self.master.after(0, lambda: self._on_models_failed(str(e)))
    
    def _on_models_loaded(self):
        """Called when models are successfully loaded"""
        self.models_preloaded = True
        
        # Enable UI controls
        self.upload_btn.config(state='normal')
        self.delete_btn.config(state='normal')
        self.chunking_btn.config(state='normal')
        self.embeddings_btn.config(state='normal')
        self.vectordb_btn.config(state='normal')
        
        # Refresh document list
        self._refresh_document_list()
        
        # Check if all processing steps were completed in this session
        all_steps_done = (
            self.session_steps_completed['chunking'] and 
            self.session_steps_completed['embeddings'] and 
            self.session_steps_completed['vectordb']
        )
        
        if all_steps_done and self.qa_chain:
            self.ask_btn.config(state='normal')
            self._update_status("✅ System ready! Knowledge base is up-to-date.", '#4CAF50')
            self._append_chat("System: All processing steps completed. Ready to answer questions! 🚀", 'success')
        else:
            # Disable chat on startup - require fresh processing
            self.ask_btn.config(state='disabled')
            
            # Check if old vector DB exists
            db_path = os.path.join(VECTOR_DB_PATH, VECTOR_INDEX_NAME)
            if os.path.exists(db_path):
                self._update_status("⚠️ Old database found. Please re-process documents (Steps 1→2→3).", '#FF9800')
                self._append_chat(
                    "System: Models loaded. An old vector database was found, but you must complete "
                    "all processing steps (1→2→3) with current documents before testing.", 
                    'system'
                )
            else:
                self._update_status("✅ Models loaded. Upload & process documents to enable chat.", '#2196F3')
                self._append_chat(
                    "System: Models loaded. Please upload documents and complete all processing steps "
                    "(1→2→3) to enable chat.", 
                    'system'
                )
    
    def _on_models_failed(self, error_msg="Unknown error"):
        self._update_status(f"❌ Failed to load models: {error_msg}", '#F44336')
        messagebox.showerror("Initialization Failed", 
                           f"Failed to load models:\n\n{error_msg}\n\n"
                           "Please check:\n"
                           "1. Ollama is running: ollama serve\n"
                           "2. Model is pulled: ollama pull llama3.2:1b-instruct-q4_K_M\n"
                           "3. Embedding model exists in: local_embedding_models/")

    # Continue in next artifact...
    # Document management functions
    def _refresh_document_list(self):
        os.makedirs(self.documents_dir, exist_ok=True)
        self.doc_listbox.delete(0, tk.END)
        files = [f for f in os.listdir(self.documents_dir) 
                if f.lower().endswith(('.pdf', '.docx', '.txt'))]
        for file in sorted(files):
            self.doc_listbox.insert(tk.END, file)
    
    def _upload_document(self):
        """Handle document upload with atomic operations"""
        file_paths = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=[
                ("Supported Files", "*.pdf *.docx *.txt"),
                ("PDF Files", "*.pdf"),
                ("Word Documents", "*.docx"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        os.makedirs(self.documents_dir, exist_ok=True)
        uploaded_count = 0
        failed_files = []
        
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                destination = os.path.join(self.documents_dir, filename)
                
                if os.path.exists(destination):
                    response = messagebox.askyesno(
                        "File Exists",
                        f"'{filename}' already exists. Overwrite?"
                    )
                    if not response:
                        continue
                
                # Atomic copy operation
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f".temp_{filename}")
                
                shutil.copy2(file_path, temp_path)
                shutil.move(temp_path, destination)
                uploaded_count += 1
                
            except Exception as e:
                failed_files.append(f"{filename}: {str(e)}")
        
        if uploaded_count > 0:
            self._refresh_document_list()
            self._update_status(f"✅ Uploaded {uploaded_count} document(s) successfully.", '#4CAF50')
            self._append_chat(f"System: Uploaded {uploaded_count} new document(s).", 'system')
            self._reset_processing_steps("new documents added")
        
        if failed_files:
            messagebox.showerror(
                "Upload Errors",
                f"Failed to upload:\n\n" + "\n".join(failed_files[:5])
            )
    
    def _delete_selected(self):
        """Move selected documents to removed_doc folder"""
        selected_indices = self.doc_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select documents to remove.")
            return
        
        selected_files = [self.doc_listbox.get(i) for i in selected_indices]
        
        if not messagebox.askyesno("Confirm Remove", 
                                f"Move {len(selected_files)} document(s) to removed_doc folder?"):
            return
        
        project_root = os.path.dirname(os.path.abspath(self.documents_dir))
        removed_doc_dir = os.path.join(project_root, 'removed_doc')
        os.makedirs(removed_doc_dir, exist_ok=True)
        
        moved_count = 0
        failed_files = []
        
        for filename in selected_files:
            try:
                source_path = os.path.join(self.documents_dir, filename)
                dest_path = os.path.join(removed_doc_dir, filename)
                
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    dest_path = os.path.join(removed_doc_dir, f"{name}_{timestamp}{ext}")
                
                shutil.move(source_path, dest_path)
                moved_count += 1
                
            except Exception as e:
                failed_files.append(f"{filename}: {str(e)}")
        
        if moved_count > 0:
            self._refresh_document_list()
            self._update_status(f"📦 Moved {moved_count} document(s) to removed_doc folder.", '#FF9800')
            self._append_chat(f"System: Moved {moved_count} document(s) to removed_doc.", 'system')
            self._reset_processing_steps("documents removed")
        
        if failed_files:
            messagebox.showerror(
                "Move Errors",
                f"Failed to move:\n\n" + "\n".join(failed_files[:5])
            )

    # Processing steps
    def _do_chunking(self):
        if self.is_processing:
            return
        
        files = list(self.doc_listbox.get(0, tk.END))
        if not files:
            messagebox.showwarning("No Documents", "Please upload documents first.")
            return
        
        with self._processing_lock:
            self.is_processing = True
            self._update_status("⚙️ Chunking documents...", '#00BCD4')
            self.chunking_btn.config(state='disabled')
            self.chunking_progress['value'] = 0
            self.chunking_percent_label.config(text="0%")
            
            threading.Thread(target=self._run_chunking, args=(files,), daemon=True).start()

    def _run_chunking(self, files):
        try:
            paths = [os.path.join(self.documents_dir, f) for f in files]
            total_files = len(paths)
            
            self.loaded_documents = []
            
            for i, path in enumerate(paths):
                docs = load_document(path)
                self.loaded_documents.extend(docs)
                progress = int(((i + 1) / total_files) * 50)
                self.master.after(0, lambda p=progress: self._update_progress(
                    self.chunking_progress, self.chunking_percent_label, p))
            
            self.master.after(0, lambda: self._update_progress(
                self.chunking_progress, self.chunking_percent_label, 55))
            
            self.text_chunks = chunk_text(self.loaded_documents)
            
            for progress in range(60, 100, 5):
                self.master.after(0, lambda p=progress: self._update_progress(
                    self.chunking_progress, self.chunking_percent_label, p))
                time.sleep(0.1)
            
            self.master.after(0, lambda: self._update_progress(
                self.chunking_progress, self.chunking_percent_label, 100))
            
            self.master.after(0, lambda: self._finish_step(
                self.chunking_btn, self.chunking_progress, 
                f"Created {len(self.text_chunks)} chunks from {total_files} document(s).", 
                self.chunking_percent_label, 'chunking'))
                
        except Exception as e:
            self.master.after(0, lambda: self._fail_step(
                self.chunking_btn, self.chunking_progress, str(e), 
                self.chunking_percent_label))

    def _do_embeddings(self):
        if self.is_processing:
            return
        
        if not self.text_chunks:
            messagebox.showwarning(
                "Step 1 Required", 
                "Please complete Step 1 (Chunking) first!\n\n"
                "Processing steps must be completed sequentially."
            )
            return
        
        with self._processing_lock:
            self.is_processing = True
            self._update_status("⚙️ Preparing embeddings...", '#9C27B0')
            self.embeddings_btn.config(state='disabled')
            self.embeddings_progress['value'] = 0
            self.embeddings_percent_label.config(text="0%")
            
            threading.Thread(target=self._run_embeddings, daemon=True).start()

    def _run_embeddings(self):
        try:
            self.master.after(0, lambda: self._update_progress(
                self.embeddings_progress, self.embeddings_percent_label, 30))
            
            if get_embedding_model():
                self.master.after(0, lambda: self._update_progress(
                    self.embeddings_progress, self.embeddings_percent_label, 70))
                self.embeddings_ready = True
                
                self.master.after(0, lambda: self._update_progress(
                    self.embeddings_progress, self.embeddings_percent_label, 100))
                
                self.master.after(0, lambda: self._finish_step(
                    self.embeddings_btn, self.embeddings_progress, 
                    "Embeddings ready.", 
                    self.embeddings_percent_label, 'embeddings'))
            else:
                raise Exception("Embedding model error")
        except Exception as e:
            self.master.after(0, lambda: self._fail_step(
                self.embeddings_btn, self.embeddings_progress, str(e), 
                self.embeddings_percent_label))

    def _do_vector_db(self):
        if self.is_processing:
            return
        
        if not self.text_chunks:
            messagebox.showwarning(
                "Step 1 Required", 
                "Please complete Step 1 (Chunking) first!"
            )
            return
        
        if not self.embeddings_ready:
            messagebox.showwarning(
                "Step 2 Required", 
                "Please complete Step 2 (Embeddings) first!"
            )
            return
        
        with self._processing_lock:
            self.is_processing = True
            self._update_status("⚙️ Creating Vector DB...", '#4CAF50')
            self.vectordb_btn.config(state='disabled')
            self.vectordb_progress['value'] = 0
            self.vectordb_percent_label.config(text="0%")
            
            threading.Thread(target=self._run_vectordb, daemon=True).start()

    def _run_vectordb(self):
        try:
            for progress in range(0, 16, 3):
                self.master.after(0, lambda p=progress: self._update_progress(
                    self.vectordb_progress, self.vectordb_percent_label, p))
                time.sleep(0.1)
            
            def progress_callback(current, total):
                percent = 15 + int((current / total) * 60)
                self.master.after(0, lambda p=percent: self._update_progress(
                    self.vectordb_progress, self.vectordb_percent_label, p))
            
            self.vector_store = create_vector_store(
                self.text_chunks, 
                index_name=self.vector_db_index,
                progress_callback=progress_callback
            )
            
            for progress in range(76, 86, 2):
                self.master.after(0, lambda p=progress: self._update_progress(
                    self.vectordb_progress, self.vectordb_percent_label, p))
                time.sleep(0.1)
            
            self.qa_chain = create_chatbot(self.llm, self.vector_store, k=RETRIEVAL_K)
            
            for progress in range(86, 100, 3):
                self.master.after(0, lambda p=progress: self._update_progress(
                    self.vectordb_progress, self.vectordb_percent_label, p))
                time.sleep(0.1)
            
            clear_query_cache()
            
            self.master.after(0, lambda: self._update_progress(
                self.vectordb_progress, self.vectordb_percent_label, 100))
            
            self.master.after(0, self._on_vectordb_complete)
            
        except Exception as e:
            self.master.after(0, lambda: self._fail_step(
                self.vectordb_btn, self.vectordb_progress, str(e), 
                self.vectordb_percent_label))

    # Continue in final artifact...
    def _finish_step(self, btn, progress, msg, percent_label=None, step_name=None):
        """Complete a processing step"""
        progress.stop()
        progress['value'] = 100
        if percent_label:
            percent_label.config(text="100%")
        btn.config(state='normal')
        
        if step_name:
            self.session_steps_completed[step_name] = True
        
        self.is_processing = False
        self._update_status(f"✅ {msg}", '#4CAF50')
        self._append_chat(f"System: {msg}", 'system')

    def _fail_step(self, btn, progress, err, percent_label=None):
        progress['value'] = 0
        if percent_label:
            percent_label.config(text="0%")
        btn.config(state='normal')
        self.is_processing = False
        self._update_status(f"❌ Error: {err}", '#F44336')
        messagebox.showerror("Processing Error", f"An error occurred:\n\n{err}")

    def _on_vectordb_complete(self):
        self._finish_step(
            self.vectordb_btn, 
            self.vectordb_progress, 
            "Vector DB Created. Knowledge base is ready!", 
            self.vectordb_percent_label, 'vectordb'
        )
        
        if (self.session_steps_completed['chunking'] and 
            self.session_steps_completed['embeddings'] and 
            self.session_steps_completed['vectordb']):
            
            self.ask_btn.config(state='normal')
            messagebox.showinfo("Success", 
                              "Knowledge base is ready!\n\n"
                              "You can now:\n"
                              "• Test queries in the chat interface\n"
                              "• Start bot_server.py for client connections")
            self._append_chat("System: ✅ All steps completed! Chat enabled.", 'success')

    def _finish_step(self, btn, progress, msg, percent_label=None, step_name=None):
        progress.stop()
        progress['value'] = 100
        if percent_label:
            percent_label.config(text="100%")
        btn.config(state='normal')
        
        if step_name:
            self.session_steps_completed[step_name] = True
        
        self.is_processing = False
        self._update_status(f"✅ {msg}", '#4CAF50')
        self._append_chat(f"System: {msg}", 'system')

    def _fail_step(self, btn, progress, err, percent_label=None):
        progress.stop()
        progress['value'] = 0
        if percent_label:
            percent_label.config(text="0%")
        btn.config(state='normal')
        self.is_processing = False
        self._update_status(f"❌ Error: {err}", '#F44336')
        self._append_chat(f"Error: {err}", 'error')

    def _on_vectordb_complete(self):
        self._finish_step(
            self.vectordb_btn, 
            self.vectordb_progress, 
            "Vector DB Created. Knowledge base is ready!", 
            self.vectordb_percent_label, 
            'vectordb'
        )
        
        if (self.session_steps_completed['chunking'] and 
            self.session_steps_completed['embeddings'] and 
            self.session_steps_completed['vectordb']):
            
            self.ask_btn.config(state='normal')
            messagebox.showinfo("Success", 
                              "✅ Knowledge base is ready!\n\n"
                              "You can now:\n"
                              "• Test queries in the chat interface\n"
                              "• Start bot_server.py for network clients")
            self._append_chat("System: ✅ All steps completed! Chat enabled.", 'success')
    
    def _finish_step(self, btn, progress, msg, percent_label, step_name):
        progress.stop()
        progress['value'] = 100
        percent_label.config(text="100%")
        btn.config(state='normal')
        
        self.session_steps_completed[step_name] = True
        self.is_processing = False
        
        self._update_status(f"✅ {msg}", '#4CAF50')
        self._append_chat(f"System: {msg}", 'system')

    def _fail_step(self, btn, progress, err, percent_label=None):
        progress.stop()
        progress['value'] = 0
        if percent_label:
            percent_label.config(text="0%")
        btn.config(state='normal')
        self.is_processing = False
        self._update_status(f"❌ Error: {err}", '#F44336')
        self._append_chat(f"System: Error - {err}", 'error')

    def _finish_step(self, btn, progress, msg, percent_label=None, step_name=None):
        progress.stop()
        progress['value'] = 100
        if percent_label:
            percent_label.config(text="100%")
        btn.config(state='normal')
        
        if step_name:
            self.session_steps_completed[step_name] = True
        
        self.is_processing = False
        self._update_status(f"✅ {msg}", '#4CAF50')
        self._append_chat(f"System: {msg}", 'system')

    def _fail_step(self, btn, progress, err, percent_label=None):
        progress['value'] = 0
        if percent_label:
            percent_label.config(text="0%")
        btn.config(state='normal')
        self.is_processing = False
        self._update_status(f"❌ Error: {err}", '#F44336')
        self._append_chat(f"System: Error - {err}", 'error')

    def _on_vectordb_complete(self):
        self._finish_step(
            self.vectordb_btn, 
            self.vectordb_progress, 
            "Vector DB Created. Knowledge base ready!", 
            self.vectordb_percent_label,
            'vectordb'
        )
        
        if (self.session_steps_completed['chunking'] and 
            self.session_steps_completed['embeddings'] and 
            self.session_steps_completed['vectordb']):
            
            self.ask_btn.config(state='normal')
            messagebox.showinfo("Success", 
                              "✅ Knowledge base is ready!\n\n"
                              "You can now:\n"
                              "1. Test queries in the chat interface\n"
                              "2. Start bot_server.py for client connections")
            self._append_chat("System: ✅ All steps completed! You can now test queries.", 'success')

    def _finish_step(self, btn, progress, msg, percent_label=None, step_name=None):
        progress['value'] = 100
        if percent_label:
            percent_label.config(text="100%")
        btn.config(state='normal')
        
        if step_name:
            self.session_steps_completed[step_name] = True
        
        self.is_processing = False
        self._update_status(f"✅ {msg}", '#4CAF50')
        self._append_chat(f"System: {msg}", 'system')

    def _fail_step(self, btn, progress, err, percent_label=None):
        progress['value'] = 0
        if percent_label:
            percent_label.config(text="0%")
        btn.config(state='normal')
        self.is_processing = False
        self._update_status(f"❌ Error: {err}", '#F44336')

    def _on_vectordb_complete(self):
        self._finish_step(
            self.vectordb_btn, 
            self.vectordb_progress, 
            "Vector DB Created. Ready for chat!", 
            self.vectordb_percent_label,
            'vectordb'
        )
        
        if (self.session_steps_completed['chunking'] and 
            self.session_steps_completed['embeddings'] and 
            self.session_steps_completed['vectordb']):
            
            self.ask_btn.config(state='normal')
            messagebox.showinfo("Success", 
                              "✅ Knowledge base is ready!\n\n"
                              "You can now:\n"
                              "1. Test queries in this GUI\n"
                              "2. Start bot_server.py for network access")
            self._append_chat("System: ✅ All steps completed! Ready for queries.", 'success')

    # Chat functionality
    def _ask_question(self):
        if self.is_processing or self.placeholder_active:
            return

        question = self.query_entry.get().strip()
        if not question:
            return
        
        if not self.qa_chain:
            messagebox.showwarning("Not Ready", "Please complete processing steps first.")
            return
        
        self.is_processing = True
        self.ask_btn.config(state='disabled')
        self._update_status("🔍 Searching knowledge base...", '#2196F3')

        self._append_chat(f"You: {question}", 'user')
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, self.placeholder_text)
        self.query_entry.config(fg='#999999')
        self.placeholder_active = True
        
        self.chat_history.config(state='normal')
        self._timer_mark = self.chat_history.index(tk.END + "-1c")
        self.chat_history.insert(tk.END, '⏱️ 0.0s\n\n', 'timer')
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)
        
        threading.Thread(target=self._process_query, args=(question,), daemon=True).start()

    def _process_query(self, question):
        start_time = time.time()
        self._timer_running = True
        
        def update_timer():
            if self._timer_running:
                try:
                    elapsed = time.time() - start_time
                    self.chat_history.config(state='normal')
                    self.chat_history.delete(f"{self._timer_mark}", f"{self._timer_mark} lineend")
                    self.chat_history.insert(self._timer_mark, f'⏱️ {elapsed:.1f}s', 'timer')
                    self.chat_history.config(state='disabled')
                    self.master.after(100, update_timer)
                except:
                    pass
        
        self.master.after(100, update_timer)
        
        try:
            response = ask_question(self.qa_chain, question)
            self._timer_running = False
            self.master.after(0, lambda: self._display_answer(response))
        except Exception as e:
            self._timer_running = False
            self.master.after(0, lambda: self._display_answer(str(e), is_error=True))

    def _display_answer(self, response, is_error=False):
        if is_error:
            self._append_chat(f"Bot: ❌ {response}", 'error')
            self._update_status("❌ Error occurred.", '#F44336')
        else:
            self._append_chat(f"Bot: {response}", 'bot')
            self._update_status("✅ Answer delivered.", '#4CAF50')
        
        self.is_processing = False
        self.ask_btn.config(state='normal')
        self.query_entry.focus()

    def _clear_chat(self):
        self.chat_history.config(state='normal')
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state='disabled')

    # Helper functions
    def _update_progress(self, progress_bar, percent_label, value):
        progress_bar['value'] = value
        percent_label.config(text=f"{int(value)}%")
        self.master.update_idletasks()

    def _finish_step(self, btn, progress, msg, percent_label=None, step_name=None):
        progress['value'] = 100
        if percent_label:
            percent_label.config(text="100%")
        btn.config(state='normal')
        
        if step_name:
            self.session_steps_completed[step_name] = True
        
        self.is_processing = False
        self._update_status(f"✅ {msg}", '#4CAF50')
        self._append_chat(f"System: {msg}", 'system')

    def _append_chat(self, message, tag=None):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, message + '\n\n', tag)
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)
    
    def _update_status(self, message, color='#666'):
        self.status_label.config(text=message, fg=color)

    def _reset_processing_steps(self, reason="document change"):
        self.loaded_documents = []
        self.text_chunks = []
        self.embeddings_ready = False
        self.qa_chain = None
        
        self.session_steps_completed = {
            'chunking': False,
            'embeddings': False,
            'vectordb': False
        }
        
        self.chunking_progress['value'] = 0
        self.chunking_percent_label.config(text="0%")
        self.embeddings_progress['value'] = 0
        self.embeddings_percent_label.config(text="0%")
        self.vectordb_progress['value'] = 0
        self.vectordb_percent_label.config(text="0%")
        
        self.ask_btn.config(state='disabled')
        
        self._update_status(f"⚠️ Processing reset: {reason}. Complete steps 1→2→3.", '#FF9800')
        self._append_chat(f"System: ⚠️ {reason.capitalize()}! Re-process documents (Steps 1→2→3).", 'system')

def start_server_gui():
    root = tk.Tk()
    app = ServerSideRAGGUI(root)
    root.mainloop()

if __name__ == "__main__":
    start_server_gui()