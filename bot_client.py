# bot_client.py

import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
import socket
import json
import threading
import sys

# Default Configuration
DEFAULT_SERVER_IP = "127.0.0.1"  # localhost for testing
SERVER_PORT = 9999
BUFFER_SIZE = 16384

class TCPClientGUI:
    def __init__(self, master):
        self.master = master
        
        # Ask user for server IP on startup
        self.server_ip = simpledialog.askstring(
            "Server Connection",
            "Enter RAG Server IP Address:",
            initialvalue=DEFAULT_SERVER_IP
        )
        
        # Robustness: Remove port if user added it (e.g. 1.2.3.4:9999 -> 1.2.3.4)
        if self.server_ip and ':' in self.server_ip:
            self.server_ip = self.server_ip.split(':')[0]
        
        if not self.server_ip:
            messagebox.showerror("Error", "Server IP is required!")
            master.destroy()
            return
        
        master.title(f"RAG Chatbot Client - {self.server_ip}:{SERVER_PORT}")
        master.geometry("1000x700")
        
        # Colors
        self.bg_color = '#F5F5F5'
        self.user_color = '#E3F2FD'
        self.bot_color = '#FFFFFF'
        self.system_color = '#FFF9C4'
        
        master.configure(bg=self.bg_color)
        self._create_ui()
        self._check_connection_async()

    def _create_ui(self):
        # Header
        top_frame = tk.Frame(self.master, bg='#263238', height=40)
        top_frame.pack(fill=tk.X)
        
        self.status_lbl = tk.Label(
            top_frame, 
            text=f"Connecting to {self.server_ip}:{SERVER_PORT}...", 
            fg="yellow", 
            bg="#263238", 
            font=("Arial", 10)
        )
        self.status_lbl.pack(side=tk.LEFT, padx=10, pady=8)
        
        # Connection info button
        info_btn = tk.Button(
            top_frame,
            text="ℹ️ Info",
            command=self._show_info,
            bg="#455A64",
            fg="white",
            font=("Arial", 9),
            relief='flat',
            padx=10
        )
        info_btn.pack(side=tk.RIGHT, padx=10)

        # Chat Body
        self.chat_history = scrolledtext.ScrolledText(
            self.master, 
            state='disabled', 
            wrap=tk.WORD, 
            font=("Segoe UI", 10), 
            bg=self.bg_color,
            relief='flat'
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure tags
        self.chat_history.tag_config('user', background=self.user_color, 
                                     lmargin1=50, lmargin2=50, rmargin=10,
                                     spacing1=5, spacing3=5)
        self.chat_history.tag_config('bot', background=self.bot_color, 
                                     lmargin1=10, lmargin2=10, rmargin=50,
                                     spacing1=5, spacing3=5)
        self.chat_history.tag_config('system', background=self.system_color,
                                     lmargin1=10, lmargin2=10, rmargin=10,
                                     spacing1=5, spacing3=5, font=("Arial", 9, "italic"))
        
        # Input Frame
        input_frame = tk.Frame(self.master, bg='white', pady=5)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.query_entry = tk.Entry(
            input_frame, 
            font=("Segoe UI", 11), 
            bg="#FAFAFA",
            relief='flat',
            bd=2
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=10)
        self.query_entry.bind('<Return>', lambda e: self._send_query())
        
        self.send_btn = tk.Button(
            input_frame, 
            text="Send ➤", 
            command=self._send_query, 
            bg="#009688", 
            fg="white", 
            font=("Arial", 10, "bold"),
            relief='flat',
            padx=20,
            cursor='hand2'
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5)
        
        # Clear button
        clear_btn = tk.Button(
            input_frame,
            text="Clear",
            command=self._clear_chat,
            bg="#FF5722",
            fg="white",
            font=("Arial", 9),
            relief='flat',
            padx=15,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Welcome message
        self._append_text(
            "Welcome to RAG Chatbot Client!\n"
            f"Connecting to server at {self.server_ip}:{SERVER_PORT}...",
            'system'
        )

    def _check_connection_async(self):
        threading.Thread(target=self._ping_server, daemon=True).start()

    def _ping_server(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((self.server_ip, SERVER_PORT))
            sock.close()
            
            if result == 0:
                self.master.after(0, lambda: self.status_lbl.config(
                    text=f"✅ Connected to {self.server_ip}:{SERVER_PORT}", 
                    fg="#66BB6A"
                ))
                self.master.after(0, lambda: self._append_text(
                    "✓ Successfully connected to server! You can start asking questions.",
                    'system'
                ))
            else:
                self.master.after(0, lambda: self.status_lbl.config(
                    text=f"❌ Server Unreachable ({self.server_ip}:{SERVER_PORT})", 
                    fg="#EF5350"
                ))
                self.master.after(0, lambda: self._append_text(
                    "✗ Cannot connect to server. Please check:\n"
                    "  1. Server is running (bot_server.py)\n"
                    "  2. IP address is correct\n"
                    "  3. Firewall allows port 9999",
                    'system'
                ))
        except Exception as e:
            self.master.after(0, lambda: self._append_text(
                f"Connection error: {e}",
                'system'
            ))

    def _append_text(self, text, tag):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"\n{text}\n", tag)
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)

    def _send_query(self):
        q = self.query_entry.get().strip()
        if not q:
            return
        
        self._append_text(f"You: {q}", 'user')
        self.query_entry.delete(0, tk.END)
        
        # Disable send button while processing
        self.send_btn.config(state='disabled', text="Sending...")
        
        threading.Thread(target=self._send_tcp_packet, args=(q,), daemon=True).start()

    def _send_tcp_packet(self, question):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(120)
            client_socket.connect((self.server_ip, SERVER_PORT))
            
            # Send request
            client_socket.sendall(json.dumps({"question": question}).encode('utf-8'))
            
            # Receive response
            data = client_socket.recv(BUFFER_SIZE)
            resp = json.loads(data.decode('utf-8'))
            
            client_socket.close()
            
            # Process response
            if resp.get("status") == "success":
                ans = resp.get("answer", "")
                time_taken = resp.get("time", 0)
                
                self.master.after(0, lambda: self._append_text(
                    f"Bot: {ans}\n⏱️ Response time: {time_taken}s",
                    'bot'
                ))
            else:
                error_msg = resp.get("answer", "Unknown error")
                self.master.after(0, lambda: self._append_text(
                    f"Error: {error_msg}",
                    'system'
                ))
            
            # Re-enable send button
            self.master.after(0, lambda: self.send_btn.config(
                state='normal', 
                text="Send ➤"
            ))
            
        except socket.timeout:
            self.master.after(0, lambda: self._append_text(
                "Error: Request timed out. The server might be processing a complex query.",
                'system'
            ))
            self.master.after(0, lambda: self.send_btn.config(
                state='normal', 
                text="Send ➤"
            ))
            
        except ConnectionRefusedError:
            self.master.after(0, lambda: self._append_text(
                "Error: Connection refused. Please ensure the server is running.",
                'system'
            ))
            self.master.after(0, lambda: self.send_btn.config(
                state='normal', 
                text="Send ➤"
            ))
            
        except Exception as e:
            self.master.after(0, lambda: self._append_text(
                f"Error: {e}",
                'system'
            ))
            self.master.after(0, lambda: self.send_btn.config(
                state='normal', 
                text="Send ➤"
            ))

    def _clear_chat(self):
        self.chat_history.config(state='normal')
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state='disabled')
        self._append_text("Chat cleared.", 'system')

    def _show_info(self):
        info_text = f"""RAG Chatbot Client

Server: {self.server_ip}:{SERVER_PORT}
Model: Llama 3.2 1B Instruct

Instructions:
1. Type your question in the input box
2. Press Enter or click Send
3. Wait for the bot's response

Note: The bot only answers based on the 
documents uploaded to the server."""
        
        messagebox.showinfo("Client Information", info_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TCPClientGUI(root)
    root.mainloop()