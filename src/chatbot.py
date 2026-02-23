# src/chatbot.py

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import hashlib
import re
from collections import OrderedDict
from src.config import RETRIEVAL_K, ENABLE_QUERY_CACHE, CACHE_SIZE

# Query result cache (LRU)
_query_cache = OrderedDict()

def create_chatbot(llm, vector_store, k=RETRIEVAL_K):
    """
    Create RAG chain optimized for Llama 3.2 1B Instruct
    """
    if llm is None or vector_store is None:
        print("✗ LLM or vector store not available")
        return None

    print(f"\n{'='*60}")
    print("CREATING RAG CHAIN (Llama 3.2 1B Instruct)")
    print(f"{'='*60}")
    print(f"Retrieval k: {k} documents")
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k}
    )
    
    # Optimized prompt for Llama 3.2 1B - short and direct
    template = """You are a helpful AI assistant.
Use the following pieces of context to answer the user's question.
If the answer is present in the context, output it directly.
If the answer is NOT in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Stop sequences to prevent repetition and prompt leakage
    llm_with_stop = llm.bind(
        stop=[
            "\n\nQuestion:", 
            "\n\nContext:", 
            "\nQuestion:",
            "User:",
            "Assistant:",
        ]
    )
    
    rag_chain = (
        {
            "context": retriever | _format_docs | (lambda x: (_debug_print_context(x), x)[1]),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_with_stop
        | StrOutputParser()
    )
    
    print(f"{'='*60}")
    print("✓ RAG CHAIN READY")
    print(f"{'='*60}\n")
    
    return rag_chain

def _format_docs(docs):
    """
    Format retrieved documents with deduplication
    """
    if not docs:
        return "No relevant documents found."
    
    print(f"\n[DEBUG] Retrieved {len(docs)} chunks")
    
    # Deduplicate similar chunks
    unique_docs = []
    seen_content = set()
    
    for doc in docs:
        fingerprint = doc.page_content[:200].strip()  # Use 200 chars for better dedup
        if fingerprint not in seen_content:
            unique_docs.append(doc)
            seen_content.add(fingerprint)
    
    print(f"[DEBUG] After deduplication: {len(unique_docs)} unique chunks")
    
    # Reduced context for 1B model (4K tokens = ~16K chars max)
    MAX_CONTEXT_LENGTH = 12000  
    
    formatted_docs = []
    for doc in unique_docs:
        content = doc.page_content.strip()
        content = ' '.join(content.split())  # Remove excessive whitespace
        formatted_docs.append(f"- {content}")
    
    formatted = "\n\n".join(formatted_docs)
    
    if len(formatted) > MAX_CONTEXT_LENGTH:
        formatted = formatted[:MAX_CONTEXT_LENGTH]
        formatted += "\n[... truncated ...]"
    
    return formatted

def _debug_print_context(formatted_text):
    print("\n" + "="*40)
    print("DEBUG: RETRIEVED CONTEXT START")
    print("="*40)
    print(formatted_text[:1000] + "... [truncated]" if len(formatted_text) > 1000 else formatted_text)
    print("="*40)
    print("DEBUG: RETRIEVED CONTEXT END")
    print("="*40 + "\n")

def ask_question(chain, query, timeout=60):
    """
    Ask question with timeout and repetition detection
    """
    if not chain:
        return "⚠️ Chatbot not initialized."
    
    if not query or not query.strip():
        return "⚠️ Please provide a valid question."
    
    query_normalized = query.strip().lower()
    query_hash = hashlib.md5(query_normalized.encode()).hexdigest()
    
    # Check cache
    if ENABLE_QUERY_CACHE and query_hash in _query_cache:
        print("⚡ Cache hit")
        return _query_cache[query_hash]
    
    try:
        print(f"🔍 Processing query: '{query}'")
        
        # Add timeout wrapper
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(chain.invoke, query)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print("⏱️ Query timeout - no relevant context found")
                return "I don't have that information in the provided documents."
        
        if isinstance(result, str):
            result = _clean_response(result)
            result = _detect_repetition(result)
            result = _validate_context_response(result)
        
        # Cache result with LRU eviction
        if ENABLE_QUERY_CACHE and result:
            _query_cache[query_hash] = result
            if len(_query_cache) > CACHE_SIZE:
                _query_cache.popitem(last=False)  # Remove oldest
        
        return result
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return "I encountered an error processing your request."

def _detect_repetition(response):
    """
    Detect and truncate repetitive responses
    """
    if not response or len(response) < 100:
        return response
    
    # Split into sentences
    sentences = response.split('.')
    
    # Check if same sentence pattern repeats
    if len(sentences) > 3:
        last_three = [s.strip().lower() for s in sentences[-4:-1] if s.strip()]
        
        if len(last_three) == 3:
            # Check for repetition patterns
            if (last_three[0] == last_three[1] or 
                last_three[1] == last_three[2] or
                all(len(set(s.split()[:5])) < 3 for s in last_three)):
                
                print("⚠️ Repetition detected - truncating response")
                return '. '.join(sentences[:len(sentences)//2]) + '.'
    
    # Check for phrase repetition
    words = response.lower().split()
    if len(words) > 20:
        phrases = [' '.join(words[i:i+5]) for i in range(len(words)-5)]
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        if any(count > 2 for count in phrase_counts.values()):
            print("⚠️ Phrase repetition detected - truncating")
            for i, phrase in enumerate(phrases):
                if phrase_counts[phrase] > 2:
                    cut_point = i * 5 + len(phrases[i].split())
                    return ' '.join(words[:cut_point]) + '.'
    
    return response

def _clean_response(response):
    """
    Clean up response to remove any leaked prompts
    """
    response = response.strip()
    
    # Remove leaked prompt patterns
    leak_patterns = [
        r'\n\s*Question:',
        r'\n\s*Context:',
        r'\n\s*User:',
        r'\n\s*Assistant:',
        r'\n\s*Rules:',
        r'\n\s*Answer based'
    ]
    
    for pattern in leak_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            response = response[:match.start()].strip()
    
    return response.strip()

def _validate_context_response(response):
    """
    Ensure response is from context, not general knowledge
    """
    generic_indicators = [
        "as an ai",
        "i'm an ai",
        "i can help you with",
        "here are some general",
        "typically,",
        "generally speaking",
        "in general,",
        "commonly,",
        "it is widely known",
        "based on my training",
        "from what i know"
    ]
    
    response_lower = response.lower()
    
    for indicator in generic_indicators:
        if indicator in response_lower:
            return "I don't have that information in the provided documents."
    
    # Relaxed validation for 1B model which might be chatty
    # if len(response) < 50 and "document" not in response_lower:
    #     if not any(keyword in response_lower for keyword in 
    #                ["according to", "states that", "mentions", "shows"]):
    #         return "I don't have that information in the provided documents."
    
    return response

def clear_query_cache():
    """
    Clear the query cache
    """
    global _query_cache
    _query_cache.clear()
    print("✓ Query cache cleared")