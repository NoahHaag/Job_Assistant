import os

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

def get_embedding_function():
    # In GitHub Actions, we don't have Ollama, so return None
    # The consumer (tools_2.py) must handle this gracefully
    if os.getenv("GITHUB_ACTIONS") == "true" or OllamaEmbeddings is None:
        return None
        
    embeddings = OllamaEmbeddings(model="qwen3-embedding")
    return embeddings
