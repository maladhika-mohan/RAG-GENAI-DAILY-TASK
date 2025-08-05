"""
Configuration file for RAG System Evaluation
"""

import os
from typing import Dict, List, Any

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "RAG System Evaluation Platform",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Chunking Methods Configuration
CHUNKING_METHODS = {
    "fixed_size": {
        "name": "Fixed-Size Chunking",
        "description": "Splits text into fixed-size chunks with overlap",
        "chunk_size": 500,
        "overlap": 50,
        "unit": "tokens"
    },
    "semantic": {
        "name": "Semantic Chunking",
        "description": "Splits text at sentence/paragraph boundaries",
        "min_chunk_size": 100,
        "max_chunk_size": 800,
        "unit": "characters"
    },
    "recursive": {
        "name": "Recursive Character Text Splitting",
        "description": "Recursively splits text using multiple separators",
        "chunk_size": 600,
        "overlap": 100,
        "separators": ["\n\n", "\n", " ", ""]
    }
}

# Embedding Models Configuration
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Lightweight, fast model (384 dimensions)",
        "dimensions": 384,
        "speed": "Fast",
        "quality": "Good",
        "use_case": "Quick prototyping, real-time applications"
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2", 
        "description": "Balanced quality and speed (768 dimensions)",
        "dimensions": 768,
        "speed": "Medium",
        "quality": "Very Good",
        "use_case": "Production applications, balanced performance"
    },
    "all-roberta-large-v1": {
        "name": "sentence-transformers/all-roberta-large-v1",
        "description": "High quality, slower model (1024 dimensions)",
        "dimensions": 1024,
        "speed": "Slow",
        "quality": "Excellent",
        "use_case": "High-accuracy requirements, offline processing"
    }
}

# FAISS Index Configuration
FAISS_CONFIG = {
    "index_types": {
        "Flat": {
            "name": "IndexFlatIP",
            "description": "Exact search, best quality but slower for large datasets",
            "use_case": "Small to medium datasets (<100k vectors)"
        },
        "IVF": {
            "name": "IndexIVFFlat",
            "description": "Inverted file index, good balance of speed and accuracy",
            "use_case": "Medium to large datasets (100k-1M vectors)",
            "nlist": 100
        },
        "HNSW": {
            "name": "IndexHNSWFlat",
            "description": "Hierarchical navigable small world, very fast search",
            "use_case": "Large datasets (>1M vectors), real-time search",
            "M": 16,
            "efConstruction": 200
        }
    },
    "search_params": {
        "k": 5,  # Number of results to return
        "nprobe": 10  # For IVF indices
    }
}

# File Processing Configuration
FILE_CONFIG = {
    "supported_formats": [".txt", ".pdf", ".docx", ".md"],
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "encoding": "utf-8"
}

# UI Configuration
UI_CONFIG = {
    "sidebar_width": 300,
    "main_content_width": 800,
    "chart_height": 400,
    "max_display_chunks": 5,
    "max_display_results": 10
}

# Paths
PATHS = {
    "data": "data",
    "indices": "indices",
    "embeddings": "embeddings",
    "results": "results",
    "temp": "temp"
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    import ssl

    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required NLTK data
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)  # Fallback for older versions

    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
        print("You may need to download manually: nltk.download('punkt_tab')")

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "embedding_time_warning": 30,  # seconds
    "index_build_time_warning": 60,  # seconds
    "search_time_warning": 1,  # seconds
    "memory_usage_warning": 1024 * 1024 * 1024  # 1GB
}
