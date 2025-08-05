"""
Test script to verify all components work correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import OK")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from config import STREAMLIT_CONFIG, create_directories, download_nltk_data
        print("✅ Config import OK")

        # Download NLTK data
        download_nltk_data()
        print("✅ NLTK data download OK")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from src.chunking.chunking_methods import ChunkingMethods
        print("✅ Chunking methods import OK")
    except ImportError as e:
        print(f"❌ Chunking methods import failed: {e}")
        return False
    
    try:
        from src.embedding.embedding_methods import EmbeddingMethods
        print("✅ Embedding methods import OK")
    except ImportError as e:
        print(f"❌ Embedding methods import failed: {e}")
        return False
    
    try:
        from src.vector_storage.faiss_storage import FAISSStorage
        print("✅ FAISS storage import OK")
    except ImportError as e:
        print(f"❌ FAISS storage import failed: {e}")
        return False
    
    try:
        from src.utils.file_processor import FileProcessor
        print("✅ File processor import OK")
    except ImportError as e:
        print(f"❌ File processor import failed: {e}")
        return False
    
    try:
        from src.ui.sidebar import create_sidebar
        from src.ui.pages import show_home_page
        print("✅ UI components import OK")
    except ImportError as e:
        print(f"❌ UI components import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test chunking
        from src.chunking.chunking_methods import ChunkingMethods
        chunker = ChunkingMethods()
        
        sample_text = "This is a test document. It has multiple sentences. We will use it to test chunking."
        result = chunker.fixed_size_chunking(sample_text, chunk_size=10, overlap=2)
        
        print(f"✅ Chunking test OK - Created {result.total_chunks} chunks")
        
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        return False
    
    try:
        # Test file processor
        from src.utils.file_processor import FileProcessor
        processor = FileProcessor()
        
        stats = processor.get_text_statistics(sample_text)
        print(f"✅ File processor test OK - {stats['word_count']} words")
        
    except Exception as e:
        print(f"❌ File processor test failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test key dependencies"""
    print("\nTesting dependencies...")
    
    dependencies = [
        'streamlit',
        'sentence_transformers', 
        'faiss',
        'numpy',
        'pandas',
        'plotly',
        'sklearn',
        'nltk',
        'tiktoken',
        'langchain'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} OK")
        except ImportError:
            print(f"❌ {dep} missing - install with: pip install {dep}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🔍 RAG System Evaluation Platform - Test Suite")
    print("=" * 50)
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please install missing packages.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check file structure and dependencies.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality test failed. Check implementation.")
        return False
    
    print("\n🎉 All tests passed! The application should work correctly.")
    print("\nTo run the application:")
    print("streamlit run app.py")
    
    return True

if __name__ == "__main__":
    main()
