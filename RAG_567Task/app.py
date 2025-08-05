"""
RAG System Evaluation Platform - Main Streamlit Application
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import STREAMLIT_CONFIG, create_directories, download_nltk_data
from src.ui.sidebar import create_sidebar
from src.ui.pages import (
    show_home_page,
    show_task5_page,
    show_task6_page, 
    show_task7_page,
    show_comparison_page
)

def main():
    """Main application function"""
    
    # Configure Streamlit page
    st.set_page_config(**STREAMLIT_CONFIG)
    
    # Create necessary directories and download NLTK data
    create_directories()
    download_nltk_data()
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
        
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
        
    if 'chunking_results' not in st.session_state:
        st.session_state.chunking_results = {}
        
    if 'embedding_results' not in st.session_state:
        st.session_state.embedding_results = {}
        
    if 'faiss_indices' not in st.session_state:
        st.session_state.faiss_indices = {}
    
    # Create sidebar navigation
    current_page = create_sidebar()
    st.session_state.current_page = current_page
    
    # Main content area
    st.title("🔍 RAG System Evaluation Platform")
    
    # Display appropriate page based on navigation
    if current_page == 'Home':
        show_home_page()
    elif current_page == 'Task 5: Chunking Methods':
        show_task5_page()
    elif current_page == 'Task 6: Embedding Implementation':
        show_task6_page()
    elif current_page == 'Task 7: FAISS Vector Storage':
        show_task7_page()
    elif current_page == 'Comparison & Results':
        show_comparison_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        RAG System Evaluation Platform | Built with Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
