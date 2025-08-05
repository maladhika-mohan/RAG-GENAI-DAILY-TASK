"""
Sidebar navigation for RAG System Evaluation Platform
"""

import streamlit as st
# Chat functionality removed

def create_sidebar():
    """Create and manage sidebar navigation"""
    
    with st.sidebar:
        st.title("🔍 Navigation")
        
        # Navigation menu
        pages = [
            "Home",
            "Task 5: Chunking Methods", 
            "Task 6: Embedding Implementation",
            "Task 7: FAISS Vector Storage",
            "Comparison & Results"
        ]
        
        selected_page = st.radio(
            "Select a page:",
            pages,
            index=0 if 'current_page' not in st.session_state else pages.index(st.session_state.get('current_page', 'Home'))
        )
        
        st.markdown("---")
        
        # Progress indicator
        st.subheader("📊 Progress")
        
        progress_data = {
            "Home": "✅" if st.session_state.get('uploaded_file') else "⏳",
            "Task 5: Chunking Methods": "✅" if st.session_state.get('chunking_results') else "⏳",
            "Task 6: Embedding Implementation": "✅" if st.session_state.get('embedding_results') else "⏳", 
            "Task 7: FAISS Vector Storage": "✅" if st.session_state.get('faiss_indices') else "⏳",
            "Comparison & Results": "✅" if (st.session_state.get('chunking_results') and 
                                           st.session_state.get('embedding_results') and 
                                           st.session_state.get('faiss_indices')) else "⏳"
        }
        
        for page, status in progress_data.items():
            st.write(f"{status} {page}")
        
        st.markdown("---")
        
        # System information
        st.subheader("ℹ️ System Info")
        
        if st.session_state.get('processed_text'):
            text_length = len(st.session_state.processed_text)
            st.write(f"📄 Document: {text_length:,} characters")
        
        if st.session_state.get('chunking_results'):
            total_chunks = sum(len(result.chunks) for result in st.session_state.chunking_results.values())
            st.write(f"🔪 Total chunks: {total_chunks}")
        
        if st.session_state.get('embedding_results'):
            st.write(f"🧮 Embeddings: Generated")
        
        if st.session_state.get('faiss_indices'):
            st.write(f"🗂️ FAISS indices: {len(st.session_state.faiss_indices)}")
        
        st.markdown("---")
        
        # Help section
        with st.expander("❓ Help"):
            st.markdown("""
            **Getting Started:**
            1. Upload a document on the Home page
            2. Explore chunking methods in Task 5
            3. Generate embeddings in Task 6
            4. Create FAISS indices in Task 7
            5. Compare results in the final page
            
            **Tips:**
            - Use the progress indicators to track completion
            - Each task builds on the previous ones
            - Hover over elements for additional help
            """)
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.checkbox("Show debug information", key="show_debug")
            st.checkbox("Enable performance monitoring", key="enable_monitoring")
            
            # Clear session state
            if st.button("🗑️ Clear All Data"):
                for key in list(st.session_state.keys()):
                    if key not in ['current_page']:
                        del st.session_state[key]
                st.success("All data cleared!")
                st.experimental_rerun()



    return selected_page
