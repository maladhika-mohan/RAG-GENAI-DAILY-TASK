"""
File processing utilities for RAG System Evaluation
"""

import os
import io
import streamlit as st
from typing import Optional, Tuple
import PyPDF2
from docx import Document
import pandas as pd

class FileProcessor:
    """Handles file upload and text extraction"""
    
    @staticmethod
    def extract_text_from_file(uploaded_file) -> Tuple[str, str]:
        """
        Extract text from uploaded file
        
        Returns:
            Tuple of (extracted_text, file_info)
        """
        if uploaded_file is None:
            return "", "No file uploaded"
        
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_size = uploaded_file.size
        
        try:
            if file_extension == '.txt':
                # Handle text files
                text = str(uploaded_file.read(), "utf-8")
                file_info = f"Text file: {uploaded_file.name} ({file_size:,} bytes)"
                
            elif file_extension == '.pdf':
                # Handle PDF files
                text = FileProcessor._extract_from_pdf(uploaded_file)
                file_info = f"PDF file: {uploaded_file.name} ({file_size:,} bytes)"
                
            elif file_extension == '.docx':
                # Handle Word documents
                text = FileProcessor._extract_from_docx(uploaded_file)
                file_info = f"Word document: {uploaded_file.name} ({file_size:,} bytes)"
                
            elif file_extension == '.md':
                # Handle Markdown files
                text = str(uploaded_file.read(), "utf-8")
                file_info = f"Markdown file: {uploaded_file.name} ({file_size:,} bytes)"
                
            else:
                return "", f"Unsupported file format: {file_extension}"
            
            return text, file_info
            
        except Exception as e:
            return "", f"Error processing file: {str(e)}"
    
    @staticmethod
    def _extract_from_pdf(uploaded_file) -> str:
        """Extract text from PDF file"""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text
    
    @staticmethod
    def _extract_from_docx(uploaded_file) -> str:
        """Extract text from Word document"""
        doc = Document(uploaded_file)
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Returns:
            Tuple of (is_valid, message)
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        if uploaded_file.size > max_size:
            return False, f"File too large. Maximum size is {max_size // (1024*1024)}MB"
        
        # Check file extension
        supported_extensions = ['.txt', '.pdf', '.docx', '.md']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension not in supported_extensions:
            return False, f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
        
        return True, "File is valid"
    
    @staticmethod
    def get_text_statistics(text: str) -> dict:
        """Get basic statistics about the text"""
        if not text:
            return {}
        
        lines = text.split('\n')
        words = text.split()
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "average_sentence_length": len(words) / max(text.count('.'), 1)
        }

def create_file_upload_interface():
    """Create Streamlit file upload interface"""
    st.subheader("📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=['txt', 'pdf', 'docx', 'md'],
        help="Upload a text document for RAG system evaluation. Supported formats: TXT, PDF, DOCX, MD"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, message = FileProcessor.validate_file(uploaded_file)
        
        if is_valid:
            st.success(f"✅ {message}")
            
            # Extract text
            with st.spinner("Extracting text from file..."):
                text, file_info = FileProcessor.extract_text_from_file(uploaded_file)
            
            if text:
                st.info(f"📄 {file_info}")
                
                # Show text statistics
                stats = FileProcessor.get_text_statistics(text)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Characters", f"{stats['character_count']:,}")
                with col2:
                    st.metric("Words", f"{stats['word_count']:,}")
                with col3:
                    st.metric("Lines", f"{stats['line_count']:,}")
                with col4:
                    st.metric("Paragraphs", f"{stats['paragraph_count']:,}")
                
                # Show text preview
                with st.expander("📖 Text Preview (First 1000 characters)"):
                    st.text(text[:1000] + "..." if len(text) > 1000 else text)
                
                return text, file_info
            else:
                st.error(f"❌ {file_info}")
                return None, None
        else:
            st.error(f"❌ {message}")
            return None, None
    
    return None, None
