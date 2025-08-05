"""
Chunking Methods Implementation for RAG System Evaluation
"""

import re
import tiktoken
import nltk
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class ChunkInfo:
    """Information about a text chunk"""
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    char_count: int
    chunk_id: int

@dataclass
class ChunkingResult:
    """Result of chunking operation"""
    method_name: str
    chunks: List[ChunkInfo]
    total_chunks: int
    avg_chunk_size_tokens: float
    avg_chunk_size_chars: float
    min_chunk_size: int
    max_chunk_size: int
    overlap_info: Dict[str, Any]
    processing_time: float

class ChunkingMethods:
    """Implementation of various text chunking methods"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def fixed_size_chunking(self, text: str, chunk_size: int = 500, overlap: int = 50) -> ChunkingResult:
        """
        Fixed-size chunking with token-based splitting and overlap
        """
        import time
        start_time = time.time()
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        chunks = []
        chunk_id = 0
        
        # Calculate step size (chunk_size - overlap)
        step_size = chunk_size - overlap
        
        for i in range(0, len(tokens), step_size):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + chunk_size]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Find character positions in original text
            start_pos = len(self.tokenizer.decode(tokens[:i])) if i > 0 else 0
            end_pos = start_pos + len(chunk_text)
            
            chunk_info = ChunkInfo(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                token_count=len(chunk_tokens),
                char_count=len(chunk_text),
                chunk_id=chunk_id
            )
            chunks.append(chunk_info)
            chunk_id += 1
        
        # Calculate statistics
        token_counts = [chunk.token_count for chunk in chunks]
        char_counts = [chunk.char_count for chunk in chunks]
        
        # Calculate overlap information
        overlap_info = {
            "overlap_tokens": overlap,
            "overlap_percentage": (overlap / chunk_size) * 100,
            "actual_overlaps": []
        }
        
        # Calculate actual overlaps between consecutive chunks
        for i in range(len(chunks) - 1):
            current_end = chunks[i].end_pos
            next_start = chunks[i + 1].start_pos
            if current_end > next_start:
                overlap_chars = current_end - next_start
                overlap_info["actual_overlaps"].append(overlap_chars)
        
        processing_time = time.time() - start_time
        
        return ChunkingResult(
            method_name="Fixed-Size Chunking",
            chunks=chunks,
            total_chunks=len(chunks),
            avg_chunk_size_tokens=sum(token_counts) / len(token_counts) if token_counts else 0,
            avg_chunk_size_chars=sum(char_counts) / len(char_counts) if char_counts else 0,
            min_chunk_size=min(token_counts) if token_counts else 0,
            max_chunk_size=max(token_counts) if token_counts else 0,
            overlap_info=overlap_info,
            processing_time=processing_time
        )
    
    def semantic_chunking(self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 800) -> ChunkingResult:
        """
        Semantic chunking based on sentence and paragraph boundaries
        """
        import time
        start_time = time.time()
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        chunk_id = 0
        current_pos = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                current_pos += len(paragraph) + 2  # +2 for \n\n
                continue
            
            # If paragraph is within size limits, use as chunk
            if min_chunk_size <= len(paragraph) <= max_chunk_size:
                chunk_info = ChunkInfo(
                    text=paragraph.strip(),
                    start_pos=current_pos,
                    end_pos=current_pos + len(paragraph),
                    token_count=self.count_tokens(paragraph),
                    char_count=len(paragraph),
                    chunk_id=chunk_id
                )
                chunks.append(chunk_info)
                chunk_id += 1
            
            # If paragraph is too long, split by sentences
            elif len(paragraph) > max_chunk_size:
                sentences = nltk.sent_tokenize(paragraph)
                current_chunk = ""
                chunk_start = current_pos
                
                for sentence in sentences:
                    # Check if adding this sentence would exceed max size
                    if len(current_chunk + sentence) > max_chunk_size and current_chunk:
                        # Save current chunk
                        chunk_info = ChunkInfo(
                            text=current_chunk.strip(),
                            start_pos=chunk_start,
                            end_pos=chunk_start + len(current_chunk),
                            token_count=self.count_tokens(current_chunk),
                            char_count=len(current_chunk),
                            chunk_id=chunk_id
                        )
                        chunks.append(chunk_info)
                        chunk_id += 1
                        
                        # Start new chunk
                        current_chunk = sentence
                        chunk_start = chunk_start + len(current_chunk) - len(sentence)
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                # Add remaining chunk if it meets minimum size
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunk_info = ChunkInfo(
                        text=current_chunk.strip(),
                        start_pos=chunk_start,
                        end_pos=chunk_start + len(current_chunk),
                        token_count=self.count_tokens(current_chunk),
                        char_count=len(current_chunk),
                        chunk_id=chunk_id
                    )
                    chunks.append(chunk_info)
                    chunk_id += 1
            
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        # Calculate statistics
        token_counts = [chunk.token_count for chunk in chunks]
        char_counts = [chunk.char_count for chunk in chunks]
        
        overlap_info = {
            "overlap_tokens": 0,
            "overlap_percentage": 0,
            "method": "Semantic boundaries - no artificial overlap"
        }
        
        processing_time = time.time() - start_time
        
        return ChunkingResult(
            method_name="Semantic Chunking",
            chunks=chunks,
            total_chunks=len(chunks),
            avg_chunk_size_tokens=sum(token_counts) / len(token_counts) if token_counts else 0,
            avg_chunk_size_chars=sum(char_counts) / len(char_counts) if char_counts else 0,
            min_chunk_size=min(token_counts) if token_counts else 0,
            max_chunk_size=max(token_counts) if token_counts else 0,
            overlap_info=overlap_info,
            processing_time=processing_time
        )
    
    def recursive_character_splitting(self, text: str, chunk_size: int = 600, overlap: int = 100) -> ChunkingResult:
        """
        Recursive character text splitting using LangChain
        """
        import time
        start_time = time.time()
        
        # Initialize the recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the text
        split_texts = text_splitter.split_text(text)
        
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(split_texts):
            # Find the position of this chunk in the original text
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            chunk_info = ChunkInfo(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                token_count=self.count_tokens(chunk_text),
                char_count=len(chunk_text),
                chunk_id=i
            )
            chunks.append(chunk_info)
            current_pos = start_pos + len(chunk_text) - overlap
        
        # Calculate statistics
        token_counts = [chunk.token_count for chunk in chunks]
        char_counts = [chunk.char_count for chunk in chunks]
        
        overlap_info = {
            "overlap_chars": overlap,
            "overlap_percentage": (overlap / chunk_size) * 100,
            "separators_used": ["\n\n", "\n", " ", ""]
        }
        
        processing_time = time.time() - start_time
        
        return ChunkingResult(
            method_name="Recursive Character Splitting",
            chunks=chunks,
            total_chunks=len(chunks),
            avg_chunk_size_tokens=sum(token_counts) / len(token_counts) if token_counts else 0,
            avg_chunk_size_chars=sum(char_counts) / len(char_counts) if char_counts else 0,
            min_chunk_size=min(token_counts) if token_counts else 0,
            max_chunk_size=max(token_counts) if token_counts else 0,
            overlap_info=overlap_info,
            processing_time=processing_time
        )
