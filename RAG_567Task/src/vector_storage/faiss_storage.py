"""
FAISS Vector Storage Implementation for RAG System Evaluation
"""

import time
import pickle
import numpy as np
import faiss
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

@dataclass
class SearchResult:
    """Result of similarity search"""
    chunk_id: int
    chunk_text: str
    similarity_score: float
    distance: float

@dataclass
class FAISSIndexResult:
    """Result of FAISS index creation"""
    index_name: str
    index: faiss.Index
    index_type: str
    embeddings: np.ndarray
    chunk_texts: List[str]
    build_time: float
    index_size_mb: float
    search_params: Dict[str, Any]
    performance_metrics: Dict[str, float]

class FAISSStorage:
    """FAISS vector storage implementation with multiple index types"""
    
    def __init__(self):
        self.indices = {}
        self.index_configs = {
            'Flat': {
                'name': 'IndexFlatIP',
                'description': 'Exact search, best quality but slower for large datasets',
                'use_case': 'Small to medium datasets (<100k vectors)',
                'build_function': self._build_flat_index
            },
            'IVF': {
                'name': 'IndexIVFFlat',
                'description': 'Inverted file index, good balance of speed and accuracy',
                'use_case': 'Medium to large datasets (100k-1M vectors)',
                'build_function': self._build_ivf_index
            },
            'HNSW': {
                'name': 'IndexHNSWFlat',
                'description': 'Hierarchical navigable small world, very fast search',
                'use_case': 'Large datasets (>1M vectors), real-time search',
                'build_function': self._build_hnsw_index
            }
        }
    
    def create_index(self, embeddings: np.ndarray, chunk_texts: List[str], 
                    index_type: str, **kwargs) -> FAISSIndexResult:
        """
        Create FAISS index with specified type
        """
        start_time = time.time()
        
        # Normalize embeddings for cosine similarity (inner product after normalization)
        normalized_embeddings = embeddings.astype('float32')
        faiss.normalize_L2(normalized_embeddings)
        
        # Build index based on type
        config = self.index_configs[index_type]
        index = config['build_function'](normalized_embeddings, **kwargs)
        
        build_time = time.time() - start_time
        
        # Calculate index size
        index_size_mb = self._estimate_index_size(index, embeddings.shape)
        
        # Performance metrics
        performance_metrics = {
            'build_time': build_time,
            'vectors_per_second': len(embeddings) / build_time,
            'index_size_mb': index_size_mb,
            'dimension': embeddings.shape[1],
            'total_vectors': len(embeddings)
        }
        
        # Search parameters
        search_params = {
            'k': 5,  # Default number of results
            'nprobe': kwargs.get('nprobe', 10) if index_type == 'IVF' else None
        }
        
        result = FAISSIndexResult(
            index_name=f"{index_type}_{int(time.time())}",
            index=index,
            index_type=index_type,
            embeddings=normalized_embeddings,
            chunk_texts=chunk_texts,
            build_time=build_time,
            index_size_mb=index_size_mb,
            search_params=search_params,
            performance_metrics=performance_metrics
        )
        
        return result
    
    def _build_flat_index(self, embeddings: np.ndarray, **kwargs) -> faiss.Index:
        """Build flat (exact search) index"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings)
        return index
    
    def _build_ivf_index(self, embeddings: np.ndarray, **kwargs) -> faiss.Index:
        """Build IVF (Inverted File) index"""
        dimension = embeddings.shape[1]
        nlist = kwargs.get('nlist', min(100, len(embeddings) // 10))  # Number of clusters
        
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Train the index
        index.train(embeddings)
        index.add(embeddings)
        
        # Set search parameters
        index.nprobe = kwargs.get('nprobe', 10)
        
        return index
    
    def _build_hnsw_index(self, embeddings: np.ndarray, **kwargs) -> faiss.Index:
        """Build HNSW (Hierarchical Navigable Small World) index"""
        dimension = embeddings.shape[1]
        M = kwargs.get('M', 16)  # Number of connections
        
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = kwargs.get('efConstruction', 200)
        
        index.add(embeddings)
        
        # Set search parameter
        index.hnsw.efSearch = kwargs.get('efSearch', 50)
        
        return index
    
    def _estimate_index_size(self, index: faiss.Index, embeddings_shape: Tuple[int, int]) -> float:
        """Estimate index size in MB"""
        # Basic estimation based on embeddings size and index overhead
        base_size = embeddings_shape[0] * embeddings_shape[1] * 4  # 4 bytes per float32
        
        # Add overhead based on index type
        if hasattr(index, 'nlist'):  # IVF index
            overhead = base_size * 0.1  # ~10% overhead for IVF
        elif hasattr(index, 'hnsw'):  # HNSW index
            overhead = base_size * 0.3  # ~30% overhead for HNSW
        else:  # Flat index
            overhead = base_size * 0.01  # ~1% overhead for flat
        
        return (base_size + overhead) / (1024 * 1024)  # Convert to MB
    
    def search(self, index_result: FAISSIndexResult, query_embedding: np.ndarray, 
              k: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform similarity search
        """
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Set search parameters for IVF
        if index_result.index_type == 'IVF':
            nprobe = kwargs.get('nprobe', index_result.search_params.get('nprobe', 10))
            index_result.index.nprobe = nprobe
        
        # Perform search
        start_time = time.time()
        distances, indices = index_result.index.search(query_embedding, k)
        search_time = time.time() - start_time
        
        # Convert results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                # Convert distance to similarity score (for inner product)
                similarity_score = float(distance)
                
                result = SearchResult(
                    chunk_id=int(idx),
                    chunk_text=index_result.chunk_texts[idx],
                    similarity_score=similarity_score,
                    distance=float(distance)
                )
                results.append(result)
        
        return results, search_time
    
    def benchmark_search_performance(self, index_result: FAISSIndexResult, 
                                   query_embeddings: np.ndarray, 
                                   k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """
        Benchmark search performance with different k values
        """
        benchmark_results = {
            'k_values': k_values,
            'search_times': [],
            'throughput': [],
            'avg_search_time': 0
        }
        
        total_time = 0
        num_queries = len(query_embeddings)
        
        for k in k_values:
            k_times = []
            
            for query_embedding in query_embeddings:
                _, search_time = self.search(index_result, query_embedding, k=k)
                k_times.append(search_time)
            
            avg_k_time = np.mean(k_times)
            benchmark_results['search_times'].append(avg_k_time)
            benchmark_results['throughput'].append(num_queries / sum(k_times))
            total_time += sum(k_times)
        
        benchmark_results['avg_search_time'] = total_time / (len(k_values) * num_queries)
        
        return benchmark_results
    
    def save_index(self, index_result: FAISSIndexResult, filepath: str):
        """Save FAISS index to disk"""
        # Save FAISS index
        faiss.write_index(index_result.index, f"{filepath}.faiss")
        
        # Save metadata
        metadata = {
            'index_name': index_result.index_name,
            'index_type': index_result.index_type,
            'chunk_texts': index_result.chunk_texts,
            'build_time': index_result.build_time,
            'index_size_mb': index_result.index_size_mb,
            'search_params': index_result.search_params,
            'performance_metrics': index_result.performance_metrics
        }
        
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_index(self, filepath: str) -> FAISSIndexResult:
        """Load FAISS index from disk"""
        # Load FAISS index
        index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        # Reconstruct result object
        result = FAISSIndexResult(
            index_name=metadata['index_name'],
            index=index,
            index_type=metadata['index_type'],
            embeddings=None,  # Not saved to reduce file size
            chunk_texts=metadata['chunk_texts'],
            build_time=metadata['build_time'],
            index_size_mb=metadata['index_size_mb'],
            search_params=metadata['search_params'],
            performance_metrics=metadata['performance_metrics']
        )
        
        return result
    
    def compare_indices(self, index_results: List[FAISSIndexResult]) -> pd.DataFrame:
        """
        Compare multiple FAISS indices
        """
        comparison_data = []
        
        for result in index_results:
            comparison_data.append({
                'Index Type': result.index_type,
                'Build Time (s)': f"{result.build_time:.3f}",
                'Index Size (MB)': f"{result.index_size_mb:.2f}",
                'Vectors/Second': f"{result.performance_metrics['vectors_per_second']:.1f}",
                'Total Vectors': result.performance_metrics['total_vectors'],
                'Dimensions': result.performance_metrics['dimension'],
                'Use Case': self.index_configs[result.index_type]['use_case']
            })
        
        return pd.DataFrame(comparison_data)
    
    def create_performance_chart(self, benchmark_results: Dict[str, Any], 
                               index_type: str) -> go.Figure:
        """
        Create performance visualization chart
        """
        fig = go.Figure()
        
        # Search time vs k
        fig.add_trace(go.Scatter(
            x=benchmark_results['k_values'],
            y=benchmark_results['search_times'],
            mode='lines+markers',
            name='Search Time',
            yaxis='y',
            line=dict(color='blue')
        ))
        
        # Throughput vs k
        fig.add_trace(go.Scatter(
            x=benchmark_results['k_values'],
            y=benchmark_results['throughput'],
            mode='lines+markers',
            name='Throughput (queries/sec)',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"Search Performance - {index_type} Index",
            xaxis_title="Number of Results (k)",
            yaxis=dict(
                title="Search Time (seconds)",
                side="left",
                color="blue"
            ),
            yaxis2=dict(
                title="Throughput (queries/sec)",
                side="right",
                overlaying="y",
                color="red"
            ),
            legend=dict(x=0.7, y=0.95)
        )
        
        return fig
