"""
Embedding Methods Implementation for RAG System Evaluation
"""

import time
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    model_name: str
    embeddings: np.ndarray
    chunk_texts: List[str]
    embedding_time: float
    model_info: Dict[str, Any]
    similarity_matrix: np.ndarray
    performance_metrics: Dict[str, float]

class EmbeddingMethods:
    """Implementation of various embedding methods using SentenceTransformers"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'all-MiniLM-L6-v2': {
                'name': 'all-MiniLM-L6-v2',
                'description': 'Lightweight, fast model (384 dimensions)',
                'dimensions': 384,
                'speed': 'Fast',
                'quality': 'Good',
                'use_case': 'Quick prototyping, real-time applications'
            },
            'all-mpnet-base-v2': {
                'name': 'all-mpnet-base-v2',
                'description': 'Balanced quality and speed (768 dimensions)',
                'dimensions': 768,
                'speed': 'Medium',
                'quality': 'Very Good',
                'use_case': 'Production applications, balanced performance'
            },
            'all-roberta-large-v1': {
                'name': 'sentence-transformers/all-roberta-large-v1',
                'description': 'High quality, slower model (1024 dimensions)',
                'dimensions': 1024,
                'speed': 'Slow',
                'quality': 'Excellent',
                'use_case': 'High-accuracy requirements, offline processing'
            }
        }
    
    @st.cache_resource
    def load_model(_self, model_name: str) -> SentenceTransformer:
        """Load and cache SentenceTransformer model"""
        if model_name not in _self.models:
            with st.spinner(f"Loading model {model_name}..."):
                _self.models[model_name] = SentenceTransformer(model_name)
        return _self.models[model_name]
    
    def generate_embeddings(self, chunks: List[str], model_name: str, 
                          show_progress: bool = True) -> EmbeddingResult:
        """
        Generate embeddings for text chunks using specified model
        """
        start_time = time.time()
        
        # Load model
        model = self.load_model(model_name)
        
        # Generate embeddings with progress tracking
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"Generating embeddings with {model_name}...")
        
        # Batch processing for better performance
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                progress = min((i + batch_size) / len(chunks), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        embeddings = np.array(all_embeddings)
        embedding_time = time.time() - start_time
        
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        
        # Calculate similarity matrix (for sample chunks to avoid memory issues)
        sample_size = min(50, len(chunks))
        sample_embeddings = embeddings[:sample_size]
        similarity_matrix = cosine_similarity(sample_embeddings)
        
        # Calculate performance metrics
        performance_metrics = {
            'embedding_time': embedding_time,
            'chunks_per_second': len(chunks) / embedding_time,
            'avg_time_per_chunk': embedding_time / len(chunks),
            'memory_usage_mb': embeddings.nbytes / (1024 * 1024),
            'embedding_dimensions': embeddings.shape[1]
        }
        
        # Model information
        model_info = self.model_configs.get(model_name, {})
        model_info.update({
            'actual_dimensions': embeddings.shape[1],
            'total_chunks': len(chunks),
            'model_size_estimate': self._estimate_model_size(model_name)
        })
        
        return EmbeddingResult(
            model_name=model_name,
            embeddings=embeddings,
            chunk_texts=chunks,
            embedding_time=embedding_time,
            model_info=model_info,
            similarity_matrix=similarity_matrix,
            performance_metrics=performance_metrics
        )
    
    def _estimate_model_size(self, model_name: str) -> str:
        """Estimate model size based on known model specifications"""
        size_estimates = {
            'all-MiniLM-L6-v2': '90MB',
            'all-mpnet-base-v2': '420MB',
            'sentence-transformers/all-roberta-large-v1': '1.3GB'
        }
        return size_estimates.get(model_name, 'Unknown')
    
    def compare_embeddings(self, embedding_results: List[EmbeddingResult]) -> Dict[str, Any]:
        """
        Compare multiple embedding results
        """
        comparison = {
            'models': [],
            'performance_comparison': pd.DataFrame(),
            'quality_metrics': {},
            'recommendations': {}
        }
        
        # Performance comparison
        perf_data = []
        for result in embedding_results:
            perf_data.append({
                'Model': result.model_name,
                'Embedding Time (s)': f"{result.embedding_time:.2f}",
                'Chunks/Second': f"{result.performance_metrics['chunks_per_second']:.1f}",
                'Memory Usage (MB)': f"{result.performance_metrics['memory_usage_mb']:.1f}",
                'Dimensions': result.performance_metrics['embedding_dimensions'],
                'Quality': result.model_info.get('quality', 'Unknown'),
                'Speed': result.model_info.get('speed', 'Unknown')
            })
        
        comparison['performance_comparison'] = pd.DataFrame(perf_data)
        
        # Quality metrics (based on similarity distribution)
        for result in embedding_results:
            sim_matrix = result.similarity_matrix
            # Remove diagonal (self-similarity)
            sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            
            comparison['quality_metrics'][result.model_name] = {
                'mean_similarity': np.mean(sim_values),
                'std_similarity': np.std(sim_values),
                'similarity_range': np.max(sim_values) - np.min(sim_values),
                'distinct_representations': np.sum(sim_values < 0.8) / len(sim_values)
            }
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_recommendations(embedding_results)
        
        return comparison
    
    def _generate_recommendations(self, embedding_results: List[EmbeddingResult]) -> Dict[str, str]:
        """Generate recommendations based on embedding results"""
        recommendations = {}
        
        # Find fastest model
        fastest = min(embedding_results, key=lambda x: x.embedding_time)
        recommendations['fastest'] = f"{fastest.model_name} - Best for real-time applications"
        
        # Find most balanced model
        balanced_scores = []
        for result in embedding_results:
            # Simple scoring: normalize time (lower is better) and quality (higher is better)
            time_score = 1 / result.embedding_time
            quality_score = {'Good': 1, 'Very Good': 2, 'Excellent': 3}.get(
                result.model_info.get('quality', 'Good'), 1
            )
            balanced_scores.append((result, time_score + quality_score))
        
        balanced = max(balanced_scores, key=lambda x: x[1])[0]
        recommendations['balanced'] = f"{balanced.model_name} - Best overall balance"
        
        # Find highest quality model
        quality_order = {'Good': 1, 'Very Good': 2, 'Excellent': 3}
        highest_quality = max(embedding_results, 
                            key=lambda x: quality_order.get(x.model_info.get('quality', 'Good'), 1))
        recommendations['highest_quality'] = f"{highest_quality.model_name} - Best for accuracy-critical applications"
        
        return recommendations
    
    def visualize_embeddings(self, embedding_result: EmbeddingResult, 
                           method: str = 'PCA', sample_size: int = 100) -> go.Figure:
        """
        Create visualization of embeddings using dimensionality reduction
        """
        embeddings = embedding_result.embeddings
        chunk_texts = embedding_result.chunk_texts
        
        # Sample data if too large
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings_sample = embeddings[indices]
            texts_sample = [chunk_texts[i] for i in indices]
        else:
            embeddings_sample = embeddings
            texts_sample = chunk_texts
        
        # Apply dimensionality reduction
        if method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings_sample)
            title = f"PCA Visualization of {embedding_result.model_name} Embeddings"
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
            reduced_embeddings = reducer.fit_transform(embeddings_sample)
            title = f"t-SNE Visualization of {embedding_result.model_name} Embeddings"
        
        # Create hover text (first 100 chars of each chunk)
        hover_texts = [text[:100] + "..." if len(text) > 100 else text for text in texts_sample]
        
        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=range(len(reduced_embeddings)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Chunk Index")
            ),
            text=hover_texts,
            hovertemplate='<b>Chunk %{marker.color}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         '<b>Text:</b> %{text}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"{method} Component 1",
            yaxis_title=f"{method} Component 2",
            width=800,
            height=600
        )
        
        return fig
    
    def create_similarity_heatmap(self, embedding_result: EmbeddingResult, 
                                sample_size: int = 20) -> go.Figure:
        """
        Create similarity heatmap for sample chunks
        """
        similarity_matrix = embedding_result.similarity_matrix
        
        # Use only a sample for visualization
        if similarity_matrix.shape[0] > sample_size:
            similarity_matrix = similarity_matrix[:sample_size, :sample_size]
        
        # Create chunk labels (first 30 chars)
        chunk_labels = [
            f"Chunk {i}: {embedding_result.chunk_texts[i][:30]}..."
            for i in range(min(sample_size, len(embedding_result.chunk_texts)))
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=chunk_labels,
            y=chunk_labels,
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Cosine Similarity")
        ))
        
        fig.update_layout(
            title=f"Chunk Similarity Heatmap - {embedding_result.model_name}",
            xaxis_title="Chunks",
            yaxis_title="Chunks",
            width=800,
            height=600
        )
        
        return fig
