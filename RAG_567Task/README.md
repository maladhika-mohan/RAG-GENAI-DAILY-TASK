# RAG System Evaluation Platform 🔍

A comprehensive Streamlit-based platform for evaluating and comparing Retrieval-Augmented Generation (RAG) system components, focusing on chunking methods, embedding models, and FAISS vector storage.

## 🌟 Features

### Task 5: Chunking Methods Analysis
- **Fixed-Size Chunking**: Token-based splitting with configurable overlap
- **Semantic Chunking**: Intelligent splitting at sentence/paragraph boundaries  
- **Recursive Character Splitting**: Hierarchical text splitting using multiple separators
- Interactive comparison with visual analytics and performance metrics

### Task 6: Embedding Implementation
- **Multiple SentenceTransformer Models**:
  - `all-MiniLM-L6-v2`: Lightweight, fast (384 dimensions)
  - `all-mpnet-base-v2`: Balanced quality/speed (768 dimensions)
  - `all-roberta-large-v1`: High quality (1024 dimensions)
- Real-time embedding generation with progress tracking
- Dimensionality reduction visualization (PCA, t-SNE)
- Similarity heatmaps and performance benchmarking

### Task 7: FAISS Vector Storage
- **Multiple Index Types**:
  - `IndexFlatIP`: Exact search, best quality
  - `IndexIVFFlat`: Balanced speed/accuracy with clustering
  - `IndexHNSWFlat`: Ultra-fast hierarchical search
- Interactive similarity search with configurable parameters
- Performance benchmarking and index comparison
- Export/import functionality for indices

### Interactive RAG Configuration
- Real-time parameter adjustment with sliders and controls
- Visual feedback for all processing steps
- Comprehensive comparison and recommendation system
- Export functionality for results and configurations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended for larger documents
- Internet connection for downloading models

### Installation

1. **Clone or download the project**:
```bash
git clone <repository-url>
cd RAG_567Task
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (if needed):
```python
import nltk
nltk.download('punkt')
```

### Running the Application

1. **Start the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Follow the workflow**:
   - Upload a document on the Home page
   - Configure and run chunking analysis (Task 5)
   - Generate embeddings with different models (Task 6)
   - Create FAISS indices and test search (Task 7)
   - Review comprehensive results and recommendations

## 📁 Project Structure

```
RAG_567Task/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── sample_document.txt             # Sample document for testing
├── README.md                       # This file
├── src/
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunking_methods.py     # Chunking implementations
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedding_methods.py    # Embedding implementations
│   ├── vector_storage/
│   │   ├── __init__.py
│   │   └── faiss_storage.py        # FAISS storage implementations
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── sidebar.py              # Navigation sidebar
│   │   └── pages.py                # Main application pages
│   └── utils/
│       ├── __init__.py
│       └── file_processor.py       # File upload and processing
└── data/                           # Created automatically for data storage
    ├── indices/                    # FAISS indices storage
    ├── embeddings/                 # Embedding cache
    └── results/                    # Export results
```

## 📖 User Guide

### 1. Document Upload (Home Page)
- **Supported formats**: TXT, PDF, DOCX, MD
- **File size limit**: 50MB
- **Features**: Automatic text extraction, statistics display, preview

### 2. Chunking Analysis (Task 5)
- **Configure parameters** for each chunking method
- **Run analysis** to generate chunks and statistics
- **Compare methods** using distribution charts and sample chunks
- **Select 2 best methods** for further evaluation

### 3. Embedding Generation (Task 6)
- **Choose embedding models** (1-3 models for comparison)
- **Select chunking method** from Task 5 results
- **Generate embeddings** with real-time progress tracking
- **Visualize results** with PCA/t-SNE and similarity heatmaps
- **Select best model** for FAISS indexing

### 4. FAISS Vector Storage (Task 7)
- **Configure index type** and parameters
- **Build indices** with performance monitoring
- **Test search functionality** with interactive queries
- **Benchmark performance** across different configurations
- **Export indices** and results

### 5. Comprehensive Results (Comparison Page)
- **View complete evaluation** summary
- **Compare all methods** side-by-side
- **Get recommendations** for different use cases
- **Export complete results** in JSON format

## ⚙️ Configuration Options

### Chunking Parameters
- **Fixed-Size**: Chunk size (200-1000 tokens), Overlap (0-200 tokens)
- **Semantic**: Min size (50-500 chars), Max size (500-1500 chars)
- **Recursive**: Chunk size (300-1000 chars), Overlap (0-300 chars)

### Embedding Models
- **all-MiniLM-L6-v2**: Fast, 384 dimensions, good for prototyping
- **all-mpnet-base-v2**: Balanced, 768 dimensions, production ready
- **all-roberta-large-v1**: Slow, 1024 dimensions, highest quality

### FAISS Index Types
- **Flat**: Exact search, best for <100k vectors
- **IVF**: Clustered search, good for 100k-1M vectors
- **HNSW**: Graph-based search, best for >1M vectors

## 🎯 Use Case Recommendations

### Speed-Optimized Setup
- **Chunking**: Fixed-Size (fast processing)
- **Embedding**: all-MiniLM-L6-v2 (lightweight)
- **Index**: HNSW (fast search)
- **Best for**: Real-time applications, chatbots

### Balanced Setup
- **Chunking**: Recursive Character Splitting
- **Embedding**: all-mpnet-base-v2 (balanced)
- **Index**: IVF (good speed/accuracy trade-off)
- **Best for**: Production applications, general use

### Quality-Optimized Setup
- **Chunking**: Semantic (context-aware)
- **Embedding**: all-roberta-large-v1 (high quality)
- **Index**: Flat (exact search)
- **Best for**: High-accuracy requirements, research

## 🔧 Troubleshooting

### Common Issues

1. **Memory errors with large documents**:
   - Reduce chunk size or use smaller embedding models
   - Process documents in smaller sections

2. **Slow embedding generation**:
   - Use lighter models (all-MiniLM-L6-v2)
   - Reduce batch size in configuration

3. **FAISS index build failures**:
   - Ensure sufficient memory
   - Reduce index parameters (nlist, M values)

4. **Model download issues**:
   - Check internet connection
   - Clear Streamlit cache and retry

### Performance Tips

- **Use caching**: Models and embeddings are cached automatically
- **Optimize parameters**: Start with default values and adjust based on results
- **Monitor memory**: Check system resources during processing
- **Batch processing**: Large documents are processed in batches automatically

## 📊 Sample Results

The platform provides comprehensive analytics including:
- Chunk distribution histograms
- Embedding similarity heatmaps
- Search performance benchmarks
- Model comparison tables
- Interactive visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **SentenceTransformers** for embedding models
- **FAISS** for efficient similarity search
- **LangChain** for text splitting utilities
- **Plotly** for interactive visualizations

## 📞 Support

For questions, issues, or feature requests:
1. Check the troubleshooting section above
2. Review the user guide for detailed instructions
3. Open an issue on the project repository
4. Consult the Streamlit and FAISS documentation for advanced usage

---

**Happy RAG Evaluation!** 🚀
