# RAG System Evaluation Platform - Setup Guide 🚀

## Quick Start (5 Minutes)

### 1. Prerequisites
- **Python 3.8+** (tested with Python 3.13)
- **4GB+ RAM** recommended for embedding models
- **Internet connection** for downloading models and dependencies

### 2. Installation

```bash
# Navigate to project directory
cd RAG_567Task

# Install dependencies
pip install streamlit sentence-transformers faiss-cpu plotly nltk tiktoken langchain python-docx PyPDF2

# Run the application
streamlit run app.py
```

### 3. Access the Application
- **URL**: http://localhost:8501
- **Browser**: Chrome, Firefox, Safari, or Edge

## Detailed Setup

### Step 1: Environment Setup

**Option A: Using pip (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Individual packages (if requirements.txt fails)**
```bash
pip install streamlit>=1.28.0
pip install sentence-transformers>=2.2.0
pip install faiss-cpu>=1.7.0
pip install plotly>=5.0.0
pip install nltk>=3.8.0
pip install tiktoken>=0.4.0
pip install langchain>=0.0.300
pip install python-docx>=0.8.0
pip install PyPDF2>=3.0.0
```

### Step 2: NLTK Data Download
The application automatically downloads required NLTK data on first run. If you encounter issues:

```python
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
```

### Step 3: Test Installation
```bash
python test_app.py
```

Expected output:
```
🔍 RAG System Evaluation Platform - Test Suite
==================================================
✅ All dependencies OK
✅ All imports OK  
✅ Basic functionality OK
🎉 All tests passed!
```

### Step 4: Launch Application
```bash
streamlit run app.py
```

## First-Time Usage

### 1. Upload Document
- Go to **Home** page
- Upload a document (TXT, PDF, DOCX, MD)
- Maximum file size: 50MB
- Review document statistics

### 2. Task 5: Chunking Analysis
- Configure chunking parameters with sliders
- Run analysis for all three methods
- Compare results and select 2 best methods

### 3. Task 6: Embedding Generation
- Select 1-3 embedding models
- Choose chunking method from Task 5
- Generate embeddings and review performance
- Select best model for FAISS indexing

### 4. Task 7: FAISS Vector Storage
- Configure index type and parameters
- Build FAISS index
- Test similarity search
- Run performance benchmarks

### 5. Results & Comparison
- Review comprehensive evaluation
- Get recommendations for your use case
- Export results and configurations

## Troubleshooting

### Common Issues

**1. NLTK punkt_tab Error**
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

**2. Memory Issues**
- Use lighter embedding models (all-MiniLM-L6-v2)
- Process smaller documents
- Close other applications

**3. Slow Performance**
- Models are downloaded once and cached
- Subsequent runs will be faster
- Use default parameters initially

**4. Import Errors**
```bash
pip install --upgrade streamlit sentence-transformers
```

**5. Port Already in Use**
```bash
streamlit run app.py --server.port 8502
```

### Performance Optimization

**For Large Documents (>1MB)**:
- Start with Fixed-Size chunking
- Use all-MiniLM-L6-v2 embedding model
- Use Flat FAISS index initially

**For Production Use**:
- Use Recursive Character Splitting
- Use all-mpnet-base-v2 embedding model
- Use IVF FAISS index

**For Research/High Accuracy**:
- Use Semantic chunking
- Use all-roberta-large-v1 embedding model
- Use Flat FAISS index

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 5GB free space
- **Python**: 3.10+

### Model Sizes
- **all-MiniLM-L6-v2**: ~90MB
- **all-mpnet-base-v2**: ~420MB
- **all-roberta-large-v1**: ~1.3GB

## Configuration Options

### Chunking Parameters
```python
# Fixed-Size Chunking
chunk_size: 200-1000 tokens (default: 500)
overlap: 0-200 tokens (default: 50)

# Semantic Chunking  
min_chunk_size: 50-500 chars (default: 100)
max_chunk_size: 500-1500 chars (default: 800)

# Recursive Character Splitting
chunk_size: 300-1000 chars (default: 600)
overlap: 0-300 chars (default: 100)
```

### FAISS Index Parameters
```python
# IVF Index
nlist: 10-500 (default: 100)
nprobe: 1-50 (default: 10)

# HNSW Index
M: 4-64 (default: 16)
efConstruction: 50-500 (default: 200)
```

## Data Storage

The application creates the following directories:
```
RAG_567Task/
├── data/
│   ├── indices/     # FAISS indices
│   ├── embeddings/  # Cached embeddings
│   ├── results/     # Export results
│   └── temp/        # Temporary files
```

## Security Notes

- **Local Processing**: All data stays on your machine
- **No External APIs**: No data sent to external services
- **Model Downloads**: Only from HuggingFace (trusted source)
- **File Uploads**: Processed locally, not stored permanently

## Support

### Getting Help
1. **Check this guide** for common solutions
2. **Run test script**: `python test_app.py`
3. **Check logs**: Look at terminal output for errors
4. **Review documentation**: README.md and USER_MANUAL.md

### Reporting Issues
Include the following information:
- Python version: `python --version`
- Operating system
- Error message (full traceback)
- Steps to reproduce

### Performance Monitoring
The application provides real-time metrics:
- Processing times
- Memory usage estimates
- Model loading status
- Search performance

---

**Ready to evaluate your RAG system!** 🎯

For detailed usage instructions, see [USER_MANUAL.md](USER_MANUAL.md)
For technical details, see [README.md](README.md)
