# RAG Daily Tasks - Comprehensive NLP & RAG System Implementation 🚀

A comprehensive collection of Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) implementations developed as part of daily learning tasks. This repository showcases various NLP techniques, from basic sentiment analysis to advanced RAG system evaluation platforms.

## 📋 Project Overview

This repository contains multiple interconnected projects focusing on different aspects of NLP and RAG systems:

- **Task 2**: Sentiment Analysis Comparison (VADER, BERT, Logistic Regression)
- **Task 3**: Article Summarization using RAG with Groq API
- **Task 4**: Advanced Article Summarization with Enhanced Features
- **Tasks 5-7**: Comprehensive RAG System Evaluation Platform

## 🗂️ Repository Structure

```
RAG_daily_task/
├── README.md                           # This comprehensive guide
├── Data_set/                           # Datasets for all tasks
│   ├── Day_2task_data/
│   │   └── IMDB Dataset.csv           # IMDB movie reviews dataset
│   ├── Day_3_4task_data/              # Articles for summarization
│   │   ├── ChatGPT Search A Guide With Example.txt
│   │   ├── Context Engineering A Guide With Ex.txt
│   │   └── Small Language Models A Guide With.txt
│   └── Day_5_6_taskData/
│       └── International Journal of Women's De.txt
├── V_B_L_ RAG_Task_2.ipynb           # Sentiment Analysis Comparison
├── Article_summarizer_RAG_task_3.ipynb # Basic Article Summarization
├── Article_summarizer_day4.ipynb      # Enhanced Article Summarization
├── day5_6TASK.ipynb                   # RAG Components Analysis
└── RAG_567Task/                       # Complete RAG Evaluation Platform
    ├── app.py                         # Streamlit web application
    ├── config.py                      # Configuration settings
    ├── requirements.txt               # Python dependencies
    ├── src/                           # Source code modules
    ├── data/                          # Generated data storage
    └── docs/                          # Documentation
```

## 🎯 Task Descriptions

### Task 2: Sentiment Analysis Comparison 📊
**File**: `V_B_L_ RAG_Task_2.ipynb`

Comprehensive comparison of three sentiment analysis approaches:
- **VADER Sentiment**: Rule-based sentiment analysis
- **BERT Model**: Transformer-based deep learning approach
- **Logistic Regression**: Traditional ML with TF-IDF features

**Features**:
- Performance comparison on IMDB movie reviews dataset
- Accuracy metrics and classification reports
- Processing time analysis
- Model strengths and weaknesses evaluation

### Task 3: Basic Article Summarization 📝
**File**: `Article_summarizer_RAG_task_3.ipynb`

RAG-based article summarization using Groq API:
- Document processing and chunking
- Vector embeddings for semantic search
- Context-aware summarization
- Interactive Jupyter notebook interface

**Key Technologies**:
- Groq API for LLM inference
- Sentence transformers for embeddings
- FAISS for vector similarity search

### Task 4: Enhanced Article Summarization ⚡
**File**: `Article_summarizer_day4.ipynb`

Advanced version with improved features:
- Enhanced chunking strategies
- Better context retrieval
- Improved summarization quality
- Performance optimizations

### Tasks 5-7: RAG System Evaluation Platform 🔍
**Directory**: `RAG_567Task/`

A comprehensive Streamlit-based platform for evaluating RAG system components:

#### Task 5: Chunking Methods Analysis
- Fixed-size chunking with configurable overlap
- Semantic chunking at sentence/paragraph boundaries
- Recursive character splitting with multiple separators
- Interactive comparison and visualization

#### Task 6: Embedding Implementation
- Multiple SentenceTransformer models comparison
- Real-time embedding generation
- Dimensionality reduction visualization (PCA, t-SNE)
- Performance benchmarking

#### Task 7: FAISS Vector Storage
- Multiple index types (Flat, IVF, HNSW)
- Interactive similarity search
- Performance benchmarking
- Export/import functionality

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection for model downloads

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd RAG_daily_task
```

2. **Set up virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies for Jupyter notebooks**:
```bash
pip install jupyter pandas numpy scikit-learn transformers torch
pip install vaderSentiment groq sentence-transformers faiss-cpu
```

4. **For the RAG Evaluation Platform**:
```bash
cd RAG_567Task
pip install -r requirements.txt
```

### Running the Projects

#### Jupyter Notebooks
```bash
jupyter notebook
# Open any of the .ipynb files to run individual tasks
```

#### RAG Evaluation Platform
```bash
cd RAG_567Task
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

## 📊 Key Features & Capabilities

### Sentiment Analysis (Task 2)
- **Multi-model comparison** with performance metrics
- **Large dataset processing** (5000+ IMDB reviews)
- **Comprehensive evaluation** with classification reports
- **Processing time analysis** for different approaches

### Article Summarization (Tasks 3-4)
- **RAG-based approach** for context-aware summaries
- **Multiple document support** with various formats
- **Semantic search** for relevant context retrieval
- **Interactive interfaces** for easy experimentation

### RAG System Evaluation (Tasks 5-7)
- **Complete RAG pipeline** evaluation and comparison
- **Interactive web interface** with real-time feedback
- **Multiple configuration options** for different use cases
- **Performance benchmarking** and recommendations
- **Export functionality** for results and configurations

## 🛠️ Technologies Used

### Core Libraries
- **Streamlit**: Web application framework
- **Transformers**: Hugging Face transformer models
- **SentenceTransformers**: Embedding models
- **FAISS**: Efficient similarity search
- **LangChain**: Text processing utilities

### Machine Learning
- **scikit-learn**: Traditional ML algorithms
- **PyTorch**: Deep learning framework
- **NLTK**: Natural language processing
- **Pandas/NumPy**: Data manipulation

### Visualization
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Statistical plots
- **Streamlit components**: Web-based charts

## 📈 Performance Insights

### Sentiment Analysis Results
- **VADER**: Fast, rule-based, good for social media text
- **BERT**: High accuracy, slower processing, best for complex text
- **Logistic Regression**: Balanced performance, interpretable results

### RAG System Recommendations
- **Speed-optimized**: Fixed chunking + MiniLM + HNSW
- **Balanced**: Recursive chunking + MPNet + IVF
- **Quality-optimized**: Semantic chunking + RoBERTa + Flat

## 🔧 Configuration & Customization

### Environment Variables
```bash
# For Groq API (Task 3-4)
export GROQ_API_KEY="your-groq-api-key"
```

### Customizable Parameters
- Chunk sizes and overlap ratios
- Embedding model selection
- FAISS index configurations
- Similarity search parameters

## 📚 Learning Outcomes

This repository demonstrates:
- **Progressive complexity** from basic to advanced NLP tasks
- **Comparative analysis** of different approaches
- **Production-ready implementations** with web interfaces
- **Performance optimization** techniques
- **Best practices** in RAG system development

