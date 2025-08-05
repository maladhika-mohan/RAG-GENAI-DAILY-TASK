# RAG System Evaluation Platform - User Manual 📚

## Table of Contents
1. [Getting Started](#getting-started)
2. [Navigation Guide](#navigation-guide)
3. [Task 5: Chunking Methods](#task-5-chunking-methods)
4. [Task 6: Embedding Implementation](#task-6-embedding-implementation)
5. [Task 7: FAISS Vector Storage](#task-7-faiss-vector-storage)
6. [Comparison & Results](#comparison--results)
7. [Tips & Best Practices](#tips--best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### First Launch
1. **Start the application**: Run `streamlit run app.py`
2. **Open your browser**: Navigate to `http://localhost:8501`
3. **Upload a document**: Begin with the Home page to upload your test document

### Understanding the Interface
- **Sidebar Navigation**: Use the left sidebar to move between tasks
- **Progress Indicators**: Track your completion status in the sidebar
- **Help Sections**: Click on expandable help sections for detailed guidance
- **Settings**: Access debug options and data clearing in the sidebar

## Navigation Guide

### Sidebar Features
- **Page Selection**: Radio buttons to navigate between tasks
- **Progress Tracking**: Visual indicators showing completion status
- **System Information**: Real-time stats about your current session
- **Help Section**: Quick tips and getting started guide
- **Settings**: Debug options and data management

### Main Content Areas
- **Headers**: Clear task identification and descriptions
- **Configuration Sections**: Interactive parameter controls
- **Results Display**: Comprehensive analytics and visualizations
- **Action Buttons**: Primary actions clearly marked with icons

## Task 5: Chunking Methods

### Overview
Compare three different text chunking approaches to find the most effective methods for your use case.

### Step-by-Step Process

#### 1. Configuration
**Fixed-Size Chunking**:
- **Chunk Size**: 200-1000 tokens (default: 500)
- **Overlap**: 0-200 tokens (default: 50)
- **Best for**: Consistent processing, predictable memory usage

**Semantic Chunking**:
- **Min Chunk Size**: 50-500 characters (default: 100)
- **Max Chunk Size**: 500-1500 characters (default: 800)
- **Best for**: Preserving context, natural language boundaries

**Recursive Character Splitting**:
- **Chunk Size**: 300-1000 characters (default: 600)
- **Overlap**: 0-300 characters (default: 100)
- **Best for**: Balanced approach, mixed content types

#### 2. Running Analysis
1. **Adjust parameters** using the sliders
2. **Click "Run Chunking Analysis"** to process all methods
3. **Monitor progress** with the real-time progress bar
4. **Review results** in the comprehensive display

#### 3. Interpreting Results
**Summary Statistics Table**:
- Compare total chunks, average sizes, and processing times
- Look for balanced chunk distributions

**Chunk Distribution Charts**:
- Histogram showing size distribution for each method
- Identify methods with consistent chunk sizes

**Sample Chunks Display**:
- Review first 3 chunks from each method
- Assess semantic completeness and readability

#### 4. Method Selection
- **Select exactly 2 methods** for further evaluation
- **Consider your use case**:
  - Speed: Fixed-Size
  - Quality: Semantic
  - Balance: Recursive
- **Read recommendations** for each selected method

### Key Metrics to Consider
- **Chunk Coherence**: Do chunks maintain semantic meaning?
- **Information Coverage**: Is important information preserved?
- **Retrieval Suitability**: Are chunks appropriate for search?

## Task 6: Embedding Implementation

### Overview
Generate embeddings using multiple SentenceTransformer models and compare their performance characteristics.

### Step-by-Step Process

#### 1. Model Selection
Choose 1-3 models for comparison:

**all-MiniLM-L6-v2**:
- 384 dimensions, fast processing
- Best for: Prototyping, real-time applications
- Memory usage: Low

**all-mpnet-base-v2**:
- 768 dimensions, balanced performance
- Best for: Production applications
- Memory usage: Medium

**all-roberta-large-v1**:
- 1024 dimensions, highest quality
- Best for: Accuracy-critical applications
- Memory usage: High

#### 2. Chunking Method Selection
- Choose from your selected methods in Task 5
- The system will use chunks from this method for embedding

#### 3. Embedding Generation
1. **Click "Generate Embeddings"** to start processing
2. **Monitor progress** for each model
3. **Wait for completion** (time varies by model and document size)

#### 4. Results Analysis
**Performance Comparison Table**:
- Compare processing time, memory usage, and throughput
- Identify the fastest and most efficient models

**Model Recommendations**:
- System provides recommendations for different use cases
- Consider speed vs. quality trade-offs

**Detailed Analysis Tabs**:
- **Statistics**: Dimensions, processing metrics
- **Sample Embeddings**: View actual embedding vectors
- **Visualizations**: PCA/t-SNE plots, similarity heatmaps
- **Quality Metrics**: Similarity distribution analysis

#### 5. Model Selection
- **Choose the best model** for FAISS indexing
- **Consider your requirements**:
  - Speed: all-MiniLM-L6-v2
  - Balance: all-mpnet-base-v2
  - Quality: all-roberta-large-v1

### Understanding Visualizations
**PCA/t-SNE Plots**:
- Show embedding distribution in 2D space
- Clusters indicate similar content
- Spread indicates diversity

**Similarity Heatmaps**:
- Show relationships between chunks
- Diagonal should be high (self-similarity)
- Off-diagonal patterns show content relationships

## Task 7: FAISS Vector Storage

### Overview
Create FAISS indices for efficient similarity search and compare different index types.

### Step-by-Step Process

#### 1. Index Type Selection
**IndexFlatIP (Flat)**:
- Exact search, best quality
- Best for: <100k vectors, highest accuracy needs
- Trade-off: Slower for large datasets

**IndexIVFFlat (IVF)**:
- Clustered search, balanced performance
- Best for: 100k-1M vectors, production use
- Parameters: nlist (clusters), nprobe (search clusters)

**IndexHNSWFlat (HNSW)**:
- Graph-based search, fastest
- Best for: >1M vectors, real-time search
- Parameters: M (connections), efConstruction (build quality)

#### 2. Parameter Configuration
**IVF Parameters**:
- **nlist**: Number of clusters (10-500)
- **nprobe**: Clusters to search (1-50)
- Higher values = better accuracy, slower search

**HNSW Parameters**:
- **M**: Connections per node (4-64)
- **efConstruction**: Build quality (50-500)
- Higher values = better quality, slower build

#### 3. Search Configuration
- **Number of results (k)**: How many similar chunks to return
- **Similarity threshold**: Minimum score for results
- **Max results to display**: UI display limit

#### 4. Index Building
1. **Click "Build FAISS Index"** to create the index
2. **Monitor progress** and build statistics
3. **Review index metrics**: Build time, size, throughput

#### 5. Interactive Search
**Search Interface**:
- Enter queries in the text area
- Adjust search parameters with sliders
- Click "Search" to find similar chunks

**Search Results**:
- Results ranked by similarity score
- Expandable sections with highlighted text
- Similarity scores and metadata

#### 6. Performance Benchmarking
- **Run benchmarks** to test search performance
- **View performance charts** showing speed vs. accuracy
- **Compare different k values** and their impact

### Search Tips
- **Use natural language queries** for best results
- **Adjust similarity threshold** to filter results
- **Try different k values** to see more/fewer results
- **Use specific terms** from your document for targeted search

## Comparison & Results

### Overview
Comprehensive summary of all evaluation results with recommendations for different use cases.

### Understanding the Results
**System Overview Metrics**:
- Total chunks created across all methods
- Number of models and indices tested
- Selected configuration summary

**Detailed Comparison Tabs**:
- **Chunking**: Compare all chunking methods
- **Embedding**: Compare all embedding models
- **FAISS**: Compare all index types

**Final Recommendations**:
- **Speed-Optimized**: Best for real-time applications
- **Balanced**: Best for production use
- **Quality-Optimized**: Best for accuracy-critical tasks

### Export Options
- **CSV exports**: Individual component results
- **JSON export**: Complete evaluation results
- **Index summaries**: FAISS index configurations

## Tips & Best Practices

### Document Preparation
- **Clean text**: Remove unnecessary formatting
- **Reasonable size**: 10KB-10MB works best
- **Clear structure**: Well-formatted documents chunk better

### Parameter Selection
- **Start with defaults**: Adjust based on results
- **Consider your use case**: Speed vs. quality trade-offs
- **Test incrementally**: Small changes, observe impact

### Performance Optimization
- **Use caching**: Models are cached automatically
- **Monitor memory**: Large models need more RAM
- **Batch processing**: System handles large documents automatically

### Result Interpretation
- **Look for patterns**: Consistent results across methods
- **Consider context**: Your specific use case requirements
- **Test with queries**: Use the search interface to validate

## Troubleshooting

### Common Issues

**"Please upload a document first"**:
- Go to Home page and upload a supported file format
- Ensure file is under 50MB size limit

**"Please complete previous tasks"**:
- Tasks must be completed in order (5 → 6 → 7)
- Check progress indicators in sidebar

**Memory errors during processing**:
- Use smaller documents or lighter models
- Close other applications to free memory
- Restart the application if needed

**Slow processing**:
- Choose lighter embedding models
- Reduce chunk sizes
- Use smaller documents for testing

**Search returns no results**:
- Lower the similarity threshold
- Try different query terms
- Check if embeddings were generated correctly

### Performance Issues

**Application feels slow**:
- Clear browser cache
- Restart Streamlit application
- Check system memory usage

**Models taking long to download**:
- Ensure stable internet connection
- Models are downloaded once and cached
- Subsequent runs will be faster

### Getting Help
1. **Check this manual** for detailed instructions
2. **Review error messages** for specific guidance
3. **Use help sections** in the application
4. **Check system requirements** and dependencies

---

**Need more help?** Refer to the main README.md file or check the troubleshooting section for additional guidance.
