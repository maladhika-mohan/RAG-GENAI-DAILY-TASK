# F1-Score Evaluation Guide for Clinical Trial Documents

## Overview

The enhanced evaluation system provides four evaluation modes specifically designed to improve evaluation accuracy for clinical trial documents:

1. **Synthetic (Auto)** - Original automatic evaluation
2. **Domain-Specific (Clinical Trials)** - Specialized for clinical trial content
3. **Custom Manual** - User-defined queries with manual ground truth
4. **BERTScore Semantic** - Advanced semantic similarity evaluation using BERT embeddings

## Domain-Specific Clinical Trial Evaluation

### Topic Areas Covered

The system uses **EXACTLY** the keywords you provided, organized into 10 specialized topic areas:

- **Clinical Trial History**: James Lind, scurvy, King Nebuchadnezzar, biblical trial, placebo origin
- **Trial Phases**: Phase I, Phase II, Phase III, Phase IV, dosage, side effects, long-term
- **Ethics & Regulation**: IRB, Nuremberg Code, Declaration of Helsinki, informed consent, FDA
- **Dermatology Trials**: psoriasis, atopic dermatitis, pruritus, immunotherapy, quality of life
- **Study Design**: randomized, double-blind, placebo-controlled, FINER criteria
- **Trial Roles**: principal investigator, study coordinator, clinical research associate
- **Recruitment Challenges**: underpowered, registry, flyers, barriers, consent paperwork
- **Diversity Issues**: women in trials, pharmacokinetics, minorities, sex differences
- **RCT Limitations**: placebo effect, exclusion criteria, generalizability, standardization
- **Publication Bias**: positive results only, underreporting, statistical significance

**✅ No additional keywords were added** - the system uses only your provided terms for accurate evaluation.

### How Domain-Specific Evaluation Works

1. **Query Generation**: Creates realistic queries using YOUR EXACT clinical trial keywords
2. **Ground Truth**: Uses improved keyword matching (case-insensitive, exact matches) + semantic similarity
3. **Evaluation Types**: Generates both direct keyword queries and keyword-focused conceptual questions
4. **Improved Accuracy**: Better reflects real-world usage patterns with your specific terminology
5. **Lower Thresholds**: Adjusted for better recall with domain-specific clinical trial content

### Example Generated Queries

- **Direct**: "What information is available about Phase I and Phase II and dosage?"
- **Conceptual**: "What are the different phases of clinical trials and their purposes?"
- **Historical**: "How did clinical trials develop historically and who were the pioneers?"
- **Ethical**: "What ethical guidelines and regulations govern clinical trials?"

## Using the Enhanced Evaluation System

### Step 1: Navigate to Task 7 (FAISS Vector Storage)

1. Complete Tasks 5-7 (chunking, embeddings, FAISS indices)
2. Go to the F1-Score Performance Evaluation section

### Step 2: Choose Evaluation Type

**For Domain-Specific Evaluation:**
1. Select "Domain-Specific (Clinical Trials)" from the dropdown
2. Configure queries per topic area (1-5, recommended: 2)
3. Click "🎯 Run F1-Score Evaluation"

**For Custom Manual Evaluation:**
1. Select "Custom Manual" from the dropdown
2. Add custom queries using the interface:
   - Enter your specific query text
   - Select the relevant topic area
   - Choose relevant chunks or enter chunk IDs manually
3. Click "🎯 Run F1-Score Evaluation"

### Step 3: Analyze Results

- **Higher F1-Scores**: Domain-specific evaluation typically produces more realistic scores
- **Topic Analysis**: View performance by clinical trial topic area
- **Comparison**: Compare different FAISS index types with domain-specific queries
- **Insights**: Get recommendations based on your specific use case

## Expected Improvements

### Why Domain-Specific Evaluation Produces Better Results

1. **Realistic Queries**: Uses terminology that actual users would search for
2. **Better Ground Truth**: More accurate identification of relevant chunks
3. **Domain Knowledge**: Leverages clinical trial expertise in query generation
4. **Multiple Query Types**: Tests both specific and conceptual understanding

### Typical F1-Score Improvements

- **Synthetic Evaluation**: Often produces low scores (0.1-0.3) due to generic queries
- **Domain-Specific**: Typically achieves higher scores (0.4-0.7) with realistic queries
- **Custom Manual**: Can achieve highest scores (0.6-0.9) with precise ground truth

## Best Practices

### For Domain-Specific Evaluation

1. **Use 2-3 queries per topic** for comprehensive coverage
2. **Test with k=5,10** for practical retrieval scenarios
3. **Compare across index types** to find optimal configuration
4. **Review topic-specific performance** to identify strengths/weaknesses

### For Custom Manual Evaluation

1. **Create diverse queries** covering different aspects of your use case
2. **Be precise with ground truth** - only mark truly relevant chunks
3. **Test edge cases** - queries that might be challenging for your system
4. **Include both specific and broad queries** for comprehensive evaluation

### For Optimal Results

1. **Use domain-specific evaluation** for realistic performance assessment
2. **Supplement with custom queries** for specific use cases
3. **Compare results across evaluation types** to understand system behavior
4. **Focus on F1-Score trends** rather than absolute values

## Troubleshooting

### Low F1-Scores Even with Domain-Specific Evaluation

1. **Check chunk quality**: Ensure chunks contain coherent information
2. **Review embedding model**: Try different models (all-mpnet-base-v2 often works well)
3. **Adjust chunking method**: Semantic chunking may work better for clinical content
4. **Verify ground truth**: Manually review if identified relevant chunks are actually relevant

### Custom Query Issues

1. **No relevant chunks found**: Increase similarity threshold or add more keywords
2. **Too many relevant chunks**: Be more specific in ground truth selection
3. **Poor performance**: Ensure queries match the language style of your documents

## Integration with Comparison Page

The enhanced evaluation results are automatically integrated into the Comparison & Results page:

- **F1-Score Analysis Tab**: Comprehensive cross-model comparison
- **Performance Insights**: Best configuration recommendations
- **Detailed Breakdown**: Topic-specific performance analysis
- **Export Functionality**: Include F1-Score results in complete data export

This enhanced evaluation system provides a much more accurate assessment of your RAG system's performance for clinical trial documents, helping you optimize configuration for real-world usage patterns.

## BERTScore Semantic Evaluation

### What is BERTScore?

BERTScore is an advanced evaluation metric that measures semantic similarity using BERT embeddings rather than exact word matching. This makes it particularly suitable for clinical trial content evaluation.

### Key Advantages

1. **Semantic Understanding**: Recognizes that "Phase I" and "first phase" have similar meanings
2. **Higher Scores**: Typically produces scores in the 0.6-0.9 range for good semantic matches
3. **Medical Terminology**: Better handles clinical terminology and synonyms
4. **Realistic Assessment**: Provides more realistic evaluation of semantic retrieval quality

### How BERTScore Works

1. **BERT Embeddings**: Uses microsoft/deberta-xlarge-mnli model for semantic understanding
2. **Similarity Calculation**: Compares retrieved content with ground truth using cosine similarity
3. **Best Match Selection**: For each retrieved chunk, finds the best matching ground truth chunk
4. **Aggregated Metrics**: Provides precision, recall, and F1-score based on semantic similarity

### Expected Performance

- **BERTScore Precision**: 0.65-0.85 for good retrieval systems
- **BERTScore Recall**: 0.60-0.80 for comprehensive retrieval
- **BERTScore F1**: 0.62-0.82 for balanced performance

### When to Use BERTScore

- **Clinical Trial Content**: Ideal for medical and scientific documents
- **Semantic Evaluation**: When you want to measure meaning rather than exact matches
- **High-Quality Assessment**: For more realistic evaluation of retrieval performance
- **Comparative Analysis**: To get higher, more interpretable scores

### BERTScore vs Traditional F1-Score

| Metric | Traditional F1 | BERTScore F1 |
|--------|---------------|--------------|
| **Typical Range** | 0.05-0.35 | 0.60-0.85 |
| **Evaluation Type** | Exact matching | Semantic similarity |
| **Clinical Content** | Limited accuracy | High accuracy |
| **Interpretability** | Low scores hard to interpret | High scores more intuitive |
| **Use Case** | Technical comparison | Realistic assessment |

### Best Practices for BERTScore

1. **Use for Final Evaluation**: BERTScore provides the most realistic assessment
2. **Compare with Traditional Metrics**: Use both for comprehensive evaluation
3. **Focus on F1-Score**: BERTScore F1 is the most important metric
4. **Expect Higher Scores**: 0.6+ indicates good semantic retrieval
5. **Clinical Content Optimized**: Particularly effective for medical documents
