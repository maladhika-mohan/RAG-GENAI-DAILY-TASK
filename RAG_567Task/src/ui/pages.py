"""
Main pages for RAG System Evaluation Platform
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from src.utils.file_processor import create_file_upload_interface
from src.chunking.chunking_methods import ChunkingMethods
from config import CHUNKING_METHODS, EMBEDDING_MODELS, FAISS_CONFIG

# Chat memory functionality disabled
CHAT_MEMORY_AVAILABLE = False

def show_home_page():
    """Display the home page with file upload"""



    st.markdown("""
    ## Welcome to the RAG System Evaluation Platform! 🚀

    This platform helps you evaluate and compare different components of a Retrieval-Augmented Generation (RAG) system:
    
    ### 📋 Simple 4-Step Process:

    **Task 5: Chunking** 🔪
    - Automatically processes text with optimal settings

    **Task 6: Embeddings** 🧮
    - Uses best-performing model (all-mpnet-base-v2)

    **Task 7: Vector Storage** 🗂️
    - Creates optimized FAISS index for search

    **Task 8: Results** 📊
    - View performance metrics and comparisons

    ### 🎯 Getting Started:
    Upload a document below to begin the evaluation process.
    """)
    
    # File upload interface
    text, file_info = create_file_upload_interface()
    
    if text:
        st.session_state.processed_text = text
        st.session_state.file_info = file_info
        
        st.success("✅ Document uploaded successfully! You can now proceed to Task 5.")
        
        # Show next steps
        st.info("📍 **Next Steps:** Navigate to 'Task 5: Chunking Methods' in the sidebar to begin chunking analysis.")



def show_task5_page():
    """Display Task 5 - Chunking Methods page"""



    st.header("🔪 Task 5: Chunking Methods Analysis")

    if not st.session_state.get('processed_text'):
        st.warning("⚠️ Please upload a document on the Home page first.")
        return
    
    # Use smart defaults - no configuration needed
    fixed_chunk_size = 500
    fixed_overlap = 50
    semantic_min = 100
    semantic_max = 800
    recursive_chunk_size = 600
    recursive_overlap = 100
    
    # Auto-run chunking with smart defaults
    if st.button("🔄 Run Chunking", type="primary"):
        
        chunking_methods = ChunkingMethods()
        text = st.session_state.processed_text
        
        with st.spinner("Processing chunking methods..."):
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            
            # Fixed-size chunking
            status_text.text("Processing Fixed-Size Chunking...")
            progress_bar.progress(0.33)
            results['fixed_size'] = chunking_methods.fixed_size_chunking(
                text, fixed_chunk_size, fixed_overlap
            )
            
            # Semantic chunking
            status_text.text("Processing Semantic Chunking...")
            progress_bar.progress(0.66)
            results['semantic'] = chunking_methods.semantic_chunking(
                text, semantic_min, semantic_max
            )
            
            # Recursive splitting
            status_text.text("Processing Recursive Character Splitting...")
            progress_bar.progress(1.0)
            results['recursive'] = chunking_methods.recursive_character_splitting(
                text, recursive_chunk_size, recursive_overlap
            )
            
            status_text.text("Chunking analysis complete!")
            progress_bar.empty()
            status_text.empty()
        
        st.session_state.chunking_results = results
        st.success("✅ Chunking analysis completed!")


    
    # Display results if available
    if st.session_state.get('chunking_results'):
        display_chunking_results(st.session_state.chunking_results)

def display_chunking_results(results):
    """Display chunking analysis results"""
    
    st.subheader("📊 Chunking Results")
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    summary_data = []
    for method_key, result in results.items():
        summary_data.append({
            'Method': result.method_name,
            'Total Chunks': result.total_chunks,
            'Avg Tokens': f"{result.avg_chunk_size_tokens:.1f}",
            'Avg Characters': f"{result.avg_chunk_size_chars:.1f}",
            'Min Size (tokens)': result.min_chunk_size,
            'Max Size (tokens)': result.max_chunk_size,
            'Processing Time (s)': f"{result.processing_time:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Visualization
    st.markdown("### 📈 Chunk Distribution Analysis")
    
    # Create chunk size distribution plots
    fig_dist = go.Figure()
    
    for method_key, result in results.items():
        chunk_sizes = [chunk.token_count for chunk in result.chunks]
        fig_dist.add_trace(go.Histogram(
            x=chunk_sizes,
            name=result.method_name,
            opacity=0.7,
            nbinsx=20
        ))
    
    fig_dist.update_layout(
        title="Chunk Size Distribution (Tokens)",
        xaxis_title="Chunk Size (Tokens)",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Sample chunks display
    st.markdown("### 📝 Sample Chunks")
    
    method_tabs = st.tabs([result.method_name for result in results.values()])
    
    for i, (method_key, result) in enumerate(results.items()):
        with method_tabs[i]:
            st.markdown(f"**{result.method_name}** - Showing first 3 chunks:")
            
            for j, chunk in enumerate(result.chunks[:3]):
                with st.expander(f"Chunk {j+1} ({chunk.token_count} tokens, {chunk.char_count} chars)"):
                    st.text(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)
                    
                    # Chunk metadata
                    st.markdown(f"""
                    **Metadata:**
                    - Chunk ID: {chunk.chunk_id}
                    - Position: {chunk.start_pos} - {chunk.end_pos}
                    - Token Count: {chunk.token_count}
                    - Character Count: {chunk.char_count}
                    """)
    
    # Method comparison and selection
    st.markdown("### 🎯 Auto-Selected Best Methods")

    # Auto-select the best 2 methods (use first 2 as default)
    method_options = [result.method_name for result in results.values()]
    selected_methods = method_options[:2] if len(method_options) >= 2 else method_options
    
    if len(selected_methods) >= 2:
        st.success(f"✅ Selected methods: {', '.join(selected_methods[:2])}")
        st.session_state.selected_chunking_methods = selected_methods[:2]
    
    # Next steps
    if st.session_state.get('selected_chunking_methods'):
        st.info("📍 **Next Steps:** Navigate to 'Task 6: Embedding Implementation' to generate embeddings for your selected chunking methods.")



def show_task6_page():
    """Display Task 6 - Embedding Implementation page"""



    st.header("🧮 Task 6: Embedding Implementation")

    if not st.session_state.get('chunking_results'):
        st.warning("⚠️ Please complete Task 5 (Chunking Methods) first.")
        return

    if not st.session_state.get('selected_chunking_methods'):
        st.warning("⚠️ Please select 2 chunking methods in Task 5 first.")
        return

    st.markdown("""
    ### Overview
    This section implements embedding generation using multiple SentenceTransformer models:
    1. **all-MiniLM-L6-v2**: Lightweight, fast model (384 dimensions)
    2. **all-mpnet-base-v2**: Balanced quality and speed (768 dimensions)
    3. **all-roberta-large-v1**: High quality, slower model (1024 dimensions)
    """)

    from src.embedding.embedding_methods import EmbeddingMethods

    # Use best default model - all-mpnet-base-v2 (balanced performance)
    selected_models = ["all-mpnet-base-v2"]
    st.info("Using all-mpnet-base-v2 (balanced performance and quality)")

    # Auto-use first selected chunking method
    selected_chunking_methods = st.session_state.get('selected_chunking_methods', [])
    chunking_results = st.session_state.get('chunking_results', {})

    if not selected_chunking_methods:
        st.error("No chunking methods found. Please complete Task 5 first.")
        return

    # Map method names to keys
    method_name_to_key = {}
    for key, result in chunking_results.items():
        method_name_to_key[result.method_name] = key

    # Use first available method
    selected_chunking_method = selected_chunking_methods[0]
    method_key = method_name_to_key[selected_chunking_method]
    selected_chunks = [chunk.text for chunk in chunking_results[method_key].chunks]

    st.info(f"📊 Using {len(selected_chunks)} chunks from {selected_chunking_method}")

    # Generate embeddings
    if st.button("🚀 Generate Embeddings", type="primary"):

        embedding_methods = EmbeddingMethods()
        embedding_results = {}

        total_models = len(selected_models)
        main_progress = st.progress(0)
        main_status = st.empty()

        for i, model_name in enumerate(selected_models):
            main_status.text(f"Processing model {i+1}/{total_models}: {model_name}")

            try:
                result = embedding_methods.generate_embeddings(
                    chunks=selected_chunks,
                    model_name=model_name,
                    show_progress=True
                )
                embedding_results[model_name] = result

                main_progress.progress((i + 1) / total_models)

            except Exception as e:
                st.error(f"Error processing {model_name}: {str(e)}")
                continue

        main_progress.empty()
        main_status.empty()

        if embedding_results:
            st.session_state.embedding_results = embedding_results
            st.success(f"✅ Successfully generated embeddings for {len(embedding_results)} models!")


        else:
            st.error("❌ Failed to generate embeddings for any model.")

    # Display results if available
    if st.session_state.get('embedding_results'):
        display_embedding_results(st.session_state.embedding_results)

def display_embedding_results(embedding_results):
    """Display embedding generation results"""

    st.subheader("📊 Embedding Results")

    from src.embedding.embedding_methods import EmbeddingMethods
    embedding_methods = EmbeddingMethods()

    # Performance comparison
    st.markdown("### ⚡ Performance Comparison")

    comparison = embedding_methods.compare_embeddings(list(embedding_results.values()))
    st.dataframe(comparison['performance_comparison'], use_container_width=True)

    # Recommendations
    st.markdown("### 💡 Model Recommendations")

    recommendations = comparison['recommendations']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🏃 **Fastest**: {recommendations.get('fastest', 'N/A')}")
    with col2:
        st.info(f"⚖️ **Balanced**: {recommendations.get('balanced', 'N/A')}")
    with col3:
        st.warning(f"🎯 **Highest Quality**: {recommendations.get('highest_quality', 'N/A')}")

    # Detailed results for each model
    st.markdown("### 📈 Detailed Analysis")

    model_tabs = st.tabs(list(embedding_results.keys()))

    for i, (model_name, result) in enumerate(embedding_results.items()):
        with model_tabs[i]:

            # Model statistics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Statistics")
                st.metric("Embedding Dimensions", result.performance_metrics['embedding_dimensions'])
                st.metric("Processing Time", f"{result.embedding_time:.2f}s")
                st.metric("Chunks per Second", f"{result.performance_metrics['chunks_per_second']:.1f}")
                st.metric("Memory Usage", f"{result.performance_metrics['memory_usage_mb']:.1f} MB")

            with col2:
                st.markdown("#### 🔍 Sample Embeddings")
                st.markdown("First 10 dimensions of first embedding:")
                sample_embedding = result.embeddings[0][:10]
                st.code(f"[{', '.join([f'{x:.4f}' for x in sample_embedding])}]")

                st.markdown("Embedding shape:")
                st.code(f"{result.embeddings.shape}")

            # Visualizations
            st.markdown("#### 📊 Visualizations")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                # Dimensionality reduction visualization
                viz_method = st.selectbox(
                    "Visualization method:",
                    ["PCA", "t-SNE"],
                    key=f"viz_method_{model_name}"
                )

                if st.button(f"Generate {viz_method} Plot", key=f"viz_btn_{model_name}"):
                    with st.spinner(f"Generating {viz_method} visualization..."):
                        fig = embedding_methods.visualize_embeddings(result, method=viz_method)
                        st.plotly_chart(fig, use_container_width=True)

            with viz_col2:
                # Similarity heatmap
                if st.button(f"Generate Similarity Heatmap", key=f"heatmap_btn_{model_name}"):
                    with st.spinner("Generating similarity heatmap..."):
                        fig = embedding_methods.create_similarity_heatmap(result)
                        st.plotly_chart(fig, use_container_width=True)

            # Quality metrics
            st.markdown("#### 🎯 Quality Metrics")
            quality_metrics = comparison['quality_metrics'].get(model_name, {})

            if quality_metrics:
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:
                    st.metric("Mean Similarity", f"{quality_metrics['mean_similarity']:.3f}")
                with metric_col2:
                    st.metric("Similarity Std", f"{quality_metrics['std_similarity']:.3f}")
                with metric_col3:
                    st.metric("Similarity Range", f"{quality_metrics['similarity_range']:.3f}")
                with metric_col4:
                    st.metric("Distinct Representations", f"{quality_metrics['distinct_representations']:.1%}")

    # Model selection for next step
    st.markdown("### 🎯 Select Model for FAISS Indexing")

    model_options = list(embedding_results.keys())
    selected_embedding_model = st.selectbox(
        "Choose the best embedding model for FAISS indexing:",
        model_options,
        help="Select the model that best balances your requirements for speed, quality, and memory usage"
    )

    if selected_embedding_model:
        st.session_state.selected_embedding_model = selected_embedding_model
        st.success(f"✅ Selected {selected_embedding_model} for FAISS indexing")

        # Show model summary
        selected_result = embedding_results[selected_embedding_model]
        st.info(f"""
        **Selected Model Summary:**
        - Model: {selected_embedding_model}
        - Dimensions: {selected_result.performance_metrics['embedding_dimensions']}
        - Processing Time: {selected_result.embedding_time:.2f}s
        - Memory Usage: {selected_result.performance_metrics['memory_usage_mb']:.1f} MB
        - Quality: {selected_result.model_info.get('quality', 'Unknown')}
        """)

        st.info("📍 **Next Steps:** Navigate to 'Task 7: FAISS Vector Storage' to create searchable indices.")



def show_task7_page():
    """Display Task 7 - FAISS Vector Storage page"""



    st.header("🗂️ Task 7: FAISS Vector Storage")

    if not st.session_state.get('embedding_results'):
        st.warning("⚠️ Please complete Task 6 (Embedding Implementation) first.")
        return

    if not st.session_state.get('selected_embedding_model'):
        st.warning("⚠️ Please select an embedding model in Task 6 first.")
        return

    from src.vector_storage.faiss_storage import FAISSStorage

    # Get selected embedding results
    selected_model = st.session_state.selected_embedding_model
    embedding_result = st.session_state.embedding_results[selected_model]

    st.info(f"📊 Using embeddings from: **{selected_model}** ({len(embedding_result.embeddings)} vectors, {embedding_result.embeddings.shape[1]} dimensions)")

    # Use smart default - IVF (balanced performance)
    faiss_storage = FAISSStorage()
    selected_index_type = "IVF"
    index_config = faiss_storage.index_configs[selected_index_type]

    st.info(f"Using {index_config['name']} - {index_config['description']}")

    # Use smart defaults for index parameters
    index_params = {}
    if selected_index_type == 'IVF':
        index_params['nlist'] = min(100, len(embedding_result.embeddings) // 10)
        index_params['nprobe'] = 10

    # Use smart defaults for search configuration
    default_k = 5
    similarity_threshold = 0.5
    max_results = 10

    # Build index
    if st.button("🚀 Build FAISS Index", type="primary"):

        with st.spinner(f"Building {selected_index_type} index..."):

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Preparing embeddings...")
            progress_bar.progress(0.2)

            # Create index
            status_text.text(f"Building {selected_index_type} index...")
            progress_bar.progress(0.5)

            try:
                index_result = faiss_storage.create_index(
                    embeddings=embedding_result.embeddings,
                    chunk_texts=embedding_result.chunk_texts,
                    index_type=selected_index_type,
                    **index_params
                )

                progress_bar.progress(0.8)
                status_text.text("Finalizing index...")

                # Store in session state
                if 'faiss_indices' not in st.session_state:
                    st.session_state.faiss_indices = {}

                st.session_state.faiss_indices[selected_index_type] = index_result
                st.session_state.search_config = {
                    'default_k': default_k,
                    'similarity_threshold': similarity_threshold,
                    'max_results': max_results
                }



                progress_bar.progress(1.0)
                status_text.text("Index built successfully!")

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                st.success(f"✅ {selected_index_type} index created successfully!")

                # Show index statistics
                st.markdown("#### 📊 Index Statistics")

                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

                with stat_col1:
                    st.metric("Build Time", f"{index_result.build_time:.2f}s")
                with stat_col2:
                    st.metric("Index Size", f"{index_result.index_size_mb:.1f} MB")
                with stat_col3:
                    st.metric("Vectors/Second", f"{index_result.performance_metrics['vectors_per_second']:.1f}")
                with stat_col4:
                    st.metric("Total Vectors", f"{len(index_result.embeddings):,}")

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error building index: {str(e)}")

    # Display existing indices and search interface
    if st.session_state.get('faiss_indices'):
        display_faiss_results(st.session_state.faiss_indices, st.session_state.get('search_config', {}))

def display_faiss_results(faiss_indices, search_config):
    """Display FAISS indices and interactive search interface"""

    st.subheader("🗂️ FAISS Indices")

    from src.vector_storage.faiss_storage import FAISSStorage
    from src.evaluation.rag_evaluation import RAGEvaluator
    faiss_storage = FAISSStorage()
    rag_evaluator = RAGEvaluator()

    # Index comparison table
    if len(faiss_indices) > 1:
        st.markdown("### 📊 Index Comparison")
        comparison_df = faiss_storage.compare_indices(list(faiss_indices.values()))
        st.dataframe(comparison_df, use_container_width=True)

    # Evaluation Section
    st.markdown("### 🎯 RAG Evaluation")

    # Clear old results button
    if st.button("🗑️ Clear Results"):
        if 'f1_evaluation_results' in st.session_state:
            del st.session_state.f1_evaluation_results
        st.success("✅ Results cleared!")

    # Use smart defaults
    evaluation_type = "Hit Rate + MRR"
    k_values_for_eval = [5, 10]

    run_evaluation = st.button("🎯 Run Evaluation", type="primary")

    # No additional options needed for Hit Rate + MRR evaluation

    if run_evaluation and k_values_for_eval:
        st.markdown("#### 📈 Results")

        # Get embedding model and chunking method info
        selected_model = st.session_state.get('selected_embedding_model', 'Unknown')
        selected_methods = st.session_state.get('selected_chunking_methods', ['Unknown'])
        chunking_method = ', '.join(selected_methods) if selected_methods else 'Unknown'

        evaluation_results = []

        # Progress bar for evaluation
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare evaluation based on type
        if evaluation_type.startswith("Hit Rate"):
            include_generation = "Generation" in evaluation_type
            st.info(f"🎯 Running evaluation...")

            # Generate domain-specific queries for component evaluation
            first_index = list(faiss_indices.values())[0]
            custom_queries = rag_evaluator.generate_domain_specific_queries(
                chunk_texts=first_index.chunk_texts,
                num_queries_per_topic=2
            )

            st.success(f"✅ Generated {len(custom_queries)} queries")

        elif evaluation_type == "Domain-Specific":
            st.info(f"🏥 Running evaluation...")

            # Generate domain-specific queries for the first index (they'll be reused)
            first_index = list(faiss_indices.values())[0]
            custom_queries = rag_evaluator.generate_domain_specific_queries(
                chunk_texts=first_index.chunk_texts,
                num_queries_per_topic=queries_per_topic
            )

            st.success(f"✅ Generated {len(custom_queries)} queries")

        elif evaluation_type == "Custom Manual":
            if not st.session_state.get('custom_evaluation_queries'):
                st.error("❌ No custom queries defined. Please add custom queries first.")
                st.stop()

            st.info(f"✏️ Running evaluation...")

            # Convert session state queries to CustomQuery objects
            from src.evaluation.rag_evaluation import CustomQuery
            custom_queries = []

            for i, query_data in enumerate(st.session_state.custom_evaluation_queries):
                custom_query = CustomQuery(
                    query_id=f"manual_{i+1}",
                    query_text=query_data['query_text'],
                    topic_area=query_data['topic_area'],
                    keywords=rag_evaluator.clinical_trial_keywords.get(query_data['topic_area'], []),
                    relevant_chunk_ids=query_data['relevant_chunks'],
                    description=query_data['description']
                )
                custom_queries.append(custom_query)

        elif evaluation_type == "BERTScore Semantic":
            st.info("🧠 Running BERTScore semantic evaluation with domain-specific queries...")

            # Get the first FAISS index result for chunk texts
            first_index_result = list(faiss_indices.values())[0]

            # Generate domain-specific queries for BERTScore evaluation
            custom_queries = rag_evaluator.generate_domain_specific_queries(
                chunk_texts=first_index_result.chunk_texts,
                num_queries_per_topic=2  # Use 2 queries per topic for comprehensive evaluation
            )

            st.success(f"✅ Generated {len(custom_queries)} domain-specific queries for BERTScore evaluation")

        else:  # Synthetic evaluation
            st.info("🤖 Running synthetic evaluation with automatically generated queries...")
            custom_queries = None

        for i, (index_name, index_result) in enumerate(faiss_indices.items()):
            status_text.text(f"Evaluating {index_name} with {evaluation_type.lower()} queries...")
            progress = (i + 1) / len(faiss_indices)
            progress_bar.progress(progress)

            try:
                if evaluation_type.startswith("Hit Rate"):
                    # Run component-specific evaluation (Hit Rate + MRR + optional BLEU/ROUGE)
                    include_generation = "Generation" in evaluation_type
                    metrics = rag_evaluator.evaluate_with_component_metrics(
                        index_result=index_result,
                        embedding_model=selected_model,
                        chunking_method=chunking_method,
                        evaluation_queries=custom_queries,
                        k_values=k_values_for_eval,
                        include_generation=include_generation
                    )
                elif evaluation_type == "Domain-Specific":
                    # Run domain-specific evaluation
                    metrics = rag_evaluator.evaluate_with_custom_queries(
                        index_result=index_result,
                        embedding_model=selected_model,
                        chunking_method=chunking_method,
                        evaluation_queries=custom_queries,
                        k_values=k_values_for_eval
                    )
                else:
                    # Run custom manual evaluation
                    metrics = rag_evaluator.evaluate_with_custom_queries(
                        index_result=index_result,
                        embedding_model=selected_model,
                        chunking_method=chunking_method,
                        custom_queries=custom_queries,
                        k_values=k_values_for_eval
                    )

                evaluation_results.append(metrics)

            except Exception as e:
                st.error(f"❌ Error evaluating {index_name}: {str(e)}")
                continue

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        if evaluation_results:
            # Display evaluation type information
            eval_type = evaluation_results[0].evaluation_type if evaluation_results else "unknown"

            if eval_type == "component_specific":
                st.success(f"✅ Evaluation completed!")

                # Create metrics display
                metrics_cols = st.columns(4)

                with metrics_cols[0]:
                    avg_hit_rate = evaluation_results[0].avg_hit_rate
                    st.metric("Hit Rate", f"{avg_hit_rate:.3f}")

                with metrics_cols[1]:
                    avg_mrr = evaluation_results[0].avg_mrr
                    st.metric("MRR", f"{avg_mrr:.3f}")

                with metrics_cols[2]:
                    if hasattr(evaluation_results[0], 'avg_bleu_score') and evaluation_results[0].avg_bleu_score is not None:
                        st.metric("BLEU", f"{evaluation_results[0].avg_bleu_score:.3f}")
                    else:
                        st.metric("BLEU", "N/A")

                with metrics_cols[3]:
                    if hasattr(evaluation_results[0], 'avg_rouge_l') and evaluation_results[0].avg_rouge_l is not None:
                        st.metric("ROUGE-L", f"{evaluation_results[0].avg_rouge_l:.3f}")
                    else:
                        st.metric("ROUGE-L", "N/A")

            elif eval_type == "custom":
                st.success(f"✅ Evaluation completed!")






            # Display evaluation comparison table
            comparison_df = rag_evaluator.compare_indices(evaluation_results)
            st.dataframe(comparison_df, use_container_width=True)

            # Display detailed metrics for each index
            st.markdown("#### 📊 Detailed Performance Metrics")

            for metrics in evaluation_results:
                with st.expander(f"📋 {metrics.index_type} - Detailed Metrics"):

                    # Summary metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                    with metric_col1:
                        if hasattr(metrics, 'avg_hit_rate'):
                            st.metric("Hit Rate", f"{metrics.avg_hit_rate:.4f}")
                        elif hasattr(metrics, 'avg_precision'):
                            st.metric("Avg Precision", f"{metrics.avg_precision:.4f}")
                        else:
                            st.metric("Precision", "N/A")
                    with metric_col2:
                        if hasattr(metrics, 'avg_mrr'):
                            st.metric("MRR", f"{metrics.avg_mrr:.4f}")
                        elif hasattr(metrics, 'avg_recall'):
                            st.metric("Avg Recall", f"{metrics.avg_recall:.4f}")
                        else:
                            st.metric("Recall", "N/A")
                    with metric_col3:
                        if hasattr(metrics, 'avg_bleu_score') and metrics.avg_bleu_score is not None:
                            st.metric("BLEU Score", f"{metrics.avg_bleu_score:.4f}")
                        elif hasattr(metrics, 'avg_f1_score'):
                            st.metric("Avg F1-Score", f"{metrics.avg_f1_score:.4f}")
                        else:
                            st.metric("F1-Score", "N/A")
                    with metric_col4:
                        st.metric("Total Queries", metrics.total_queries)

                    # Performance distribution
                    if metrics.evaluation_results:
                        if hasattr(metrics, 'avg_hit_rate'):
                            # Component-specific metrics
                            hit_rates = [r.hit_rate for r in metrics.evaluation_results if hasattr(r, 'hit_rate')]
                            mrr_scores = [r.mrr_score for r in metrics.evaluation_results if hasattr(r, 'mrr_score')]

                            perf_data = {
                                'Query': [r.query_id for r in metrics.evaluation_results],
                                'Hit Rate': [f"{h:.4f}" for h in hit_rates] if hit_rates else ["N/A"] * len(metrics.evaluation_results),
                                'MRR': [f"{m:.4f}" for m in mrr_scores] if mrr_scores else ["N/A"] * len(metrics.evaluation_results),
                                'Retrieval Time (ms)': [f"{r.retrieval_time * 1000:.2f}" for r in metrics.evaluation_results]
                            }

                            # Add generation metrics if available
                            if any(hasattr(r, 'bleu_score') and r.bleu_score is not None for r in metrics.evaluation_results):
                                bleu_scores = [r.bleu_score if hasattr(r, 'bleu_score') and r.bleu_score is not None else 0.0 for r in metrics.evaluation_results]
                                rouge_l_scores = [r.rouge_l_score if hasattr(r, 'rouge_l_score') and r.rouge_l_score is not None else 0.0 for r in metrics.evaluation_results]
                                perf_data.update({
                                    'BLEU Score': [f"{b:.4f}" for b in bleu_scores],
                                    'ROUGE-L': [f"{r:.4f}" for r in rouge_l_scores]
                                })
                        else:
                            # Legacy F1-score metrics
                            f1_scores = [r.f1_score for r in metrics.evaluation_results if hasattr(r, 'f1_score')]
                            precisions = [r.precision for r in metrics.evaluation_results if hasattr(r, 'precision')]
                            recalls = [r.recall for r in metrics.evaluation_results if hasattr(r, 'recall')]

                            perf_data = {
                                'Query': [r.query_id for r in metrics.evaluation_results],
                                'Precision': [f"{p:.4f}" for p in precisions] if precisions else ["N/A"] * len(metrics.evaluation_results),
                                'Recall': [f"{r:.4f}" for r in recalls] if recalls else ["N/A"] * len(metrics.evaluation_results),
                                'F1-Score': [f"{f:.4f}" for f in f1_scores] if f1_scores else ["N/A"] * len(metrics.evaluation_results),
                                'Retrieval Time (ms)': [f"{r.retrieval_time * 1000:.2f}" for r in metrics.evaluation_results]
                            }

                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)

            # Store evaluation results in session state for comparison page
            st.session_state.component_evaluation_results = evaluation_results

            st.success("✅ Component-specific evaluation completed!")
        else:
            st.error("❌ No evaluation results generated. Please check your indices and try again.")

    # Interactive search interface
    st.markdown("### 🔍 Interactive Search")

    # Select index for search
    index_options = list(faiss_indices.keys())
    selected_index = st.selectbox(
        "Select index for search:",
        index_options,
        help="Choose which FAISS index to use for similarity search"
    )

    if selected_index:
        index_result = faiss_indices[selected_index]

        # Search query input
        st.markdown("#### 💬 Search Query")

        # Text area for query (similar to document text area in your image)
        query_text = st.text_area(
            "Enter your search query:",
            height=100,
            placeholder="Type your question or search query here...",
            help="Enter text to search for similar chunks in the document"
        )

        # Search configuration (similar to RAG Configuration in your image)
        st.markdown("#### ⚙️ Search Configuration")

        search_col1, search_col2, search_col3 = st.columns(3)

        with search_col1:
            k_value = st.slider(
                "Number of results (k)",
                min_value=1,
                max_value=20,
                value=search_config.get('default_k', 5),
                help="Number of similar chunks to retrieve",
                key="search_k_value_slider"
            )

        with search_col2:
            similarity_threshold = st.slider(
                "Similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=search_config.get('similarity_threshold', 0.5),
                step=0.05,
                help="Minimum similarity score for results",
                key="search_similarity_threshold_slider"
            )

        with search_col3:
            show_scores = st.checkbox(
                "Show similarity scores",
                value=True,
                help="Display similarity scores with results"
            )

        # Search button and results
        if st.button("🔍 Search", type="primary") and query_text.strip():

            # Generate query embedding
            selected_model = st.session_state.selected_embedding_model
            embedding_result = st.session_state.embedding_results[selected_model]

            with st.spinner("Generating query embedding..."):
                from src.embedding.embedding_methods import EmbeddingMethods
                embedding_methods = EmbeddingMethods()
                model = embedding_methods.load_model(selected_model)
                query_embedding = model.encode([query_text.strip()])[0]

            # Perform search
            with st.spinner("Searching similar chunks..."):
                search_results, search_time = faiss_storage.search(
                    index_result=index_result,
                    query_embedding=query_embedding,
                    k=k_value
                )

            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result.similarity_score >= similarity_threshold
            ]

            # Display results
            st.markdown(f"#### 📋 Search Results ({len(filtered_results)} found in {search_time:.3f}s)")

            if filtered_results:
                for i, result in enumerate(filtered_results):
                    with st.expander(
                        f"Result {i+1}: Chunk {result.chunk_id}" +
                        (f" (Score: {result.similarity_score:.3f})" if show_scores else ""),
                        expanded=i < 3  # Expand first 3 results
                    ):
                        # Highlight query terms in text (basic highlighting)
                        display_text = result.chunk_text
                        query_words = query_text.lower().split()

                        for word in query_words:
                            if len(word) > 2:  # Only highlight words longer than 2 chars
                                display_text = display_text.replace(
                                    word, f"**{word}**"
                                ).replace(
                                    word.capitalize(), f"**{word.capitalize()}**"
                                )

                        st.markdown(display_text)

                        if show_scores:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Similarity Score", f"{result.similarity_score:.4f}")
                            with col2:
                                st.metric("Chunk ID", result.chunk_id)
                            with col3:
                                st.metric("Distance", f"{result.distance:.4f}")
            else:
                st.warning(f"No results found above similarity threshold {similarity_threshold:.2f}")

        elif query_text.strip() == "":
            st.info("💡 Enter a search query above to find similar chunks in your document.")

    # Performance benchmarking
    if st.button("📊 Run Performance Benchmark"):

        selected_index = st.selectbox(
            "Select index for benchmarking:",
            index_options,
            key="benchmark_index"
        )

        if selected_index:
            index_result = faiss_indices[selected_index]

            with st.spinner("Running performance benchmark..."):
                # Create sample queries from existing chunks
                sample_size = min(10, len(index_result.chunk_texts))
                sample_indices = np.random.choice(len(index_result.embeddings), sample_size, replace=False)
                sample_embeddings = index_result.embeddings[sample_indices]

                # Run benchmark
                benchmark_results = faiss_storage.benchmark_search_performance(
                    index_result=index_result,
                    query_embeddings=sample_embeddings,
                    k_values=[1, 5, 10, 20]
                )

                # Display benchmark results
                st.markdown("#### 🏃 Performance Benchmark Results")

                # Create performance chart
                fig = faiss_storage.create_performance_chart(benchmark_results, selected_index)
                st.plotly_chart(fig, use_container_width=True)

                # Performance metrics table
                perf_data = []
                for i, k in enumerate(benchmark_results['k_values']):
                    perf_data.append({
                        'k': k,
                        'Avg Search Time (ms)': f"{benchmark_results['search_times'][i] * 1000:.2f}",
                        'Throughput (queries/sec)': f"{benchmark_results['throughput'][i]:.1f}"
                    })

                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)

    # Index management
    st.markdown("### 🛠️ Index Management")

    mgmt_col1, mgmt_col2 = st.columns(2)

    with mgmt_col1:
        # Export functionality removed as requested
        pass

    with mgmt_col2:
        if st.button("🗑️ Clear All Indices"):
            st.session_state.faiss_indices = {}
            st.success("All indices cleared!")
            st.experimental_rerun()



def show_comparison_page():
    """Display comparison and results page"""



    st.header("📊 Comparison & Results")

    # Check if all tasks are completed
    has_chunking = bool(st.session_state.get('chunking_results'))
    has_embeddings = bool(st.session_state.get('embedding_results'))
    has_faiss = bool(st.session_state.get('faiss_indices'))

    if not all([has_chunking, has_embeddings, has_faiss]):
        st.warning("⚠️ Please complete all previous tasks (5, 6, and 7) to view comprehensive results.")

        # Show completion status
        st.markdown("### 📋 Task Completion Status")

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            status = "✅ Complete" if has_chunking else "❌ Incomplete"
            st.markdown(f"**Task 5 - Chunking**: {status}")

        with status_col2:
            status = "✅ Complete" if has_embeddings else "❌ Incomplete"
            st.markdown(f"**Task 6 - Embeddings**: {status}")

        with status_col3:
            status = "✅ Complete" if has_faiss else "❌ Incomplete"
            st.markdown(f"**Task 7 - FAISS**: {status}")

        return

    # Comprehensive results display
    st.markdown("""
    ### 🎉 RAG System Evaluation Complete!

    All components of your RAG system have been successfully implemented and evaluated.
    Below is a comprehensive summary of your results.
    """)

    # Summary metrics
    st.markdown("### 📊 System Overview")

    chunking_results = st.session_state.chunking_results
    embedding_results = st.session_state.embedding_results
    faiss_indices = st.session_state.faiss_indices

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

    with overview_col1:
        total_chunks = sum(len(result.chunks) for result in chunking_results.values())
        st.metric("Total Chunks Created", f"{total_chunks:,}")

    with overview_col2:
        st.metric("Embedding Models Tested", len(embedding_results))

    with overview_col3:
        st.metric("FAISS Indices Built", len(faiss_indices))

    with overview_col4:
        selected_model = st.session_state.get('selected_embedding_model', 'N/A')
        st.metric("Selected Model", selected_model)

    # Detailed comparison tables
    st.markdown("### 📋 Detailed Results")

    # Create tabs for different comparisons
    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs(["Chunking Comparison", "Embedding Comparison", "FAISS Comparison", "Performance Analysis"])

    with comp_tab1:
        # Chunking results summary
        chunking_summary = []
        for method_key, result in chunking_results.items():
            chunking_summary.append({
                'Method': result.method_name,
                'Total Chunks': result.total_chunks,
                'Avg Size (tokens)': f"{result.avg_chunk_size_tokens:.1f}",
                'Processing Time': f"{result.processing_time:.3f}s",
                'Min Size': result.min_chunk_size,
                'Max Size': result.max_chunk_size
            })

        chunking_df = pd.DataFrame(chunking_summary)
        st.dataframe(chunking_df, use_container_width=True)

    with comp_tab2:
        # Embedding results summary
        if embedding_results:
            from src.embedding.embedding_methods import EmbeddingMethods
            embedding_methods = EmbeddingMethods()
            comparison = embedding_methods.compare_embeddings(list(embedding_results.values()))
            st.dataframe(comparison['performance_comparison'], use_container_width=True)

    with comp_tab3:
        # FAISS results summary
        if faiss_indices:
            from src.vector_storage.faiss_storage import FAISSStorage
            faiss_storage = FAISSStorage()
            comparison_df = faiss_storage.compare_indices(list(faiss_indices.values()))
            st.dataframe(comparison_df, use_container_width=True)

    with comp_tab4:
        # Component-Specific Analysis and Cross-Model Comparison
        st.markdown("#### 🎯 Component-Specific Performance Analysis")

        # Check if component-specific evaluation has been run
        component_results = st.session_state.get('component_evaluation_results')
        f1_results = st.session_state.get('f1_evaluation_results')  # Keep for backward compatibility

        # Use component results if available, otherwise fall back to F1 results
        evaluation_results = component_results if component_results else f1_results

        if evaluation_results:
            # Determine evaluation type
            eval_type = "component_specific" if component_results else "f1_score"

            if eval_type == "component_specific":
                st.markdown("""
                **Component-Specific Analysis Summary:**
                This section provides comprehensive component-specific analysis across different configurations,
                helping you understand which combination of chunking methods, embedding models, and
                FAISS index types provides the best retrieval and generation performance for your RAG system.
                """)
            else:
                st.markdown("""
                **F1-Score Analysis Summary:**
                This section provides comprehensive F1-Score analysis across different configurations,
                helping you understand which combination of chunking methods, embedding models, and
                FAISS index types provides the best retrieval performance for your RAG system.
                """)

            # Evaluation comparison table
            from src.evaluation.rag_evaluation import RAGEvaluator
            rag_evaluator = RAGEvaluator()
            comparison_df = rag_evaluator.compare_indices(evaluation_results)

            table_title = "📊 Component-Specific Comparison Table" if eval_type == "component_specific" else "📊 F1-Score Comparison Table"
            st.markdown(f"##### {table_title}")
            st.dataframe(comparison_df, use_container_width=True)

            # Performance insights
            st.markdown("##### 🔍 Performance Insights")

            # Find best performing configuration
            best_idx = 0
            best_score = 0.0

            if eval_type == "component_specific":
                # Use Hit Rate as primary metric for component-specific evaluation
                for i, metrics in enumerate(evaluation_results):
                    if hasattr(metrics, 'avg_hit_rate') and metrics.avg_hit_rate > best_score:
                        best_score = metrics.avg_hit_rate
                        best_idx = i
            else:
                # Use F1-score for legacy evaluation
                for i, metrics in enumerate(evaluation_results):
                    if hasattr(metrics, 'avg_f1_score') and metrics.avg_f1_score > best_score:
                        best_score = metrics.avg_f1_score
                        best_idx = i

            if evaluation_results:
                best_config = evaluation_results[best_idx]

                insight_col1, insight_col2 = st.columns(2)

                with insight_col1:
                    if eval_type == "component_specific":
                        config_info = f"""
                        **🏆 Best Performing Configuration:**
                        - **Index Type**: {best_config.index_type}
                        - **Embedding Model**: {best_config.embedding_model}
                        - **Chunking Method**: {best_config.chunking_method}
                        """
                        if hasattr(best_config, 'avg_hit_rate'):
                            config_info += f"\n- **Hit Rate**: {best_config.avg_hit_rate:.4f}"
                        if hasattr(best_config, 'avg_mrr'):
                            config_info += f"\n- **MRR**: {best_config.avg_mrr:.4f}"
                        if hasattr(best_config, 'avg_bleu_score') and best_config.avg_bleu_score is not None:
                            config_info += f"\n- **BLEU Score**: {best_config.avg_bleu_score:.4f}"
                        st.success(config_info)
                    else:
                        st.success(f"""
                        **🏆 Best Performing Configuration:**
                        - **Index Type**: {best_config.index_type}
                        - **F1-Score**: {best_config.avg_f1_score:.4f}
                        - **Precision**: {best_config.avg_precision:.4f}
                        - **Recall**: {best_config.avg_recall:.4f}
                        """)

                with insight_col2:
                    if eval_type == "component_specific":
                        # Calculate component-specific statistics
                        hit_rates = [m.avg_hit_rate for m in evaluation_results if hasattr(m, 'avg_hit_rate')]
                        mrr_scores = [m.avg_mrr for m in evaluation_results if hasattr(m, 'avg_mrr')]

                        stats_info = f"""
                        **📈 Overall Statistics:**
                        - **Configurations Tested**: {len(evaluation_results)}
                        """
                        if hit_rates:
                            avg_hit_rate = sum(hit_rates) / len(hit_rates)
                            stats_info += f"\n- **Average Hit Rate**: {avg_hit_rate:.4f}"
                            stats_info += f"\n- **Best Hit Rate**: {max(hit_rates):.4f}"
                        if mrr_scores:
                            avg_mrr = sum(mrr_scores) / len(mrr_scores)
                            stats_info += f"\n- **Average MRR**: {avg_mrr:.4f}"
                        st.info(stats_info)
                    else:
                        # Calculate F1-score statistics
                        f1_scores = [m.avg_f1_score for m in evaluation_results if hasattr(m, 'avg_f1_score')]
                        if f1_scores:
                            avg_f1 = sum(f1_scores) / len(f1_scores)
                            st.info(f"""
                            **📈 Overall Statistics:**
                            - **Average F1-Score**: {avg_f1:.4f}
                            - **Best F1-Score**: {max(f1_scores):.4f}
                            - **F1-Score Range**: {max(f1_scores) - min(f1_scores):.4f}
                            - **Configurations Tested**: {len(evaluation_results)}
                            """)

            # Detailed performance breakdown
            st.markdown("##### 📋 Detailed Performance Breakdown")

            for metrics in evaluation_results:
                with st.expander(f"📊 {metrics.index_type} - Performance Details"):

                    # Performance metrics
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                    with perf_col1:
                        if hasattr(metrics, 'avg_hit_rate'):
                            st.metric("Hit Rate", f"{metrics.avg_hit_rate:.4f}")
                        elif hasattr(metrics, 'avg_f1_score'):
                            st.metric("Avg F1-Score", f"{metrics.avg_f1_score:.4f}")
                        else:
                            st.metric("Primary Metric", "N/A")
                    with perf_col2:
                        if hasattr(metrics, 'avg_mrr'):
                            st.metric("MRR", f"{metrics.avg_mrr:.4f}")
                        elif hasattr(metrics, 'std_f1_score'):
                            st.metric("Std Deviation", f"{metrics.std_f1_score:.4f}")
                        else:
                            st.metric("Secondary Metric", "N/A")
                    with perf_col3:
                        st.metric("Total Queries", metrics.total_queries)
                    with perf_col4:
                        st.metric("Avg Retrieval Time", f"{metrics.avg_retrieval_time * 1000:.2f}ms")

                    # Performance interpretation
                    if hasattr(metrics, 'avg_hit_rate'):
                        # Component-specific interpretation
                        if metrics.avg_hit_rate >= 0.8:
                            performance_level = "🟢 Excellent"
                            interpretation = "This configuration shows excellent retrieval performance with high hit rate."
                        elif metrics.avg_hit_rate >= 0.6:
                            performance_level = "🟡 Good"
                            interpretation = "This configuration shows good retrieval performance with room for improvement."
                        elif metrics.avg_hit_rate >= 0.4:
                            performance_level = "🟠 Fair"
                            interpretation = "This configuration shows fair retrieval performance. Consider optimization."
                        else:
                            performance_level = "🔴 Poor"
                            interpretation = "This configuration shows poor retrieval performance. Optimization needed."
                    elif hasattr(metrics, 'avg_f1_score'):
                        # Legacy F1-score interpretation
                        if metrics.avg_f1_score >= 0.7:
                            performance_level = "🟢 Excellent"
                            interpretation = "This configuration shows excellent retrieval performance."
                        elif metrics.avg_f1_score >= 0.5:
                            performance_level = "🟡 Good"
                            interpretation = "This configuration shows good retrieval performance with room for improvement."
                        elif metrics.avg_f1_score >= 0.3:
                            performance_level = "🟠 Fair"
                            interpretation = "This configuration shows fair performance but may need optimization."
                        else:
                            performance_level = "🔴 Poor"
                            interpretation = "This configuration shows poor performance and should be reconsidered."
                    else:
                        performance_level = "❓ Unknown"
                        interpretation = "Performance metrics not available for this configuration."

                    st.markdown(f"**Performance Level**: {performance_level}")
                    st.markdown(f"**Interpretation**: {interpretation}")

            # Cross-model recommendations
            st.markdown("##### 🎯 Configuration Recommendations")

            # Sort by primary metric for recommendations
            if eval_type == "component_specific":
                sorted_results = sorted(evaluation_results, key=lambda x: getattr(x, 'avg_hit_rate', 0), reverse=True)
            else:
                sorted_results = sorted(evaluation_results, key=lambda x: getattr(x, 'avg_f1_score', 0), reverse=True)

            rec_col1, rec_col2, rec_col3 = st.columns(3)

            with rec_col1:
                if len(sorted_results) >= 1:
                    top_config = sorted_results[0]
                    if hasattr(top_config, 'avg_hit_rate'):
                        score_text = f"Hit Rate: {top_config.avg_hit_rate:.4f}"
                    elif hasattr(top_config, 'avg_f1_score'):
                        score_text = f"F1-Score: {top_config.avg_f1_score:.4f}"
                    else:
                        score_text = "Score: N/A"

                    st.markdown(f"""
                    **🥇 Top Performer**
                    - **Index**: {top_config.index_type}
                    - **{score_text}**
                    - **Use Case**: Best overall retrieval performance
                    """)

            with rec_col2:
                if len(sorted_results) >= 2:
                    second_config = sorted_results[1]
                    if hasattr(second_config, 'avg_hit_rate'):
                        score_text = f"Hit Rate: {second_config.avg_hit_rate:.4f}"
                    elif hasattr(second_config, 'avg_f1_score'):
                        score_text = f"F1-Score: {second_config.avg_f1_score:.4f}"
                    else:
                        score_text = "Score: N/A"

                    st.markdown(f"""
                    **🥈 Second Best**
                    - **Index**: {second_config.index_type}
                    - **{score_text}**
                    - **Use Case**: Alternative high-performance option
                    """)

            with rec_col3:
                # Find fastest configuration
                fastest_config = min(evaluation_results, key=lambda x: x.avg_retrieval_time)
                st.markdown(f"""
                **⚡ Fastest Retrieval**
                - **Index**: {fastest_config.index_type}
                - **Time**: {fastest_config.avg_retrieval_time * 1000:.2f}ms
                - **Use Case**: Speed-critical applications
                """)

        else:
            st.info("""
            🔍 **Evaluation not yet performed.**

            To see analysis:
            1. Go to Task 7: FAISS Vector Storage
            2. Click "🎯 Run Evaluation"
            3. Return here to view results
            """)

    # Final recommendations
    st.markdown("### 🎯 Final Recommendations")

    st.markdown("""
    Based on your evaluation results, here are the recommended configurations for different use cases:
    """)

    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        st.markdown("""
        **🏃 Speed-Optimized Setup**
        - Chunking: Fixed-Size (fast processing)
        - Embedding: all-MiniLM-L6-v2 (lightweight)
        - Index: HNSW (fast search)
        - Best for: Real-time applications
        """)

    with rec_col2:
        st.markdown("""
        **⚖️ Balanced Setup**
        - Chunking: Recursive Character Splitting
        - Embedding: all-mpnet-base-v2 (balanced)
        - Index: IVF (good speed/accuracy)
        - Best for: Production applications
        """)

    with rec_col3:
        st.markdown("""
        **🎯 Quality-Optimized Setup**
        - Chunking: Semantic (context-aware)
        - Embedding: all-roberta-large-v1 (high quality)
        - Index: Flat (exact search)
        - Best for: High-accuracy requirements
        """)

    # Export all results
    if st.button("📥 Export Complete Results"):
        # Create comprehensive export
        export_data = {
            'chunking_summary': chunking_df.to_dict(),
            'embedding_summary': comparison['performance_comparison'].to_dict() if embedding_results else {},
            'faiss_summary': comparison_df.to_dict() if faiss_indices else {},
            'selected_configuration': {
                'embedding_model': st.session_state.get('selected_embedding_model'),
                'chunking_methods': st.session_state.get('selected_chunking_methods', []),
                'faiss_indices': list(faiss_indices.keys())
            }
        }

        # Add F1-Score results if available
        f1_results = st.session_state.get('f1_evaluation_results')
        if f1_results:
            from src.evaluation.rag_evaluation import RAGEvaluator
            rag_evaluator = RAGEvaluator()
            f1_comparison_df = rag_evaluator.compare_indices(f1_results)
            export_data['f1_score_summary'] = f1_comparison_df.to_dict()

            # Add detailed F1-Score metrics
            f1_detailed = []
            for metrics in f1_results:
                f1_detailed.append({
                    'index_type': metrics.index_type,
                    'embedding_model': metrics.embedding_model,
                    'chunking_method': metrics.chunking_method,
                    'avg_precision': metrics.avg_precision,
                    'avg_recall': metrics.avg_recall,
                    'avg_f1_score': metrics.avg_f1_score,
                    'std_f1_score': metrics.std_f1_score,
                    'total_queries': metrics.total_queries,
                    'avg_retrieval_time': metrics.avg_retrieval_time
                })
            export_data['f1_score_detailed'] = f1_detailed

        import json
        st.download_button(
            label="📥 Download Complete Results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="rag_evaluation_results.json",
            mime="application/json"
        )


