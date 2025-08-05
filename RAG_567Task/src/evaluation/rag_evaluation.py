"""
RAG System Evaluation Module

This module provides evaluation metrics for Retrieval-Augmented Generation (RAG) systems,
including F1-Score, precision, and recall calculations for retrieval performance assessment.
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

# BERTScore imports
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️ BERTScore not available. Install with: pip install bert-score")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalEvaluationResult:
    """Results from retrieval evaluation with component-specific metrics"""
    query_id: str
    relevant_chunks: List[int]  # Ground truth relevant chunk IDs
    retrieved_chunks: List[int]  # Retrieved chunk IDs
    # Retrieval Component Metrics
    hit_rate: float  # Did we retrieve at least one relevant document? (0 or 1)
    mrr_score: float  # Mean Reciprocal Rank (1/rank of first relevant doc)
    retrieval_time: float
    similarity_scores: List[float]
    # Generation Component Metrics (if applicable)
    bleu_score: Optional[float] = None
    rouge_l_score: Optional[float] = None
    rouge_1_score: Optional[float] = None
    rouge_2_score: Optional[float] = None

@dataclass
class CustomQuery:
    """Custom evaluation query with manual ground truth"""
    query_id: str
    query_text: str
    topic_area: str
    keywords: List[str]
    relevant_chunk_ids: List[int]
    description: str = ""

@dataclass
class RAGEvaluationMetrics:
    """Component-specific RAG evaluation metrics"""
    index_type: str
    embedding_model: str
    chunking_method: str
    # Retrieval Component Metrics (Higher is Better)
    avg_hit_rate: float  # Average hit rate across all queries (0.0-1.0)
    avg_mrr: float  # Average Mean Reciprocal Rank (0.0-1.0)
    std_hit_rate: float
    std_mrr: float
    total_queries: int
    avg_retrieval_time: float
    # Generation Component Metrics (Higher is Better)
    avg_bleu_score: Optional[float] = None  # 0.0-1.0
    avg_rouge_l: Optional[float] = None  # 0.0-1.0
    avg_rouge_1: Optional[float] = None  # 0.0-1.0
    avg_rouge_2: Optional[float] = None  # 0.0-1.0
    std_bleu_score: Optional[float] = None
    std_rouge_l: Optional[float] = None
    std_rouge_1: Optional[float] = None
    std_rouge_2: Optional[float] = None
    evaluation_results: List[RetrievalEvaluationResult] = None
    evaluation_type: str = "component_specific"

class RAGEvaluator:
    """
    RAG System Evaluator for calculating F1-Score, precision, and recall
    """

    def __init__(self):
        self.evaluation_cache = {}
        self.clinical_trial_keywords = self._load_clinical_trial_keywords()

    def _load_clinical_trial_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords for clinical trial documents - EXACT match to user's keywords"""
        return {
            "clinical_trial_history": [
                "James Lind", "scurvy", "King Nebuchadnezzar", "biblical trial", "placebo origin"
            ],
            "trial_phases": [
                "Phase I", "Phase II", "Phase III", "Phase IV", "dosage", "side effects", "long-term"
            ],
            "ethics_regulation": [
                "IRB", "Nuremberg Code", "Declaration of Helsinki", "informed consent", "FDA"
            ],
            "dermatology_trials": [
                "psoriasis", "atopic dermatitis", "pruritus", "immunotherapy", "quality of life"
            ],
            "study_design": [
                "randomized", "double-blind", "placebo-controlled", "FINER criteria"
            ],
            "trial_roles": [
                "principal investigator", "study coordinator", "clinical research associate"
            ],
            "recruitment_challenges": [
                "underpowered", "registry", "flyers", "barriers", "consent paperwork"
            ],
            "diversity_issues": [
                "women in trials", "pharmacokinetics", "minorities", "sex differences"
            ],
            "rct_limitations": [
                "placebo effect", "exclusion criteria", "generalizability", "standardization"
            ],
            "publication_bias": [
                "positive results only", "underreporting", "statistical significance"
            ]
        }
    
    def calculate_retrieval_metrics(self, 
                                  relevant_chunks: List[int], 
                                  retrieved_chunks: List[int],
                                  k: int = 5) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1-score for a single query
        
        Args:
            relevant_chunks: List of ground truth relevant chunk IDs
            retrieved_chunks: List of retrieved chunk IDs (top-k)
            k: Number of top results to consider
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not relevant_chunks or not retrieved_chunks:
            return 0.0, 0.0, 0.0
        
        # Take only top-k retrieved chunks
        retrieved_k = retrieved_chunks[:k]
        
        # Convert to sets for intersection calculation
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_k)
        
        # Calculate intersection
        true_positives = len(relevant_set.intersection(retrieved_set))
        
        # Calculate precision: TP / (TP + FP) = TP / retrieved_count
        precision = true_positives / len(retrieved_k) if retrieved_k else 0.0
        
        # Calculate recall: TP / (TP + FN) = TP / relevant_count
        recall = true_positives / len(relevant_set) if relevant_set else 0.0
        
        # Calculate F1-score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

    def calculate_hit_rate(self, relevant_chunks: List[int], retrieved_chunks: List[int], k: int) -> float:
        """
        Calculate Hit Rate: Did we retrieve at least one relevant document in top-k?

        Args:
            relevant_chunks: List of relevant chunk IDs
            retrieved_chunks: List of retrieved chunk IDs
            k: Number of top results to consider

        Returns:
            Hit rate (1.0 if hit, 0.0 if miss)
        """
        if not retrieved_chunks or k == 0 or not relevant_chunks:
            return 0.0

        top_k_retrieved = set(retrieved_chunks[:k])
        relevant_set = set(relevant_chunks)

        # Hit if any relevant document is in top-k
        return 1.0 if top_k_retrieved.intersection(relevant_set) else 0.0

    def calculate_mrr(self, relevant_chunks: List[int], retrieved_chunks: List[int], k: int) -> float:
        """
        Calculate Mean Reciprocal Rank: 1/rank of first relevant document

        Args:
            relevant_chunks: List of relevant chunk IDs
            retrieved_chunks: List of retrieved chunk IDs
            k: Number of top results to consider

        Returns:
            MRR score (1/rank of first relevant doc, 0 if no relevant doc found)
        """
        if not retrieved_chunks or k == 0 or not relevant_chunks:
            return 0.0

        relevant_set = set(relevant_chunks)

        # Find rank of first relevant document
        for rank, chunk_id in enumerate(retrieved_chunks[:k], 1):
            if chunk_id in relevant_set:
                return 1.0 / rank

        return 0.0  # No relevant document found in top-k

    def calculate_bleu_score(self, candidate: str, references: List[str]) -> float:
        """
        Calculate BLEU score for generation evaluation

        Args:
            candidate: Generated text
            references: List of reference texts

        Returns:
            BLEU score (0.0-1.0)
        """
        if not BLEU_AVAILABLE or not candidate or not references:
            return 0.0

        try:
            # Tokenize candidate and references
            candidate_tokens = candidate.split()
            reference_tokens = [ref.split() for ref in references]

            # Use smoothing to handle edge cases
            smoothing = SmoothingFunction().method1

            # Calculate BLEU score
            bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
            return bleu
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            return 0.0

    def calculate_rouge_scores(self, candidate: str, references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores for generation evaluation

        Args:
            candidate: Generated text
            references: List of reference texts

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        if not ROUGE_AVAILABLE or not candidate or not references:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            # Calculate ROUGE against all references and take the best score
            best_scores = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

            for reference in references:
                scores = scorer.score(reference, candidate)
                best_scores["rouge_1"] = max(best_scores["rouge_1"], scores['rouge1'].fmeasure)
                best_scores["rouge_2"] = max(best_scores["rouge_2"], scores['rouge2'].fmeasure)
                best_scores["rouge_l"] = max(best_scores["rouge_l"], scores['rougeL'].fmeasure)

            return best_scores
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {e}")
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    def calculate_bert_score(self,
                           candidate_texts: List[str],
                           reference_texts: List[str],
                           model_type: str = "microsoft/deberta-xlarge-mnli") -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate BERTScore for semantic similarity evaluation

        Args:
            candidate_texts: List of generated/retrieved text chunks
            reference_texts: List of reference/ground truth text chunks
            model_type: BERT model to use for scoring

        Returns:
            Tuple of (precision_scores, recall_scores, f1_scores) lists
        """
        if not BERT_SCORE_AVAILABLE:
            logger.warning("BERTScore not available. Install with: pip install bert-score")
            return [], [], []

        if not candidate_texts or not reference_texts:
            return [], [], []

        try:
            # Calculate BERTScore
            P, R, F1 = bert_score(
                cands=candidate_texts,
                refs=reference_texts,
                model_type=model_type,
                verbose=False,
                device='cpu'  # Use CPU for compatibility
            )

            # Convert tensors to lists
            precision_scores = P.tolist()
            recall_scores = R.tolist()
            f1_scores = F1.tolist()

            return precision_scores, recall_scores, f1_scores

        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return [], [], []
    
    def generate_synthetic_ground_truth(self, 
                                      chunk_texts: List[str], 
                                      num_queries: int = 10,
                                      relevance_threshold: float = 0.7) -> Dict[str, List[int]]:
        """
        Generate synthetic ground truth for evaluation when real ground truth is not available.
        This creates queries based on existing chunks and determines relevant chunks using text similarity.
        
        Args:
            chunk_texts: List of chunk texts
            num_queries: Number of synthetic queries to generate
            relevance_threshold: Similarity threshold for determining relevance
            
        Returns:
            Dictionary mapping query_id to list of relevant chunk IDs
        """
        from sentence_transformers import SentenceTransformer
        import random
        
        # Load a model for similarity calculation
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        ground_truth = {}
        
        # Generate queries by selecting random chunks and creating variations
        import random
        selected_indices = random.sample(range(len(chunk_texts)), min(num_queries, len(chunk_texts)))
        
        for i, chunk_idx in enumerate(selected_indices):
            query_id = f"synthetic_query_{i+1}"
            source_chunk = chunk_texts[chunk_idx]
            
            # Create a query by taking the first sentence or part of the chunk
            sentences = source_chunk.split('.')
            query_text = sentences[0].strip() if sentences else source_chunk[:100]
            
            # Find relevant chunks using similarity
            query_embedding = model.encode([query_text])
            chunk_embeddings = model.encode(chunk_texts)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
            
            # Find chunks above threshold
            relevant_indices = [idx for idx, sim in enumerate(similarities) 
                              if sim >= relevance_threshold]
            
            # Ensure at least the source chunk is included
            if chunk_idx not in relevant_indices:
                relevant_indices.append(chunk_idx)
            
            ground_truth[query_id] = relevant_indices
            
        return ground_truth

    def generate_domain_specific_queries(self,
                                       chunk_texts: List[str],
                                       num_queries_per_topic: int = 2) -> List[CustomQuery]:
        """
        Generate domain-specific evaluation queries for clinical trial documents

        Args:
            chunk_texts: List of chunk texts
            num_queries_per_topic: Number of queries to generate per topic area

        Returns:
            List of CustomQuery objects with domain-specific queries
        """
        from sentence_transformers import SentenceTransformer

        # Load model for similarity calculation
        model = SentenceTransformer('all-MiniLM-L6-v2')

        custom_queries = []

        for topic_area, keywords in self.clinical_trial_keywords.items():
            # Generate queries for this topic area
            for i in range(num_queries_per_topic):
                # Create different types of queries using EXACT keywords
                if i == 0:
                    # Direct keyword query - use exact keywords from your list
                    if len(keywords) >= 2:
                        primary_keywords = keywords[:2]  # Use first 2 keywords for better precision
                        query_text = f"What information is available about {' and '.join(primary_keywords)}?"
                    else:
                        query_text = f"What information is available about {keywords[0]}?"
                    query_type = "direct"
                else:
                    # Keyword-focused conceptual queries using your exact terms
                    if topic_area == "clinical_trial_history":
                        query_text = f"Tell me about {keywords[0]} and the history of clinical trials"
                    elif topic_area == "trial_phases":
                        query_text = f"What are {keywords[0]}, {keywords[1]}, {keywords[2]}, and {keywords[3]} in clinical trials?"
                    elif topic_area == "ethics_regulation":
                        query_text = f"What are {keywords[0]} and {keywords[1]} requirements for clinical trials?"
                    elif topic_area == "dermatology_trials":
                        query_text = f"How are {keywords[0]} and {keywords[1]} treated in clinical trials?"
                    elif topic_area == "study_design":
                        query_text = f"What does {keywords[0]} and {keywords[1]} mean in clinical trial design?"
                    elif topic_area == "trial_roles":
                        query_text = f"What are the responsibilities of a {keywords[0]} and {keywords[1]}?"
                    elif topic_area == "recruitment_challenges":
                        query_text = f"What are {keywords[0]} studies and {keywords[1]} challenges in clinical trials?"
                    elif topic_area == "diversity_issues":
                        query_text = f"What issues exist with {keywords[0]} and {keywords[1]} in clinical trials?"
                    elif topic_area == "rct_limitations":
                        query_text = f"What are the problems with {keywords[0]} and {keywords[1]} in clinical trials?"
                    elif topic_area == "publication_bias":
                        query_text = f"How does {keywords[0]} and {keywords[1]} affect clinical trial research?"
                    else:
                        query_text = f"Tell me about {keywords[0]} in clinical trials"
                    query_type = "keyword_focused"

                # Find relevant chunks using keyword matching and semantic similarity
                relevant_chunks = self._find_relevant_chunks_for_query(
                    query_text, keywords, chunk_texts, model
                )

                query_id = f"{topic_area}_{query_type}_{i+1}"

                custom_query = CustomQuery(
                    query_id=query_id,
                    query_text=query_text,
                    topic_area=topic_area,
                    keywords=keywords,
                    relevant_chunk_ids=relevant_chunks,
                    description=f"{query_type.title()} query for {topic_area.replace('_', ' ')}"
                )

                custom_queries.append(custom_query)

        return custom_queries

    def _find_relevant_chunks_for_query(self,
                                      query_text: str,
                                      keywords: List[str],
                                      chunk_texts: List[str],
                                      model) -> List[int]:
        """
        Find relevant chunks for a query using improved keyword matching and semantic similarity

        Args:
            query_text: The query text
            keywords: List of relevant keywords
            chunk_texts: List of all chunk texts
            model: SentenceTransformer model for similarity calculation

        Returns:
            List of relevant chunk indices
        """
        relevant_indices = []

        # Method 1: Enhanced keyword-based matching
        for i, chunk_text in enumerate(chunk_texts):
            chunk_lower = chunk_text.lower()
            keyword_score = 0

            # Score based on exact keyword matches (case-insensitive)
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in chunk_lower:
                    # Give higher score for exact matches
                    keyword_score += 2

                    # Additional score if keyword appears multiple times
                    keyword_count = chunk_lower.count(keyword_lower)
                    if keyword_count > 1:
                        keyword_score += keyword_count - 1

            # Consider chunk relevant if it has good keyword coverage
            # Lower threshold for better recall with your specific keywords
            if keyword_score >= 2:  # At least one strong keyword match
                relevant_indices.append(i)

        # Method 2: Semantic similarity (if keyword matching finds few results)
        if len(relevant_indices) < 2:  # Lower threshold to ensure we find relevant chunks
            query_embedding = model.encode([query_text])
            chunk_embeddings = model.encode(chunk_texts)

            # Calculate similarities
            similarities = np.dot(query_embedding, chunk_embeddings.T)[0]

            # Find semantically similar chunks with lower threshold for clinical content
            similarity_threshold = 0.4  # Lower threshold for better recall
            similar_indices = [idx for idx, sim in enumerate(similarities)
                             if sim >= similarity_threshold]

            # Combine with keyword-based results
            relevant_indices.extend(similar_indices)
            relevant_indices = list(set(relevant_indices))  # Remove duplicates

        # Ensure we have at least some relevant chunks (fallback with top similar)
        if len(relevant_indices) < 1:
            query_embedding = model.encode([query_text])
            chunk_embeddings = model.encode(chunk_texts)
            similarities = np.dot(query_embedding, chunk_embeddings.T)[0]

            # Get top 5 most similar chunks as fallback
            top_indices = np.argsort(similarities)[-5:]
            relevant_indices.extend(top_indices.tolist())
            relevant_indices = list(set(relevant_indices))

        return relevant_indices

    def evaluate_with_bert_score(self,
                                index_result,
                                embedding_model: str,
                                chunking_method: str,
                                evaluation_queries: Optional[List[CustomQuery]] = None,
                                k_values: List[int] = [5, 10],
                                bert_model: str = "microsoft/deberta-xlarge-mnli") -> RAGEvaluationMetrics:
        """
        Evaluate RAG system using BERTScore for semantic similarity

        Args:
            index_result: FAISSIndexResult object
            embedding_model: Name of the embedding model used
            chunking_method: Name of the chunking method used
            evaluation_queries: List of CustomQuery objects (if None, generates domain-specific)
            k_values: List of k values to evaluate
            bert_model: BERT model for BERTScore calculation

        Returns:
            RAGEvaluationMetrics with BERTScore results
        """
        from src.vector_storage.faiss_storage import FAISSStorage
        from src.embedding.embedding_methods import EmbeddingMethods

        if not BERT_SCORE_AVAILABLE:
            logger.error("BERTScore not available. Install with: pip install bert-score")
            return self._create_empty_metrics(index_result, embedding_model, chunking_method)

        # Generate domain-specific queries if not provided
        if evaluation_queries is None:
            logger.info("Generating domain-specific queries for BERTScore evaluation...")
            evaluation_queries = self.generate_domain_specific_queries(
                index_result.chunk_texts,
                num_queries_per_topic=2
            )

        faiss_storage = FAISSStorage()
        embedding_methods = EmbeddingMethods()
        model = embedding_methods.load_model(embedding_model)

        evaluation_results = []

        for query in evaluation_queries:
            # Generate query embedding
            query_embedding = model.encode([query.query_text])[0]

            # Perform search
            start_time = time.time()
            search_results, search_time = faiss_storage.search(
                index_result=index_result,
                query_embedding=query_embedding,
                k=max(k_values)
            )
            retrieval_time = time.time() - start_time

            # Extract retrieved chunk IDs and texts
            retrieved_chunks = [result.chunk_id for result in search_results]
            similarity_scores = [result.similarity_score for result in search_results]

            # Get retrieved chunk texts for BERTScore
            retrieved_texts = [index_result.chunk_texts[chunk_id] for chunk_id in retrieved_chunks[:max(k_values)]]

            # Get ground truth texts for BERTScore
            ground_truth_texts = [index_result.chunk_texts[chunk_id] for chunk_id in query.relevant_chunk_ids]

            # Calculate BERTScore for retrieved vs ground truth
            if retrieved_texts and ground_truth_texts:
                # For BERTScore, we compare each retrieved text with the best matching ground truth
                bert_precisions, bert_recalls, bert_f1s = [], [], []

                for retrieved_text in retrieved_texts[:max(k_values)]:
                    # Calculate BERTScore against all ground truth texts and take the best
                    if ground_truth_texts:
                        P_scores, R_scores, F1_scores = self.calculate_bert_score(
                            [retrieved_text] * len(ground_truth_texts),
                            ground_truth_texts,
                            bert_model
                        )

                        if P_scores:
                            # Take the best score (highest F1) among ground truth texts
                            best_idx = np.argmax(F1_scores)
                            bert_precisions.append(P_scores[best_idx])
                            bert_recalls.append(R_scores[best_idx])
                            bert_f1s.append(F1_scores[best_idx])
                        else:
                            bert_precisions.append(0.0)
                            bert_recalls.append(0.0)
                            bert_f1s.append(0.0)
                    else:
                        bert_precisions.append(0.0)
                        bert_recalls.append(0.0)
                        bert_f1s.append(0.0)
            else:
                bert_precisions = [0.0] * max(k_values)
                bert_recalls = [0.0] * max(k_values)
                bert_f1s = [0.0] * max(k_values)

            # Calculate metrics for different k values
            for k in k_values:
                # Traditional retrieval metrics
                precision, recall, f1 = self.calculate_retrieval_metrics(
                    query.relevant_chunk_ids, retrieved_chunks, k
                )

                # BERTScore metrics (average of top-k)
                bert_precision = np.mean(bert_precisions[:k]) if bert_precisions[:k] else 0.0
                bert_recall = np.mean(bert_recalls[:k]) if bert_recalls[:k] else 0.0
                bert_f1 = np.mean(bert_f1s[:k]) if bert_f1s[:k] else 0.0

                result = RetrievalEvaluationResult(
                    query_id=f"{query.query_id}_k{k}",
                    relevant_chunks=query.relevant_chunk_ids,
                    retrieved_chunks=retrieved_chunks[:k],
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    retrieval_time=retrieval_time,
                    similarity_scores=similarity_scores[:k],
                    bert_score_precision=bert_precision,
                    bert_score_recall=bert_recall,
                    bert_score_f1=bert_f1
                )
                evaluation_results.append(result)

        # Calculate aggregate metrics
        return self._calculate_aggregate_metrics_with_bert(
            evaluation_results, index_result, embedding_model, chunking_method
        )
    
    def evaluate_faiss_index(self, 
                           index_result,
                           embedding_model: str,
                           chunking_method: str,
                           ground_truth: Optional[Dict[str, List[int]]] = None,
                           k_values: List[int] = [5, 10, 20]) -> RAGEvaluationMetrics:
        """
        Evaluate a FAISS index using F1-Score, precision, and recall
        
        Args:
            index_result: FAISSIndexResult object
            embedding_model: Name of the embedding model used
            chunking_method: Name of the chunking method used
            ground_truth: Dictionary mapping query_id to relevant chunk IDs
            k_values: List of k values to evaluate
            
        Returns:
            RAGEvaluationMetrics object with comprehensive evaluation results
        """
        from src.vector_storage.faiss_storage import FAISSStorage
        from src.embedding.embedding_methods import EmbeddingMethods
        
        # Generate synthetic ground truth if not provided
        if ground_truth is None:
            logger.info("Generating synthetic ground truth for evaluation...")
            ground_truth = self.generate_synthetic_ground_truth(
                index_result.chunk_texts, 
                num_queries=10
            )
        
        faiss_storage = FAISSStorage()
        embedding_methods = EmbeddingMethods()
        model = embedding_methods.load_model(embedding_model)
        
        evaluation_results = []
        
        for query_id, relevant_chunks in ground_truth.items():
            # Create query from the first relevant chunk (simplified approach)
            if relevant_chunks:
                query_chunk_idx = relevant_chunks[0]
                query_text = index_result.chunk_texts[query_chunk_idx]
                
                # Generate query embedding
                query_embedding = model.encode([query_text])[0]
                
                # Perform search
                start_time = time.time()
                search_results, search_time = faiss_storage.search(
                    index_result=index_result,
                    query_embedding=query_embedding,
                    k=max(k_values)
                )
                retrieval_time = time.time() - start_time
                
                # Extract retrieved chunk IDs
                retrieved_chunks = [result.chunk_id for result in search_results]
                similarity_scores = [result.similarity_score for result in search_results]
                
                # Calculate metrics for different k values
                for k in k_values:
                    precision, recall, f1 = self.calculate_retrieval_metrics(
                        relevant_chunks, retrieved_chunks, k
                    )
                    
                    result = RetrievalEvaluationResult(
                        query_id=f"{query_id}_k{k}",
                        relevant_chunks=relevant_chunks,
                        retrieved_chunks=retrieved_chunks[:k],
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                        retrieval_time=retrieval_time,
                        similarity_scores=similarity_scores[:k]
                    )
                    evaluation_results.append(result)
        
        # Calculate aggregate metrics
        if evaluation_results:
            precisions = [r.precision for r in evaluation_results]
            recalls = [r.recall for r in evaluation_results]
            f1_scores = [r.f1_score for r in evaluation_results]
            retrieval_times = [r.retrieval_time for r in evaluation_results]
            
            metrics = RAGEvaluationMetrics(
                index_type=index_result.index_type,
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                avg_precision=np.mean(precisions),
                avg_recall=np.mean(recalls),
                avg_f1_score=np.mean(f1_scores),
                std_precision=np.std(precisions),
                std_recall=np.std(recalls),
                std_f1_score=np.std(f1_scores),
                total_queries=len(evaluation_results),
                avg_retrieval_time=np.mean(retrieval_times),
                evaluation_results=evaluation_results
            )
        else:
            # Return empty metrics if no results
            metrics = RAGEvaluationMetrics(
                index_type=index_result.index_type,
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_f1_score=0.0,
                std_precision=0.0,
                std_recall=0.0,
                std_f1_score=0.0,
                total_queries=0,
                avg_retrieval_time=0.0,
                evaluation_results=[]
            )
        
        return metrics

    def evaluate_with_custom_queries(self,
                                   index_result,
                                   embedding_model: str,
                                   chunking_method: str,
                                   custom_queries: List[CustomQuery],
                                   k_values: List[int] = [5, 10, 20]) -> RAGEvaluationMetrics:
        """
        Evaluate FAISS index using custom domain-specific queries

        Args:
            index_result: FAISSIndexResult object
            embedding_model: Name of the embedding model used
            chunking_method: Name of the chunking method used
            custom_queries: List of CustomQuery objects
            k_values: List of k values to evaluate

        Returns:
            RAGEvaluationMetrics object with evaluation results
        """
        from src.vector_storage.faiss_storage import FAISSStorage
        from src.embedding.embedding_methods import EmbeddingMethods

        faiss_storage = FAISSStorage()
        embedding_methods = EmbeddingMethods()
        model = embedding_methods.load_model(embedding_model)

        evaluation_results = []

        for custom_query in custom_queries:
            # Generate query embedding
            query_embedding = model.encode([custom_query.query_text])[0]

            # Perform search
            start_time = time.time()
            search_results, search_time = faiss_storage.search(
                index_result=index_result,
                query_embedding=query_embedding,
                k=max(k_values)
            )
            retrieval_time = time.time() - start_time

            # Extract retrieved chunk IDs
            retrieved_chunks = [result.chunk_id for result in search_results]
            similarity_scores = [result.similarity_score for result in search_results]

            # Calculate metrics for different k values
            for k in k_values:
                precision, recall, f1 = self.calculate_retrieval_metrics(
                    custom_query.relevant_chunk_ids, retrieved_chunks, k
                )

                result = RetrievalEvaluationResult(
                    query_id=f"{custom_query.query_id}_k{k}",
                    relevant_chunks=custom_query.relevant_chunk_ids,
                    retrieved_chunks=retrieved_chunks[:k],
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    retrieval_time=retrieval_time,
                    similarity_scores=similarity_scores[:k]
                )
                evaluation_results.append(result)

        # Calculate aggregate metrics
        if evaluation_results:
            precisions = [r.precision for r in evaluation_results]
            recalls = [r.recall for r in evaluation_results]
            f1_scores = [r.f1_score for r in evaluation_results]
            retrieval_times = [r.retrieval_time for r in evaluation_results]

            metrics = RAGEvaluationMetrics(
                index_type=index_result.index_type,
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                avg_precision=np.mean(precisions),
                avg_recall=np.mean(recalls),
                avg_f1_score=np.mean(f1_scores),
                std_precision=np.std(precisions),
                std_recall=np.std(recalls),
                std_f1_score=np.std(f1_scores),
                total_queries=len(evaluation_results),
                avg_retrieval_time=np.mean(retrieval_times),
                evaluation_results=evaluation_results,
                evaluation_type="custom"
            )
        else:
            # Return empty metrics if no results
            metrics = RAGEvaluationMetrics(
                index_type=index_result.index_type,
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_f1_score=0.0,
                std_precision=0.0,
                std_recall=0.0,
                std_f1_score=0.0,
                total_queries=0,
                avg_retrieval_time=0.0,
                evaluation_results=[],
                evaluation_type="custom"
            )

        return metrics

    def create_manual_query(self,
                          query_text: str,
                          topic_area: str,
                          relevant_chunk_ids: List[int],
                          description: str = "") -> CustomQuery:
        """
        Create a manual custom query for evaluation

        Args:
            query_text: The query text
            topic_area: Topic area category
            relevant_chunk_ids: List of chunk IDs that should be considered relevant
            description: Optional description of the query

        Returns:
            CustomQuery object
        """
        # Get keywords for the topic area if it exists
        keywords = self.clinical_trial_keywords.get(topic_area, [])

        query_id = f"manual_{topic_area}_{len(relevant_chunk_ids)}chunks"

        return CustomQuery(
            query_id=query_id,
            query_text=query_text,
            topic_area=topic_area,
            keywords=keywords,
            relevant_chunk_ids=relevant_chunk_ids,
            description=description or f"Manual query for {topic_area}"
        )
    
    def compare_indices(self, evaluation_results: List[RAGEvaluationMetrics]) -> pd.DataFrame:
        """
        Compare multiple index evaluation results
        
        Args:
            evaluation_results: List of RAGEvaluationMetrics
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for metrics in evaluation_results:
            row_data = {
                'Index Type': metrics.index_type,
                'Embedding Model': metrics.embedding_model,
                'Chunking Method': metrics.chunking_method,
                'Total Queries': metrics.total_queries,
                'Avg Retrieval Time (ms)': f"{metrics.avg_retrieval_time * 1000:.2f}"
            }

            # Add component-specific metrics if available
            if hasattr(metrics, 'avg_hit_rate'):
                row_data.update({
                    'Hit Rate': f"{metrics.avg_hit_rate:.4f}",
                    'MRR': f"{metrics.avg_mrr:.4f}",
                    'Std Hit Rate': f"{metrics.std_hit_rate:.4f}",
                    'Std MRR': f"{metrics.std_mrr:.4f}"
                })

                # Add generation metrics if available
                if hasattr(metrics, 'avg_bleu_score') and metrics.avg_bleu_score is not None:
                    row_data.update({
                        'BLEU Score': f"{metrics.avg_bleu_score:.4f}",
                        'ROUGE-L': f"{metrics.avg_rouge_l:.4f}" if metrics.avg_rouge_l else "N/A",
                        'ROUGE-1': f"{metrics.avg_rouge_1:.4f}" if metrics.avg_rouge_1 else "N/A",
                        'ROUGE-2': f"{metrics.avg_rouge_2:.4f}" if metrics.avg_rouge_2 else "N/A"
                    })
            else:
                # Legacy F1-score metrics for backward compatibility
                if hasattr(metrics, 'avg_precision'):
                    row_data.update({
                        'Avg Precision': f"{metrics.avg_precision:.4f}",
                        'Avg Recall': f"{metrics.avg_recall:.4f}",
                        'Avg F1-Score': f"{metrics.avg_f1_score:.4f}",
                        'Std F1-Score': f"{metrics.std_f1_score:.4f}"
                    })

                # Add BERTScore metrics if available
                if hasattr(metrics, 'avg_bert_f1') and metrics.avg_bert_f1 is not None:
                    row_data.update({
                        'BERTScore Precision': f"{metrics.avg_bert_precision:.4f}",
                        'BERTScore Recall': f"{metrics.avg_bert_recall:.4f}",
                        'BERTScore F1': f"{metrics.avg_bert_f1:.4f}",
                    'Evaluation Type': metrics.evaluation_type
                })

            comparison_data.append(row_data)
        
        return pd.DataFrame(comparison_data)

    def _create_empty_metrics(self, index_result, embedding_model: str, chunking_method: str) -> RAGEvaluationMetrics:
        """Create empty metrics object"""
        return RAGEvaluationMetrics(
            index_type=index_result.index_type,
            embedding_model=embedding_model,
            chunking_method=chunking_method,
            avg_precision=0.0,
            avg_recall=0.0,
            avg_f1_score=0.0,
            std_precision=0.0,
            std_recall=0.0,
            std_f1_score=0.0,
            total_queries=0,
            avg_retrieval_time=0.0,
            evaluation_results=[],
            evaluation_type="bert_score"
        )

    def _calculate_aggregate_metrics_with_bert(self,
                                             evaluation_results: List[RetrievalEvaluationResult],
                                             index_result,
                                             embedding_model: str,
                                             chunking_method: str) -> RAGEvaluationMetrics:
        """Calculate aggregate metrics including BERTScore"""
        if not evaluation_results:
            return self._create_empty_metrics(index_result, embedding_model, chunking_method)

        # Traditional metrics
        precisions = [r.precision for r in evaluation_results]
        recalls = [r.recall for r in evaluation_results]
        f1_scores = [r.f1_score for r in evaluation_results]
        retrieval_times = [r.retrieval_time for r in evaluation_results]

        # BERTScore metrics
        bert_precisions = [r.bert_score_precision for r in evaluation_results if r.bert_score_precision is not None]
        bert_recalls = [r.bert_score_recall for r in evaluation_results if r.bert_score_recall is not None]
        bert_f1s = [r.bert_score_f1 for r in evaluation_results if r.bert_score_f1 is not None]

        metrics = RAGEvaluationMetrics(
            index_type=index_result.index_type,
            embedding_model=embedding_model,
            chunking_method=chunking_method,
            avg_precision=np.mean(precisions),
            avg_recall=np.mean(recalls),
            avg_f1_score=np.mean(f1_scores),
            std_precision=np.std(precisions),
            std_recall=np.std(recalls),
            std_f1_score=np.std(f1_scores),
            total_queries=len(evaluation_results),
            avg_retrieval_time=np.mean(retrieval_times),
            evaluation_results=evaluation_results,
            evaluation_type="bert_score",
            # BERTScore metrics
            avg_bert_precision=np.mean(bert_precisions) if bert_precisions else None,
            avg_bert_recall=np.mean(bert_recalls) if bert_recalls else None,
            avg_bert_f1=np.mean(bert_f1s) if bert_f1s else None,
            std_bert_precision=np.std(bert_precisions) if bert_precisions else None,
            std_bert_recall=np.std(bert_recalls) if bert_recalls else None,
            std_bert_f1=np.std(bert_f1s) if bert_f1s else None
        )

        return metrics

    def evaluate_with_component_metrics(self,
                                      index_result,
                                      embedding_model: str,
                                      chunking_method: str,
                                      evaluation_queries: Optional[List[CustomQuery]] = None,
                                      k_values: List[int] = [5, 10],
                                      include_generation: bool = False) -> RAGEvaluationMetrics:
        """
        Evaluate RAG system using component-specific metrics (Hit Rate, MRR, BLEU, ROUGE)

        Args:
            index_result: FAISSIndexResult object
            embedding_model: Name of the embedding model used
            chunking_method: Name of the chunking method used
            evaluation_queries: List of CustomQuery objects (if None, generates domain-specific)
            k_values: List of k values to evaluate
            include_generation: Whether to include generation metrics (BLEU/ROUGE)

        Returns:
            RAGEvaluationMetrics with component-specific results
        """
        from src.vector_storage.faiss_storage import FAISSStorage
        from src.embedding.embedding_methods import EmbeddingMethods

        # Generate domain-specific queries if not provided
        if evaluation_queries is None:
            logger.info("Generating domain-specific queries for component-specific evaluation...")
            evaluation_queries = self.generate_domain_specific_queries(
                index_result.chunk_texts,
                num_queries_per_topic=2
            )

        faiss_storage = FAISSStorage()
        embedding_methods = EmbeddingMethods()
        model = embedding_methods.load_model(embedding_model)

        evaluation_results = []

        for query in evaluation_queries:
            # Generate query embedding
            query_embedding = model.encode([query.query_text])[0]

            # Perform search
            start_time = time.time()
            search_results, search_time = faiss_storage.search(
                index_result=index_result,
                query_embedding=query_embedding,
                k=max(k_values)
            )
            retrieval_time = time.time() - start_time

            # Extract retrieved chunk IDs
            retrieved_chunks = [result.chunk_id for result in search_results]
            similarity_scores = [result.similarity_score for result in search_results]

            # Calculate metrics for different k values
            for k in k_values:
                # Retrieval Component Metrics
                hit_rate = self.calculate_hit_rate(query.relevant_chunk_ids, retrieved_chunks, k)
                mrr_score = self.calculate_mrr(query.relevant_chunk_ids, retrieved_chunks, k)

                # Generation Component Metrics (if enabled)
                bleu_score = None
                rouge_scores = {"rouge_1": None, "rouge_2": None, "rouge_l": None}

                if include_generation:
                    # Get retrieved texts for generation evaluation
                    retrieved_texts = [index_result.chunk_texts[chunk_id] for chunk_id in retrieved_chunks[:k]]
                    ground_truth_texts = [index_result.chunk_texts[chunk_id] for chunk_id in query.relevant_chunk_ids]

                    if retrieved_texts and ground_truth_texts:
                        # Use the best retrieved text as candidate
                        candidate = retrieved_texts[0] if retrieved_texts else ""

                        # Calculate BLEU and ROUGE scores
                        bleu_score = self.calculate_bleu_score(candidate, ground_truth_texts)
                        rouge_scores = self.calculate_rouge_scores(candidate, ground_truth_texts)

                result = RetrievalEvaluationResult(
                    query_id=f"{query.query_id}_k{k}",
                    relevant_chunks=query.relevant_chunk_ids,
                    retrieved_chunks=retrieved_chunks[:k],
                    hit_rate=hit_rate,
                    mrr_score=mrr_score,
                    retrieval_time=retrieval_time,
                    similarity_scores=similarity_scores[:k],
                    bleu_score=bleu_score,
                    rouge_l_score=rouge_scores["rouge_l"],
                    rouge_1_score=rouge_scores["rouge_1"],
                    rouge_2_score=rouge_scores["rouge_2"]
                )
                evaluation_results.append(result)

        # Calculate aggregate metrics
        return self._calculate_component_aggregate_metrics(
            evaluation_results, index_result, embedding_model, chunking_method, include_generation
        )

    def _calculate_component_aggregate_metrics(self,
                                             evaluation_results: List[RetrievalEvaluationResult],
                                             index_result,
                                             embedding_model: str,
                                             chunking_method: str,
                                             include_generation: bool) -> RAGEvaluationMetrics:
        """Calculate aggregate metrics for component-specific evaluation"""
        if not evaluation_results:
            return RAGEvaluationMetrics(
                index_type=index_result.index_type,
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                avg_hit_rate=0.0,
                avg_mrr=0.0,
                std_hit_rate=0.0,
                std_mrr=0.0,
                total_queries=0,
                avg_retrieval_time=0.0,
                evaluation_results=[]
            )

        # Retrieval metrics
        hit_rates = [r.hit_rate for r in evaluation_results]
        mrr_scores = [r.mrr_score for r in evaluation_results]
        retrieval_times = [r.retrieval_time for r in evaluation_results]

        # Generation metrics (if applicable)
        bleu_scores = [r.bleu_score for r in evaluation_results if r.bleu_score is not None]
        rouge_l_scores = [r.rouge_l_score for r in evaluation_results if r.rouge_l_score is not None]
        rouge_1_scores = [r.rouge_1_score for r in evaluation_results if r.rouge_1_score is not None]
        rouge_2_scores = [r.rouge_2_score for r in evaluation_results if r.rouge_2_score is not None]

        metrics = RAGEvaluationMetrics(
            index_type=index_result.index_type,
            embedding_model=embedding_model,
            chunking_method=chunking_method,
            # Retrieval Component Metrics
            avg_hit_rate=np.mean(hit_rates),
            avg_mrr=np.mean(mrr_scores),
            std_hit_rate=np.std(hit_rates),
            std_mrr=np.std(mrr_scores),
            total_queries=len(evaluation_results),
            avg_retrieval_time=np.mean(retrieval_times),
            # Generation Component Metrics
            avg_bleu_score=np.mean(bleu_scores) if bleu_scores else None,
            avg_rouge_l=np.mean(rouge_l_scores) if rouge_l_scores else None,
            avg_rouge_1=np.mean(rouge_1_scores) if rouge_1_scores else None,
            avg_rouge_2=np.mean(rouge_2_scores) if rouge_2_scores else None,
            std_bleu_score=np.std(bleu_scores) if bleu_scores else None,
            std_rouge_l=np.std(rouge_l_scores) if rouge_l_scores else None,
            std_rouge_1=np.std(rouge_1_scores) if rouge_1_scores else None,
            std_rouge_2=np.std(rouge_2_scores) if rouge_2_scores else None,
            evaluation_results=evaluation_results,
            evaluation_type="component_specific"
        )

        return metrics
