"""
Latent Semantic Indexing (LSI) Implementation

This module implements Latent Semantic Indexing/Analysis, a technique that uses
Singular Value Decomposition (SVD) to analyze relationships between documents and
terms by identifying latent semantic patterns.

LSI reduces the dimensionality of a term-document matrix to capture the underlying
semantic structure, handling synonymy and polysemy issues in text analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import re
from dataclasses import dataclass


@dataclass
class LSIConfig:
    """Configuration for LSI model."""
    n_components: int = 100  # Number of latent dimensions
    use_idf: bool = True  # Use inverse document frequency weighting
    normalize: bool = True  # Normalize document vectors
    min_word_length: int = 2  # Minimum word length to consider
    max_word_length: int = 50  # Maximum word length to consider


class LatentSemanticIndexing:
    """
    Latent Semantic Indexing for document analysis and retrieval.
    
    This class implements LSI using Singular Value Decomposition (SVD) to
    discover latent semantic relationships between terms and documents.
    
    Key Features:
    - TF-IDF weighting with logarithmic term frequency
    - Dimensionality reduction via truncated SVD
    - Document similarity using cosine similarity
    - Query projection into semantic space
    - Incremental document folding
    
    Attributes:
        config: LSI configuration parameters
        vocabulary: Mapping from terms to indices
        term_doc_matrix: Original term-document matrix
        U: Left singular vectors (term-concept matrix)
        S: Singular values (diagonal matrix)
        V: Right singular vectors (document-concept matrix)
        n_docs: Number of documents in the index
    """
    
    def __init__(self, config: Optional[LSIConfig] = None):
        """
        Initialize LSI model.
        
        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or LSIConfig()
        self.vocabulary: Dict[str, int] = {}
        self.idf_weights: Optional[np.ndarray] = None
        self.term_doc_matrix: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.n_docs: int = 0
        self.doc_ids: List[str] = []
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing and cleaning.
        
        Args:
            text: Input text string
            
        Returns:
            List of cleaned tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter by length
        tokens = [
            t for t in tokens 
            if self.config.min_word_length <= len(t) <= self.config.max_word_length
        ]
        
        return tokens
    
    def _build_vocabulary(self, documents: List[str]) -> None:
        """
        Build vocabulary from documents.
        
        Args:
            documents: List of document texts
        """
        all_terms = set()
        for doc in documents:
            terms = self._preprocess_text(doc)
            all_terms.update(terms)
        
        # Create term to index mapping
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
    
    def _compute_tf(self, term_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Compute logarithmic term frequency.
        
        Args:
            term_counts: Dictionary of term counts in a document
            
        Returns:
            Dictionary of term frequencies
        """
        tf = {}
        for term, count in term_counts.items():
            # Logarithmic term frequency: log(1 + count)
            tf[term] = np.log(1 + count)
        return tf
    
    def _compute_idf(self, documents: List[str]) -> np.ndarray:
        """
        Compute inverse document frequency for all terms.
        
        IDF formula: log(N / df_i) where N is number of docs, df_i is document frequency
        
        Args:
            documents: List of document texts
            
        Returns:
            Array of IDF weights for each term in vocabulary
        """
        n_docs = len(documents)
        doc_freq = np.zeros(len(self.vocabulary))
        
        for doc in documents:
            terms = set(self._preprocess_text(doc))
            for term in terms:
                if term in self.vocabulary:
                    doc_freq[self.vocabulary[term]] += 1
        
        # IDF with smoothing to avoid division by zero
        idf = np.log(n_docs / (1 + doc_freq))
        return idf
    
    def _build_term_doc_matrix(self, documents: List[str]) -> np.ndarray:
        """
        Build weighted term-document matrix.
        
        Args:
            documents: List of document texts
            
        Returns:
            Term-document matrix (terms x documents)
        """
        n_terms = len(self.vocabulary)
        n_docs = len(documents)
        
        matrix = np.zeros((n_terms, n_docs))
        
        for doc_idx, doc in enumerate(documents):
            # Count term frequencies
            terms = self._preprocess_text(doc)
            term_counts = Counter(terms)
            
            # Compute TF
            tf = self._compute_tf(term_counts)
            
            # Build column for this document
            for term, tf_value in tf.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    if self.config.use_idf:
                        # TF-IDF weighting
                        matrix[term_idx, doc_idx] = tf_value * self.idf_weights[term_idx]
                    else:
                        # Just TF
                        matrix[term_idx, doc_idx] = tf_value
        
        # Normalize document vectors if requested
        if self.config.normalize:
            col_norms = np.linalg.norm(matrix, axis=0, keepdims=True)
            # Avoid division by zero
            col_norms[col_norms == 0] = 1
            matrix = matrix / col_norms
        
        return matrix
    
    def fit(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> 'LatentSemanticIndexing':
        """
        Fit LSI model to documents using truncated SVD.
        
        Process:
        1. Build vocabulary from all documents
        2. Compute IDF weights
        3. Create weighted term-document matrix
        4. Apply truncated SVD for dimensionality reduction
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document identifiers
            
        Returns:
            Self for method chaining
        """
        if not documents:
            raise ValueError("Cannot fit LSI model with empty document list")
        
        self.n_docs = len(documents)
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(self.n_docs)]
        
        # Build vocabulary
        self._build_vocabulary(documents)
        
        # Compute IDF weights
        if self.config.use_idf:
            self.idf_weights = self._compute_idf(documents)
        
        # Build term-document matrix
        self.term_doc_matrix = self._build_term_doc_matrix(documents)
        
        # Perform truncated SVD
        # U: term-concept matrix (m x k)
        # S: singular values (k,)
        # Vt: concept-document matrix transposed (k x n)
        k = min(self.config.n_components, min(self.term_doc_matrix.shape) - 1)
        self.U, self.S, Vt = np.linalg.svd(self.term_doc_matrix, full_matrices=False)
        
        # Truncate to k components
        self.U = self.U[:, :k]
        self.S = self.S[:k]
        self.V = Vt[:k, :].T  # Transpose to get V (n x k)
        
        return self
    
    def transform_query(self, query: str) -> np.ndarray:
        """
        Transform a query into the LSI semantic space.
        
        The query is projected into the reduced space using:
        q_reduced = query_vector^T * U * S^-1
        
        Args:
            query: Query text string
            
        Returns:
            Query vector in reduced dimensional space
        """
        if self.U is None or self.S is None:
            raise ValueError("Model must be fitted before transforming queries")
        
        # Build query vector
        query_vec = np.zeros(len(self.vocabulary))
        terms = self._preprocess_text(query)
        term_counts = Counter(terms)
        tf = self._compute_tf(term_counts)
        
        for term, tf_value in tf.items():
            if term in self.vocabulary:
                term_idx = self.vocabulary[term]
                if self.config.use_idf:
                    query_vec[term_idx] = tf_value * self.idf_weights[term_idx]
                else:
                    query_vec[term_idx] = tf_value
        
        # Normalize query vector
        if self.config.normalize:
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
        
        # Project into reduced space: q_k = q^T * U * S^-1
        # This is equivalent to q_k = U^T * q / S
        S_inv = 1.0 / self.S
        query_reduced = self.U.T @ query_vec * S_inv
        
        return query_reduced
    
    def transform_documents(self, documents: List[str]) -> np.ndarray:
        """
        Transform new documents into LSI space (folding-in).
        
        Args:
            documents: List of new document texts
            
        Returns:
            Matrix of document vectors in reduced space (n_docs x k)
        """
        if self.U is None or self.S is None:
            raise ValueError("Model must be fitted before transforming documents")
        
        doc_matrix = np.zeros((len(self.vocabulary), len(documents)))
        
        for doc_idx, doc in enumerate(documents):
            terms = self._preprocess_text(doc)
            term_counts = Counter(terms)
            tf = self._compute_tf(term_counts)
            
            for term, tf_value in tf.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    if self.config.use_idf:
                        doc_matrix[term_idx, doc_idx] = tf_value * self.idf_weights[term_idx]
                    else:
                        doc_matrix[term_idx, doc_idx] = tf_value
        
        # Normalize
        if self.config.normalize:
            col_norms = np.linalg.norm(doc_matrix, axis=0, keepdims=True)
            col_norms[col_norms == 0] = 1
            doc_matrix = doc_matrix / col_norms
        
        # Project into reduced space
        S_inv = 1.0 / self.S
        docs_reduced = (self.U.T @ doc_matrix).T * S_inv
        
        return docs_reduced
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value between -1 and 1
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for documents most similar to query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, similarity_score) tuples, sorted by relevance
        """
        if self.V is None:
            raise ValueError("Model must be fitted before searching")
        
        # Transform query to reduced space
        query_vec = self.transform_query(query)
        
        # Compute similarities with all documents
        similarities = []
        for doc_idx in range(self.n_docs):
            doc_vec = self.V[doc_idx, :]
            sim = self.cosine_similarity(query_vec, doc_vec)
            similarities.append((self.doc_ids[doc_idx], sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_similar_documents(self, doc_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id: Document identifier
            top_k: Number of similar documents to return
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        if doc_id not in self.doc_ids:
            raise ValueError(f"Document ID '{doc_id}' not found in index")
        
        doc_idx = self.doc_ids.index(doc_id)
        doc_vec = self.V[doc_idx, :]
        
        # Compute similarities
        similarities = []
        for idx in range(self.n_docs):
            if idx != doc_idx:  # Skip the query document itself
                sim = self.cosine_similarity(doc_vec, self.V[idx, :])
                similarities.append((self.doc_ids[idx], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_term_concepts(self, term: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Get the top latent concepts associated with a term.
        
        Args:
            term: Term to analyze
            top_k: Number of top concepts to return
            
        Returns:
            List of (concept_index, weight) tuples
        """
        if term not in self.vocabulary:
            raise ValueError(f"Term '{term}' not in vocabulary")
        
        term_idx = self.vocabulary[term]
        term_vec = self.U[term_idx, :]
        
        # Get top concepts by absolute weight
        concept_weights = [(i, abs(term_vec[i])) for i in range(len(term_vec))]
        concept_weights.sort(key=lambda x: x[1], reverse=True)
        
        return concept_weights[:top_k]
    
    def reconstruct_document(self, doc_idx: int) -> np.ndarray:
        """
        Reconstruct a document from its LSI representation.
        
        Args:
            doc_idx: Document index
            
        Returns:
            Reconstructed term vector
        """
        if self.U is None or self.S is None or self.V is None:
            raise ValueError("Model must be fitted before reconstruction")
        
        # Reconstruction: A_k = U * S * V^T
        doc_reduced = self.V[doc_idx, :]
        reconstructed = self.U @ np.diag(self.S) @ doc_reduced
        
        return reconstructed
    
    def get_explained_variance(self) -> np.ndarray:
        """
        Get the proportion of variance explained by each component.
        
        Returns:
            Array of explained variance ratios
        """
        if self.S is None:
            raise ValueError("Model must be fitted first")
        
        total_variance = np.sum(self.S ** 2)
        explained_var = (self.S ** 2) / total_variance
        
        return explained_var
    
    def save(self, filepath: str) -> None:
        """
        Save LSI model to file.
        
        Args:
            filepath: Path to save the model
        """
        np.savez(
            filepath,
            vocabulary=self.vocabulary,
            idf_weights=self.idf_weights,
            U=self.U,
            S=self.S,
            V=self.V,
            doc_ids=self.doc_ids,
            config_n_components=self.config.n_components,
            config_use_idf=self.config.use_idf,
            config_normalize=self.config.normalize
        )
    
    @staticmethod
    def load(filepath: str) -> 'LatentSemanticIndexing':
        """
        Load LSI model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded LSI model
        """
        data = np.load(filepath, allow_pickle=True)
        
        config = LSIConfig(
            n_components=int(data['config_n_components']),
            use_idf=bool(data['config_use_idf']),
            normalize=bool(data['config_normalize'])
        )
        
        lsi = LatentSemanticIndexing(config)
        lsi.vocabulary = data['vocabulary'].item()
        lsi.idf_weights = data['idf_weights']
        lsi.U = data['U']
        lsi.S = data['S']
        lsi.V = data['V']
        lsi.doc_ids = list(data['doc_ids'])
        lsi.n_docs = len(lsi.doc_ids)
        
        return lsi
    
    def fit_context_vectors(self, context_vectors: np.ndarray, n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply LSI to a stack of context vectors and return Q, K, V matrices.
        
        This method performs SVD on context vectors (e.g., from transformer hidden states)
        to discover latent semantic structure and generate query, key, value matrices
        suitable for attention mechanisms.
        
        Process:
        1. Normalize context vectors (optional)
        2. Apply SVD: X = U * S * V^T
        3. Compute Q, K, V matrices from SVD components
        
        Args:
            context_vectors: Array of shape (n_vectors, hidden_dim)
                            Each row is a context vector (e.g., token representation)
            n_components: Number of latent dimensions (uses config if None)
            
        Returns:
            Tuple of (Q, K, V) matrices:
                Q: Query matrix (n_vectors, n_components)
                K: Key matrix (n_vectors, n_components) 
                V: Value matrix (n_vectors, n_components)
        
        Example:
            >>> context_vecs = np.random.randn(100, 512)  # 100 tokens, 512-dim
            >>> Q, K, V = lsi.fit_context_vectors(context_vecs, n_components=64)
            >>> # Use Q, K, V in attention: scores = Q @ K.T
        """
        if context_vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {context_vectors.shape}")
        
        n_vectors, hidden_dim = context_vectors.shape
        k = n_components or self.config.n_components
        k = min(k, min(n_vectors, hidden_dim) - 1)
        
        # Normalize context vectors if configured
        X = context_vectors.copy()
        if self.config.normalize:
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1
            X = X / row_norms
        
        # Perform SVD on transposed matrix
        # X^T = V * S * U^T where X is (n_vectors x hidden_dim)
        # So we get U (hidden_dim x k), S (k,), V^T (k x n_vectors)
        U, S, Vt = np.linalg.svd(X.T, full_matrices=False)
        
        # Truncate to k components
        U_k = U[:, :k]  # (hidden_dim, k)
        S_k = S[:k]     # (k,)
        V_k = Vt[:k, :].T  # (n_vectors, k)
        
        # Generate Q, K, V matrices
        # Q: Query matrix - projects vectors to semantic space
        # Q = X * U_k (n_vectors x k)
        Q = X @ U_k
        
        # K: Key matrix - weighted by singular values for importance
        # K = V_k * S_k (n_vectors x k)
        K = V_k * S_k[np.newaxis, :]
        
        # V: Value matrix - semantic content representation
        # V = V_k (n_vectors x k)
        V = V_k
        
        return Q, K, V
    
    def context_attention(self, context_vectors: np.ndarray, n_components: Optional[int] = None, 
                         scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LSI-based attention weights and attended values for context vectors.
        
        This combines LSI decomposition with attention mechanism to find
        semantic relationships between context vectors.
        
        Args:
            context_vectors: Array of shape (n_vectors, hidden_dim)
            n_components: Number of latent dimensions (uses config if None)
            scale: Whether to scale attention scores by sqrt(dim)
            
        Returns:
            Tuple of (attention_weights, attended_values):
                attention_weights: (n_vectors, n_vectors) attention matrix
                attended_values: (n_vectors, n_components) weighted combination
        
        Example:
            >>> context_vecs = model.context_state.unsqueeze(0).expand(seq_len, -1)
            >>> weights, attended = lsi.context_attention(context_vecs.numpy())
        """
        Q, K, V = self.fit_context_vectors(context_vectors, n_components)
        
        # Compute attention scores: Q @ K^T
        attention_scores = Q @ K.T
        
        # Scale by sqrt(d_k) if requested
        if scale:
            d_k = Q.shape[1]
            attention_scores = attention_scores / np.sqrt(d_k)
        
        # Apply softmax to get attention weights
        # Subtract max for numerical stability
        attention_scores_exp = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = attention_scores_exp / np.sum(attention_scores_exp, axis=1, keepdims=True)
        
        # Compute attended values: attention_weights @ V
        attended_values = attention_weights @ V
        
        return attention_weights, attended_values
    
    def semantic_clustering(self, context_vectors: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Cluster context vectors based on semantic similarity in LSI space.
        
        Args:
            context_vectors: Array of shape (n_vectors, hidden_dim)
            n_components: Number of latent dimensions (uses config if None)
            
        Returns:
            Cluster assignments for each vector based on dominant concept
        """
        Q, K, V = self.fit_context_vectors(context_vectors, n_components)
        
        # Assign each vector to its most dominant latent concept
        clusters = np.argmax(np.abs(V), axis=1)
        
        return clusters


if __name__ == "__main__":
    # Example 1: Document-based LSI
    print("="*60)
    print("Example 1: Document-based LSI")
    print("="*60)
    
    documents = [
        "The car is driven on the road",
        "The truck is driven on the highway",
        "The rose is a beautiful flower",
        "The daisy is a flower in the garden",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks for AI"
    ]
    
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Create and fit LSI model
    config = LSIConfig(n_components=3)
    lsi = LatentSemanticIndexing(config)
    lsi.fit(documents, doc_ids)
    
    # Search for similar documents
    query = "vehicle on street"
    print(f"\nQuery: '{query}'")
    results = lsi.search(query, top_k=3)
    print("\nTop results:")
    for doc_id, score in results:
        idx = lsi.doc_ids.index(doc_id)
        print(f"  {doc_id}: {score:.3f} - {documents[idx]}")
    
    # Find similar documents
    print(f"\nDocuments similar to {doc_ids[0]}:")
    similar = lsi.get_similar_documents(doc_ids[0], top_k=2)
    for doc_id, score in similar:
        idx = lsi.doc_ids.index(doc_id)
        print(f"  {doc_id}: {score:.3f} - {documents[idx]}")
    
    # Show explained variance
    explained_var = lsi.get_explained_variance()
    print(f"\nExplained variance by components:")
    for i, var in enumerate(explained_var):
        print(f"  Component {i}: {var:.3f} ({var*100:.1f}%)")
    
    # Example 2: Context Vector LSI with Q, K, V matrices
    print("\n" + "="*60)
    print("Example 2: LSI on Context Vectors (Q, K, V)")
    print("="*60)
    
    # Simulate context vectors (e.g., from transformer hidden states)
    np.random.seed(42)
    n_tokens = 50
    hidden_dim = 128
    context_vectors = np.random.randn(n_tokens, hidden_dim)
    
    # Add some structure: make certain vectors semantically similar
    context_vectors[10:15] = context_vectors[10:15] + np.random.randn(5, hidden_dim) * 0.1
    context_vectors[20:25] = context_vectors[20:25] + np.random.randn(5, hidden_dim) * 0.1
    
    config2 = LSIConfig(n_components=16, normalize=True)
    lsi2 = LatentSemanticIndexing(config2)
    
    # Get Q, K, V matrices
    Q, K, V = lsi2.fit_context_vectors(context_vectors, n_components=16)
    
    print(f"\nContext vectors shape: {context_vectors.shape}")
    print(f"Q (Query) matrix shape: {Q.shape}")
    print(f"K (Key) matrix shape: {K.shape}")
    print(f"V (Value) matrix shape: {V.shape}")
    
    # Compute attention
    attention_weights, attended_values = lsi2.context_attention(context_vectors, n_components=16)
    print(f"\nAttention weights shape: {attention_weights.shape}")
    print(f"Attended values shape: {attended_values.shape}")
    
    # Show attention pattern for first token
    print(f"\nTop 5 attention weights for token 0:")
    top_indices = np.argsort(attention_weights[0])[-5:][::-1]
    for idx in top_indices:
        print(f"  Token {idx}: {attention_weights[0, idx]:.4f}")
    
    # Semantic clustering
    clusters = lsi2.semantic_clustering(context_vectors, n_components=16)
    print(f"\nSemantic clusters assigned:")
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"  Cluster {cluster_id}: {count} vectors")
