"""
Fast nearest neighbor search for embedding-based inference.
Uses FAISS for efficient similarity search.
"""

import numpy as np
import torch
from pathlib import Path
import pickle


try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using numpy (slower).")


class EmbeddingIndex:
    """Fast similarity search for embeddings."""
    
    def __init__(self, dimension: int = 128, use_gpu: bool = False):
        """
        Initialize embedding index.
        
        Args:
            dimension: Embedding dimension
            use_gpu: Use GPU for search (if available)
        """
        self.dimension = dimension
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        
        if FAISS_AVAILABLE:
            # Create FAISS index
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors)
            
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("✓ Using GPU for search")
        else:
            self.index = None
            self.embeddings = None
        
        self.texts = []
        self.metadata = []
    
    def add_embeddings(self, embeddings: np.ndarray, texts: list, metadata: list = None):
        """
        Add embeddings to index.
        
        Args:
            embeddings: (N, dimension) array
            texts: List of corresponding texts
            metadata: Optional metadata for each embedding
        """
        # Normalize embeddings (for cosine similarity)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if FAISS_AVAILABLE:
            self.index.add(embeddings.astype('float32'))
        else:
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.texts.extend(texts)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
        
        print(f"✓ Added {len(texts)} embeddings to index")
    
    def search(self, query: np.ndarray, k: int = 5) -> list:
        """
        Search for nearest neighbors.
        
        Args:
            query: (dimension,) or (batch, dimension) array
            k: Number of neighbors to return
            
        Returns:
            List of (text, similarity, metadata) tuples
        """
        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize query
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        query = query.astype('float32')
        
        if FAISS_AVAILABLE:
            # FAISS search
            similarities, indices = self.index.search(query, k)
            
            results = []
            for sims, idxs in zip(similarities, indices):
                batch_results = [
                    (self.texts[idx], float(sim), self.metadata[idx])
                    for sim, idx in zip(sims, idxs)
                    if idx < len(self.texts)  # Valid index
                ]
                results.append(batch_results)
        else:
            # Numpy search (slower)
            similarities = np.dot(query, self.embeddings.T)
            
            results = []
            for sims in similarities:
                top_k_indices = np.argsort(sims)[::-1][:k]
                batch_results = [
                    (self.texts[idx], float(sims[idx]), self.metadata[idx])
                    for idx in top_k_indices
                ]
                results.append(batch_results)
        
        return results[0] if len(results) == 1 else results
    
    def save(self, path: str):
        """Save index to disk."""
        save_dict = {
            'texts': self.texts,
            'metadata': self.metadata,
            'dimension': self.dimension,
        }
        
        if FAISS_AVAILABLE:
            # Save FAISS index
            index_path = Path(path).parent / 'faiss_index.bin'
            faiss.write_index(self.index, str(index_path))
            save_dict['index_path'] = str(index_path)
        else:
            save_dict['embeddings'] = self.embeddings
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Saved index to {path}")
    
    @classmethod
    def load(cls, path: str, use_gpu: bool = False):
        """Load index from disk."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        index = cls(dimension=save_dict['dimension'], use_gpu=use_gpu)
        index.texts = save_dict['texts']
        index.metadata = save_dict['metadata']
        
        if FAISS_AVAILABLE and 'index_path' in save_dict:
            # Load FAISS index
            index.index = faiss.read_index(save_dict['index_path'])
            
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index.index = faiss.index_cpu_to_gpu(res, 0, index.index)
        else:
            index.embeddings = save_dict.get('embeddings')
        
        print(f"✓ Loaded index with {len(index.texts)} embeddings")
        return index


def build_index_from_training_data(
    audio_embeddings_path: str,
    texts_path: str,
    output_path: str = 'models/embedding_index.pkl'
):
    """
    Build embedding index from training data.
    
    Args:
        audio_embeddings_path: Path to audio embeddings .npy
        texts_path: Path to texts .txt
        output_path: Where to save index
    """
    print("Building embedding index from training data...")
    
    # Load embeddings
    embeddings = np.load(audio_embeddings_path)
    print(f"✓ Loaded {len(embeddings)} audio embeddings")
    
    # Load texts
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    print(f"✓ Loaded {len(texts)} texts")
    
    # Create index
    index = EmbeddingIndex(dimension=embeddings.shape[1])
    index.add_embeddings(embeddings, texts)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output_path))
    
    print(f"\n✓ Built and saved index to {output_path}")
    return index


def benchmark_search(index: EmbeddingIndex, num_queries: int = 100):
    """Benchmark search speed."""
    import time
    
    print(f"\nBenchmarking search with {num_queries} queries...")
    
    # Random queries
    queries = np.random.randn(num_queries, index.dimension).astype('float32')
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Warm up
    index.search(queries[0], k=5)
    
    # Benchmark
    start = time.time()
    for query in queries:
        index.search(query, k=5)
    elapsed = time.time() - start
    
    avg_latency = (elapsed / num_queries) * 1000  # ms
    
    print(f"✓ Average search latency: {avg_latency:.2f} ms")
    print(f"✓ Throughput: {num_queries / elapsed:.1f} queries/sec")
    
    return avg_latency


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build and benchmark embedding index')
    parser.add_argument('--embeddings', default='data/features/audio_embeddings.npy',
                       help='Path to audio embeddings')
    parser.add_argument('--texts', default='data/features/audio_texts.txt',
                       help='Path to texts')
    parser.add_argument('--output', default='models/embedding_index.pkl',
                       help='Output path')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark')
    
    args = parser.parse_args()
    
    # Build index
    if Path(args.embeddings).exists() and Path(args.texts).exists():
        index = build_index_from_training_data(
            args.embeddings,
            args.texts,
            args.output
        )
        
        # Benchmark
        if args.benchmark:
            benchmark_search(index)
    else:
        print("Error: Embeddings or texts not found!")
        print("Please run preprocessing first.")