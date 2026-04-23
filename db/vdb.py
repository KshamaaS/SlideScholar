import json
import os
import zipfile
import tempfile
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

class vdb:
    def __init__(self, model_name='intfloat/e5-large-v2', device=None, verbose=False):
        """
        Initialize the vector database with the specified embedding model.
        E-01: Build embed_texts() — e5-large-v2 setup on GPU.
        
        Args:
            model_name (str): The name of the sentence-transformers model to use.
            device (str, optional): The device to run the model on ('cuda', 'mps', 'cpu'). Auto-detected if None.
            verbose (bool): Whether to print verbose output statements during operations.
            
        Returns:
            None
        """
        if device is None:
            # Automatically detect GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        self.verbose = verbose
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = 1024  # e5-large-v2 embedding dimension
        self.index = None
        self.chunks = []

        self.vprint(f"[VDB.__init__] Model {model_name} loaded on {self.device}.")

    def _embed_texts(self, texts, prefix, batch_size=32, normalize_embeddings=True):
        """
        E-01 & E-02 & E-03: Embed texts with appropriate prefix.
        Batch embed all chunks (batch_size=32, normalize_embeddings=True).
        
        Args:
            texts (list of str): The list of text strings to embed.
            prefix (str): The prefix to prepend to each text before embedding (e.g., 'passage: ', 'query: ').
            batch_size (int): The batch size for encoding.
            normalize_embeddings (bool): Whether to normalize the resulting embeddings.
            
        Returns:
            numpy.ndarray: The generated embeddings.
        """
        self.vprint(f"[VDB._embed_texts] Embedding {len(texts)} texts with prefix '{prefix}'...")
        
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings

    def _build_index(self, embeddings, index_type='ivfflat', nlist=100):
        """
        E-04: Build FAISS IndexFlatIP from embeddings (dim=1024)
        E-08: Switch to IVFFlat index for HuggingFace Spaces 16 GB RAM cap
        
        Args:
            embeddings (numpy.ndarray): The embeddings to add to the index.
            index_type (str): The type of FAISS index to build ('flat' or 'ivfflat').
            nlist (int): The number of cells/clusters for IVFFlat index.
        """
        
        self.vprint(f"[VDB._build_index] Building index of type '{index_type}' with {len(embeddings)} embeddings...")

        if self.index is not None:
            print("[VDB._build_index] WARNING: Index already exists. Index will be overwritten.")
        
        if index_type.lower() == 'flat':
            # E-04: FAISS IndexFlatIP
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
            
        elif index_type.lower() == 'ivfflat':
            # E-08: FAISS IndexIVFFlat
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # IVFFlat index must be trained before adding data
            if not self.index.is_trained:
                self.vprint("[VDB.build_index] Training IVFFlat index...")
                
                try:
                    self.index.train(embeddings)
                except Exception as e:
                    print("[VDB.build_index] ERROR: Failed to train IVFFlat index. reset to None:", e)
                    self.index = None
                    
                
            self.index.add(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def save_index(self, index_path, chunks_path):
        """
        E-05: Save index to Drive with faiss.write_index()
        
        Args:
            index_path (str): The file path where the FAISS index will be saved.
            chunks_path (str): The file path where the chunks JSON will be saved.
        """
        if self.index is None:
            raise ValueError("No index to save. Please build the index first.")
            
        self.vprint(f"[VDB.save_index] Saving index to {index_path}...")
        faiss.write_index(self.index, index_path)
        
        self.vprint(f"[VDB.save_index] Saving chunks to {chunks_path}...")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, index_path, chunks_path):
        """
        E-06: Build index loader — faiss.read_index() + load chunks.json
        
        Args:
            index_path (str): The file path from which the FAISS index will be loaded.
            chunks_path (str): The file path from which the chunks JSON will be loaded.
        """
        self.vprint(f"[VDB.load_index] Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        self.vprint(f"[VDB.load_index] Loading chunks from {chunks_path}...")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)


    def add_documents(self, chunks, index_type='ivfflat', nlist=100):
        """
        Process a list of chunks, embed them, and add to the index.
        `chunks` should be a list of dicts, e.g., [{"text": "...", "metadata": {...}}, ...]
        
        Args:
            chunks (list of dict or str): The documents to add to the database.
            index_type (str): The type of FAISS index to build if one doesn't exist ('flat' or 'ivfflat').
            nlist (int): The number of cells/clusters for IVFFlat index.
        """

        self.vprint(f"[VDB.add_documents] Adding {len(chunks)} documents...")
        
        self.chunks.extend(chunks)
        texts = [chunk['text'] if isinstance(chunk, dict) else chunk for chunk in chunks]
        
        # E-02: Add e5 prefixes correctly (passage: for chunks)
        self.vprint(f"[VDB.add_documents] Embedding {len(texts)} passages...")
        embeddings = self._embed_texts(texts, prefix="passage: ", batch_size=32, normalize_embeddings=True)
        
        if self.index is None:
            self.vprint(f"[VDB.add_documents] Building {index_type} index...")
            # If dataset is smaller than nlist, faiss will throw error for IVFFlat, so we handle it 
            if index_type == 'ivfflat' and len(embeddings) < nlist:
                self.vprint(f"[VDB.add_documents] Warning: Not enough embeddings ({len(embeddings)}) for nlist={nlist}. Falling back to Flat index.")
                index_type = 'flat'
                
            self._build_index(embeddings, index_type=index_type, nlist=nlist)
        else:
            if len(self.chunks) > nlist * 10 and not isinstance(self.index, faiss.IndexFlat):
                self.vprint(f"[VDB.add_documents] Too many chunks ({len(self.chunks)}). Convert to IVFFlat index to save memory.")
                self.convert_to_ivfflat(nlist=nlist)
            self.index.add(embeddings)

    def search(self, query, top_k=5):
        """
        Search for documents relevant to the query.
        
        Args:
            query (str): The search query string.
            top_k (int): The number of top relevant documents to retrieve.
            
        Returns:
            list of dict: A list of dictionaries representing the top_k search results, 
                          each containing 'score' and 'chunk'.
        """
        self.vprint(f"[VDB.search] Searching for query: '{query}' (top_k={top_k})...")
            
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
            
        # E-02: Add e5 prefixes correctly (query: for queries)
        query_embedding = self.embed_texts([query], prefix="query: ", batch_size=1, normalize_embeddings=True)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    "score": float(dist),
                    "chunk": self.chunks[idx]
                })
        return results

    def migrate_to_ivf(self, nlist=100):
        """
        E-04: Migrate index from Flat to IVFFlat if not already IVF.

        Args:
            nlist (int): The number of cells/clusters for IVFFlat index.
        """
        self.vprint("[VDB.migrate_to_ivf] Migrating index from Flat to IVFFlat...")

        if not isinstance(self.index, faiss.IndexFlat):
            self.vprint("Index is already IVF or not initialized.")
            return

        n_total = self.index.ntotal
        all_vectors = self.index.reconstruct_n(0, n_total)

        quantizer = faiss.IndexFlatIP(self.dim)
        new_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)

        new_index.train(all_vectors)
        new_index.add(all_vectors)

        self.index = new_index

        self.vprint(f"Successfully migrated {n_total} vectors to IVFFlat.")

    def test_cross_lecture_retrieval(self, query, top_k=5):
        """
        E-07: Test cross-lecture retrieval — queries that span multiple lecture weeks
        
        Args:
            query (str): The search query string.
            top_k (int): The number of top relevant documents to retrieve.
            
        Returns:
            list of dict: A list of the search results with their scores and chunks.
        """
        self.vprint(f"--- Cross-Lecture Retrieval Test ---")
        self.vprint(f"Query: '{query}'")
        
        results = self.search(query, top_k=top_k)
        
        self.vprint(f"\nTop {top_k} Results:")
        for i, res in enumerate(results):
            score = res['score']
            chunk_data = res['chunk']
            
            # Extract text safely
            text = chunk_data.get('text', str(chunk_data)) if isinstance(chunk_data, dict) else str(chunk_data)
            preview = text[:150] + "..." if len(text) > 150 else text
            
            metadata = ""
            if isinstance(chunk_data, dict) and 'metadata' in chunk_data:
                metadata = f" | Meta: {chunk_data['metadata']}"
                
            self.vprint(f"Rank {i+1} (Score: {score:.4f}){metadata}\n   {preview}\n")
            
        return results

    def vprint(self, msg):
        if self.verbose:
            print(msg)