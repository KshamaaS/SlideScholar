import os
import unittest
import numpy as np
import tempfile
import json
from unittest.mock import patch, MagicMock

# Import the vdb class
from .vdb import vdb

class TestVDB(unittest.TestCase):
    
    @patch('db.vdb.SentenceTransformer')
    def setUp(self, MockModel):
        # Setup mock model's encode method to return random vectors of dim 1024
        self.mock_model_instance = MockModel.return_value
        
        # Lambda to return proper shape of embeddings
        def mock_encode(texts, **kwargs):
            return np.random.rand(len(texts), 1024).astype('float32')
            
        self.mock_model_instance.encode.side_effect = mock_encode
        
        # Initialize db with mocked device
        self.db = vdb(device="cpu")
        
        # Create a temporary directory for saving and loading index
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_path = os.path.join(self.temp_dir.name, "index.faiss")
        self.chunks_path = os.path.join(self.temp_dir.name, "chunks.json")

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    @patch('db.vdb.SentenceTransformer')
    def test_init(self, MockModel):
        """Test proper initialization of the vdb class"""
        db_instance = vdb(device="cpu")
        self.assertEqual(db_instance.dim, 1024)
        self.assertIsNone(db_instance.index)
        self.assertEqual(db_instance.chunks, [])
        self.assertEqual(db_instance.device, "cpu")

    def test_embed_texts_prefixes(self):
        """Test whether text prefixes are passed correctly into the embedding generation"""
        texts = ["text1", "text2"]
        embeddings = self.db.embed_texts(texts, prefix="passage: ")
        
        self.assertEqual(embeddings.shape, (2, 1024))
        # Ensure model.encode was called
        self.mock_model_instance.encode.assert_called_once()
        args, kwargs = self.mock_model_instance.encode.call_args
        
        # The texts array should be prefixed
        self.assertEqual(args[0], ["passage: text1", "passage: text2"])
        # Should normalize embeddings by default as per E-03
        self.assertTrue(kwargs.get('normalize_embeddings', False))
        
    def test_add_documents_flat_index(self):
        """Test adding documents and building a flat index"""
        docs = [{"text": "doc1", "metadata": {"id": 1}}, {"text": "doc2", "metadata": {"id": 2}}]
        self.db.add_documents(docs, index_type='flat')
        
        self.assertIsNotNone(self.db.index)
        self.assertEqual(self.db.index.ntotal, 2)
        self.assertEqual(len(self.db.chunks), 2)
        
    def test_add_documents_ivfflat_fallback(self):
        """Test adding fewer documents than nlist causes fallback to Flat index"""
        docs = [{"text": "doc1"}] * 10 
        # Since 10 documents < 100 nlist, it should cleanly fall back to flat index
        self.db.add_documents(docs, index_type='ivfflat', nlist=100)
        self.assertEqual(self.db.index.ntotal, 10)

    def test_save_and_load_index(self):
        """Test persisting and reloading the faiss index + json chunks"""
        docs = [{"text": "doc1", "metadata": {"id": 1}}, {"text": "doc2", "metadata": {"id": 2}}]
        self.db.add_documents(docs, index_type='flat')
        
        # Save mechanism
        self.db.save_index(self.index_path, self.chunks_path)
        
        self.assertTrue(os.path.exists(self.index_path))
        self.assertTrue(os.path.exists(self.chunks_path))
        
        # Reloading mechanism onto a new DB instance
        with patch('db.vdb.SentenceTransformer'):
            new_db = vdb(device="cpu")
            new_db.load_index(self.index_path, self.chunks_path)
            
            self.assertIsNotNone(new_db.index)
            self.assertEqual(new_db.index.ntotal, 2)
            self.assertEqual(len(new_db.chunks), 2)
            self.assertEqual(new_db.chunks[0]['metadata']['id'], 1)

    def test_search(self):
        """Test similarity search retrieves expected formatting and queries use the query prefix format"""
        docs = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
        self.db.add_documents(docs, index_type='flat')
        
        results = self.db.search("query text", top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn('score', results[0])
        self.assertIn('chunk', results[0])
        self.assertIn(results[0]['chunk']['text'], ["doc1", "doc2", "doc3"])  # Any doc is valid since embeddings are random
        
        # Verify query had correct prefix applied
        args, kwargs = self.mock_model_instance.encode.call_args
        self.assertEqual(args[0], ["query: query text"])

if __name__ == '__main__':
    unittest.main()
