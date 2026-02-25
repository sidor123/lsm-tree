import unittest
import os
import logging
from inverted_index import (
    InvertedIndex, 
    RoaringBitmapWrapper,
    TextPreprocessor
)

logger = logging.getLogger(__name__)


class TestRoaringBitmapWrapper(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
    
    def tearDown(self):
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_basic_operations(self):
        bitmap = RoaringBitmapWrapper()
        bitmap.add(1)
        bitmap.add(2)
        bitmap.add(3)
        
        self.assertEqual(len(bitmap), 3)
        self.assertIn(1, bitmap)
        self.assertIn(2, bitmap)
        self.assertNotIn(4, bitmap)
    
    def test_union(self):
        bitmap1 = RoaringBitmapWrapper([1, 2, 3])
        bitmap2 = RoaringBitmapWrapper([3, 4, 5])
        
        result = bitmap1.union(bitmap2)
        
        self.assertEqual(len(result), 5)
        self.assertEqual(result.to_list(), [1, 2, 3, 4, 5])
    
    def test_intersection(self):
        bitmap1 = RoaringBitmapWrapper([1, 2, 3, 4])
        bitmap2 = RoaringBitmapWrapper([3, 4, 5, 6])
        
        result = bitmap1.intersection(bitmap2)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result.to_list(), [3, 4])
    
    def test_difference(self):
        bitmap1 = RoaringBitmapWrapper([1, 2, 3, 4])
        bitmap2 = RoaringBitmapWrapper([3, 4, 5])
        
        result = bitmap1.difference(bitmap2)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result.to_list(), [1, 2])
    
    def test_serialization(self):
        bitmap = RoaringBitmapWrapper([1, 2, 3, 4, 5])
        
        data = bitmap.serialize()
        
        restored = RoaringBitmapWrapper.deserialize(data)
        
        self.assertEqual(len(restored), 5)
        self.assertEqual(restored.to_list(), [1, 2, 3, 4, 5])


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
    
    def tearDown(self):
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_tokenization(self):
        preprocessor = TextPreprocessor(use_stemming=False, remove_stopwords=False)
        
        text = "Hello World! This is a test."
        tokens = preprocessor.tokenize(text)
        
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
    
    def test_stopword_removal(self):
        preprocessor = TextPreprocessor(use_stemming=False, remove_stopwords=True)
        
        text = "the quick brown fox"
        tokens = preprocessor.preprocess(text)
        
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
    
    def test_stemming(self):
        preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=False)
        
        text = "running runner runs"
        tokens = preprocessor.preprocess(text)
        
        unique_stems = set(tokens)
        self.assertLessEqual(len(unique_stems), 2)  # Should have 1-2 unique stems


class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.index = InvertedIndex(use_stemming=True, remove_stopwords=True)
    
    def tearDown(self):
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_add_document(self):
        doc_id = self.index.add_document("Python is great")
        
        self.assertEqual(doc_id, 0)
        self.assertEqual(len(self.index.documents), 1)
        self.assertGreater(len(self.index.index), 0)
    
    def test_get_document(self):
        text = "Python programming language"
        doc_id = self.index.add_document(text)
        
        retrieved = self.index.get_document(doc_id)
        self.assertEqual(retrieved, text)
    
    def test_search_term(self):
        self.index.add_document("Python is great")
        self.index.add_document("Java is also good")
        self.index.add_document("Python and Java are popular")
        
        results = self.index.search_term("python")
        doc_ids = results.to_list()
        
        self.assertEqual(len(doc_ids), 2)
        self.assertIn(0, doc_ids)
        self.assertIn(2, doc_ids)
    
    def test_boolean_and(self):
        self.index.add_document("Python is great for machine learning")
        self.index.add_document("Java is used in enterprise")
        self.index.add_document("Python and machine learning")
        
        results = self.index.search_boolean("python AND machine")
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(2, results)
    
    def test_boolean_or(self):
        self.index.add_document("Python programming")
        self.index.add_document("Java development")
        self.index.add_document("JavaScript coding")
        
        results = self.index.search_boolean("python OR java")
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(1, results)
    
    def test_boolean_not(self):
        self.index.add_document("Python programming")
        self.index.add_document("Java programming")
        self.index.add_document("Python and Java")
        
        results = self.index.search_boolean("python AND NOT java")
        
        self.assertEqual(len(results), 1)
        self.assertIn(0, results)
    
    def test_complex_boolean_query(self):
        self.index.add_document("Python for web development")
        self.index.add_document("Java for mobile apps")
        self.index.add_document("Python for machine learning")
        self.index.add_document("JavaScript for web frontend")
        
        results = self.index.search_boolean("(python OR java) AND NOT web")
        
        self.assertEqual(len(results), 2)
        self.assertIn(1, results)  # Java for mobile
        self.assertIn(2, results)  # Python for machine learning
    
    def test_empty_query(self):
        self.index.add_document("Test document")
        
        results = self.index.search_boolean("")
        
        self.assertEqual(len(results), 0)
    
    def test_nonexistent_term(self):
        self.index.add_document("Python programming")
        
        results = self.index.search_term("nonexistent")
        
        self.assertEqual(len(results.to_list()), 0)
    
    def test_get_stats(self):
        self.index.add_document("Python is great")
        self.index.add_document("Java is good")
        
        stats = self.index.get_stats()
        
        self.assertEqual(stats['num_documents'], 2)
        self.assertGreater(stats['num_terms'], 0)
        self.assertGreater(stats['total_postings'], 0)


if __name__ == '__main__':
    log_dir = 'tests/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'inverted_index_test.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )

    unittest.main()
