import unittest
import os
import shutil
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
    
    def test_preprocessing(self):
        preprocessor = TextPreprocessor(use_stemming=False, remove_stopwords=False)
        
        tokens = ["Hello", "World", "This", "is", "a", "test"]
        result = preprocessor.preprocess(tokens)
        
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("test", result)
    
    def test_stopword_removal(self):
        preprocessor = TextPreprocessor(use_stemming=False, remove_stopwords=True)
        
        tokens = ["the", "quick", "brown", "fox"]
        result = preprocessor.preprocess(tokens)
        
        self.assertNotIn("the", result)
        self.assertIn("quick", result)
        self.assertIn("brown", result)
    
    def test_stemming(self):
        preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=False)
        
        tokens = ["running", "runner", "runs"]
        result = preprocessor.preprocess(tokens)
        
        unique_stems = set(result)
        self.assertLessEqual(len(unique_stems), 2)  # Should have 1-2 unique stems


class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "test_inverted_storage"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.index = InvertedIndex(storage_dir=self.test_dir, use_stemming=True, remove_stopwords=True)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_add_document(self):
        doc_id = self.index.add_document("Python is great", doc_id=0)
        
        self.assertEqual(doc_id, 0)
    
    def test_get_document(self):
        text = "Python programming language"
        doc_id = self.index.add_document(text, doc_id=0)
        
        retrieved = self.index.get_document(doc_id)
        self.assertEqual(retrieved, text)
    
    def test_search_term(self):
        self.index.add_document("Python is great", doc_id=0)
        self.index.add_document("Java is also good", doc_id=1)
        self.index.add_document("Python and Java are popular", doc_id=2)
        
        results = self.index.search_term(["python"])
        doc_ids = results.to_list()
        
        self.assertEqual(len(doc_ids), 2)
        self.assertIn(0, doc_ids)
        self.assertIn(2, doc_ids)
    
    def test_boolean_and(self):
        self.index.add_document("Python is great for machine learning", doc_id=0)
        self.index.add_document("Java is used in enterprise", doc_id=1)
        self.index.add_document("Python and machine learning", doc_id=2)
        
        results = self.index.search_boolean("python AND machine")
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(2, results)
    
    def test_boolean_or(self):
        self.index.add_document("Python programming", doc_id=0)
        self.index.add_document("Java development", doc_id=1)
        self.index.add_document("JavaScript coding", doc_id=2)
        
        results = self.index.search_boolean("python OR java")
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(1, results)
    
    def test_boolean_not(self):
        self.index.add_document("Python programming", doc_id=0)
        self.index.add_document("Java programming", doc_id=1)
        self.index.add_document("Python and Java", doc_id=2)
        
        results = self.index.search_boolean("python AND NOT java")
        
        self.assertEqual(len(results), 1)
        self.assertIn(0, results)
    
    def test_complex_boolean_query(self):
        self.index.add_document("Python for web development", doc_id=0)
        self.index.add_document("Java for mobile apps", doc_id=1)
        self.index.add_document("Python for machine learning", doc_id=2)
        self.index.add_document("JavaScript for web frontend", doc_id=3)
        
        results = self.index.search_boolean("(python OR java) AND NOT web")
        
        self.assertEqual(len(results), 2)
        self.assertIn(1, results)  # Java for mobile
        self.assertIn(2, results)  # Python for machine learning
    
    def test_empty_query(self):
        self.index.add_document("Test document", doc_id=0)
        
        results = self.index.search_boolean("")
        
        self.assertEqual(len(results), 0)
    
    def test_nonexistent_term(self):
        self.index.add_document("Python programming", doc_id=0)
        
        results = self.index.search_term(["nonexistent"])
        
        self.assertEqual(len(results.to_list()), 0)
    
    def test_get_stats(self):
        self.index.add_document("Python is great", doc_id=0)
        self.index.add_document("Java is good", doc_id=1)
        
        stats = self.index.get_stats()
        
        self.assertEqual(stats['num_documents'], 2)
        self.assertGreaterEqual(stats['term_lsm_layers'], 1)


class TestPrefixSearch(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "test_prefix_storage"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.index = InvertedIndex(storage_dir=self.test_dir, use_stemming=True, remove_stopwords=True, enable_kgram=True)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_prefix_search_basic(self):
        self.index.add_document("Python programming is great", doc_id=0)
        self.index.add_document("I love programming in Python", doc_id=1)
        self.index.add_document("Java programming language", doc_id=2)
        
        results = self.index.search_prefix(["prog"])
        
        self.assertEqual(len(results), 3)
        self.assertIn(0, results)
        self.assertIn(1, results)
        self.assertIn(2, results)
    
    def test_prefix_search_no_matches(self):
        self.index.add_document("Python is great", doc_id=0)
        self.index.add_document("Java is good", doc_id=1)
        
        results = self.index.search_prefix(["xyz"])
        
        self.assertEqual(len(results), 0)
    
    def test_prefix_search_single_char(self):
        self.index.add_document("Python programming", doc_id=0)
        self.index.add_document("Java development", doc_id=1)
        
        results = self.index.search_prefix(["p"])
        
        self.assertGreater(len(results), 0)
        self.assertIn(0, results)
    
    def test_prefix_search_full_term(self):
        self.index.add_document("Python programming", doc_id=0)
        self.index.add_document("Java programming", doc_id=1)
        
        results = self.index.search_prefix(["python"])
        
        self.assertEqual(len(results), 1)
        self.assertIn(0, results)
    
    def test_prefix_search_empty(self):
        self.index.add_document("Python programming", doc_id=0)
        
        results = self.index.search_prefix([])
        
        self.assertEqual(len(results), 0)
    
    def test_prefix_search_with_stemming(self):
        self.index.add_document("running runner runs", doc_id=0)
        
        results = self.index.search_prefix(["run"])
        
        self.assertEqual(len(results), 1)
        self.assertIn(0, results)


class TestWildcardSearch(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "test_wildcard_storage"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.index = InvertedIndex(storage_dir=self.test_dir, use_stemming=True, remove_stopwords=True, enable_kgram=True)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_wildcard_search_prefix(self):
        self.index.add_document("Python programming is great", doc_id=0)
        self.index.add_document("I love programming in Python", doc_id=1)
        self.index.add_document("Java programming language", doc_id=2)
        
        results = self.index.search_wildcard(["prog*"])
        
        self.assertEqual(len(results), 3)
    
    def test_wildcard_search_suffix(self):
        self.index.add_document("Python programming is great", doc_id=0)
        self.index.add_document("I love programming in Python", doc_id=1)
        self.index.add_document("Java development", doc_id=2)
        
        results = self.index.search_wildcard(["*thon"])
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(1, results)
    
    def test_wildcard_search_middle(self):
        self.index.add_document("Python programming is great", doc_id=0)
        self.index.add_document("I love programming in Python", doc_id=1)
        self.index.add_document("Java development", doc_id=2)
        
        results = self.index.search_wildcard(["pro*am"])
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(1, results)
    
    def test_wildcard_search_no_matches(self):
        self.index.add_document("Python is great", doc_id=0)
        self.index.add_document("Java is good", doc_id=1)
        
        results = self.index.search_wildcard(["xyz*abc"])
        
        self.assertEqual(len(results), 0)
    
    def test_wildcard_search_no_wildcard(self):
        self.index.add_document("Python programming", doc_id=0)
        
        with self.assertRaises(ValueError):
            self.index.search_wildcard(["python"])
    
    def test_wildcard_search_multiple_wildcards(self):
        self.index.add_document("Python programming", doc_id=0)
        
        with self.assertRaises(ValueError):
            self.index.search_wildcard(["p*o*g"])
    
    def test_wildcard_search_short_pattern(self):
        self.index.add_document("Python programming", doc_id=0)
        self.index.add_document("Java development", doc_id=1)
        
        results = self.index.search_wildcard(["p*"])
        
        self.assertGreater(len(results), 0)
    
    def test_wildcard_search_disabled_kgram(self):
        test_dir_no_kgram = "test_wildcard_no_kgram"
        if os.path.exists(test_dir_no_kgram):
            shutil.rmtree(test_dir_no_kgram)
        index_no_kgram = InvertedIndex(storage_dir=test_dir_no_kgram, enable_kgram=False)
        index_no_kgram.add_document("Python programming", doc_id=0)
        
        with self.assertRaises(RuntimeError):
            index_no_kgram.search_wildcard(["prog*"])
        
        if os.path.exists(test_dir_no_kgram):
            shutil.rmtree(test_dir_no_kgram)
    
    def test_wildcard_search_with_stemming(self):
        self.index.add_document("running runner runs", doc_id=0)
        
        results = self.index.search_wildcard(["run*"])
        
        self.assertEqual(len(results), 1)
        self.assertIn(0, results)
    
    def test_kgram_index_stats(self):
        self.index.add_document("Python programming", doc_id=0)
        self.index.add_document("Java development", doc_id=1)
        
        stats = self.index.get_stats()
        
        self.assertIn('kgram_lsm_layers', stats)
        self.assertIn('term_mapping_layers', stats)
        self.assertGreaterEqual(stats['kgram_lsm_layers'], 1)
        self.assertGreaterEqual(stats['term_mapping_layers'], 1)


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
