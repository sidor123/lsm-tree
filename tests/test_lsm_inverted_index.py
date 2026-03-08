import unittest
import shutil
import os
from inverted_index import LSMInvertedIndex
import logging

logger = logging.getLogger(__name__)


class TestLSMInvertedIndex(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "lsm_inverted_storage_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        self.index = LSMInvertedIndex(
            storage_dir=self.test_dir,
            use_stemming=True,
            remove_stopwords=True
        )
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_add_document(self):
        doc_id = self.index.add_document("Python is great")
        
        self.assertEqual(doc_id, 0)
        self.assertEqual(self.index.next_doc_id, 1)
    
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
    
    # test that index persists across restarts
    def test_persistence(self):
        docs = ["Python", "Java", "JavaScript"]
        for doc in docs:
            self.index.add_document(doc)
        
        new_index = LSMInvertedIndex(storage_dir=self.test_dir)
        
        for i, doc in enumerate(docs):
            retrieved = new_index.get_document(i)
            self.assertEqual(retrieved, doc)
        
        results = new_index.search_boolean("python")
        self.assertGreater(len(results), 0)
    
    def test_lsm_tree_integration(self):
        for i in range(10):
            self.index.add_document(f"Document {i} with Python content")
        
        stats = self.index.get_stats()
        self.assertGreaterEqual(stats['term_lsm_layers'], 1)
        self.assertGreaterEqual(stats['doc_lsm_layers'], 1)
        
        results = self.index.search_boolean("python")
        self.assertEqual(len(results), 10)
    
    def test_multiple_terms_per_document(self):
        self.index.add_document("Python machine learning data science")
        
        python_results = self.index.search_term("python")
        machine_results = self.index.search_term("machine")
        data_results = self.index.search_term("data")
        
        self.assertEqual(len(python_results.to_list()), 1)
        self.assertEqual(len(machine_results.to_list()), 1)
        self.assertEqual(len(data_results.to_list()), 1)
    
    def test_get_stats(self):
        for i in range(5):
            self.index.add_document(f"Document {i}")
        
        stats = self.index.get_stats()
        
        self.assertEqual(stats['num_documents'], 5)
        self.assertEqual(stats['next_doc_id'], 5)
        self.assertGreaterEqual(stats['term_lsm_layers'], 1)
        self.assertGreaterEqual(stats['doc_lsm_layers'], 1)
    
    def test_empty_query(self):
        self.index.add_document("Test document")
        
        results = self.index.search_boolean("")
        
        self.assertEqual(len(results), 0)
    
    def test_nonexistent_term(self):
        self.index.add_document("Python programming")
        
        results = self.index.search_term("nonexistent")
        
        self.assertEqual(len(results.to_list()), 0)
    
    def test_stemming_effect(self):
        self.index.add_document("running runner runs")
        
        running_results = self.index.search_term("running")
        runner_results = self.index.search_term("runner")
        run_results = self.index.search_term("run")
        
        self.assertEqual(running_results.to_list(), runner_results.to_list())
        self.assertEqual(running_results.to_list(), run_results.to_list())
    
    def test_stopword_removal(self):
        self.index.add_document("the quick brown fox")
        
        the_results = self.index.search_term("the")
        self.assertEqual(len(the_results.to_list()), 0)
        
        quick_results = self.index.search_term("quick")
        self.assertGreater(len(quick_results.to_list()), 0)


class TestLSMPrefixSearch(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "lsm_prefix_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        self.index = LSMInvertedIndex(
            storage_dir=self.test_dir,
            use_stemming=True,
            remove_stopwords=True,
            enable_kgram=True
        )
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_prefix_search_basic(self):
        self.index.add_document("Python programming is great")
        self.index.add_document("I love programming in Python")
        self.index.add_document("Java programming language")
        
        results = self.index.search_prefix("prog")
        
        self.assertEqual(len(results), 3)
    
    def test_prefix_search_no_matches(self):
        self.index.add_document("Python is great")
        self.index.add_document("Java is good")
        
        results = self.index.search_prefix("xyz")
        
        self.assertEqual(len(results), 0)
    
    def test_prefix_search_lsm_range(self):
        for i in range(10):
            self.index.add_document(f"Python programming document {i}")
        
        results = self.index.search_prefix("prog")
        
        self.assertEqual(len(results), 10)
    
    def test_prefix_search_persistence(self):
        self.index.add_document("Python programming")
        self.index.add_document("Java development")
        
        new_index = LSMInvertedIndex(storage_dir=self.test_dir, enable_kgram=True)
        
        results = new_index.search_prefix("prog")
        
        self.assertGreater(len(results), 0)


class TestLSMWildcardSearch(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "lsm_wildcard_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        self.index = LSMInvertedIndex(
            storage_dir=self.test_dir,
            use_stemming=True,
            remove_stopwords=True,
            enable_kgram=True
        )
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_wildcard_search_prefix(self):
        self.index.add_document("Python programming is great")
        self.index.add_document("I love programming in Python")
        self.index.add_document("Java programming language")
        
        results = self.index.search_wildcard("prog*")
        
        self.assertEqual(len(results), 3)
    
    def test_wildcard_search_suffix(self):
        self.index.add_document("Python programming is great")
        self.index.add_document("I love programming in Python")
        self.index.add_document("Java development")
        
        results = self.index.search_wildcard("*thon")
        
        self.assertEqual(len(results), 2)
    
    def test_wildcard_search_middle(self):
        self.index.add_document("Python programming is great")
        self.index.add_document("I love programming in Python")
        self.index.add_document("Java development")
        
        results = self.index.search_wildcard("pro*am")
        
        self.assertEqual(len(results), 2)
    
    def test_wildcard_search_no_matches(self):
        self.index.add_document("Python is great")
        self.index.add_document("Java is good")
        
        results = self.index.search_wildcard("xyz*abc")
        
        self.assertEqual(len(results), 0)
    
    def test_wildcard_search_disabled_kgram(self):
        index_no_kgram = LSMInvertedIndex(
            storage_dir=self.test_dir + "_no_kgram",
            enable_kgram=False
        )
        index_no_kgram.add_document("Python programming")
        
        with self.assertRaises(RuntimeError):
            index_no_kgram.search_wildcard("prog*")
        
        if os.path.exists(self.test_dir + "_no_kgram"):
            shutil.rmtree(self.test_dir + "_no_kgram")
    
    def test_wildcard_search_persistence(self):
        self.index.add_document("Python programming")
        self.index.add_document("Java development")
        
        new_index = LSMInvertedIndex(storage_dir=self.test_dir, enable_kgram=True)
        
        results = new_index.search_wildcard("prog*")
        
        self.assertGreater(len(results), 0)
    
    def test_wildcard_search_many_documents(self):
        for i in range(20):
            self.index.add_document(f"Python programming document {i}")
        
        results = self.index.search_wildcard("prog*")
        
        self.assertEqual(len(results), 20)
    
    def test_kgram_index_stats(self):
        self.index.add_document("Python programming")
        self.index.add_document("Java development")
        
        stats = self.index.get_stats()
        
        self.assertIn('kgram_lsm_layers', stats)
        self.assertIn('term_mapping_layers', stats)
        self.assertGreaterEqual(stats['kgram_lsm_layers'], 1)
        self.assertGreaterEqual(stats['term_mapping_layers'], 1)


if __name__ == '__main__':
    log_dir = 'tests/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'lsm_inverted_index_test.log')
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
