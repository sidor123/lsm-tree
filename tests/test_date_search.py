import unittest
import os
import shutil
import logging
from datetime import date
from inverted_index import InvertedIndex

logger = logging.getLogger(__name__)


class TestDateSearchPartA(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "test_date_search_storage"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.index = InvertedIndex(storage_dir=self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_add_document_with_date(self):
        doc_id = self.index.add_document("Python programming", doc_id=0, doc_date=date(2024, 1, 15))
        self.assertEqual(doc_id, 0)
    
    def test_search_date_range_basic(self):
        self.index.add_document("Doc 1", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Doc 2", doc_id=1, doc_date=date(2024, 2, 20))
        self.index.add_document("Doc 3", doc_id=2, doc_date=date(2024, 3, 10))
        
        results = self.index.search_date_range(date(2024, 2, 1), date(2024, 2, 28))
        self.assertEqual(results, [1])
    
    def test_search_date_range_open_start(self):
        self.index.add_document("Doc 1", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Doc 2", doc_id=1, doc_date=date(2024, 2, 20))
        self.index.add_document("Doc 3", doc_id=2, doc_date=date(2024, 3, 10))
        
        results = self.index.search_date_range(end_date=date(2024, 2, 15))
        self.assertEqual(results, [0])
    
    def test_search_date_range_open_end(self):
        self.index.add_document("Doc 1", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Doc 2", doc_id=1, doc_date=date(2024, 2, 20))
        self.index.add_document("Doc 3", doc_id=2, doc_date=date(2024, 3, 10))
        
        results = self.index.search_date_range(start_date=date(2024, 2, 15))
        self.assertEqual(results, [1, 2])
    
    def test_search_date_range_no_matches(self):
        self.index.add_document("Doc 1", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Doc 2", doc_id=1, doc_date=date(2024, 2, 20))
        
        results = self.index.search_date_range(date(2024, 5, 1), date(2024, 5, 31))
        self.assertEqual(results, [])
    
    def test_boolean_with_date_condition(self):
        self.index.add_document("Python programming", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Java development", doc_id=1, doc_date=date(2024, 2, 20))
        self.index.add_document("Python machine learning", doc_id=2, doc_date=date(2024, 3, 10))
        
        results = self.index.search_boolean_with_dates("python AND DATE[2024-01-01:2024-01-31]")
        self.assertEqual(results, [0])
    
    def test_boolean_with_date_or_condition(self):
        self.index.add_document("Python programming", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Java development", doc_id=1, doc_date=date(2024, 2, 20))
        self.index.add_document("Python machine learning", doc_id=2, doc_date=date(2024, 3, 10))
        
        results = self.index.search_boolean_with_dates("(python OR java) AND DATE[2024-01-01:2024-02-28]")
        self.assertEqual(sorted(results), [0, 1])
    
    def test_boolean_with_open_date_range(self):
        self.index.add_document("Python programming", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Java development", doc_id=1, doc_date=date(2024, 2, 20))
        self.index.add_document("Python machine learning", doc_id=2, doc_date=date(2024, 3, 10))
        
        results = self.index.search_boolean_with_dates("python AND DATE[2024-02-01:]")
        self.assertEqual(results, [2])


class TestDateSearchPartB(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "test_lifecycle_storage"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.index = InvertedIndex(storage_dir=self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_add_document_with_lifecycle(self):
        doc_id = self.index.add_document("Project Alpha", doc_id=0, start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))
        self.assertEqual(doc_id, 0)
    
    def test_add_document_ongoing(self):
        doc_id = self.index.add_document("Ongoing Project", doc_id=0, start_date=date(2024, 1, 1))
        self.assertEqual(doc_id, 0)
    
    def test_search_valid_in_range_basic(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 1), end_date=date(2024, 3, 31))
        
        results = self.index.search_valid_in_range(date(2024, 2, 1), date(2024, 2, 28))
        self.assertEqual(results, [0])
    
    def test_search_valid_in_range_ongoing(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 1), end_date=date(2024, 3, 31))
        self.index.add_document("Doc 2", doc_id=1, start_date=date(2024, 2, 1), end_date=date(2024, 5, 31))
        self.index.add_document("Doc 3", doc_id=2, start_date=date(2024, 1, 1))
        
        results = self.index.search_valid_in_range(date(2024, 4, 1), date(2024, 4, 30))
        self.assertEqual(sorted(results), [1, 2])
    
    def test_search_valid_in_range_no_overlap(self):
        self.index.add_document("Ongoing 1", doc_id=0, start_date=date(2024, 1, 1))
        self.index.add_document("Ended", doc_id=1, start_date=date(2024, 1, 1), end_date=date(2024, 2, 28))
        self.index.add_document("Ongoing 2", doc_id=2, start_date=date(2024, 6, 1))
        
        results = self.index.search_valid_in_range(date(2024, 3, 1), date(2024, 3, 31))
        self.assertEqual(results, [0])
    
    def test_search_created_in_range_basic(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 15))
        self.index.add_document("Doc 2", doc_id=1, start_date=date(2024, 2, 20))
        
        results = self.index.search_created_in_range(date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(results, [0])
    
    def test_search_created_in_range_multiple(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 15))
        self.index.add_document("Doc 2", doc_id=1, start_date=date(2024, 1, 20))
        self.index.add_document("Doc 3", doc_id=2, start_date=date(2024, 2, 10))
        
        results = self.index.search_created_in_range(date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(sorted(results), [0, 1])
    
    def test_boolean_with_valid_condition(self):
        self.index.add_document("Python project", doc_id=0, start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))
        self.index.add_document("Java project", doc_id=1, start_date=date(2024, 3, 1), end_date=date(2024, 12, 31))
        self.index.add_document("Python service", doc_id=2, start_date=date(2024, 1, 1))
        
        results = self.index.search_boolean_with_dates("python AND VALID[2024-05-01:2024-05-31]")
        self.assertEqual(sorted(results), [0, 2])
    
    def test_boolean_with_created_condition(self):
        self.index.add_document("Python project", doc_id=0, start_date=date(2024, 1, 15))
        self.index.add_document("Java project", doc_id=1, start_date=date(2024, 2, 20))
        self.index.add_document("Python service", doc_id=2, start_date=date(2024, 1, 25))
        
        results = self.index.search_boolean_with_dates("python AND CREATED[2024-01-01:2024-01-31]")
        self.assertEqual(sorted(results), [0, 2])
    
    def test_complex_boolean_with_multiple_conditions(self):
        self.index.add_document("Python ML project", doc_id=0, doc_date=date(2024, 2, 15), start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))
        self.index.add_document("Java web service", doc_id=1, doc_date=date(2024, 3, 20), start_date=date(2024, 3, 1))
        self.index.add_document("Python data pipeline", doc_id=2, doc_date=date(2024, 1, 10), start_date=date(2024, 1, 5), end_date=date(2024, 3, 31))
        
        results = self.index.search_boolean_with_dates(
            "python AND DATE[2024-01-01:2024-03-31] AND VALID[2024-04-01:2024-06-30] AND CREATED[2024-01-01:2024-03-31]"
        )
        self.assertEqual(results, [0])


class TestDateSearchEdgeCases(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "test_edge_cases_storage"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.index = InvertedIndex(storage_dir=self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_document_without_dates(self):
        self.index.add_document("Python programming", doc_id=0)
        self.index.add_document("Java development", doc_id=1, doc_date=date(2024, 1, 15))
        
        results = self.index.search_date_range(date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(results, [1])
    
    def test_boundary_dates_inclusive(self):
        self.index.add_document("Doc 1", doc_id=0, doc_date=date(2024, 1, 1))
        self.index.add_document("Doc 2", doc_id=1, doc_date=date(2024, 1, 31))
        self.index.add_document("Doc 3", doc_id=2, doc_date=date(2024, 2, 1))
        
        results = self.index.search_date_range(date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(sorted(results), [0, 1])
    
    def test_same_start_and_end_date(self):
        self.index.add_document("Doc 1", doc_id=0, doc_date=date(2024, 1, 15))
        self.index.add_document("Doc 2", doc_id=1, doc_date=date(2024, 1, 16))
        
        results = self.index.search_date_range(date(2024, 1, 15), date(2024, 1, 15))
        self.assertEqual(results, [0])
    
    def test_lifecycle_exact_overlap(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        
        results = self.index.search_valid_in_range(date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(results, [0])
    
    def test_lifecycle_partial_overlap_start(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 15), end_date=date(2024, 2, 15))
        
        results = self.index.search_valid_in_range(date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(results, [0])
    
    def test_lifecycle_partial_overlap_end(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 15), end_date=date(2024, 2, 15))
        
        results = self.index.search_valid_in_range(date(2024, 2, 1), date(2024, 2, 28))
        self.assertEqual(results, [0])
    
    def test_lifecycle_contains_range(self):
        self.index.add_document("Doc 1", doc_id=0, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))
        
        results = self.index.search_valid_in_range(date(2024, 6, 1), date(2024, 6, 30))
        self.assertEqual(results, [0])
    
    def test_multiple_date_types_same_document(self):
        doc_id = self.index.add_document("Python project", doc_id=0, doc_date=date(2024, 2, 15), start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))
        
        date_results = self.index.search_date_range(date(2024, 2, 1), date(2024, 2, 28))
        self.assertIn(doc_id, date_results)
        
        valid_results = self.index.search_valid_in_range(date(2024, 3, 1), date(2024, 3, 31))
        self.assertIn(doc_id, valid_results)
        
        created_results = self.index.search_created_in_range(date(2024, 1, 1), date(2024, 1, 31))
        self.assertIn(doc_id, created_results)


class TestDateSearchPersistence(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.test_dir = "test_persistence_storage"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_date_persistence(self):
        index1 = InvertedIndex(storage_dir=self.test_dir)
        index1.add_document("Python programming", doc_id=0, doc_date=date(2024, 1, 15))
        index1.add_document("Java development", doc_id=1, start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))
        
        del index1
        index2 = InvertedIndex(storage_dir=self.test_dir)
        
        date_results = index2.search_date_range(date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(date_results, [0])
        
        valid_results = index2.search_valid_in_range(date(2024, 3, 1), date(2024, 3, 31))
        self.assertEqual(valid_results, [1])


if __name__ == '__main__':
    log_dir = 'tests/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'date_search_test.log')
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