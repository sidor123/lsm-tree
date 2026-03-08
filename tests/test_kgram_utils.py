import unittest
import logging
from inverted_index.kgram_utils import KGramGenerator

logger = logging.getLogger(__name__)


class TestKGramGenerator(unittest.TestCase):
    def setUp(self):
        logger.info("=" * 80)
        logger.info(f"    Starting test: {self._testMethodName}")
        logger.info("=" * 80)
        self.generator = KGramGenerator(k=2)
    
    def tearDown(self):
        logger.info("=" * 80)
        logger.info(f"    Completed test: {self._testMethodName}")
        logger.info("=" * 80)
        logger.info("")
    
    def test_initialization(self):
        gen = KGramGenerator(k=2)
        self.assertEqual(gen.k, 2)
        self.assertEqual(gen.boundary_char, '$')
    
    def test_initialization_invalid_k(self):
        with self.assertRaises(ValueError):
            KGramGenerator(k=0)
        
        with self.assertRaises(ValueError):
            KGramGenerator(k=-1)
    
    def test_generate_kgrams_simple(self):
        kgrams = self.generator.generate_kgrams("program")
        expected = ['$p', 'pr', 'ro', 'og', 'gr', 'ra', 'am', 'm$']
        self.assertEqual(kgrams, expected)
    
    def test_generate_kgrams_short_term(self):
        kgrams = self.generator.generate_kgrams("ab")
        expected = ['$a', 'ab', 'b$']
        self.assertEqual(kgrams, expected)
    
    def test_generate_kgrams_single_char(self):
        kgrams = self.generator.generate_kgrams("a")
        expected = ['$a', 'a$']
        self.assertEqual(kgrams, expected)
    
    def test_generate_kgrams_empty(self):
        kgrams = self.generator.generate_kgrams("")
        self.assertEqual(kgrams, [])
    
    def test_wildcard_to_kgrams_prefix(self):
        kgrams = self.generator.wildcard_to_kgrams("prog*")
        expected = ['$p', 'pr', 'ro', 'og']
        self.assertEqual(kgrams, expected)
    
    def test_wildcard_to_kgrams_suffix(self):
        kgrams = self.generator.wildcard_to_kgrams("*gram")
        expected = ['gr', 'ra', 'am', 'm$']
        self.assertEqual(kgrams, expected)
    
    def test_wildcard_to_kgrams_middle(self):
        kgrams = self.generator.wildcard_to_kgrams("pro*ing")
        expected = ['$p', 'pr', 'ro', 'in', 'ng', 'g$']
        self.assertEqual(kgrams, expected)
    
    def test_wildcard_to_kgrams_short_prefix(self):
        kgrams = self.generator.wildcard_to_kgrams("p*")
        expected = ['$p']
        self.assertEqual(kgrams, expected)
    
    def test_wildcard_to_kgrams_short_suffix(self):
        kgrams = self.generator.wildcard_to_kgrams("*p")
        expected = ['p$']
        self.assertEqual(kgrams, expected)
    
    def test_wildcard_to_kgrams_wildcard_only(self):
        kgrams = self.generator.wildcard_to_kgrams("*")
        self.assertEqual(kgrams, [])
    
    def test_wildcard_to_kgrams_no_wildcard(self):
        with self.assertRaises(ValueError):
            self.generator.wildcard_to_kgrams("program")
    
    def test_wildcard_to_kgrams_multiple_wildcards(self):
        with self.assertRaises(ValueError):
            self.generator.wildcard_to_kgrams("p*o*g")
    
    def test_pattern_to_regex_prefix(self):
        regex = self.generator.pattern_to_regex("prog*")
        self.assertEqual(regex, "^prog.*$")
    
    def test_pattern_to_regex_suffix(self):
        regex = self.generator.pattern_to_regex("*gram")
        self.assertEqual(regex, "^.*gram$")
    
    def test_pattern_to_regex_middle(self):
        regex = self.generator.pattern_to_regex("pro*ing")
        self.assertEqual(regex, "^pro.*ing$")
    
    def test_pattern_to_regex_special_chars(self):
        regex = self.generator.pattern_to_regex("test.*")
        self.assertIn(r'\.\*', regex)
    
    def test_kgram_generation_with_k3(self):
        gen = KGramGenerator(k=3)
        kgrams = gen.generate_kgrams("program")
        expected = ['$pr', 'pro', 'rog', 'ogr', 'gra', 'ram', 'am$']
        self.assertEqual(kgrams, expected)
    
    def test_wildcard_to_kgrams_with_k3(self):
        gen = KGramGenerator(k=3)
        kgrams = gen.wildcard_to_kgrams("pro*ing")
        expected = ['$pr', 'pro', 'ing', 'ng$']
        self.assertEqual(kgrams, expected)


if __name__ == '__main__':
    import os
    log_dir = 'tests/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'kgram_utils_test.log')
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