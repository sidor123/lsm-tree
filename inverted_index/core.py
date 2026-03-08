from typing import Dict, List, Optional
import re
import logging

from pyroaring import BitMap

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from inverted_index.kgram_utils import KGramGenerator

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text.lower())

    def preprocess(self, text: str) -> List[str]:
        tokens = self.tokenize(text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens


class RoaringBitmapWrapper:
    def __init__(self, initial_values: Optional[List[int]] = None):
        self.bitmap = BitMap(initial_values or [])

    def add(self, doc_id: int):
        self.bitmap.add(doc_id)

    def __contains__(self, doc_id: int) -> bool:
        return doc_id in self.bitmap

    def __len__(self) -> int:
        return len(self.bitmap)

    def __iter__(self):
        return iter(self.bitmap)

    # OR operation
    def union(self, other: RoaringBitmapWrapper) -> RoaringBitmapWrapper:
        result = RoaringBitmapWrapper()
        result.bitmap = self.bitmap | other.bitmap
        return result

    # AND operation
    def intersection(self, other: RoaringBitmapWrapper) -> RoaringBitmapWrapper:
        result = RoaringBitmapWrapper()
        result.bitmap = self.bitmap & other.bitmap
        return result

    # NOT operation
    def difference(self, other: RoaringBitmapWrapper) -> RoaringBitmapWrapper:
        result = RoaringBitmapWrapper()
        result.bitmap = self.bitmap - other.bitmap
        return result

    def to_list(self) -> List[int]:
        return sorted(list(self.bitmap))

    def serialize(self) -> bytes:
        return self.bitmap.serialize()

    @classmethod
    def deserialize(cls, data: bytes) -> RoaringBitmapWrapper:
        wrapper = cls()
        wrapper.bitmap = BitMap.deserialize(data)
        return wrapper


class InvertedIndex:
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True, enable_kgram: bool = True):
        self.index: Dict[str, RoaringBitmapWrapper] = {}
        self.documents: Dict[int, str] = {}
        self.next_doc_id = 0
        self.preprocessor = TextPreprocessor(use_stemming, remove_stopwords)
        
        # k-gram index
        self.enable_kgram = enable_kgram
        if enable_kgram:
            self.kgram_generator = KGramGenerator(k=2)
            self.kgram_index: Dict[str, RoaringBitmapWrapper] = {}
            self.term_to_id: Dict[str, int] = {}
            self.id_to_term: Dict[int, str] = {}
            self.next_term_id = 0
            logger.debug("K-gram index enabled")

    def add_document(self, text: str) -> int:
        doc_id = self.next_doc_id
        self.next_doc_id += 1
        
        logger.debug(f"Adding document {doc_id}: '{text[:50]}...'")
        self.documents[doc_id] = text

        tokens = self.preprocessor.preprocess(text)
        unique_tokens = set(tokens)
        logger.debug(f"  Preprocessed into {len(unique_tokens)} unique tokens")

        for token in unique_tokens:
            if token not in self.index:
                self.index[token] = RoaringBitmapWrapper()
            self.index[token].add(doc_id)
            
            if self.enable_kgram:
                self._add_term_to_kgram_index(token)
        
        logger.debug(f"  Document {doc_id} indexed successfully")
        return doc_id
    
    def _add_term_to_kgram_index(self, term: str):
        if term not in self.term_to_id:
            term_id = self.next_term_id
            self.next_term_id += 1
            self.term_to_id[term] = term_id
            self.id_to_term[term_id] = term
            logger.debug(f"  Assigned term ID {term_id} to term '{term}'")
            
            kgrams = self.kgram_generator.generate_kgrams(term)
            for kgram in kgrams:
                if kgram not in self.kgram_index:
                    self.kgram_index[kgram] = RoaringBitmapWrapper()
                self.kgram_index[kgram].add(term_id)
            
            logger.debug(f"  Added {len(kgrams)} k-grams for term '{term}'")

    def get_document(self, doc_id: int) -> Optional[str]:
        return self.documents.get(doc_id)

    def search_term(self, term: str) -> RoaringBitmapWrapper:
        tokens = self.preprocessor.preprocess(term)
        if not tokens:
            logger.debug(f"Search term '{term}' resulted in no tokens after preprocessing")
            return RoaringBitmapWrapper()

        processed_term = tokens[0]
        result = self.index.get(processed_term, RoaringBitmapWrapper())
        logger.debug(f"Search for term '{term}' (processed: '{processed_term}') found {len(result)} documents")
        return result

    def search_boolean(self, query: str) -> List[int]:
        logger.debug(f"Executing boolean query: '{query}'")
        result_bitmap = self._evaluate_query(query)
        results = result_bitmap.to_list()
        logger.debug(f"  Query returned {len(results)} documents")
        return results

    def _evaluate_query(self, query: str) -> RoaringBitmapWrapper:
        tokens = self._tokenize_query(query)
        return self._parse_expression(tokens)

    def _tokenize_query(self, query: str) -> List[str]:
        pattern = r'\(|\)|AND|OR|NOT|\w+'
        tokens = re.findall(pattern, query.upper())
        return tokens

    def _parse_expression(self, tokens: List[str]) -> RoaringBitmapWrapper:
        if not tokens:
            return RoaringBitmapWrapper()

        result = self._parse_or(tokens)
        return result

    def _parse_or(self, tokens: List[str]) -> RoaringBitmapWrapper:
        left = self._parse_and(tokens)

        while tokens and tokens[0] == 'OR':
            tokens.pop(0)
            right = self._parse_and(tokens)
            left = left.union(right)

        return left

    def _parse_and(self, tokens: List[str]) -> RoaringBitmapWrapper:
        left = self._parse_not(tokens)

        while tokens and tokens[0] == 'AND':
            tokens.pop(0)
            right = self._parse_not(tokens)
            left = left.intersection(right)

        return left

    def _parse_not(self, tokens: List[str]) -> RoaringBitmapWrapper:
        if tokens and tokens[0] == 'NOT':
            tokens.pop(0)
            operand = self._parse_primary(tokens)
            all_docs = RoaringBitmapWrapper(list(self.documents.keys()))
            return all_docs.difference(operand)

        return self._parse_primary(tokens)

    def _parse_primary(self, tokens: List[str]) -> RoaringBitmapWrapper:
        if not tokens:
            return RoaringBitmapWrapper()

        token = tokens.pop(0)

        if token == '(':
            result = self._parse_or(tokens)
            if tokens and tokens[0] == ')':
                tokens.pop(0)
            return result

        return self.search_term(token)

    def search_prefix(self, prefix: str) -> List[int]:
        tokens = self.preprocessor.preprocess(prefix)
        if not tokens:
            logger.debug(f"Prefix '{prefix}' resulted in no tokens after preprocessing")
            return []
        
        processed_prefix = tokens[0]
        logger.debug(f"Searching for prefix '{prefix}' (processed: '{processed_prefix}')")
        
        matching_terms = [term for term in self.index.keys() if term.startswith(processed_prefix)]
        logger.debug(f"Found {len(matching_terms)} terms matching prefix '{processed_prefix}'")
        
        if not matching_terms:
            return []
        
        result_bitmap = RoaringBitmapWrapper()
        for term in matching_terms:
            result_bitmap = result_bitmap.union(self.index[term])
        
        results = result_bitmap.to_list()
        logger.debug(f"Prefix search returned {len(results)} documents")
        return results
    
    def search_wildcard(self, pattern: str) -> List[int]:
        if not self.enable_kgram:
            raise RuntimeError("Wildcard search requires k-gram index to be enabled")
        
        logger.debug(f"Executing wildcard search: '{pattern}'")
        
        if '*' not in pattern:
            raise ValueError("Pattern must contain at least one wildcard (*)")
        
        wildcard_count = pattern.count('*')
        if wildcard_count > 1:
            raise ValueError(f"Pattern must contain exactly one wildcard, found {wildcard_count}")
        
        parts = pattern.split('*')
        prefix_part = parts[0]
        suffix_part = parts[1]
        
        processed_parts = []
        for part in [prefix_part, suffix_part]:
            if part:
                tokens = self.preprocessor.preprocess(part)
                if tokens:
                    processed_parts.append(tokens[0])
                else:
                    processed_parts.append('')
            else:
                processed_parts.append('')
        
        processed_pattern = processed_parts[0] + '*' + processed_parts[1]
        logger.debug(f"Preprocessed pattern: '{pattern}' → '{processed_pattern}'")
        
        try:
            kgrams = self.kgram_generator.wildcard_to_kgrams(processed_pattern)
        except ValueError as e:
            logger.error(f"Failed to extract k-grams: {e}")
            raise
        
        logger.debug(f"Extracted {len(kgrams)} k-grams: {kgrams}")
        
        if not kgrams:
            logger.debug("No k-grams extracted, returning all documents")
            return list(self.documents.keys())
        
        candidate_term_ids = None
        for kgram in kgrams:
            if kgram in self.kgram_index:
                kgram_term_ids = self.kgram_index[kgram]
                if candidate_term_ids is None:
                    candidate_term_ids = kgram_term_ids
                else:
                    candidate_term_ids = candidate_term_ids.intersection(kgram_term_ids)
            else:
                logger.debug(f"K-gram '{kgram}' not found in index")
                return []
        
        if candidate_term_ids is None or len(candidate_term_ids) == 0:
            logger.debug("No candidate terms found")
            return []
        
        candidate_term_id_list = candidate_term_ids.to_list()
        logger.debug(f"Found {len(candidate_term_id_list)} candidate terms")
        
        regex_pattern = self.kgram_generator.pattern_to_regex(processed_pattern)
        import re
        regex = re.compile(regex_pattern)
        
        matching_terms = []
        for term_id in candidate_term_id_list:
            term = self.id_to_term[term_id]
            if regex.match(term):
                matching_terms.append(term)
                logger.debug(f"Term '{term}' matches pattern")
        
        logger.debug(f"After regex filtering: {len(matching_terms)} matching terms")
        
        if not matching_terms:
            return []
        
        result_bitmap = RoaringBitmapWrapper()
        for term in matching_terms:
            if term in self.index:
                result_bitmap = result_bitmap.union(self.index[term])
        
        results = result_bitmap.to_list()
        logger.debug(f"Wildcard search returned {len(results)} documents")
        return results
    
    def get_stats(self) -> Dict[str, int]:
        stats = {
            'num_documents': len(self.documents),
            'num_terms': len(self.index),
            'total_postings': sum(len(bitmap) for bitmap in self.index.values())
        }
        
        if self.enable_kgram:
            stats['num_unique_terms'] = len(self.term_to_id)
            stats['num_kgrams'] = len(self.kgram_index)
            stats['kgram_postings'] = sum(len(bitmap) for bitmap in self.kgram_index.values())
        
        return stats

    def __str__(self) -> str:
        stats = self.get_stats()
        base_str = f"InvertedIndex(docs={stats['num_documents']}, terms={stats['num_terms']}, postings={stats['total_postings']}"
        if self.enable_kgram:
            base_str += f", kgrams={stats['num_kgrams']}"
        base_str += ")"
        return base_str
