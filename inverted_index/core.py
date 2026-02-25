from typing import Dict, List, Optional
import re
import logging

from pyroaring import BitMap

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

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
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.index: Dict[str, RoaringBitmapWrapper] = {}
        self.documents: Dict[int, str] = {}
        self.next_doc_id = 0
        self.preprocessor = TextPreprocessor(use_stemming, remove_stopwords)

    def add_document(self, text: str) -> int:
        doc_id = self.next_doc_id
        self.next_doc_id += 1
        
        logger.debug(f"Adding document {doc_id}: '{text[:50]}...'")
        self.documents[doc_id] = text

        tokens = self.preprocessor.preprocess(text)
        logger.debug(f"  Preprocessed into {len(set(tokens))} unique tokens")

        for token in set(tokens):
            if token not in self.index:
                self.index[token] = RoaringBitmapWrapper()
            self.index[token].add(doc_id)
        
        logger.debug(f"  Document {doc_id} indexed successfully")
        return doc_id

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

    def get_stats(self) -> Dict[str, int]:
        return {
            'num_documents': len(self.documents),
            'num_terms': len(self.index),
            'total_postings': sum(len(bitmap) for bitmap in self.index.values())
        }

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"InvertedIndex(docs={stats['num_documents']}, terms={stats['num_terms']}, postings={stats['total_postings']})"
