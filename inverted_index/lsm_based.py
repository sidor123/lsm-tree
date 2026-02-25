from typing import Dict, List, Optional
import pickle
import logging
from lsm_tree import LSMTree
from inverted_index.core import RoaringBitmapWrapper, TextPreprocessor

logger = logging.getLogger(__name__)


class LSMInvertedIndex:
    def __init__(self,
                 storage_dir: str = "lsm_inverted_storage",
                 use_stemming: bool = True,
                 remove_stopwords: bool = True):
        logger.info(f"Initializing LSM Inverted Index with storage_dir='{storage_dir}'")
        self.term_index = LSMTree(storage_dir=f"{storage_dir}/terms")
        self.doc_store = LSMTree(storage_dir=f"{storage_dir}/docs")
        self.preprocessor = TextPreprocessor(use_stemming, remove_stopwords)
        self.next_doc_id = self._load_next_doc_id()
        logger.info(f"  Loaded next_doc_id={self.next_doc_id}")

    def _load_next_doc_id(self) -> int:
        metadata = self.doc_store.get("__metadata__")
        if metadata:
            try:
                meta_dict = pickle.loads(metadata.encode('latin1'))
                return meta_dict.get('next_doc_id', 0)
            except:
                return 0
        return 0

    def _save_next_doc_id(self):
        metadata = {'next_doc_id': self.next_doc_id}
        self.doc_store.add("__metadata__", pickle.dumps(metadata).decode('latin1'))

    def add_document(self, text: str, doc_id: Optional[int] = None) -> int:
        if doc_id is None:
            doc_id = self.next_doc_id
            self.next_doc_id += 1
            self._save_next_doc_id()
        
        logger.debug(f"Adding document {doc_id}: '{text[:50]}...'")
        doc_key = f"doc_{doc_id}"
        self.doc_store.add(doc_key, text)

        tokens = self.preprocessor.preprocess(text)
        unique_tokens = set(tokens)
        logger.debug(f"  Preprocessed into {len(unique_tokens)} unique tokens")

        for token in unique_tokens:
            existing_data = self.term_index.get(token)

            if existing_data:
                try:
                    bitmap = RoaringBitmapWrapper.deserialize(existing_data.encode('latin1'))
                except:
                    bitmap = RoaringBitmapWrapper()
            else:
                bitmap = RoaringBitmapWrapper()

            bitmap.add(doc_id)
            serialized = bitmap.serialize().decode('latin1')
            self.term_index.add(token, serialized)
        
        logger.debug(f"Document {doc_id} indexed successfully in LSM tree")
        return doc_id

    def get_document(self, doc_id: int) -> Optional[str]:
        doc_key = f"doc_{doc_id}"
        return self.doc_store.get(doc_key)

    def search_term(self, term: str) -> RoaringBitmapWrapper:
        tokens = self.preprocessor.preprocess(term)
        if not tokens:
            logger.debug(f"Search term '{term}' resulted in no tokens after preprocessing")
            return RoaringBitmapWrapper()

        processed_term = tokens[0]
        logger.debug(f"Searching for term '{term}' (processed: '{processed_term}') in LSM tree")

        data = self.term_index.get(processed_term)
        if data:
            try:
                result = RoaringBitmapWrapper.deserialize(data.encode('latin1'))
                logger.debug(f"Found {len(result)} documents for term '{processed_term}'")
                return result
            except:
                logger.debug(f"Failed to deserialize bitmap for term '{processed_term}'")
                return RoaringBitmapWrapper()

        logger.debug(f"Term '{processed_term}' not found in index")
        return RoaringBitmapWrapper()

    def search_boolean(self, query: str) -> List[int]:
        logger.debug(f"Executing boolean query: '{query}'")
        result_bitmap = self._evaluate_query(query)
        results = result_bitmap.to_list()
        logger.debug(f"Query returned {len(results)} documents")
        return results

    def _evaluate_query(self, query: str) -> RoaringBitmapWrapper:
        import re

        pattern = r'\(|\)|AND|OR|NOT|\w+'
        tokens = re.findall(pattern, query.upper())

        return self._parse_expression(tokens)

    def _parse_expression(self, tokens: List[str]) -> RoaringBitmapWrapper:
        if not tokens:
            return RoaringBitmapWrapper()

        return self._parse_or(tokens)

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

            all_docs = self._get_all_doc_ids()
            all_bitmap = RoaringBitmapWrapper(all_docs)

            return all_bitmap.difference(operand)

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

    def _get_all_doc_ids(self) -> List[int]:
        doc_ids = []
        for i in range(self.next_doc_id):
            if self.get_document(i) is not None:
                doc_ids.append(i)
        return doc_ids

    def get_stats(self) -> Dict[str, int]:
        return {
            'num_documents': len(self._get_all_doc_ids()),
            'next_doc_id': self.next_doc_id,
            'term_lsm_layers': len(self.term_index.layers),
            'doc_lsm_layers': len(self.doc_store.layers)
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f"LSM Inverted Index Statistics:")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Next Doc ID: {stats['next_doc_id']}")
        print(f"  Term Index LSM Layers: {stats['term_lsm_layers']}")
        print(f"  Doc Store LSM Layers: {stats['doc_lsm_layers']}")

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"LSMInvertedIndex(docs={stats['num_documents']}, term_layers={stats['term_lsm_layers']}, doc_layers={stats['doc_lsm_layers']})"
