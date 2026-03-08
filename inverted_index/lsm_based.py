from typing import Dict, List, Optional
import pickle
import logging
import re
from lsm_tree import LSMTree
from inverted_index.core import RoaringBitmapWrapper, TextPreprocessor
from inverted_index.kgram_utils import KGramGenerator

logger = logging.getLogger(__name__)


class LSMInvertedIndex:
    def __init__(self,
                 storage_dir: str = "lsm_inverted_storage",
                 use_stemming: bool = True,
                 remove_stopwords: bool = True,
                 enable_kgram: bool = True):
        logger.info(f"Initializing LSM Inverted Index with storage_dir='{storage_dir}'")
        self.term_index = LSMTree(storage_dir=f"{storage_dir}/terms")
        self.doc_store = LSMTree(storage_dir=f"{storage_dir}/docs")
        self.preprocessor = TextPreprocessor(use_stemming, remove_stopwords)
        self.next_doc_id = self._load_next_doc_id()
        logger.info(f"  Loaded next_doc_id={self.next_doc_id}")
        
        # k-gram index
        self.enable_kgram = enable_kgram
        if enable_kgram:
            self.kgram_generator = KGramGenerator(k=2)
            self.kgram_index = LSMTree(storage_dir=f"{storage_dir}/kgrams")
            self.term_mapping = LSMTree(storage_dir=f"{storage_dir}/term_mapping")
            self.next_term_id = self._load_next_term_id()
            logger.info(f"  K-gram index enabled, loaded next_term_id={self.next_term_id}")

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
    
    def _load_next_term_id(self) -> int:
        metadata = self.term_mapping.get("__term_metadata__")
        if metadata:
            try:
                meta_dict = pickle.loads(metadata.encode('latin1'))
                return meta_dict.get('next_term_id', 0)
            except:
                return 0
        return 0
    
    def _save_next_term_id(self):
        metadata = {'next_term_id': self.next_term_id}
        self.term_mapping.add("__term_metadata__", pickle.dumps(metadata).decode('latin1'))

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
            
            if self.enable_kgram:
                self._add_term_to_kgram_index(token)
        
        logger.debug(f"Document {doc_id} indexed successfully in LSM tree")
        return doc_id
    
    def _add_term_to_kgram_index(self, term: str):
        term_key = f"term_{term}"
        existing_id = self.term_mapping.get(term_key)
        
        if existing_id is None:
            term_id = self.next_term_id
            self.next_term_id += 1
            self._save_next_term_id()
            
            self.term_mapping.add(term_key, str(term_id))
            self.term_mapping.add(f"id_{term_id}", term)
            
            logger.debug(f"  Assigned term ID {term_id} to term '{term}'")
            
            kgrams = self.kgram_generator.generate_kgrams(term)
            for kgram in kgrams:
                kgram_key = f"kg_{kgram}"
                existing_data = self.kgram_index.get(kgram_key)
                
                if existing_data:
                    try:
                        bitmap = RoaringBitmapWrapper.deserialize(existing_data.encode('latin1'))
                    except:
                        bitmap = RoaringBitmapWrapper()
                else:
                    bitmap = RoaringBitmapWrapper()
                
                bitmap.add(term_id)
                serialized = bitmap.serialize().decode('latin1')
                self.kgram_index.add(kgram_key, serialized)
            
            logger.debug(f"  Added {len(kgrams)} k-grams for term '{term}'")

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
    
    def search_prefix(self, prefix: str) -> List[int]:
        tokens = self.preprocessor.preprocess(prefix)
        if not tokens:
            logger.debug(f"Prefix '{prefix}' resulted in no tokens after preprocessing")
            return []
        
        processed_prefix = tokens[0]
        logger.debug(f"Searching for prefix '{prefix}' (processed: '{processed_prefix}')")
        
        end_key = processed_prefix + '\uffff'
        matching_terms = self.term_index.range_get(processed_prefix, end_key)
        
        logger.debug(f"Found {len(matching_terms)} terms matching prefix '{processed_prefix}'")
        
        if not matching_terms:
            return []
        
        result_bitmap = RoaringBitmapWrapper()
        for term, serialized_bitmap in matching_terms.items():
            try:
                bitmap = RoaringBitmapWrapper.deserialize(serialized_bitmap.encode('latin1'))
                result_bitmap = result_bitmap.union(bitmap)
            except:
                logger.warning(f"Failed to deserialize bitmap for term '{term}'")
                continue
        
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
            return self._get_all_doc_ids()
        
        candidate_term_ids = None
        for kgram in kgrams:
            kgram_key = f"kg_{kgram}"
            existing_data = self.kgram_index.get(kgram_key)
            
            if existing_data:
                try:
                    kgram_bitmap = RoaringBitmapWrapper.deserialize(existing_data.encode('latin1'))
                    if candidate_term_ids is None:
                        candidate_term_ids = kgram_bitmap
                    else:
                        candidate_term_ids = candidate_term_ids.intersection(kgram_bitmap)
                except:
                    logger.warning(f"Failed to deserialize k-gram '{kgram}'")
                    return []
            else:
                logger.debug(f"K-gram '{kgram}' not found in index")
                return []
        
        if candidate_term_ids is None or len(candidate_term_ids) == 0:
            logger.debug("No candidate terms found")
            return []
        
        candidate_term_id_list = candidate_term_ids.to_list()
        logger.debug(f"Found {len(candidate_term_id_list)} candidate terms")
        
        regex_pattern = self.kgram_generator.pattern_to_regex(processed_pattern)
        regex = re.compile(regex_pattern)
        
        matching_terms = []
        for term_id in candidate_term_id_list:
            term = self.term_mapping.get(f"id_{term_id}")
            if term and regex.match(term):
                matching_terms.append(term)
                logger.debug(f"Term '{term}' matches pattern")
        
        logger.debug(f"After regex filtering: {len(matching_terms)} matching terms")
        
        if not matching_terms:
            return []
        
        result_bitmap = RoaringBitmapWrapper()
        for term in matching_terms:
            data = self.term_index.get(term)
            if data:
                try:
                    bitmap = RoaringBitmapWrapper.deserialize(data.encode('latin1'))
                    result_bitmap = result_bitmap.union(bitmap)
                except:
                    logger.warning(f"Failed to deserialize bitmap for term '{term}'")
                    continue
        
        results = result_bitmap.to_list()
        logger.debug(f"Wildcard search returned {len(results)} documents")
        return results

    def get_stats(self) -> Dict[str, int]:
        stats = {
            'num_documents': len(self._get_all_doc_ids()),
            'next_doc_id': self.next_doc_id,
            'term_lsm_layers': len(self.term_index.layers),
            'doc_lsm_layers': len(self.doc_store.layers)
        }
        
        if self.enable_kgram:
            stats['next_term_id'] = self.next_term_id
            stats['kgram_lsm_layers'] = len(self.kgram_index.layers)
            stats['term_mapping_layers'] = len(self.term_mapping.layers)
        
        return stats

    def print_stats(self):
        stats = self.get_stats()
        print(f"LSM Inverted Index Statistics:")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Next Doc ID: {stats['next_doc_id']}")
        print(f"  Term Index LSM Layers: {stats['term_lsm_layers']}")
        print(f"  Doc Store LSM Layers: {stats['doc_lsm_layers']}")
        if self.enable_kgram:
            print(f"  K-gram Index LSM Layers: {stats['kgram_lsm_layers']}")
            print(f"  Term Mapping LSM Layers: {stats['term_mapping_layers']}")

    def __str__(self) -> str:
        stats = self.get_stats()
        base_str = f"LSMInvertedIndex(docs={stats['num_documents']}, term_layers={stats['term_lsm_layers']}, doc_layers={stats['doc_lsm_layers']}"
        if self.enable_kgram:
            base_str += f", kgram_layers={stats['kgram_lsm_layers']}"
        base_str += ")"
        return base_str
