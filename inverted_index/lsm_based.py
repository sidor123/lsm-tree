from typing import Dict, List, Optional
import pickle
import logging
import re
from datetime import date
from lsm_tree import LSMTree
from inverted_index.core import RoaringBitmapWrapper, TextPreprocessor
from inverted_index.kgram_utils import KGramGenerator

logger = logging.getLogger(__name__)


class InvertedIndex:
    def __init__(self,
                 storage_dir: str = "lsm_inverted_storage",
                 use_stemming: bool = True,
                 remove_stopwords: bool = True,
                 enable_kgram: bool = True):
        logger.info(f"Initializing LSM Inverted Index with storage_dir='{storage_dir}'")
        self.term_index = LSMTree(storage_dir=f"{storage_dir}/terms")
        self.doc_store = LSMTree(storage_dir=f"{storage_dir}/docs")
        self.preprocessor = TextPreprocessor(use_stemming, remove_stopwords)
        
        # k-gram index
        self.enable_kgram = enable_kgram
        if enable_kgram:
            self.kgram_generator = KGramGenerator(k=2)
            self.kgram_index = LSMTree(storage_dir=f"{storage_dir}/kgrams")
            self.term_mapping = LSMTree(storage_dir=f"{storage_dir}/term_mapping")
            self.next_term_id = self._load_next_term_id()
            logger.info(f"  K-gram index enabled, loaded next_term_id={self.next_term_id}")

    def _load_next_term_id(self) -> int:
        metadata = self.term_mapping.get("__term_metadata__")
        if metadata:
            try:
                return pickle.loads(metadata.encode('latin1')).get('next_term_id', 0)
            except:
                pass
        return 0

    def _save_next_term_id(self):
        self.term_mapping.add("__term_metadata__", pickle.dumps({'next_term_id': self.next_term_id}).decode('latin1'))

    def _get_bitmap(self, index: LSMTree, key: str) -> RoaringBitmapWrapper:
        data = index.get(key)
        if data:
            try:
                return RoaringBitmapWrapper.deserialize(data.encode('latin1'))
            except:
                pass
        return RoaringBitmapWrapper()

    def _save_bitmap(self, index: LSMTree, key: str, bitmap: RoaringBitmapWrapper):
        index.add(key, bitmap.serialize().decode('latin1'))

    def _store_date(self, prefix: str, doc_id: int, d: Optional[date]):
        if d is not None:
            self.doc_store.add(f"{prefix}_{doc_id}", d.isoformat())
            logger.debug(f"  Stored {prefix}: {d.isoformat()}")

    def _get_date(self, prefix: str, doc_id: int) -> Optional[date]:
        date_str = self.doc_store.get(f"{prefix}_{doc_id}")
        return date.fromisoformat(date_str) if date_str else None

    def add_document(self, text: str, doc_id: int,
                     doc_date: Optional[date] = None, start_date: Optional[date] = None,
                     end_date: Optional[date] = None) -> int:
        logger.debug(f"Adding document {doc_id}: '{text[:50]}...'")
        self.doc_store.add(f"doc_{doc_id}", text)

        self._store_date("date", doc_id, doc_date)
        self._store_date("start", doc_id, start_date)
        self._store_date("end", doc_id, end_date)

        unique_tokens = set(self.preprocessor.preprocess(text.split()))
        logger.debug(f"  Preprocessed into {len(unique_tokens)} unique tokens")

        for token in unique_tokens:
            bitmap = self._get_bitmap(self.term_index, token)
            bitmap.add(doc_id)
            self._save_bitmap(self.term_index, token, bitmap)

            if self.enable_kgram:
                self._add_term_to_kgram_index(token)

        logger.debug(f"Document {doc_id} indexed successfully in LSM tree")
        return doc_id

    def _add_term_to_kgram_index(self, term: str):
        term_key = f"term_{term}"
        if self.term_mapping.get(term_key) is not None:
            return

        term_id = self.next_term_id
        self.next_term_id += 1
        self._save_next_term_id()

        self.term_mapping.add(term_key, str(term_id))
        self.term_mapping.add(f"id_{term_id}", term)
        logger.debug(f"  Assigned term ID {term_id} to term '{term}'")

        kgrams = self.kgram_generator.generate_kgrams(term)
        for kgram in kgrams:
            bitmap = self._get_bitmap(self.kgram_index, f"kg_{kgram}")
            bitmap.add(term_id)
            self._save_bitmap(self.kgram_index, f"kg_{kgram}", bitmap)

        logger.debug(f"  Added {len(kgrams)} k-grams for term '{term}'")

    def get_document(self, doc_id: int) -> Optional[str]:
        return self.doc_store.get(f"doc_{doc_id}")

    def search_term(self, tokens: List[str]) -> RoaringBitmapWrapper:
        processed_tokens = self.preprocessor.preprocess(tokens)
        if not processed_tokens:
            logger.debug(f"Search tokens resulted in no tokens after preprocessing")
            return RoaringBitmapWrapper()

        processed_term = processed_tokens[0]
        logger.debug(f"Searching for term '{processed_term}' in LSM tree")

        bitmap = self._get_bitmap(self.term_index, processed_term)
        logger.debug(f"Found {len(bitmap)} documents for term '{processed_term}'")
        return bitmap

    def search_boolean(self, query: str) -> List[int]:
        logger.debug(f"Executing boolean query: '{query}'")
        results = self._evaluate_query(query).to_list()
        logger.debug(f"Query returned {len(results)} documents")
        return results

    def _evaluate_query(self, query: str) -> RoaringBitmapWrapper:
        tokens = re.findall(r'\(|\)|AND|OR|NOT|\w+', query.upper())
        return self._parse_or(tokens) if tokens else RoaringBitmapWrapper()

    def _parse_or(self, tokens: List[str]) -> RoaringBitmapWrapper:
        left = self._parse_and(tokens)
        while tokens and tokens[0] == 'OR':
            tokens.pop(0)
            left = left.union(self._parse_and(tokens))
        return left

    def _parse_and(self, tokens: List[str]) -> RoaringBitmapWrapper:
        left = self._parse_not(tokens)
        while tokens and tokens[0] == 'AND':
            tokens.pop(0)
            left = left.intersection(self._parse_not(tokens))
        return left

    def _parse_not(self, tokens: List[str]) -> RoaringBitmapWrapper:
        if tokens and tokens[0] == 'NOT':
            tokens.pop(0)
            return RoaringBitmapWrapper(self._get_all_doc_ids()).difference(self._parse_primary(tokens))
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

        return self.search_term([token])

    def _get_all_doc_ids(self) -> List[int]:
        doc_ids = []
        for key in self.doc_store.range_get("doc_", "doc_\uffff").keys():
            if key.startswith("doc_"):
                try:
                    doc_ids.append(int(key[4:]))  # extract ID from "doc_123"
                except ValueError:
                    pass
        return sorted(doc_ids)

    def search_prefix(self, tokens: List[str]) -> List[int]:
        processed_tokens = self.preprocessor.preprocess(tokens)
        if not processed_tokens:
            logger.debug(f"Prefix tokens resulted in no tokens after preprocessing")
            return []

        processed_prefix = processed_tokens[0]
        logger.debug(f"Searching for prefix '{processed_prefix}'")

        matching_terms = self.term_index.range_get(processed_prefix, processed_prefix + '\uffff')
        logger.debug(f"Found {len(matching_terms)} terms matching prefix '{processed_prefix}'")

        if not matching_terms:
            return []

        result_bitmap = RoaringBitmapWrapper()
        for term, serialized_bitmap in matching_terms.items():
            try:
                result_bitmap = result_bitmap.union(RoaringBitmapWrapper.deserialize(serialized_bitmap.encode('latin1')))
            except:
                logger.warning(f"Failed to deserialize bitmap for term '{term}'")

        results = result_bitmap.to_list()
        logger.debug(f"Prefix search returned {len(results)} documents")
        return results

    def search_wildcard(self, pattern_tokens: List[str]) -> List[int]:
        if not self.enable_kgram:
            raise RuntimeError("Wildcard search requires k-gram index to be enabled")

        if not pattern_tokens:
            raise ValueError("Pattern tokens list cannot be empty")

        pattern = pattern_tokens[0]
        logger.debug(f"Executing wildcard search: '{pattern}'")

        if '*' not in pattern:
            raise ValueError("Pattern must contain at least one wildcard (*)")

        wildcard_count = pattern.count('*')
        if wildcard_count > 1:
            raise ValueError(f"Pattern must contain exactly one wildcard, found {wildcard_count}")

        prefix_part, suffix_part = pattern.split('*')

        processed_parts = []
        for part in [prefix_part, suffix_part]:
            if part:
                part_tokens = self.preprocessor.preprocess([part])
                processed_parts.append(part_tokens[0] if part_tokens else '')
            else:
                processed_parts.append('')

        processed_pattern = f"{processed_parts[0]}*{processed_parts[1]}"
        logger.debug(f"Preprocessed pattern: '{pattern}' → '{processed_pattern}'")

        kgrams = self.kgram_generator.wildcard_to_kgrams(processed_pattern)
        logger.debug(f"Extracted {len(kgrams)} k-grams: {kgrams}")

        if not kgrams:
            logger.debug("No k-grams extracted, returning all documents")
            return self._get_all_doc_ids()

        candidate_term_ids = None
        for kgram in kgrams:
            kgram_bitmap = self._get_bitmap(self.kgram_index, f"kg_{kgram}")
            if len(kgram_bitmap) == 0:
                logger.debug(f"K-gram '{kgram}' not found in index")
                return []
            candidate_term_ids = kgram_bitmap if candidate_term_ids is None else candidate_term_ids.intersection(kgram_bitmap)

        if candidate_term_ids is None or len(candidate_term_ids) == 0:
            logger.debug("No candidate terms found")
            return []

        candidate_term_id_list = candidate_term_ids.to_list()
        logger.debug(f"Found {len(candidate_term_id_list)} candidate terms")

        regex = re.compile(self.kgram_generator.pattern_to_regex(processed_pattern))

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
            result_bitmap = result_bitmap.union(self._get_bitmap(self.term_index, term))

        results = result_bitmap.to_list()
        logger.debug(f"Wildcard search returned {len(results)} documents")
        return results

    def get_stats(self) -> Dict[str, int]:
        stats = {
            'num_documents': len(self._get_all_doc_ids()),
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
        print(f"Inverted Index Statistics:")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Term Index LSM Layers: {stats['term_lsm_layers']}")
        print(f"  Doc Store LSM Layers: {stats['doc_lsm_layers']}")
        if self.enable_kgram:
            print(f"  K-gram Index LSM Layers: {stats['kgram_lsm_layers']}")
            print(f"  Term Mapping LSM Layers: {stats['term_mapping_layers']}")

    def __str__(self) -> str:
        stats = self.get_stats()
        base_str = f"InvertedIndex(docs={stats['num_documents']}, term_layers={stats['term_lsm_layers']}, doc_layers={stats['doc_lsm_layers']}"
        if self.enable_kgram:
            base_str += f", kgram_layers={stats['kgram_lsm_layers']}"
        return base_str + ")"

    def search_date_range(self, start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> List[int]:
        logger.debug(f"Searching date range: {start_date} to {end_date}")
        result_ids = [
            doc_id for doc_id in self._get_all_doc_ids()
            if (doc_date := self._get_date("date", doc_id)) is not None
            and (start_date is None or doc_date >= start_date)
            and (end_date is None or doc_date <= end_date)
        ]
        logger.debug(f"Found {len(result_ids)} documents in date range")
        return result_ids

    def search_valid_in_range(self, start_date: date, end_date: date) -> List[int]:
        logger.debug(f"Searching documents valid in range: {start_date} to {end_date}")
        result_ids = []
        for doc_id in self._get_all_doc_ids():
            doc_start = self._get_date("start", doc_id)
            if doc_start is None or doc_start > end_date:
                continue
            doc_end = self._get_date("end", doc_id)
            if doc_end is not None and doc_end < start_date:
                continue
            result_ids.append(doc_id)
        logger.debug(f"Found {len(result_ids)} documents valid in range")
        return result_ids

    def search_created_in_range(self, start_date: date, end_date: date) -> List[int]:
        logger.debug(f"Searching documents created in range: {start_date} to {end_date}")
        result_ids = [
            doc_id for doc_id in self._get_all_doc_ids()
            if (doc_start := self._get_date("start", doc_id)) is not None
            and start_date <= doc_start <= end_date
        ]
        logger.debug(f"Found {len(result_ids)} documents created in range")
        return result_ids

    def search_boolean_with_dates(self, query: str) -> List[int]:
        """
        - DATE[start:end] - documents with doc_date in range
        - VALID[start:end] - documents valid during range
        - CREATED[start:end] - documents created during range
        Date format: YYYY-MM-DD
        Open ranges: DATE[2024-01-01:] or DATE[:2024-12-31]
        """
        logger.debug(f"Executing boolean query with dates: '{query}'")

        date_conditions = {}
        counter = [0]

        def replace_date_condition(match):
            placeholder = f"__DATE_PLACEHOLDER_{counter[0]}__"
            date_conditions[placeholder] = (match.group(1), match.group(2))
            counter[0] += 1
            return placeholder

        modified_query = re.sub(r'(DATE|VALID|CREATED)\[([^\]]+)\]', replace_date_condition, query)

        results = self._parse_or_with_dates(
            re.findall(r'\(|\)|AND|OR|NOT|__DATE_PLACEHOLDER_\d+__|[\w]+', modified_query.upper()),
            date_conditions
        ).to_list()
        logger.debug(f"Query returned {len(results)} documents")
        return results

    def _parse_or_with_dates(self, tokens: List[str], date_conditions: Dict[str, tuple]) -> RoaringBitmapWrapper:
        left = self._parse_and_with_dates(tokens, date_conditions)
        while tokens and tokens[0] == 'OR':
            tokens.pop(0)
            left = left.union(self._parse_and_with_dates(tokens, date_conditions))
        return left

    def _parse_and_with_dates(self, tokens: List[str], date_conditions: Dict[str, tuple]) -> RoaringBitmapWrapper:
        left = self._parse_not_with_dates(tokens, date_conditions)
        while tokens and tokens[0] == 'AND':
            tokens.pop(0)
            left = left.intersection(self._parse_not_with_dates(tokens, date_conditions))
        return left

    def _parse_not_with_dates(self, tokens: List[str], date_conditions: Dict[str, tuple]) -> RoaringBitmapWrapper:
        if tokens and tokens[0] == 'NOT':
            tokens.pop(0)
            return RoaringBitmapWrapper(self._get_all_doc_ids()).difference(
                self._parse_primary_with_dates(tokens, date_conditions)
            )
        return self._parse_primary_with_dates(tokens, date_conditions)

    def _parse_primary_with_dates(self, tokens: List[str], date_conditions: Dict[str, tuple]) -> RoaringBitmapWrapper:
        if not tokens:
            return RoaringBitmapWrapper()

        token = tokens.pop(0)

        if token == '(':
            result = self._parse_or_with_dates(tokens, date_conditions)
            if tokens and tokens[0] == ')':
                tokens.pop(0)
            return result

        if token.startswith('__DATE_PLACEHOLDER_'):
            return self._evaluate_date_condition(*date_conditions[token])

        return self.search_term([token])

    def _evaluate_date_condition(self, condition_type: str, date_range: str) -> RoaringBitmapWrapper:
        parts = date_range.split(':')
        if len(parts) != 2:
            logger.warning(f"Invalid date range format: {date_range}")
            return RoaringBitmapWrapper()

        start_str, end_str = parts
        start_date = date.fromisoformat(start_str) if start_str else None
        end_date = date.fromisoformat(end_str) if end_str else None

        if condition_type == 'DATE':
            return RoaringBitmapWrapper(self.search_date_range(start_date, end_date))

        if start_date is None or end_date is None:
            logger.warning(f"{condition_type} condition requires both start and end dates")
            return RoaringBitmapWrapper()

        if condition_type == 'VALID':
            return RoaringBitmapWrapper(self.search_valid_in_range(start_date, end_date))
        if condition_type == 'CREATED':
            return RoaringBitmapWrapper(self.search_created_in_range(start_date, end_date))

        logger.warning(f"Unknown date condition type: {condition_type}")
        return RoaringBitmapWrapper()
