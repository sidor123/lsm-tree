from typing import List
import logging

logger = logging.getLogger(__name__)


class KGramGenerator:
    def __init__(self, k: int = 2):
        if k < 1:
            raise ValueError("k must be at least 1")
        
        self.k = k
        self.boundary_char = '$'
        logger.debug(f"Initialized KGramGenerator with k={k}")
    
    def generate_kgrams(self, term: str) -> List[str]:
        if not term:
            return []
        
        padded_term = self.boundary_char + term + self.boundary_char
        
        kgrams = []
        for i in range(len(padded_term) - self.k + 1):
            kgram = padded_term[i:i + self.k]
            kgrams.append(kgram)
        
        logger.debug(f"Generated {len(kgrams)} k-grams from term '{term}'")
        return kgrams
    
    def wildcard_to_kgrams(self, pattern: str) -> List[str]:
        if '*' not in pattern:
            raise ValueError("Pattern must contain at least one wildcard (*)")
        
        wildcard_count = pattern.count('*')
        if wildcard_count > 1:
            raise ValueError(f"Pattern must contain exactly one wildcard, found {wildcard_count}")
        
        parts = pattern.split('*')
        prefix = parts[0]
        suffix = parts[1]
        
        kgrams = []
        
        if prefix:
            prefix_with_boundary = self.boundary_char + prefix
            for i in range(len(prefix_with_boundary) - self.k + 1):
                kgram = prefix_with_boundary[i:i + self.k]
                kgrams.append(kgram)
        
        if suffix:
            suffix_with_boundary = suffix + self.boundary_char
            for i in range(len(suffix_with_boundary) - self.k + 1):
                kgram = suffix_with_boundary[i:i + self.k]
                kgrams.append(kgram)
        
        logger.debug(f"Extracted {len(kgrams)} k-grams from pattern '{pattern}'")
        return kgrams
    
    def pattern_to_regex(self, pattern: str) -> str:
        import re
        escaped = re.escape(pattern)
        regex_pattern = escaped.replace(r'\*', '.*')
        regex_pattern = '^' + regex_pattern + '$'
        
        return regex_pattern