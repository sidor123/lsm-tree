from typing import List, Optional
import logging

from pyroaring import BitMap

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, tokens: List[str]) -> List[str]:
        tokens = [t.lower() for t in tokens]

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
