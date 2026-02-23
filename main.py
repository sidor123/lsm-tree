from sortedcontainers import SortedDict
import bisect
import hashlib
import pickle
import os
from typing import Sequence, cast, Optional

BASE_SIZE = 4
R = 2
STORAGE_DIR = "lsm_storage"

class BloomFilter:
    def __init__(self, size: int = 1000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size
    
    def _hash(self, key: str, seed: int) -> int:
        hash_obj = hashlib.md5(f"{key}{seed}".encode())
        return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, key: str):
        for i in range(self.num_hashes):
            idx = self._hash(key, i)
            self.bit_array[idx] = True
    
    def might_contain(self, key: str) -> bool:
        for i in range(self.num_hashes):
            idx = self._hash(key, i)
            if not self.bit_array[idx]:
                return False
        return True


class Layer:
    size: int
    max_size: int
    objects: SortedDict
    bloom_filter: BloomFilter

    def __init__(self, max_size: int = BASE_SIZE):
        self.size = 0
        self.max_size = max_size
        self.objects = SortedDict()
        self.bloom_filter = BloomFilter(size=max_size * 10, num_hashes=3)

    def add(self, key: str, value: str):
        if self.is_full():
            print(f"Layer is full, aborting add operation...")
            return
        if key not in self.objects:
            self.size += 1
        self.objects[key] = value
        self.bloom_filter.add(key)
    
    def update(self, key: str, value: str):
        if key not in self.objects:
            self.add(key, value)
        self.objects[key] = value

    def is_full(self): 
        print(f"Layer has size {self.size} out of {self.max_size}.")
        return self.size == self.max_size

    def search(self, key: str):
        if not self.bloom_filter.might_contain(key):
            print(f"Bloom filter: key '{key}' definitely not in this layer")
            return None
        
        print(f"Bloom filter: key '{key}' might be in this layer, checking...")
        
        keys = list(self.objects.keys())
        idx = bisect.bisect_left(keys, key)
        
        if idx < len(keys) and keys[idx] == key:
            return self.objects[key]
        
        print(f"False positive from Bloom filter")
        return None
    
    def range_search(self, start_key: str, end_key: str):
        keys = list(self.objects.keys())
        start_idx = bisect.bisect_left(keys, start_key)
        end_idx = bisect.bisect_right(keys, end_key)
        
        result = {}
        for i in range(start_idx, end_idx):
            key = keys[i]
            result[key] = self.objects[key]
        
        return result
    
    def __str__(self):
        return f"Size: {self.size}, max size: {self.max_size}, objects: {self.objects}"


class DiskLayer(Layer):
    filepath: str
    storage_dir: str
    
    def __init__(self, max_size: int, layer_id: int, storage_dir: str = STORAGE_DIR):
        super().__init__(max_size)
        self.layer_id = layer_id
        self.storage_dir = storage_dir
        self.filepath = os.path.join(storage_dir, f"layer_{layer_id}.pkl")
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_to_disk(self):
        data = {
            'size': self.size,
            'max_size': self.max_size,
            'objects': dict(self.objects),
            'bloom_filter': {
                'size': self.bloom_filter.size,
                'num_hashes': self.bloom_filter.num_hashes,
                'bit_array': self.bloom_filter.bit_array
            }
        }
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved layer {self.layer_id} to disk: {self.filepath}")
    
    @classmethod
    def load_from_disk(cls, layer_id: int, storage_dir: str = STORAGE_DIR) -> Optional['DiskLayer']:
        filepath = os.path.join(storage_dir, f"layer_{layer_id}.pkl")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        layer = cls(data['max_size'], layer_id, storage_dir)
        layer.size = data['size']
        layer.objects = SortedDict(data['objects'])
        
        bf_data = data['bloom_filter']
        layer.bloom_filter = BloomFilter(bf_data['size'], bf_data['num_hashes'])
        layer.bloom_filter.bit_array = bf_data['bit_array']
        
        print(f"Loaded layer {layer_id} from disk: {filepath}")
        return layer
    
    def add(self, key: str, value: str):
        super().add(key, value)
        self.save_to_disk()
    
    def update(self, key: str, value: str):
        super().update(key, value)
        self.save_to_disk()
    
    def __str__(self):
        return f"DiskLayer {self.layer_id}: Size: {self.size}, max size: {self.max_size}, objects: {self.objects}, file: {self.filepath}"


# diffirence in storing (value, is_deleted) instead of value in on-disk components
class MemoryBuffer(Layer):
    def add(self, key: str, value: str):
        if self.is_full():
            print(f"Layer is full, aborting add operation...")
            return
        if key not in self.objects:
            self.size += 1
        self.objects[key] = (value, False)
        self.bloom_filter.add(key)

    def search(self, key: str):
        if not self.bloom_filter.might_contain(key):
            print(f"Bloom filter: key '{key}' definitely not in this layer")
            return None
        
        print(f"Bloom filter: key '{key}' might be in this layer, checking...")
        
        keys = list(self.objects.keys())
        idx = bisect.bisect_left(keys, key)
        
        if idx < len(keys) and keys[idx] == key and not self.objects[key][1]:
            return self.objects[key][0]
        
        print(f"False positive from Bloom filter")
        return None

    def range_search(self, start_key: str, end_key: str):
        keys = list(self.objects.keys())
        start_idx = bisect.bisect_left(keys, start_key)
        end_idx = bisect.bisect_right(keys, end_key)
        
        result = {}
        for i in range(start_idx, end_idx):
            if not self.objects[keys[i]][1]:
                result[keys[i]] = self.objects[keys[i]][0]
        
        return result

    def remove(self, key: str):
        if not self.search(key):
            self.objects[key] = ("tombstone", True)
            self.size += 1
            return
        self.objects[key] = (None, True)


class LSMTree:
    def __init__(self, storage_dir: str = STORAGE_DIR):
        self.storage_dir = storage_dir
        self.layers: list[Layer | MemoryBuffer | DiskLayer] = [MemoryBuffer(R)]
        self._load_existing_layers()
    
    def _load_existing_layers(self):
        layer_id = 1
        while True:
            disk_layer = DiskLayer.load_from_disk(layer_id, self.storage_dir)
            if disk_layer is None:
                break
            self.layers.append(disk_layer)
            layer_id += 1
        
        if len(self.layers) > 1:
            print(f"Loaded {len(self.layers) - 1} disk layers from storage")

    def add(self, key: str, value: str):
        self.layers[0].add(key, value)

        if self.layers[0].is_full():
            print("Memory buffer is full. Merging...")
            merge(self.layers, 0, self.storage_dir)

    def remove(self, key: str):
        if isinstance(self.layers[0], MemoryBuffer):
            self.layers[0].remove(key)

        if self.layers[0].is_full():
            print("Memory buffer is full. Merging...")
            merge(self.layers, 0, self.storage_dir)

    def get(self, key: str):
        return search_layers(self.layers, key)

    def range_get(self, start_key: str, end_key: str):
        return range_search_layers(self.layers, start_key, end_key)

    def print_layers(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}:", layer)


def merge(layers: list[Layer | MemoryBuffer | DiskLayer], layer_index: int, storage_dir: str = STORAGE_DIR):
    layer = layers[layer_index]

    if layer_index == len(layers) - 1:
        print(f"Adding new layer with max_size={R * layers[-1].max_size}")
        next_layer_id = layer_index + 1
        next_layer = DiskLayer(R * layers[-1].max_size, next_layer_id, storage_dir)
    else:
        print(f"Merging layer {layer_index} with next layer {layer_index + 1}")
        next_layer = layers[layer_index + 1]

    if isinstance(next_layer, DiskLayer):
        merged_layer = DiskLayer(next_layer.max_size, next_layer.layer_id, storage_dir)
    else:
        merged_layer = DiskLayer(next_layer.max_size, layer_index + 1, storage_dir)
    
    for key, value in next_layer.objects.items():
        if key in layer.objects:
            continue
        merged_layer.objects[key] = value
        merged_layer.size += 1
        merged_layer.bloom_filter.add(key)
    
    for key, value in layer.objects.items():
        if isinstance(layer, MemoryBuffer):
            if not value[1]:
                print(f"Merging key={key} and value={value[0]} to next layer.")
                if key not in merged_layer.objects:
                    merged_layer.size += 1
                merged_layer.objects[key] = value[0]
                merged_layer.bloom_filter.add(key)
        else:
            print(f"Merging key={key} and value={value} to next layer.")
            if key not in merged_layer.objects:
                merged_layer.size += 1
            merged_layer.objects[key] = value
            merged_layer.bloom_filter.add(key)
    
    merged_layer.save_to_disk()
    
    layers[layer_index] = MemoryBuffer(layer.max_size) if layer_index == 0 else DiskLayer(layer.max_size, layer_index, storage_dir)
    
    if layer_index == len(layers) - 1:
        layers.append(merged_layer)
    else:
        layers[layer_index + 1] = merged_layer
    
    if merged_layer.is_full():
        print("Next layer is full, merging next layer...")
        merge(layers, layer_index + 1, storage_dir)


def search_layers(layers: Sequence[Layer | MemoryBuffer | DiskLayer], key: str):
    print(f"Searching for key='{key}' across {len(layers)} layers...")
    
    for i, layer in enumerate(layers):
        print(f"Searching in layer {i}...")
        value = layer.search(key)
        if value is not None:
            print(f"Found key='{key}' with value='{value}' in layer {i}")
            return value
    
    print(f"Key='{key}' not found in any layer")
    return None


def range_search_layers(layers: Sequence[Layer | MemoryBuffer | DiskLayer], start_key: str, end_key: str):
    print(f"Range search for keys from '{start_key}' to '{end_key}' across {len(layers)} layers...")
    
    result = {}
    seen_keys = set()
    
    for i, layer in enumerate(layers):
        print(f"Searching in layer {i}...")
        layer_results = layer.range_search(start_key, end_key)
        
        for key, value in layer_results.items():
            if key not in seen_keys:
                result[key] = value
                seen_keys.add(key)
        
        print(f"Found {len(layer_results)} keys in layer {i}")
    
    print(f"Total unique keys found: {len(result)}")
    return result


def run_program():
    print("LSM leveled compaction started.")
    print(f"Using R={R} as size modifier")

    lsm = LSMTree()
    print("Created memory buffer. (layer 0)")

    while True:
        command = input("Enter command (add, remove, get, get range, print, exit): ")
        if command == "add":
            key = input("Enter key: ")
            value = input("Enter value: ")
            print(f"Adding key={key} and value={value} to memory buffer.")
            lsm.add(key, value)
        elif command == "remove":
            key = input("Enter key: ")
            print(f"Removing key={key} from memory buffer.")
            lsm.remove(key)
        elif command == "get":
            key = input("Enter key: ")
            result = lsm.get(key)
            if result is not None:
                print(f"{key} = {result}")
            else:
                print(f"Key '{key}' not found")
        elif command == "get range":
            start_key = input("Enter start key: ")
            end_key = input("Enter end key: ")
            results = lsm.range_get(start_key, end_key)
            if results:
                print(f"Found {len(results)} keys:")
                for key, value in sorted(results.items()):
                    print(f"{key} = {value}")
            else:
                print("No keys found in range")
        elif command == "print":
            lsm.print_layers()
        elif command == "exit":
            print("Exiting...")
            return 0


if __name__ == '__main__':
    run_program()
