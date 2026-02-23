import unittest
import shutil
import os
from main import LSMTree, Layer, BloomFilter, DiskLayer

TEST_STORAGE_BASE = "lsm_storage_test"


class TestLSMTree(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(TEST_STORAGE_BASE, self._testMethodName)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_simple(self):
        """Test simple add and get operations."""
        lsm = LSMTree(self.test_dir)
        
        lsm.add("key1", "value1")
        lsm.add("key2", "value2")
        
        self.assertEqual(lsm.get("key1"), "value1")
        self.assertEqual(lsm.get("key2"), "value2")
        self.assertIsNone(lsm.get("nonexistent"))
    
    def test_merge_full_layer(self):
        """Test that merge happens when layer is full."""
        lsm = LSMTree(self.test_dir)
        
        lsm.add("a", "1")
        lsm.add("b", "2")
        
        self.assertEqual(len(lsm.layers), 2)
        self.assertEqual(lsm.layers[0].size, 0)
        self.assertEqual(lsm.layers[1].size, 2)
        self.assertEqual(lsm.get("a"), "1")
        self.assertEqual(lsm.get("b"), "2")

    def test_simple_removal(self):
        """Test that remove command works correctly."""
        lsm = LSMTree(self.test_dir)

        for i in range(3):
            lsm.add(f"key{i}", f"value{i}")

        lsm.remove("key0")

        self.assertEqual(len(lsm.layers), 2)
        self.assertEqual(lsm.get("key0"), None)
        self.assertEqual(lsm.get("key1"), "value1")
        self.assertEqual(lsm.get("key2"), "value2")
    
    def test_overwrite(self):
        """Test that newer values override older ones."""
        lsm = LSMTree(self.test_dir)
        
        lsm.add("key", "old_value")
        lsm.add("key", "new_value")
        
        self.assertEqual(lsm.get("key"), "new_value")
    
    def test_layers(self):
        """Test operations across multiple layers."""
        lsm = LSMTree(self.test_dir)
        
        keys = [f"key{i}" for i in range(10)]
        values = [f"value{i}" for i in range(10)]
        
        for key, value in zip(keys, values):
            lsm.add(key, value)
        
        for key, value in zip(keys, values):
            self.assertEqual(lsm.get(key), value)
    
    def test_simple_search(self):
        """Test that binary search works correctly in layers."""
        layer = Layer(max_size=10)
        
        for i in range(5):
            layer.add(f"key{i}", f"value{i}")
        
        self.assertEqual(layer.search("key0"), "value0")
        self.assertEqual(layer.search("key2"), "value2")
        self.assertEqual(layer.search("key4"), "value4")
        self.assertIsNone(layer.search("nonexistent"))

    def test_multilayer_search(self):
        """Test that search works across multiple layers."""
        lsm = LSMTree(self.test_dir)
        
        keys = [f"key{i}" for i in range(4)]
        values = [f"value{i}" for i in range(4)]

        keys.extend([f"key{i}" for i in range(2)])
        values.extend([f"new_value{i}" for i in range(2)])

        keys.append(f"key0")
        values.append(f"newest_value0")

        for key, value in zip(keys, values):
            lsm.add(key, value)

        self.assertEqual(lsm.get("key0"), "newest_value0")
        self.assertEqual(lsm.get("key1"), "new_value1")
        self.assertEqual(lsm.get("key2"), "value2")


class TestBloomFilter(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(TEST_STORAGE_BASE, self._testMethodName)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_simple(self):
        """Test basic Bloom filter operations."""
        bf = BloomFilter(size=100, num_hashes=3)
        
        bf.add("key1")
        bf.add("key2")
        bf.add("key3")
        
        self.assertTrue(bf.might_contain("key1"))
        self.assertTrue(bf.might_contain("key2"))
        self.assertTrue(bf.might_contain("key3"))
    
    def test_fp_rate(self):
        """Test Bloom filter returns False for keys not added."""
        bf = BloomFilter(size=1000, num_hashes=3)
        
        bf.add("existing_key")
        
        self.assertTrue(bf.might_contain("existing_key"))
        
        not_added_keys = [f"key{i}" for i in range(100)]
        false_positives = 0
        for key in not_added_keys:
            if bf.might_contain(key):
                false_positives += 1
        
        self.assertLess(false_positives, 20)
    
    def test_layer(self):
        """Test Bloom filter integration in Layer."""
        layer = Layer(max_size=10)
        
        layer.add("key1", "value1")
        layer.add("key2", "value2")
        
        self.assertTrue(layer.bloom_filter.might_contain("key1"))
        self.assertTrue(layer.bloom_filter.might_contain("key2"))
        self.assertFalse(layer.bloom_filter.might_contain("nonexistent"))
    
    def test_after_merge(self):
        """Test Bloom filter is correctly rebuilt after merge."""
        lsm = LSMTree(self.test_dir)
        
        lsm.add("a", "1")
        lsm.add("b", "2")
        
        self.assertTrue(lsm.layers[1].bloom_filter.might_contain("a"))
        self.assertTrue(lsm.layers[1].bloom_filter.might_contain("b"))
        self.assertFalse(lsm.layers[1].bloom_filter.might_contain("nonexistent"))
    
    def test_fp_rate_layers(self):
        """Test Bloom filter false positive rate is reasonable."""
        bf = BloomFilter(size=1000, num_hashes=3)
        
        added_keys = [f"added_{i}" for i in range(50)]
        for key in added_keys:
            bf.add(key)
        
        test_keys = [f"test_{i}" for i in range(1000)]
        false_positives = sum(1 for key in test_keys if bf.might_contain(key))
        
        false_positive_rate = false_positives / len(test_keys)
        self.assertLess(false_positive_rate, 0.1)


class TestRangeSearch(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(TEST_STORAGE_BASE, self._testMethodName)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_simple(self):
        """Test range search within a single layer."""
        layer = Layer(max_size=20)
        
        for i in range(10):
            layer.add(f"key{i:02d}", f"value{i}")
        
        result = layer.range_search("key02", "key05")
        
        self.assertEqual(len(result), 4)
        self.assertIn("key02", result)
        self.assertIn("key03", result)
        self.assertIn("key04", result)
        self.assertIn("key05", result)
    
    def test_empty(self):
        """Test range search with no matching keys."""
        layer = Layer(max_size=20)
        
        layer.add("key10", "value10")
        layer.add("key20", "value20")
        
        result = layer.range_search("key11", "key19")
        
        self.assertEqual(len(result), 0)
    
    def test_layers(self):
        """Test range search across multiple layers."""
        lsm = LSMTree(self.test_dir)
        
        for i in range(10):
            lsm.add(f"key{i}", f"value{i}")
        
        result = lsm.range_get("key2", "key7")
        
        self.assertEqual(len(result), 6)
        for i in range(2, 8):
            key = f"key{i}"
            self.assertIn(key, result)
            self.assertEqual(result[key], f"value{i}")
    
    def test_overwrites(self):
        """Test range search returns newest values when keys are overwritten."""
        lsm = LSMTree(self.test_dir)
        
        lsm.add("key1", "old1")
        lsm.add("key2", "old2")
        lsm.add("key3", "old3")
        lsm.add("key4", "old4")
        
        lsm.add("key2", "new2")
        lsm.add("key3", "new3")
        
        result = lsm.range_get("key1", "key4")
        
        self.assertEqual(result["key1"], "old1")
        self.assertEqual(result["key2"], "new2")
        self.assertEqual(result["key3"], "new3")
        self.assertEqual(result["key4"], "old4")
    
    def test_full_range(self):
        """Test range search that covers all keys."""
        lsm = LSMTree(self.test_dir)
        
        keys = ["apple", "banana", "cherry", "date", "elderberry"]
        for key in keys:
            lsm.add(key, f"value_{key}")
        
        result = lsm.range_get("a", "z")
        
        self.assertEqual(len(result), 5)
        for key in keys:
            self.assertIn(key, result)


class TestDiskPersistence(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(TEST_STORAGE_BASE, self._testMethodName)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_disk_layer_save_and_load(self):
        """Test that DiskLayer can save and load from disk."""
        layer = DiskLayer(max_size=10, layer_id=1, storage_dir=self.test_dir)
        
        layer.add("key1", "value1")
        layer.add("key2", "value2")
        layer.add("key3", "value3")
        
        loaded_layer = DiskLayer.load_from_disk(1, self.test_dir)
        
        self.assertIsNotNone(loaded_layer)
        if loaded_layer:
            self.assertEqual(loaded_layer.size, 3)
            self.assertEqual(loaded_layer.objects["key1"], "value1")
            self.assertEqual(loaded_layer.objects["key2"], "value2")
            self.assertEqual(loaded_layer.objects["key3"], "value3")
    
    def test_lsm_tree_persistence(self):
        """Test that LSM tree persists disk layers across restarts."""
        lsm1 = LSMTree(self.test_dir)
        
        for i in range(6):
            lsm1.add(f"key{i}", f"value{i}")
        
        self.assertGreater(len(lsm1.layers), 1)
        
        lsm2 = LSMTree(self.test_dir)
        
        self.assertEqual(len(lsm2.layers), len(lsm1.layers))
        
        for i in range(6):
            self.assertEqual(lsm2.get(f"key{i}"), f"value{i}")
    
    def test_merge_creates_disk_layers(self):
        """Test that merge operation creates disk layers."""
        lsm = LSMTree(self.test_dir)
        
        lsm.add("a", "1")
        lsm.add("b", "2")
        
        self.assertEqual(len(lsm.layers), 2)
        self.assertIsInstance(lsm.layers[1], DiskLayer)
        
        disk_files = os.listdir(self.test_dir)
        self.assertGreater(len(disk_files), 0)
    
    def test_disk_layer_bloom_filter_persistence(self):
        """Test that Bloom filter is correctly saved and loaded."""
        layer = DiskLayer(max_size=10, layer_id=1, storage_dir=self.test_dir)
        
        layer.add("key1", "value1")
        layer.add("key2", "value2")
        
        loaded_layer = DiskLayer.load_from_disk(1, self.test_dir)
        
        self.assertIsNotNone(loaded_layer)
        if loaded_layer:
            self.assertTrue(loaded_layer.bloom_filter.might_contain("key1"))
            self.assertTrue(loaded_layer.bloom_filter.might_contain("key2"))
            self.assertFalse(loaded_layer.bloom_filter.might_contain("nonexistent"))
    
    def test_memory_buffer_not_persisted(self):
        """Test that MemoryBuffer (layer 0) is not persisted to disk."""
        lsm = LSMTree(self.test_dir)
        
        lsm.add("key1", "value1")
        
        disk_files = os.listdir(self.test_dir) if os.path.exists(self.test_dir) else []
        
        layer_0_file = "layer_0.pkl"
        self.assertNotIn(layer_0_file, disk_files)


if __name__ == '__main__':
    unittest.main()
