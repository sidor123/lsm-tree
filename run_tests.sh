#!/bin/bash
set -e

echo "Running all tests..."
echo ""

source venv/bin/activate

export PYTHONPATH=.

echo "Testing LSM Tree..."
python3 tests/test_lsm_tree.py
echo "LSM Tree tests passed!"
echo ""

echo "Testing Inverted Index (Standalone)..."
python3 tests/test_inverted_index.py
echo "Inverted Index tests passed!"
echo ""

echo "Testing LSM-based Inverted Index..."
python3 tests/test_lsm_inverted_index.py
echo "LSM-based Inverted Index tests passed!"
echo ""

echo "All tests passed successfully!"