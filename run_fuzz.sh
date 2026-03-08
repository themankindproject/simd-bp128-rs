#!/bin/bash

set -e

echo "========================================"
echo "SIMD-BP128 Fuzz Testing"
echo "========================================"

cd fuzz

echo ""
echo "Running roundtrip fuzz test (60 seconds)..."
timeout 60 cargo fuzz run roundtrip -- -max_total_time=60 || true

echo ""
echo "Running decompress_only fuzz test (60 seconds)..."
timeout 60 cargo fuzz run decompress_only -- -max_total_time=60 || true

echo ""
echo "Running compress_only fuzz test (60 seconds)..."
timeout 60 cargo fuzz run compress_only -- -max_total_time=60 || true

echo ""
echo "========================================"
echo "Fuzz testing complete!"
echo "========================================"
echo ""
echo "To run longer fuzz sessions:"
echo "  cd fuzz"
echo "  cargo fuzz run roundtrip"
echo "  cargo fuzz run decompress_only"
echo "  cargo fuzz run compress_only"