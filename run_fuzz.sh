#!/bin/bash

set -e

echo "========================================"
echo "SIMD-BP128 Fuzz Testing"
echo "========================================"

cd fuzz

# Configurable timeout (default 300 seconds = 5 minutes per target)
FUZZ_TIMEOUT=${FUZZ_TIMEOUT:-300}

echo ""
echo "Running roundtrip fuzz test (${FUZZ_TIMEOUT} seconds)..."
timeout $FUZZ_TIMEOUT cargo fuzz run roundtrip -- -max_total_time=$FUZZ_TIMEOUT || true

echo ""
echo "Running decompress_only fuzz test (${FUZZ_TIMEOUT} seconds)..."
timeout $FUZZ_TIMEOUT cargo fuzz run decompress_only -- -max_total_time=$FUZZ_TIMEOUT || true

echo ""
echo "Running compress_only fuzz test (${FUZZ_TIMEOUT} seconds)..."
timeout $FUZZ_TIMEOUT cargo fuzz run compress_only -- -max_total_time=$FUZZ_TIMEOUT || true

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