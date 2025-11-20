#!/bin/bash

# Script to execute all instances in instances/spprclib/*.sppcc
# and save output in solutions/spprclib/*-{algorithm}.txt

# Ensure solutions directory exists
mkdir -p solutions/spprclib

# Path to executable and set file
PATHWYSE_BIN="./bin/pathwyse"
SET_FILE="./pathwyse.set"

# Check if executable exists
if [ ! -f "$PATHWYSE_BIN" ]; then
    echo "Error: $PATHWYSE_BIN not found"
    exit 1
fi

# Extract algorithm name from set file
if [ -f "$SET_FILE" ]; then
    ALGORITHM=$(grep "^main_algorithm" "$SET_FILE" | sed 's/.*= *//' | tr -d '[:space:]')
    if [ -z "$ALGORITHM" ]; then
        echo "Warning: Could not find main_algorithm in $SET_FILE, using 'default'"
        ALGORITHM="default"
    fi
else
    echo "Warning: $SET_FILE not found, using 'default' as algorithm name"
    ALGORITHM="default"
fi

echo "Using algorithm: $ALGORITHM"

# Counter for progress
total=$(ls instances/spprclib/*.sppcc 2>/dev/null | wc -l)
current=0

echo "Found $total instance files to process"
echo "========================================="

# Process each .sppcc file
for instance in instances/spprclib/*.sppcc; do
    if [ -f "$instance" ]; then
        # Extract filename without path and extension
        filename=$(basename "$instance" .sppcc)
        
        # Define output file with algorithm name
        output="solutions/spprclib/${filename}-${ALGORITHM}.txt"
        
        # Increment counter
        current=$((current + 1))
        
        echo "[$current/$total] Processing: $filename"
        echo "  Input:  $instance"
        echo "  Output: $output"
        
        # Execute pathwyse and save output
        "$PATHWYSE_BIN" "$instance" > "$output" 2>&1
        
        # Check execution status
        if [ $? -eq 0 ]; then
            echo "  Status: SUCCESS"
        else
            echo "  Status: FAILED (exit code: $?)"
        fi
        echo ""
    fi
done

echo "========================================="
echo "Processing complete: $current instances processed"
echo "Results saved in solutions/spprclib/"
