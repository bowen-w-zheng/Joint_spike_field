#!/bin/bash
# Clear Python and Numba caches to fix module import issues
# Run this if you encounter "ModuleNotFoundError" related to numba cache

echo "Clearing Python and Numba caches..."

# Clear Python bytecode cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✓ Removed __pycache__ directories"

# Clear Numba cache files
find . -name "*.nbc" -delete 2>/dev/null
find . -name "*.nbi" -delete 2>/dev/null
echo "✓ Removed Numba cache files (*.nbc, *.nbi)"

# Clear .pyc files
find . -name "*.pyc" -delete 2>/dev/null
echo "✓ Removed .pyc files"

echo ""
echo "Cache cleared successfully!"
echo "You can now re-run your code."
