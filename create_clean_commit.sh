#!/bin/bash

echo "Creating clean BERT commit with only essential files..."
echo ""

# Safety check
if [ "$(git status --porcelain)" ]; then
    echo "WARNING: You have uncommitted changes. Please commit or stash them first."
    git status
    exit 1
fi

# Get current branch name for reference
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Create new clean branch from base
echo "Step 1: Creating new clean branch from base..."
git checkout ivoitovych/bert-model-for-ttml
git checkout -b ivoitovych/bert-dtype-fix-clean

echo ""
echo "Step 2: Cherry-picking essential files from messy branch..."

# Cherry-pick only the essential files
git checkout myfork/ivoitovych/bert-model-for-ttml-base-uncased-validation -- \
  tt-train/sources/ttml/models/bert.cpp \
  tt-train/sources/ttml/models/bert.hpp \
  tt-train/sources/ttml/modules/bert_block.cpp \
  tt-train/sources/ttml/modules/bert_block.hpp \
  tt-train/sources/ttml/modules/multi_head_attention.cpp \
  tt-train/sources/ttml/nanobind/nb_models.cpp \
  tt-train/sources/ttml/nanobind/nb_ops.cpp \
  tt-train/sources/ttml/nanobind/nb_util.cpp \
  tt-train/sources/ttml/ops/multi_head_utils.cpp \
  tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp \
  tt-train/tests/python/test_bert_isolated_layer_validation.py \
  tt-train/tests/python/test_bert_embedding_decomposition.py \
  tt-train/tests/python/test_bert_end_to_end_validation.py \
  tt-train/tests/python/test_bert_padding_mask_validation.py \
  tt-train/tests/python/test_bert_layer_pcc_report.py \
  tt-train/tests/python/BERT_DTYPE_FIX_RESULTS.md

echo ""
echo "Step 3: Adding optional C++ tests..."
git checkout myfork/ivoitovych/bert-model-for-ttml-base-uncased-validation -- \
  tt-train/tests/core/tile_layout_round_trip_test.cpp \
  tt-train/tests/model/bert_operator_test.cpp \
  tt-train/tests/model/bert_real_data_test.cpp \
  tt-train/tests/CMakeLists.txt 2>/dev/null || echo "C++ tests not found, skipping..."

echo ""
echo "Step 4: Creating commit with comprehensive message..."
git commit -F CLEAN_COMMIT_MESSAGE.txt

echo ""
echo "✅ Clean commit created successfully!"
echo ""
echo "Files included:"
git diff --name-only ivoitovych/bert-model-for-ttml..HEAD | wc -l | xargs echo "  Total files:"
echo ""

echo "Next steps:"
echo "  1. Review the commit: git show"
echo "  2. Push to your fork: git push myfork ivoitovych/bert-dtype-fix-clean"
echo "  3. Create PR from the clean branch"
echo ""
echo "To return to your original branch:"
echo "  git checkout $CURRENT_BRANCH"
