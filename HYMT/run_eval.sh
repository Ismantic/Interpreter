#!/bin/bash
PYTHON=/home/tfbao/new/HY-MT/.venv/bin/python
cd /home/tfbao/Shiyu/Interpreter/HYMT

echo "=== Phase 1 v5 ==="
$PYTHON -u eval.py --model_path ./private/phase1_v5 --testset wmt22 --direction both --max_samples 200 --batch_size 8

echo ""
echo "=== Phase 2 ==="
$PYTHON -u eval.py --model_path ./private/phase2 --testset wmt22 --direction both --max_samples 200 --batch_size 8
