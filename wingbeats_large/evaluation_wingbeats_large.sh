#!/bin/bash

DATASET=Wingbeats
MODEL=large
TEST_SPLIT_RATIO=0.25
BATCH_SIZE=32

python evaluation.py --dataset $DATASET --model $MODEL --test_split_ratio $TEST_SPLIT_RATIO --batch_size $BATCH_SIZE
