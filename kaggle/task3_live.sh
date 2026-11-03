#!/bin/bash
# Runs the 50-question live comparison as two shards, one per GPU, then grades them.
mkdir -p /kaggle/working/task3_outputs/logs
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard python -u -m task3.live --shard $shard --num-shards 2 2>&1 | tee -a /kaggle/working/task3_outputs/logs/live_shard$shard.log | grep --line-buffered -E "done|Error|Traceback" &
done
wait
python -u -m task3.live --summarize
