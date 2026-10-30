#!/bin/bash
# Trains and evaluates one Task 2 run per GPU in parallel; full logs go to task2_outputs/logs.
set -o pipefail
mkdir -p /kaggle/working/task2_outputs/logs
run() {
  log=/kaggle/working/task2_outputs/logs/$2.log
  CUDA_VISIBLE_DEVICES=$1 timeout 11h python -u -m task2.train --run $2 2>&1 | tee -a $log | grep --line-buffered -E "update [0-9]+/|dev @|resumed|Error|Traceback" &&
  CUDA_VISIBLE_DEVICES=$1 python -u -m task2.evaluate --run $2 2>&1 | tee -a $log | grep --line-buffered -E "acc=|Error|Traceback"
  echo "$2 finished with exit code $?"
}
run 0 vlm196 &
run 1 vlm49 &
wait
