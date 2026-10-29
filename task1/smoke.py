"""End-to-end smoke test (4 updates per method, 2 documents per benchmark) in a throwaway output folder.

    python -m task1.smoke
"""

import os
import subprocess
import sys
import tempfile

from task1.common import REPO_ROOT

STEPS = [
    ["-m", "task1.data"],
    ["-m", "task1.train", "--run", "full_ft", "--max-updates", "1"],
    ["-m", "task1.train", "--run", "full_ft"],
    ["-m", "task1.train", "--run", "l2sp"],
    ["-m", "task1.train", "--run", "lora4"],
    ["-m", "task1.train", "--run", "lora16"],
    ["-m", "task1.train", "--run", "oplora16"],
    ["-m", "task1.evaluate", "--condition", "baseline", "--limit", "2"],
    ["-m", "task1.evaluate", "--condition", "full_ft_mid", "--limit", "2"],
    ["-m", "task1.evaluate", "--condition", "oplora16", "--limit", "2"],
]


def main():
    scratch = tempfile.mkdtemp(prefix="atml_smoke_")
    environment = dict(os.environ, ATML_SMOKE_TEST="1", ATML_OUT=f"{scratch}/out", ATML_RESULTS=f"{scratch}/results",
                       PYTHONUNBUFFERED="1")
    for step in STEPS:
        print("==>", " ".join(step), flush=True)
        completed = subprocess.run([sys.executable] + step, cwd=REPO_ROOT, env=environment)
        if completed.returncode != 0:
            print(f"SMOKE TEST FAILED at: {' '.join(step)}", flush=True)
            sys.exit(1)
    subprocess.run(["bash", "-c", f"cat {scratch}/out/runs/*/train_summary.json | grep -E "
                                  f"'trainable_parameters|displacement|basis_bytes|peak_memory_allocated'; "
                                  f"cat {scratch}/results/eval/*/merge_check.json; "
                                  f"grep -h -A1 'gsm8k_non_thinking_prompt' {scratch}/results/eval/*/eval_info.json; "
                                  f"cat {scratch}/results/eval/*/scores.json | head -n 60"])
    print("SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
