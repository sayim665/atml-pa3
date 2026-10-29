"""Run Task 1 jobs across the available GPUs (Kaggle T4 x2) with a wall-clock budget.

    python -m task1.run_queue --jobs eval:baseline train:full_ft train:l2sp --budget-hours 11.3

Each GPU takes the next job whose inputs exist. When the budget is close, running jobs are
stopped (they resume from their last recovery checkpoint next session) and the runner exits
cleanly so Kaggle saves /kaggle/working.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import torch

from task1.common import REPO_ROOT, output_root, run_dir, save_json

POLL_SECONDS = 30
REPORT_SECONDS = 600
STOP_MARGIN_HOURS = 0.25


def job_done(job):
    kind, name = job.split(":")
    if kind == "train":
        return (run_dir(name) / "DONE").exists()
    return (output_root() / "eval" / name / "DONE").exists()


def job_ready(job):
    kind, name = job.split(":")
    if kind == "train":
        return True
    if name == "baseline":
        return True
    if name == "full_ft_mid":
        return (run_dir("full_ft") / "midpoint" / "config.json").exists()
    return (run_dir(name) / "DONE").exists()


def last_progress_line(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle.read().splitlines()[-200:] if line.strip()]
    for line in reversed(lines):
        if line.startswith("[") or "%|" in line or "Error" in line:
            return line[-200:]
    return lines[-1][-200:] if lines else "(no output yet)"


def job_command(job, extra_args):
    kind, name = job.split(":")
    if kind == "train":
        return [sys.executable, "-m", "task1.train", "--run", name] + extra_args
    return [sys.executable, "-m", "task1.evaluate", "--condition", name] + extra_args


class Queue:
    def __init__(self, jobs):
        self.pending = [job for job in jobs if not job_done(job)]
        self.lock = threading.Lock()
        self.status = {job: ("done" if job_done(job) else "pending") for job in jobs}
        self.running = 0

    def take(self):
        with self.lock:
            for job in self.pending:
                if job_ready(job):
                    self.pending.remove(job)
                    self.status[job] = "running"
                    self.running += 1
                    return job
            return None

    def finish(self, job, result):
        with self.lock:
            self.running -= 1
            self.status[job] = result

    def can_progress(self):
        """False once nothing is running and no pending job can become ready (e.g. its training failed)."""
        with self.lock:
            if not self.pending:
                return False
            return self.running > 0 or any(job_ready(job) for job in self.pending)


def worker(gpu, queue, deadline, log_dir, extra_args, children, status_path):
    while time.time() < deadline:
        job = queue.take()
        if job is None:
            if not queue.can_progress():
                return
            time.sleep(POLL_SECONDS)
            continue
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        log_path = log_dir / f"{job.replace(':', '_')}.log"
        print(f"[gpu{gpu}] start {job} -> {log_path}", flush=True)
        with open(log_path, "a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(job_command(job, extra_args), cwd=REPO_ROOT, env=environment,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            children[gpu] = process
            last_report = time.time()
            while process.poll() is None:
                if time.time() - last_report >= REPORT_SECONDS:
                    last_report = time.time()
                    print(f"[gpu{gpu}] {job} | {last_progress_line(log_path)}", flush=True)
                if time.time() >= deadline:
                    print(f"[gpu{gpu}] budget reached, stopping {job}", flush=True)
                    process.terminate()
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(5)
        children.pop(gpu, None)
        result = "done" if job_done(job) else f"stopped(rc={process.returncode})"
        queue.finish(job, result)
        save_json(status_path, queue.status)
        print(f"[gpu{gpu}] {job}: {result}", flush=True)
        if result != "done":
            with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                tail = handle.read().splitlines()[-25:]
            for line in tail:
                print(f"[gpu{gpu}] | {line}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--budget-hours", type=float, default=11.3)
    parser.add_argument("--gpus", type=int, default=None)
    args, extra_args = parser.parse_known_args()

    started = time.time()
    deadline = started + (args.budget_hours - STOP_MARGIN_HOURS) * 3600
    gpu_count = args.gpus or max(1, torch.cuda.device_count())
    log_dir = output_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_root() / "queue_status.json"

    queue = Queue(args.jobs)
    save_json(status_path, queue.status)
    children = {}
    threads = [threading.Thread(target=worker,
                                args=(gpu, queue, deadline, log_dir, extra_args, children, status_path))
               for gpu in range(gpu_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    save_json(status_path, queue.status)
    print(f"queue finished after {(time.time() - started) / 3600:.2f} h: {queue.status}", flush=True)


if __name__ == "__main__":
    main()
