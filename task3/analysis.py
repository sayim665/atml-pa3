"""Offline Task 3 experiments on the supplied rollouts (Steps 1, 2, 3 and 5).

    python -m task3.analysis [--dataset math|mathvista]

Outcomes are averaged over the ten rollout orders within a question, then across questions. Headline numbers use the
800 held-out questions; the 200 development questions are used only to choose stopping thresholds.
"""

import argparse
import gzip
import json
import multiprocessing
import platform
import time

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from task1.common import save_json
from task3.config import (BOOTSTRAP_RESAMPLES, BUDGETS, DATASETS, DEV_TOLERANCE, GAMMA_GRID, NO_ANSWER,
                          ROLLOUTS, SEED, STOP_CHECKS, TAU_GRID, results_root)
from task3.data import load_prepared
from task3.selection import method_names, select_all, stopping_features, visconf_scores

PREPARED = None
CHECKPOINTS = STOP_CHECKS + (ROLLOUTS,)


def evaluate_question(question):
    """Every method at every budget for every order, plus the stopping evidence at n = 4, 8, 16, 32."""
    record = PREPARED["questions"][question]
    vision = DATASETS[PREPARED["dataset"]]["vision"]
    correct_by_group = dict(zip(record["groups"], record["correct"]))
    selections, coverage, stopping = [], [], []
    for order_index, order in enumerate(PREPARED["orders"][question]):
        for budget in BUDGETS:
            candidates = order[:budget]
            chosen = select_all(record, candidates, vision)
            cost = int(record["tokens"][candidates].sum())
            for method, group in chosen.items():
                method_cost = int(record["tokens"][candidates[0]]) if method == "first" else cost
                selections.append((question, order_index, budget, method, group, bool(correct_by_group[group]),
                                   method_cost))
            groups = [record["groups"][candidate] for candidate in candidates]
            coverage.append((question, order_index, budget, bool(record["correct"][candidates].any()),
                             len(set(groups)), cost))
        for checkpoint in CHECKPOINTS:
            candidates = order[:checkpoint]
            features = stopping_features(record, candidates, vision)
            features.update(question=question, order=order_index, n=checkpoint,
                            correct=bool(correct_by_group[features["sc"]]),
                            cost=int(record["tokens"][candidates].sum()))
            stopping.append(features)
    return selections, coverage, stopping


def run_all(question_ids):
    if platform.system() == "Linux":
        with multiprocessing.get_context("fork").Pool() as pool:
            outputs = pool.map(evaluate_question, question_ids, chunksize=8)
    else:
        outputs = [evaluate_question(question) for question in question_ids]
    selections = [row for output in outputs for row in output[0]]
    coverage = [row for output in outputs for row in output[1]]
    stopping = [row for output in outputs for row in output[2]]
    return selections, coverage, stopping


def per_question_means(rows, key_fields, value_fields):
    """Average values over orders within (question, *keys)."""
    sums = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        entry = sums.setdefault(key, [0.0] * len(value_fields) + [0])
        for index, field in enumerate(value_fields):
            entry[index] += float(row[field])
        entry[-1] += 1
    return {key: [value / entry[-1] for value in entry[:-1]] for key, entry in sums.items()}


def bootstrap_difference(first, second, rng_indices):
    difference = first - second
    estimates = difference[rng_indices].mean(axis=1)
    return {"mean": float(difference.mean()),
            "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]}


def fixed_budget_results(selections, coverage, split_ids, vision):
    selection_rows = [dict(zip(("question", "order", "budget", "method", "group", "correct", "cost"), row))
                      for row in selections]
    coverage_rows = [dict(zip(("question", "order", "budget", "covered", "distinct", "cost"), row))
                     for row in coverage]
    method_means = per_question_means(selection_rows, ("question", "method", "budget"), ("correct", "cost"))
    coverage_means = per_question_means(coverage_rows, ("question", "budget"), ("covered", "distinct", "cost"))
    methods = method_names(vision)
    results = {}
    for split, questions in split_ids.items():
        table = {}
        for budget in BUDGETS:
            row = {"oracle_coverage": float(np.mean([coverage_means[(q, budget)][0] for q in questions])),
                   "distinct_groups": float(np.mean([coverage_means[(q, budget)][1] for q in questions])),
                   "token_cost_all_candidates": float(np.mean([coverage_means[(q, budget)][2] for q in questions]))}
            for method in methods:
                row[method] = {"accuracy": float(np.mean([method_means[(q, method, budget)][0] for q in questions])),
                               "token_cost": float(np.mean([method_means[(q, method, budget)][1] for q in questions]))}
            table[str(budget)] = row
        results[split] = table

    eval_questions = split_ids["eval"]
    indices = np.random.default_rng(SEED).integers(len(eval_questions),
                                                    size=(BOOTSTRAP_RESAMPLES, len(eval_questions)))
    differences = {}
    for budget in BUDGETS:
        accuracy = {method: np.array([method_means[(q, method, budget)][0] for q in eval_questions])
                    for method in methods}
        differences[str(budget)] = {
            f"{method}_minus_{reference}": bootstrap_difference(accuracy[method], accuracy[reference], indices)
            for reference in ("sc", "first") for method in methods if method != reference}
    results["eval_bootstrap_differences"] = differences

    covered = {(row["question"], row["order"], row["budget"]): row["covered"] for row in coverage_rows}
    eval_set = set(eval_questions)
    counts = {}
    for row in selection_rows:
        if row["question"] not in eval_set:
            continue
        if row["correct"]:
            outcome = "correct"
        elif covered[(row["question"], row["order"], row["budget"])]:
            outcome = "correct_available_but_missed"
        else:
            outcome = "no_correct_candidate"
        per_question = counts.setdefault((row["method"], row["budget"]), {}).setdefault(row["question"], {})
        per_question[outcome] = per_question.get(outcome, 0) + 1
    decomposition = {}
    for (method, budget), per_question in counts.items():
        totals = {"correct": 0.0, "correct_available_but_missed": 0.0, "no_correct_candidate": 0.0}
        for question_counts in per_question.values():
            orders = sum(question_counts.values())
            for outcome, value in question_counts.items():
                totals[outcome] += value / orders
        decomposition.setdefault(method, {})[str(budget)] = {outcome: value / len(per_question)
                                                            for outcome, value in totals.items()}
    results["eval_failure_decomposition"] = decomposition
    return results, method_means


def quartile_gap(scores, correct):
    order = np.argsort(rankdata(-scores, method="ordinal"))
    quarter = len(scores) // 4
    return float(correct[order[:quarter]].mean() - correct[order[-quarter:]].mean())


def signal_analysis(prepared, questions):
    """N = 32 within-question associations between length, scores and correctness."""
    vision = DATASETS[prepared["dataset"]]["vision"]
    names = ["logp", "selfcert", "length"] + (["vision", "visconf", "tokenfusion"] if vision else [])
    gaps = {name: [] for name in names}
    correlations = {name: [] for name in names}
    length_correlations = {name: [] for name in names if name != "length"}
    confidence_vision = []
    covariance_terms = []
    everything = np.arange(ROLLOUTS)
    for question in questions:
        record = prepared["questions"][question]
        scores = {"logp": record["L"], "selfcert": record["K"], "length": record["tokens"].astype(np.float64)}
        if vision:
            visconf, tokenfusion, confidence = visconf_scores(record, everything)
            scores.update(vision=record["V"], visconf=visconf, tokenfusion=tokenfusion)
            covariance_terms.append(float(np.mean(np.abs(tokenfusion - visconf))))
            low, high = np.quantile(np.concatenate(record["token_kl"]), (0.05, 0.95))
            spread = max(high - low, 1e-8)
            per_rollout = []
            for kl_values, vis_values in zip(record["token_kl"], record["token_vis"]):
                if len(kl_values) < 3:
                    continue
                calibrated = np.clip((kl_values - low) / spread, 0, 1)
                if calibrated.std() == 0 or vis_values.std() == 0:
                    continue
                per_rollout.append(pearsonr(calibrated, vis_values)[0])
            if per_rollout:
                confidence_vision.append(float(np.mean(per_rollout)))
        correct = record["correct"].astype(np.float64)
        for name in names:
            gaps[name].append(quartile_gap(scores[name], correct))
            if 0 < correct.mean() < 1 and np.std(scores[name]) > 0:
                correlations[name].append(spearmanr(scores[name], correct)[0])
            if name != "length" and np.std(scores[name]) > 0:
                length_correlations[name].append(spearmanr(scores[name], scores["length"])[0])
    output = {
        "top_minus_bottom_quartile_correctness": {name: float(np.mean(values)) for name, values in gaps.items()},
        "within_question_spearman_with_correctness": {
            name: {"mean": float(np.mean(values)), "questions": len(values)} for name, values in correlations.items()},
        "within_question_spearman_with_length": {name: float(np.mean(values))
                                                 for name, values in length_correlations.items()},
        "note": "length gap uses longest-minus-shortest quartiles; shortest-rollout selection is the method 'shortest'",
    }
    if vision:
        values = np.array(confidence_vision)
        output["confidence_vision_correlation_per_question"] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75)),
            "fraction_positive": float((values > 0).mean()), "questions": len(values)}
        output["mean_abs_tokenfusion_minus_visconf"] = float(np.mean(covariance_terms))
    return output


def illustrative_trajectories(prepared, questions, count=3, points=120):
    """Selection rule: the first `count` held-out questions (in split order) where max-VisConf and SC disagree at
    N = 32; for each, the top-VisConf rollout and the earliest rollout of the SC answer."""
    from task3.selection import self_consistency

    examples = []
    everything = np.arange(ROLLOUTS)
    for question in questions:
        record = prepared["questions"][question]
        visconf, _, _ = visconf_scores(record, everything)
        top = int(np.argmax(visconf))
        sc_group = self_consistency(record["groups"])[0]
        if record["groups"][top] == sc_group:
            continue
        sc_rollout = record["groups"].index(sc_group)
        low, high = np.quantile(np.concatenate(record["token_kl"]), (0.05, 0.95))
        traces = []
        for label, rollout in (("max_visconf", top), ("self_consistency", sc_rollout)):
            calibrated = np.clip((record["token_kl"][rollout] - low) / max(high - low, 1e-8), 0, 1)
            vis = record["token_vis"][rollout]
            positions = np.linspace(0, len(vis) - 1, min(points, len(vis))).astype(int) if len(vis) else []
            traces.append({"role": label, "rollout": rollout, "answer": record["groups"][rollout],
                           "correct": bool(record["correct"][rollout]), "tokens": int(record["tokens"][rollout]),
                           "visconf": float(visconf[rollout]), "positions": [int(p) for p in positions],
                           "confidence": [float(calibrated[p]) for p in positions],
                           "vision": [float(vis[p]) for p in positions]})
        examples.append({"question": question, "gold": record["gold_label"], "traces": traces})
        if len(examples) == count:
            break
    return examples


def stop_point(rows_by_n, tau, gamma, quality):
    for checkpoint in STOP_CHECKS:
        features = rows_by_n[checkpoint]
        if not features["unique"] or features["support"] < tau:
            continue
        if quality and (not features["rwc_agrees"] or features["margin"] < gamma):
            continue
        return checkpoint
    return ROLLOUTS


def evaluate_policy(stopping_index, questions, tau, gamma, quality, fixed=False):
    accuracy, cost, rollouts, stops = [], [], [], {n: 0 for n in CHECKPOINTS}
    early_wrong = early_wrong_recoverable = early_total = 0
    for question in questions:
        question_accuracy, question_cost, question_rollouts = [], [], []
        for rows_by_n in stopping_index[question]:
            stop = ROLLOUTS if fixed else stop_point(rows_by_n, tau, gamma, quality)
            row = rows_by_n[stop]
            question_accuracy.append(row["correct"])
            question_cost.append(row["cost"])
            question_rollouts.append(stop)
            stops[stop] += 1
            if stop < ROLLOUTS:
                early_total += 1
                if not row["correct"]:
                    early_wrong += 1
                    early_wrong_recoverable += int(rows_by_n[ROLLOUTS]["correct"])
        accuracy.append(np.mean(question_accuracy))
        cost.append(np.mean(question_cost))
        rollouts.append(np.mean(question_rollouts))
    total = sum(stops.values())
    return {"accuracy": float(np.mean(accuracy)), "token_cost": float(np.mean(cost)),
            "mean_rollouts": float(np.mean(rollouts)),
            "stop_distribution": {str(n): stops[n] / total for n in CHECKPOINTS},
            "early_stops": early_total, "incorrect_early_stops": early_wrong,
            "incorrect_early_stops_fixed32_correct": early_wrong_recoverable,
            "per_question_accuracy": accuracy}


def adaptive_sampling(stopping, split_ids):
    index = {}
    for row in stopping:
        index.setdefault(row["question"], {}).setdefault(row["order"], {})[row["n"]] = row
    index = {question: [orders[order] for order in sorted(orders)] for question, orders in index.items()}
    dev, held_out = split_ids["dev"], split_ids["eval"]
    fixed_dev = evaluate_policy(index, dev, None, None, False, fixed=True)
    target = fixed_dev["accuracy"] - DEV_TOLERANCE
    families = {"agreement": [(tau, None) for tau in TAU_GRID],
                "quality_aware": [(tau, gamma) for tau in TAU_GRID for gamma in GAMMA_GRID]}
    output = {"dev_fixed32_sc": strip(fixed_dev), "dev_target_accuracy": target, "families": {}}
    held_fixed = evaluate_policy(index, held_out, None, None, False, fixed=True)
    rng_indices = np.random.default_rng(SEED).integers(len(held_out), size=(BOOTSTRAP_RESAMPLES, len(held_out)))
    for family, settings in families.items():
        grid = []
        for tau, gamma in settings:
            result = evaluate_policy(index, dev, tau, gamma, family == "quality_aware")
            grid.append({"tau": tau, "gamma": gamma, **strip(result)})
        grid.append({"tau": None, "gamma": None, "fallback_fixed32": True, **strip(fixed_dev)})
        eligible = [entry for entry in grid if entry["accuracy"] >= target - 1e-12]
        chosen = min(eligible, key=lambda entry: (entry["token_cost"], -entry["accuracy"],
                                                  entry["tau"] if entry["tau"] is not None else 2.0,
                                                  entry["gamma"] if entry["gamma"] is not None else 2.0))
        if chosen.get("fallback_fixed32"):
            held = held_fixed
        else:
            held = evaluate_policy(index, held_out, chosen["tau"], chosen["gamma"], family == "quality_aware")
        difference = bootstrap_difference(np.array(held["per_question_accuracy"]),
                                          np.array(held_fixed["per_question_accuracy"]), rng_indices)
        output["families"][family] = {"dev_grid": grid, "selected": {"tau": chosen["tau"], "gamma": chosen["gamma"]},
                                      "held_out": strip(held), "held_out_minus_fixed32_sc": difference}
    output["held_out_fixed32_sc"] = strip(held_fixed)
    return output


def strip(result):
    return {key: value for key, value in result.items() if key != "per_question_accuracy"}


def write_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(json.dumps(value) if isinstance(value, str) else str(value) for value in row) + "\n")


def analyze(name):
    global PREPARED
    started = time.time()
    PREPARED = load_prepared(name)
    vision = DATASETS[name]["vision"]
    output = results_root() / "offline" / name
    question_ids = sorted(PREPARED["questions"])
    selections, coverage, stopping = run_all(question_ids)
    print(f"{name}: evaluated {len(question_ids)} questions in {time.time() - started:.0f}s", flush=True)

    fixed, method_means = fixed_budget_results(selections, coverage, PREPARED["splits"], vision)
    save_json(output / "fixed_budget.json", fixed)
    write_rows(output / "selections.csv.gz", ("question", "order", "budget", "method", "selected_group", "correct",
                                              "token_cost"), selections)
    write_rows(output / "per_question.csv.gz", ("question", "method", "budget", "accuracy", "token_cost"),
               [(q, method, budget, values[0], values[1]) for (q, method, budget), values in method_means.items()])
    save_json(output / "signals.json", signal_analysis(PREPARED, PREPARED["splits"]["eval"]))
    if vision:
        save_json(output / "trajectories.json", illustrative_trajectories(PREPARED, PREPARED["splits"]["eval"]))
    save_json(output / "adaptive.json", adaptive_sampling(stopping, PREPARED["splits"]))
    save_json(output / "run_info.json", {"seconds": time.time() - started, "scorer": PREPARED["scorer"],
                                         "alignment": PREPARED["alignment"]})
    eval_table = fixed["eval"]["32"]
    print(name, {method: round(eval_table[method]["accuracy"], 4) for method in method_names(vision)},
          "coverage", round(eval_table["oracle_coverage"], 4), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    args = parser.parse_args()
    for name in ([args.dataset] if args.dataset else DATASETS):
        analyze(name)


if __name__ == "__main__":
    main()
