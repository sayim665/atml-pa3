"""Representative Task 3 cases with fixed selection rules (selection disagreement, premature stopping, revision).

    python -m task3.examples            (offline cases need outputs/task3/prepared_*.pkl; writes results/task3/examples/)
    python -m task3.examples --live     (revision cases only, from results/task3/live)

Selection rules (stated in each output file):
- disagreement: the first 3 held-out questions in split order where, with all 32 rollouts, self-consistency and the
  quality-weighted RWC winner (self-certainty for MATH, VisConf for MathVista) differ.
- premature stop: the first 3 held-out questions in split order where the selected quality-aware policy stops early in
  order 0 with a wrong answer while fixed-32 self-consistency is correct.
- revision: every wrong-to-correct transition, then the first 2 questions whose revised answer differs from the initial
  answer without becoming correct, then the first question whose initial response was truncated.
"""

import argparse

import numpy as np

from task1.common import load_json, save_json
from task3.config import DATASETS, ROLLOUTS, STOP_CHECKS, results_root
from task3.selection import rwc, self_consistency, stopping_features, visconf_scores

CASES = 3
TEXT_CHARS = 700


def quality_scores(record, candidates, vision):
    if vision:
        return visconf_scores(record, candidates)[0]
    return record["K"][candidates]


def group_table(record, candidates, scores):
    groups = [record["groups"][candidate] for candidate in candidates]
    _, votes = rwc(groups, scores)
    table = {}
    for position, (candidate, group) in enumerate(zip(candidates, groups)):
        entry = table.setdefault(group, {"answer": group, "count": 0, "correct": bool(record["correct"][candidate]),
                                         "scores": [], "first_position": position})
        entry["count"] += 1
        entry["scores"].append(float(scores[position]))
    rows = []
    for entry in table.values():
        rows.append({"answer": entry["answer"], "count": entry["count"], "correct": entry["correct"],
                     "mean_quality": float(np.mean(entry["scores"])), "rwc_vote": round(votes[entry["answer"]], 4)})
    return sorted(rows, key=lambda row: (-row["count"], -row["rwc_vote"]))


def question_header(record, question):
    text = record["question_text"] or ""
    return {"question": question, "text": text[:TEXT_CHARS] + ("..." if len(text) > TEXT_CHARS else ""),
            "gold": record["gold_label"]}


def disagreement_cases(prepared, vision):
    everything = np.arange(ROLLOUTS)
    cases = []
    for question in prepared["splits"]["eval"]:
        record = prepared["questions"][question]
        scores = quality_scores(record, everything, vision)
        sc_answer = self_consistency(record["groups"])[0]
        rwc_answer = rwc(record["groups"], scores)[0]
        if sc_answer == rwc_answer:
            continue
        correct = dict(zip(record["groups"], record["correct"]))
        cases.append({**question_header(record, question), "self_consistency": sc_answer,
                      "self_consistency_correct": bool(correct[sc_answer]), "quality_rwc": rwc_answer,
                      "quality_rwc_correct": bool(correct[rwc_answer]),
                      "groups": group_table(record, everything, scores)})
        if len(cases) == CASES:
            break
    return cases


def premature_stop_cases(prepared, vision, tau, gamma):
    cases = []
    for question in prepared["splits"]["eval"]:
        record = prepared["questions"][question]
        order = prepared["orders"][question][0]
        stop = ROLLOUTS
        for checkpoint in STOP_CHECKS:
            features = stopping_features(record, order[:checkpoint], vision)
            if features["unique"] and features["support"] >= tau and features["rwc_agrees"] and features["margin"] >= gamma:
                stop = checkpoint
                break
        if stop == ROLLOUTS:
            continue
        correct = dict(zip(record["groups"], record["correct"]))
        stopped = stopping_features(record, order[:stop], vision)
        full = self_consistency(record["groups"])[0]
        if correct[stopped["sc"]] or not correct[full]:
            continue
        revealed = order[:stop]
        cases.append({**question_header(record, question), "order": 0, "stopped_at": int(stop),
                      "answer_at_stop": stopped["sc"], "support_at_stop": stopped["support"],
                      "rwc_margin_at_stop": stopped["margin"], "fixed32_answer": full,
                      "revealed_groups": group_table(record, revealed, quality_scores(record, revealed, vision)),
                      "all_32_groups": group_table(record, np.arange(ROLLOUTS),
                                                   quality_scores(record, np.arange(ROLLOUTS), vision))})
        if len(cases) == CASES:
            break
    return cases


def offline_examples():
    from task3.data import load_prepared

    for name in DATASETS:
        prepared = load_prepared(name)
        vision = DATASETS[name]["vision"]
        adaptive = load_json(results_root() / "offline" / name / "adaptive.json")
        selected = adaptive["families"]["quality_aware"]["selected"]
        tau = selected["tau"] if selected["tau"] is not None else 2.0
        gamma = selected["gamma"] if selected["gamma"] is not None else 0.0
        save_json(results_root() / "examples" / f"{name}_cases.json", {
            "rules": __doc__.split("Selection rules (stated in each output file):")[1].strip(),
            "quality_signal": "VisConf" if vision else "self-certainty",
            "selection_disagreement": disagreement_cases(prepared, vision),
            "premature_stop": {"policy": {"tau": tau, "gamma": gamma},
                               "cases": premature_stop_cases(prepared, vision, tau, gamma)},
        })
        print(f"{name}: examples written", flush=True)


def excerpt(text, head=350, tail=250):
    if len(text) <= head + tail:
        return text
    return text[:head] + " [...] " + text[-tail:]


def live_examples():
    per_question = load_json(results_root() / "live" / "per_question.json")
    generations = {}
    for line in open(results_root() / "live" / "generations.jsonl", encoding="utf-8"):
        row = __import__("json").loads(line)
        generations[(row["question_index"], row["strategy"], row["call"])] = row
    questions = load_json(results_root() / "splits" / "math_live_questions.json")
    chosen, reasons = [], {}
    for row in per_question:
        if row["refinement"]["transition"] == "wrong_to_correct":
            chosen.append(row)
            reasons[row["question_index"]] = "wrong_to_correct"
    changed = [row for row in per_question if row["refinement"]["answer_changed"]
               and not row["refinement"]["correct"] and row["question_index"] not in reasons][:2]
    for row in changed:
        chosen.append(row)
        reasons[row["question_index"]] = "changed_but_still_wrong"
    truncated = [row for row in per_question if row["refinement"]["initial_truncated"]
                 and row["question_index"] not in reasons][:1]
    for row in truncated:
        chosen.append(row)
        reasons[row["question_index"]] = "initial_truncated"
    cases = []
    for row in chosen:
        index = row["question_index"]
        initial = generations[(index, "parallel", 0)]
        revision = generations[(index, "revision", 0)]
        cases.append({"reason": reasons[index], "question": questions[index]["question"][:TEXT_CHARS],
                      "gold": row["gold"], "initial_answer": row["refinement"]["initial_answer"],
                      "revised_answer": row["refinement"]["answer"], "transition": row["refinement"]["transition"],
                      "single_answer": row["single"]["answer"], "parallel_answers": row["parallel"]["answers"],
                      "initial_tokens": initial["output_tokens"], "initial_truncated": initial["truncated"],
                      "revision_tokens": revision["output_tokens"],
                      "initial_excerpt": excerpt(initial["response"]), "revision_excerpt": excerpt(revision["response"])})
    save_json(results_root() / "examples" / "live_revision_cases.json",
              {"rules": "every wrong-to-correct transition; then the first 2 questions whose revised answer differs "
                        "from the initial answer without becoming correct; then the first question whose initial "
                        "1,024-token response was truncated", "cases": cases})
    print(f"live: {len(cases)} revision examples written", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.live:
        live_examples()
    else:
        offline_examples()


if __name__ == "__main__":
    main()
