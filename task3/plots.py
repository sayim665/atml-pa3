"""Report figures and LaTeX tables for Task 3, built only from the committed results/task3 files.

    python -m task3.plots
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task1.plots import INK, INK_SECONDARY, SURFACE, style

RESULTS = Path(__file__).resolve().parent.parent / "results" / "task3"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
BUDGETS = ("1", "2", "4", "8", "16", "32")
DATASET_LABELS = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
METHOD_LABELS = {"first": "FirstAns", "sc": "Self-consistency", "max_logp": "Max log-prob",
                 "max_selfcert": "Max self-certainty", "selfcert_rwc": "Self-certainty + RWC",
                 "shortest": "Shortest", "max_vision": "Max vision", "vision_rwc": "Vision + RWC",
                 "max_visconf": "Max VisConf", "visconf_rwc": "VisConf + RWC",
                 "max_tokenfusion": "Max token fusion", "tokenfusion_rwc": "Token fusion + RWC"}
SERIES = {"sc": "#2a78d6", "first": "#eb6834", "max_selfcert": "#1baf7a", "selfcert_rwc": "#eda100",
          "visconf_rwc": "#e87ba4"}
COVERAGE = "#8a8984"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    figure.savefig(FIGURES / f"{name}.png")
    plt.close(figure)


def fixed(name):
    return load(RESULTS / "offline" / name / "fixed_budget.json")


def scaling_curves():
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    for axis, name in zip(axes, DATASET_LABELS):
        table = fixed(name)["eval"]
        budgets = np.array([int(budget) for budget in BUDGETS])
        axis.plot(budgets, [100 * table[b]["oracle_coverage"] for b in BUDGETS], color=COVERAGE, linestyle=(0, (4, 2)),
                  label="Oracle coverage")
        for method, color in SERIES.items():
            if method not in table["1"]:
                continue
            axis.plot(budgets, [100 * table[b][method]["accuracy"] for b in BUDGETS], color=color, marker="o",
                      markersize=3.5, label=METHOD_LABELS[method])
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, [str(b) for b in budgets])
        axis.set_xlabel("Candidates N")
        axis.set_title(DATASET_LABELS[name], color=INK)
    axes[0].set_ylabel("Held-out accuracy (%)")
    handles, labels = {}, {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(list(handles.values()), list(handles.keys()), ncol=6, loc="upper center",
                  bbox_to_anchor=(0.5, 1.08), fontsize=6.5)
    figure.tight_layout()
    save(figure, "task3_scaling")


def cost_curves():
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for axis, name in zip(axes, DATASET_LABELS):
        table = fixed(name)["eval"]
        adaptive = load(RESULTS / "offline" / name / "adaptive.json")
        cost = [table[b]["sc"]["token_cost"] / 1000 for b in BUDGETS]
        axis.plot(cost, [100 * table[b]["sc"]["accuracy"] for b in BUDGETS], color=SERIES["sc"], marker="o",
                  markersize=3.5, label="Fixed-N self-consistency")
        for budget, x, y in zip(BUDGETS, cost, [100 * table[b]["sc"]["accuracy"] for b in BUDGETS]):
            axis.annotate(f"N={budget}", (x, y), xytext=(3, -9), textcoords="offset points", fontsize=6,
                          color=INK_SECONDARY)
        markers = {"agreement": ("s", "#eb6834", "Adaptive: agreement"),
                   "quality_aware": ("D", "#1baf7a", "Adaptive: quality-aware")}
        for family, (marker, color, label) in markers.items():
            held = adaptive["families"][family]["held_out"]
            axis.scatter([held["token_cost"] / 1000], [100 * held["accuracy"]], marker=marker, s=34, color=color,
                         edgecolor=SURFACE, linewidth=1.2, zorder=4, label=label)
        axis.set_xscale("log")
        axis.set_xlabel("Mean generated tokens per question (thousands, log)")
        axis.set_title(DATASET_LABELS[name], color=INK)
    axes[0].set_ylabel("Held-out accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7)
    figure.tight_layout()
    save(figure, "task3_accuracy_vs_cost")


def failure_decomposition():
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    parts = (("correct", "#2a78d6", "Correct"), ("correct_available_but_missed", "#eda100", "Correct available, missed"),
             ("no_correct_candidate", "#c3c2b7", "No correct candidate"))
    for axis, name in zip(axes, DATASET_LABELS):
        decomposition = fixed(name)["eval_failure_decomposition"]["sc"]
        bottom = np.zeros(len(BUDGETS))
        for key, color, label in parts:
            values = np.array([100 * decomposition[b][key] for b in BUDGETS])
            axis.bar(range(len(BUDGETS)), values, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.5,
                     width=0.7, label=label)
            bottom += values
        axis.set_xticks(range(len(BUDGETS)), [f"N={b}" for b in BUDGETS])
        axis.set_title(f"{DATASET_LABELS[name]}: self-consistency", color=INK)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Share of held-out questions (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7)
    figure.tight_layout()
    save(figure, "task3_failure_decomposition")


def trajectories():
    path = RESULTS / "offline" / "mathvista" / "trajectories.json"
    if not path.exists():
        return
    examples = load(path)
    figure, axes = plt.subplots(len(examples), 2, figsize=(7.0, 1.7 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        for column, trace in enumerate(example["traces"]):
            axis = axes[row][column]
            axis.plot(trace["positions"], trace["confidence"], color="#2a78d6", linewidth=1, label="Calibrated confidence c")
            vision = np.array(trace["vision"])
            scale = vision.max() if vision.size and vision.max() > 0 else 1
            axis.plot(trace["positions"], vision / scale, color="#eb6834", linewidth=1,
                      label="Visual engagement v (scaled to max)")
            verdict = "correct" if trace["correct"] else "wrong"
            axis.set_title(f"Q{example['question']} {trace['role'].replace('_', ' ')}: answer {trace['answer']} "
                           f"({verdict}), Z={trace['visconf']:.3f}", fontsize=7, color=INK)
            axis.set_ylim(0, 1.05)
            if row == len(examples) - 1:
                axis.set_xlabel("Generated token position")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=7)
    figure.tight_layout()
    save(figure, "task3_trajectories")


def pct(value):
    return f"{100 * value:.1f}"


def interval(entry):
    low, high = entry["ci95"]
    return f"{100 * entry['mean']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"


def method_table(name):
    results = fixed(name)
    table = results["eval"]
    differences = results["eval_bootstrap_differences"]
    methods = [method for method in METHOD_LABELS if method in table["1"]]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & $N{=}8$ & $N{=}32$ & $\Delta$ vs SC, $N{=}32$ (95\% CI) & Tokens, $N{=}32$ \\", r"\midrule"]
    for method in methods:
        delta = "--" if method == "sc" else interval(differences["32"][f"{method}_minus_sc"])
        lines.append(f"{METHOD_LABELS[method]} & {pct(table['8'][method]['accuracy'])} & "
                     f"{pct(table['32'][method]['accuracy'])} & {delta} & "
                     f"{table['32'][method]['token_cost']:.0f}" + r" \\")
    lines += [r"\midrule", f"Oracle coverage & {pct(table['8']['oracle_coverage'])} & {pct(table['32']['oracle_coverage'])}"
              r" & -- & -- \\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def adaptive_table():
    lines = [r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"Dataset & Policy & $\tau$ / $\gamma$ & Acc. & Tokens & Rollouts & Stops at 4/8/16/32 (\%) \\",
             r"\midrule"]
    for name in DATASET_LABELS:
        adaptive = load(RESULTS / "offline" / name / "adaptive.json")
        fixed32 = adaptive["held_out_fixed32_sc"]
        lines.append(f"{DATASET_LABELS[name]} & Fixed-32 SC & -- & {pct(fixed32['accuracy'])} & "
                     f"{fixed32['token_cost']:.0f} & 32 & -- " + r"\\")
        for family, label in (("agreement", "Agreement"), ("quality_aware", "Quality-aware")):
            entry = adaptive["families"][family]
            held = entry["held_out"]
            setting = entry["selected"]
            parameters = f"{setting['tau']}" + (f" / {setting['gamma']}" if setting["gamma"] is not None else "")
            stops = "/".join(f"{100 * held['stop_distribution'][n]:.0f}" for n in ("4", "8", "16", "32"))
            lines.append(f" & {label} & {parameters} & {pct(held['accuracy'])} & {held['token_cost']:.0f} & "
                         f"{held['mean_rollouts']:.1f} & {stops} " + r"\\")
        lines.append(r"\midrule" if name == "math" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def live_table():
    path = RESULTS / "live" / "summary.json"
    if not path.exists():
        return None
    summary = load(path)
    single, parallel, refinement = summary["single"], summary["parallel"], summary["refinement"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Strategy & Acc. (\%) & Output tok. & Input tok. & Time (s) & Trunc. / extr. (\%) \\", r"\midrule",
             f"Single (3,072) & {pct(single['accuracy'])} & {single['mean_output_tokens']:.0f} & "
             f"{single['mean_input_tokens']:.0f} & {single['mean_seconds']:.1f} & "
             f"{pct(single['truncation_rate'])} / {pct(single['extraction_rate'])}" + r" \\",
             f"Parallel 3$\\times$1,024 + SC & {pct(parallel['accuracy'])} & {parallel['mean_output_tokens']:.0f} & "
             f"{parallel['mean_input_tokens']:.0f} & {parallel['mean_seconds']:.1f} & "
             f"{pct(parallel['truncation_rate_per_response'])} / {pct(parallel['extraction_rate_per_response'])}" + r" \\",
             f"Refinement 1,024 + 2,048 & {pct(refinement['accuracy'])} & {refinement['mean_output_tokens']:.0f} & "
             f"{refinement['mean_input_tokens']:.0f} & {refinement['mean_seconds']:.1f} & "
             f"{pct(refinement['revision_truncation_rate'])} / {pct(refinement['revision_extraction_rate'])}" + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    style()
    scaling_curves()
    cost_curves()
    failure_decomposition()
    trajectories()
    TABLES.mkdir(parents=True, exist_ok=True)
    for name in DATASET_LABELS:
        (TABLES / f"task3_methods_{name}.tex").write_text(method_table(name), encoding="utf-8")
    (TABLES / "task3_adaptive.tex").write_text(adaptive_table(), encoding="utf-8")
    live = live_table()
    if live:
        (TABLES / "task3_live.tex").write_text(live, encoding="utf-8")
    print(f"figures in {FIGURES}, tables in {TABLES}")


if __name__ == "__main__":
    main()
