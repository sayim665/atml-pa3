"""Report figures and LaTeX tables for Task 1, built only from the committed results/task1 files.

    python -m task1.plots          (writes results/task1/figures/*.pdf|png and results/task1/tables/*.tex)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "task1"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

RUNS = ("full_ft", "l2sp", "lora4", "lora16", "oplora16")
CONDITIONS = ("baseline", "full_ft_mid") + RUNS
LABELS = {"baseline": "Baseline", "full_ft_mid": "Full FT (update 625)", "full_ft": "Full FT", "l2sp": "L2-SP",
          "lora4": "LoRA r=4", "lora16": "LoRA r=16", "oplora16": "OPLoRA r=16"}
COLORS = {"full_ft": "#2a78d6", "l2sp": "#eb6834", "lora4": "#1baf7a", "lora16": "#eda100", "oplora16": "#e87ba4",
          "full_ft_mid": "#86b6ef", "baseline": "#8a8984"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

BENCHMARKS = (("gsm8k", "exact_match,flexible-extract", "GSM8K"), ("hellaswag", "acc_norm,none", "HellaSwag"),
              ("boolq", "acc,none", "BoolQ"), ("arc_easy", "acc,none", "ARC-Easy"))
LAYERS = ("0", "13", "27")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.5, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    figure.savefig(FIGURES / f"{name}.png")
    plt.close(figure)


def scores():
    return {condition: load(RESULTS / "eval" / condition / "scores.json") for condition in CONDITIONS}


def score(table, condition, benchmark):
    for name, key, _ in BENCHMARKS:
        if name == benchmark:
            return 100 * table[condition][name][key]
    raise KeyError(benchmark)


def ema(values, weight=0.95):
    smoothed, running = [], values[0]
    for value in values:
        running = weight * running + (1 - weight) * value
        smoothed.append(running)
    return np.array(smoothed)


def training_curves():
    figure, (full, zoom) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": (1, 1.4)})
    for run in RUNS:
        rows = [json.loads(line) for line in open(RESULTS / "training" / "runs" / run / "train_log.jsonl",
                                                  encoding="utf-8")]
        updates = np.array([row["update"] for row in rows])
        smoothed = ema(np.array([row["sft_loss"] for row in rows]))
        dash = (0, (4, 2)) if run in ("l2sp", "oplora16") else "solid"
        full.plot(updates, smoothed, color=COLORS[run], linestyle=dash, label=LABELS[run])
        zoom.plot(updates[150:], smoothed[150:], color=COLORS[run], linestyle=dash, label=LABELS[run])
    full.set_title("All updates", color=INK)
    zoom.set_title("Updates 150-1250 (zoomed)", color=INK)
    zoom.axvline(625, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    zoom.text(632, zoom.get_ylim()[1], "midpoint (625)", color=INK_SECONDARY, fontsize=7, va="top")
    full.set_xlim(0, 1250)
    zoom.set_xlim(150, 1250)
    for axis in (full, zoom):
        axis.set_xlabel("Optimizer update (32 examples each)")
    full.set_ylabel("SFT loss, EMA 0.95")
    handles, labels = full.get_legend_handles_labels()
    figure.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7)
    figure.tight_layout()
    save(figure, "task1_training_loss")


def benchmark_deltas(table):
    conditions = ("full_ft_mid",) + RUNS
    figure, axes = plt.subplots(1, 4, figsize=(7.0, 2.4), sharey=False)
    for axis, (name, _, title) in zip(axes, BENCHMARKS):
        base = score(table, "baseline", name)
        deltas = [score(table, condition, name) - base for condition in conditions]
        positions = np.arange(len(conditions))
        axis.bar(positions, deltas, width=0.72, color=[COLORS[condition] for condition in conditions],
                 edgecolor=SURFACE, linewidth=1)
        axis.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        axis.set_title(f"{title} (baseline {base:.1f})", color=INK)
        axis.set_xticks(positions, [LABELS[condition].replace(" (update 625)", "\n@625") for condition in conditions],
                        rotation=60, ha="right", fontsize=6.5)
        low, high = min(deltas + [0]), max(deltas + [0])
        pad = 0.18 * (high - low + 1)
        axis.set_ylim(low - pad, high + pad)
    axes[0].set_ylabel("Change vs. baseline (points)")
    figure.tight_layout()
    save(figure, "task1_benchmark_deltas")


def adaptation_retention(table):
    figure, axis = plt.subplots(figsize=(3.6, 2.8))
    for condition in ("full_ft_mid",) + RUNS:
        gain = score(table, condition, "gsm8k") - score(table, "baseline", "gsm8k")
        retention = np.mean([score(table, condition, name) - score(table, "baseline", name)
                             for name in ("hellaswag", "boolq", "arc_easy")])
        axis.scatter(gain, retention, s=36, color=COLORS[condition], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        axis.annotate(LABELS[condition], (gain, retention), xytext=(5, 3), textcoords="offset points",
                      fontsize=7, color=INK)
    axis.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    axis.set_xlabel("GSM8K gain over baseline (points)")
    axis.set_ylabel("Mean change on HellaSwag,\nBoolQ, ARC-Easy (points)")
    axis.set_xlim(8, 24)
    save(figure, "task1_adaptation_vs_retention")


def diagnostics():
    weight_space = load(RESULTS / "diagnostics" / "weight_space.json")
    activations = load(RESULTS / "diagnostics" / "activation_ratios.json")
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    width = 0.16
    positions = np.arange(len(LAYERS))
    for index, run in enumerate(RUNS):
        offset = (index - 2) * width
        rho = [max(weight_space[run][layer]["rho_k"], 1e-9) for layer in LAYERS]
        ratio = [activations[run][layer]["math_over_non_math_mean"] for layer in LAYERS]
        left.bar(positions + offset, rho, width=width * 0.9, color=COLORS[run], label=LABELS[run])
        right.bar(positions + offset, ratio, width=width * 0.9, color=COLORS[run], label=LABELS[run])
    left.set_yscale("log")
    left.set_title(r"Update energy in protected subspace $\rho_k$ ($k=128$)", color=INK)
    left.set_ylabel(r"$\rho_k$ (log scale)")
    right.axhline(1, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    right.set_title("Activation change ratio, math / non-math prompts", color=INK)
    right.set_ylabel("Ratio of mean relative change")
    for axis in (left, right):
        axis.set_xticks(positions, [f"Layer {layer}\ndown_proj" for layer in LAYERS])
    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7)
    figure.tight_layout()
    save(figure, "task1_diagnostics")


def latex_scores(table):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Condition & GSM8K & HellaSwag & BoolQ & ARC-E & Trunc. (\%) & Resp. tokens \\", r"\midrule"]
    best = {name: max(score(table, condition, name) for condition in CONDITIONS[1:]) for name, _, _ in BENCHMARKS}
    for condition in CONDITIONS:
        cells = []
        for name, _, _ in BENCHMARKS:
            value = score(table, condition, name)
            text = f"{value:.1f}"
            cells.append(rf"\textbf{{{text}}}" if condition != "baseline" and value == best[name] else text)
        gsm8k = table[condition]["gsm8k"]
        cells.append(f"{100 * gsm8k['truncation_rate']:.1f}")
        cells.append(f"{gsm8k['mean_response_tokens']:.0f}")
        lines.append(f"{LABELS[condition]} & " + " & ".join(cells) + r" \\")
        if condition == "full_ft_mid":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_resources():
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Run & Trainable params & Train time (h) & Peak alloc. (GB) & Checkpoint (MB) \\", r"\midrule"]
    for run in RUNS:
        summary = load(RESULTS / "training" / "runs" / run / "train_summary.json")
        peak = max(segment["peak_memory_allocated_bytes"] for segment in summary["segments"]) / 2 ** 30
        checkpoint = max(entry["recovery_state_bytes"] for entry in summary["checkpoints"].values()) / 2 ** 20
        trainable = summary["trainable_parameters"]
        trainable_text = f"{trainable / 1e6:.1f}M" if trainable < 1e8 else f"{trainable / 1e6:.0f}M"
        lines.append(f"{LABELS[run]} & {trainable_text} & {summary['training_seconds'] / 3600:.2f} & "
                     f"{peak:.1f} & {checkpoint:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_diagnostics():
    weight_space = load(RESULTS / "diagnostics" / "weight_space.json")
    activations = load(RESULTS / "diagnostics" / "activation_ratios.json")
    lines = [r"\begin{tabular}{l" + "r" * 9 + "}", r"\toprule",
             r" & \multicolumn{3}{c}{$\rho_k$} & \multicolumn{3}{c}{$\eta$} & \multicolumn{3}{c}{Act. ratio} \\",
             r"Run & L0 & L13 & L27 & L0 & L13 & L27 & L0 & L13 & L27 \\", r"\midrule"]
    for run in ("full_ft_mid",) + RUNS:
        rho = [weight_space[run][layer]["rho_k"] for layer in LAYERS]
        eta = [weight_space[run][layer]["eta"] for layer in LAYERS]
        ratio = [activations[run][layer]["math_over_non_math_mean"] for layer in LAYERS]
        rho_text = [f"{value:.1e}" if value < 1e-3 else f"{value:.3f}" for value in rho]
        lines.append(f"{LABELS[run]} & " + " & ".join(rho_text + [f"{value:.3f}" for value in eta]
                                                      + [f"{value:.2f}" for value in ratio]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    style()
    table = scores()
    training_curves()
    benchmark_deltas(table)
    adaptation_retention(table)
    diagnostics()
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "task1_scores.tex").write_text(latex_scores(table), encoding="utf-8")
    (TABLES / "task1_resources.tex").write_text(latex_resources(), encoding="utf-8")
    (TABLES / "task1_diagnostics.tex").write_text(latex_diagnostics(), encoding="utf-8")
    print(f"figures in {FIGURES}, tables in {TABLES}")


if __name__ == "__main__":
    main()
