"""Report figures and LaTeX tables for Task 2, built from the committed results/task2 files.

    python -m task2.plots          (the qualitative figure downloads its few COCO val2014 images to outputs/)
"""

import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task1.plots import INK, INK_SECONDARY, SURFACE, ema, style

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "task2"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
IMAGES = REPO / "outputs" / "task2_qualitative_images"
RUNS = {"vlm196": "196 tokens", "vlm49": "49 tokens"}
RUN_COLORS = {"vlm196": "#2a78d6", "vlm49": "#eb6834"}
CONDITIONS = ("correct", "complementary", "mismatched", "none")
CONDITION_LABELS = {"correct": "Correct image", "complementary": "Complementary swap", "mismatched": "Mismatched",
                    "none": "No image"}
CONDITION_COLORS = {"correct": "#2a78d6", "complementary": "#eb6834", "mismatched": "#1baf7a", "none": "#8a8984"}
SUITES = {"grounding": "Annotation-derived grounding", "vqa": "Natural VQAv2"}
STAGE_ENDS = (2250, 3750)
STAGE_NAMES = (("Captions", 1125), ("Instruction", 3000), ("VQA", 4125))
COCO_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    figure.savefig(FIGURES / f"{name}.png")
    plt.close(figure)


def stage_marks(axis, label=False):
    for end in STAGE_ENDS:
        axis.axvline(end, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    if label:
        top = axis.get_ylim()[1]
        for name, center in STAGE_NAMES:
            axis.text(center, top, name, ha="center", va="top", fontsize=6.5, color=INK_SECONDARY)


def learning_curves():
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    train_axis, caption_axis, delta_axis = axes[0]
    grounding_axis, vqa_axis, accuracy_axis = axes[1]
    for run, label in RUNS.items():
        rows = read_jsonl(RESULTS / "training" / run / "train_log.jsonl")
        updates = np.array([row["update"] for row in rows])
        losses = np.array([row["loss"] for row in rows])
        train_axis.plot(updates, ema(losses, 0.98), color=RUN_COLORS[run], label=label)
        dev = read_jsonl(RESULTS / "training" / run / "dev_log.jsonl")
        dev_updates = [row["update"] for row in dev]
        caption_axis.plot(dev_updates, [row["caption_loss"] for row in dev], color=RUN_COLORS[run], marker="o",
                          markersize=3, label=label)
        for suite, dash in (("grounding", "solid"), ("vqa", (0, (4, 2)))):
            delta_axis.plot(dev_updates, [row[suite]["delta_nll"] for row in dev], color=RUN_COLORS[run],
                            linestyle=dash, marker="o", markersize=3,
                            label=f"{label}, {'grounding' if suite == 'grounding' else 'VQAv2'}")
    dev = read_jsonl(RESULTS / "training" / "vlm196" / "dev_log.jsonl")
    dev_updates = [row["update"] for row in dev]
    for axis, suite in ((grounding_axis, "grounding"), (vqa_axis, "vqa")):
        for condition in ("correct", "complementary", "none"):
            axis.plot(dev_updates, [row[suite][f"nll_{condition}"] for row in dev], color=CONDITION_COLORS[condition],
                      marker="o", markersize=3, label=CONDITION_LABELS[condition])
        axis.set_title(f"Dev answer NLL, {SUITES[suite].split()[-1]} (196)", color=INK)
        axis.set_ylabel("Reference-answer NLL")
    train_axis.set_title("Training loss (EMA 0.98)", color=INK)
    caption_axis.set_title("Dev caption loss", color=INK)
    delta_axis.set_title(r"Image-swap penalty $\Delta$NLL", color=INK)
    delta_axis.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    train_axis.set_ylabel("Loss")
    caption_axis.set_ylabel("Caption NLL")
    delta_axis.set_ylabel(r"NLL(partner) $-$ NLL(correct)")
    for axis in (train_axis, caption_axis, delta_axis, grounding_axis, vqa_axis):
        axis.set_xlim(0, 4500)
        axis.set_xticks([0, 1500, 3000, 4500])
        stage_marks(axis, label=axis is train_axis)
    for axis in (grounding_axis, vqa_axis):
        axis.set_xlabel("Main-run update")
    train_axis.legend(fontsize=6.5, loc="center right")
    delta_axis.legend(fontsize=5.8, loc="upper left")
    grounding_axis.legend(fontsize=6, loc="upper right")

    metrics = {run: load(RESULTS / "eval" / run / "metrics.json") for run in RUNS}
    positions = np.arange(len(CONDITIONS))
    width = 0.2
    for offset_index, (run, suite) in enumerate([(run, suite) for suite in SUITES for run in RUNS]):
        values = [100 * metrics[run][suite][condition]["accuracy"] for condition in CONDITIONS]
        low = [100 * metrics[run][suite][condition]["accuracy_ci95"][0] for condition in CONDITIONS]
        high = [100 * metrics[run][suite][condition]["accuracy_ci95"][1] for condition in CONDITIONS]
        errors = np.array([np.subtract(values, low), np.subtract(high, values)])
        offset = (offset_index - 1.5) * width
        accuracy_axis.bar(positions + offset, values, width=width * 0.92, color=RUN_COLORS[run],
                          alpha=1.0 if suite == "grounding" else 0.55, edgecolor=SURFACE, linewidth=0.8,
                          label=f"{RUNS[run]}, {'grounding' if suite == 'grounding' else 'VQAv2'}")
        accuracy_axis.errorbar(positions + offset, values, yerr=errors, fmt="none", ecolor=INK_SECONDARY,
                               elinewidth=0.7, capsize=1.5)
    accuracy_axis.set_xticks(positions, ["Correct", "Swap", "Mism.", "None"], fontsize=7)
    accuracy_axis.set_title("Final-test accuracy (95% CI)", color=INK)
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_xlabel("Image condition")
    accuracy_axis.legend(fontsize=5.6, loc="upper right")
    figure.tight_layout()
    save(figure, "task2_learning_curves")


def fetch_image(key):
    split, image_id = key.split("/")
    path = IMAGES / f"{image_id}.jpg"
    if not path.exists():
        IMAGES.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COCO_URL.format(image_id=int(image_id)), path)
    return path


def qualitative_pairs():
    """Selection rule: the first pair of each question type in the outcome-independent seeded sample saved by
    task2.evaluate (qualitative_*.json), for presence, counting, VQA yes/no and VQA other."""
    chosen = []
    for suite, kinds in (("grounding", ("presence", "counting")), ("vqa", ("yes/no", "other"))):
        pairs = load(RESULTS / "eval" / "vlm196" / f"qualitative_{suite}.json")
        for kind in kinds:
            chosen.append(next(pair for pair in pairs if pair["question_type"] == kind))
    return chosen


def qualitative_figure():
    from PIL import Image

    pairs = qualitative_pairs()
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.6))
    for column, pair in enumerate(pairs):
        for row, side in enumerate(pair["sides"]):
            axis = axes[row][column]
            with Image.open(fetch_image(side["image"])) as image:
                axis.imshow(image.convert("RGB").resize((224, 224)))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
            predictions = side["predictions"]
            axis.set_xlabel(f"ref: {side['reference']}\ncorrect img: {predictions['correct']} | "
                            f"swap: {predictions['complementary']}\nno img: {predictions['none']}",
                            fontsize=6.2, color=INK)
        question = pair["question"] if len(pair["question"]) < 46 else pair["question"][:43] + "..."
        axes[0][column].set_title(f"[{pair['question_type']}]\n{question}", fontsize=6.6, color=INK)
    figure.tight_layout()
    save(figure, "task2_qualitative")


def pct(value):
    return f"{100 * value:.1f}"


def results_table():
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Suite & Condition & \multicolumn{2}{c}{Accuracy (\%)} & \multicolumn{2}{c}{Pair acc. (\%)} & "
             r"\multicolumn{2}{c}{Ref. NLL} \\",
             r" & & 196 & 49 & 196 & 49 & 196 & 49 \\", r"\midrule"]
    metrics = {run: load(RESULTS / "eval" / run / "metrics.json") for run in RUNS}
    for suite, suite_label in SUITES.items():
        for index, condition in enumerate(CONDITIONS):
            cells = [suite_label if index == 0 else "", CONDITION_LABELS[condition]]
            cells += [pct(metrics[run][suite][condition]["accuracy"]) for run in RUNS]
            cells += [pct(metrics[run][suite][condition]["pair_accuracy"]) for run in RUNS]
            cells += [f"{metrics[run][suite][condition]['reference_nll']:.2f}" for run in RUNS]
            lines.append(" & ".join(cells) + r" \\")
        advantage = [metrics[run][suite]["advantages"]["correct_vs_complementary"] for run in RUNS]
        lines.append(r" & Correct $-$ swap & " + " & ".join(
            f"{100 * entry['accuracy']:+.1f} [{100 * entry['accuracy_ci95'][0]:.1f}, {100 * entry['accuracy_ci95'][1]:.1f}]"
            for entry in advantage) + r" & & & " + " & ".join(
            f"{entry['delta_nll']:+.2f}" for entry in advantage) + r" \\")
        lines.append(r"\midrule" if suite == "grounding" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def type_table():
    metrics = {run: load(RESULTS / "eval" / run / "metrics.json") for run in RUNS}
    kinds = (("grounding", "presence"), ("grounding", "counting"), ("vqa", "yes/no"), ("vqa", "number"),
             ("vqa", "other"))
    lines = [r"\begin{tabular}{l" + "r" * 5 + "}", r"\toprule",
             r"Condition (196 tokens) & Presence & Counting & Yes/no & Number & Other \\", r"\midrule"]
    for condition in CONDITIONS:
        cells = [pct(metrics["vlm196"][suite][condition]["by_type"][kind]["accuracy"]) for suite, kind in kinds]
        lines.append(f"{CONDITION_LABELS[condition]} & " + " & ".join(cells) + r" \\")
    follow = [metrics["vlm196"][suite]["complementary"]["follows_partner_answer_rate"] for suite in SUITES]
    lines += [r"\midrule", f"Swap follows partner answer & \\multicolumn{{2}}{{c}}{{{pct(follow[0])}}} & "
              f"\\multicolumn{{3}}{{c}}{{{pct(follow[1])}}}" + r" \\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def resources_table():
    cache = load(RESULTS / "features" / "cache_info.json")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Run & Train (h) & s/update (stage 1/2/3) & Peak GPU (GB) & Eval (s) & Gen. ms/answer \\", r"\midrule"]
    for run, label in RUNS.items():
        summary = load(RESULTS / "training" / run / "train_summary.json")
        rows = read_jsonl(RESULTS / "training" / run / "train_log.jsonl")
        per_stage = [np.median([row["seconds"] for row in rows if row["stage"] == stage])
                     for stage in ("caption", "instruction", "vqa")]
        metrics = load(RESULTS / "eval" / run / "metrics.json")
        generation = np.mean([metrics[suite]["correct"]["generation_seconds_per_answer"] for suite in SUITES]) * 1000
        lines.append(f"{label} & {summary['training_seconds'] / 3600:.2f} & "
                     + "/".join(f"{value:.2f}" for value in per_stage)
                     + f" & {summary['peak_gpu_memory_gb_this_session']:.2f} & {metrics['evaluation_seconds']:.0f} & "
                       f"{generation:.0f}" + r" \\")
    counts = load(RESULTS / "training" / "vlm196" / "train_summary.json")["trainable_parameters_by_stage"]
    lines += [r"\midrule",
              f"Feature cache (9,373 images) & {cache['total_seconds'] / 3600:.2f} & -- & "
              f"{cache['peak_gpu_memory_gb']:.2f} & -- & --" + r" \\", r"\bottomrule", r"\end{tabular}",
              f"% trainable parameters: stage 1 connector {counts['caption']['connector']:,}; "
              f"stages 2-3 connector {counts['instruction']['connector']:,} + LoRA {counts['instruction']['lora']:,}"]
    return "\n".join(lines) + "\n"


def main():
    style()
    learning_curves()
    qualitative_figure()
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "task2_results.tex").write_text(results_table(), encoding="utf-8")
    (TABLES / "task2_question_types.tex").write_text(type_table(), encoding="utf-8")
    (TABLES / "task2_resources.tex").write_text(resources_table(), encoding="utf-8")
    print(f"figures in {FIGURES}, tables in {TABLES}")


if __name__ == "__main__":
    main()
